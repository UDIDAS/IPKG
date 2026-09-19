#!/usr/bin/env python3
"""Stage 1 of the repair / closed-loop experiments: fully-autonomous SAM3 segmentation of the FLARE23
full-label cases (concept-prompted organs via base SAM3 + generic text-'tumor' model, NO boxes, tumor
kept only inside an organ — the pipeline of infer_ensemble.py / kg_guided_eval.py), cached to disk so
that repair (Tab. 12), the post-processing control (Tab. 12) and the closed-loop rebuild (Tab. 13) are
CPU-only passes over the same raw masks.

Output per case ($VKG_DATA/autonomous_raw/):
  {cid}.nii.gz            raw multi-label mask (organs 1/2/3/4/13, tumor 14 after the inside-organ rule)
  {cid}_tumor_raw.nii.gz  tumor model output BEFORE the inside-organ rule (for ablations)
  {cid}.json              raw Dice vs GT per structure + run metadata

Sharded: `--shard k/N` processes cids[k::N]; run one process per GPU (see run_autonomous_shards.sh).
Slices are batched (seg_concept_batched) — verified equal to the single-slice path (Dice 0.9999).
"""
import argparse
import glob
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sam3_autonomous_local as IE

TUMOR = IE.TUMOR_LABEL
STRUCTS = [("liver", 1), ("right_kidney", 2), ("spleen", 3), ("pancreas", 4), ("left_kidney", 13), ("tumor", TUMOR)]


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="k/N: process cids[k::N]")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(IE.VKG_DATA, "autonomous_raw"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    os.makedirs(a.out, exist_ok=True)

    cids = sorted(os.path.basename(f)[:-len("_ct.nii.gz")] for f in glob.glob(f"{IE.CASES}/*_ct.nii.gz"))
    cids = cids[k::n][:a.limit]
    todo = [c for c in cids if not os.path.exists(f"{a.out}/{c}.json")]
    print(f"shard {k}/{n}: {len(cids)} cases, {len(todo)} to do", flush=True)
    if not todo:
        return
    model, proc = IE.load_base()
    tmodel = IE._load_sam3_ckpt(IE.TUMOR_CKPT, "cuda")

    for i, cid in enumerate(todo):
        t0 = time.time()
        ct_nii = nib.load(f"{IE.CASES}/{cid}_ct.nii.gz"); ct = ct_nii.get_fdata()
        gt = np.asarray(nib.load(f"{IE.CASES}/{cid}_label.nii.gz").dataobj).astype(np.uint8)
        Z = ct.shape[2]
        rgb = [IE.hu_to_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True),
                            *IE.WIN).astype(np.uint8) for z in range(Z)]
        pred = np.zeros(ct.shape, np.uint8)
        for concept, lab in IE.ORGANS:
            m = IE.seg_concept_batched(ct, model, proc, concept, batch=a.batch, rgb_cache=rgb)
            pred[m > 0] = lab
        tm = IE.seg_concept_batched(ct, tmodel, proc, "tumor", batch=a.batch, rgb_cache=rgb)
        pred[(tm > 0) & (pred > 0)] = TUMOR                  # baseline pipeline: tumor kept inside an organ

        nib.save(nib.Nifti1Image(pred, ct_nii.affine), f"{a.out}/{cid}.nii.gz")
        nib.save(nib.Nifti1Image(tm.astype(np.uint8), ct_nii.affine), f"{a.out}/{cid}_tumor_raw.nii.gz")
        rec = {"cid": cid, "shape": list(ct.shape), "spacing": [float(s) for s in ct_nii.header.get_zooms()[:3]],
               "dice_raw": {s: dice(pred == lab, gt == lab) for s, lab in STRUCTS if (gt == lab).any()},
               "vox": {s: int((pred == lab).sum()) for s, lab in STRUCTS},
               "tumor_vox_before_organ_rule": int((tm > 0).sum()),
               "seconds": round(time.time() - t0, 1), "batch": a.batch, "gpu": torch.cuda.get_device_name(0)}
        json.dump(rec, open(f"{a.out}/{cid}.json", "w"), indent=1)
        print(f"[{i+1}/{len(todo)}] {cid} {rec['seconds']}s  " +
              "  ".join(f"{s} {v:.2f}" for s, v in rec["dice_raw"].items() if v is not None), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
