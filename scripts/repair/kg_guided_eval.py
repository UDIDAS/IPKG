#!/usr/bin/env python3
"""Quantify KG-guided segmentation: does the KG atlas repair improve autonomous masks?

For each full FLARE case: run fully-autonomous segmentation (SAM3 concept organs + generic tumor model,
no boxes) -> RAW mask; apply the KG-guided repair -> REPAIRED mask; score both per structure against GT.
Reports mean Dice raw vs repaired. Writes results/repair/kg_guided_eval.json.

Raw masks are read from the cache written by scripts/segmentation/autonomous_infer_flare.py
($VKG_DATA/autonomous_raw/{cid}.nii.gz) when present — then this is a CPU-only pass; otherwise the
SAM3 models are loaded and inference runs inline (the original behaviour).
"""
import glob
import json
import os
import sys

import numpy as np
import nibabel as nib
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "segmentation"))
import sam3_autonomous_local as IE                      # bundle-local vendored infer_ensemble
from sam3_autonomous_local import _load_sam3_ckpt, CASES, TUMOR_CKPT   # $VKG_DATA-rooted
from kg_guided_segment import repair, ORGAN_LABELS, TUMOR

RAW_CACHE = os.path.join(IE.VKG_DATA, "autonomous_raw")
OUT = os.environ.get("VKG_OUT", os.path.join(_HERE, "..", "..", "results", "repair", "kg_guided_eval.json"))
STRUCTS = [("liver", 1), ("right_kidney", 2), ("spleen", 3), ("pancreas", 4), ("left_kidney", 13), ("tumor", TUMOR)]


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else None


def main():
    cids = sorted(os.path.basename(f)[:-len("_ct.nii.gz")] for f in glob.glob(f"{CASES}/*_ct.nii.gz"))
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(cids)
    cids = cids[:n]
    print(f"KG-guided eval over {len(cids)} full FLARE cases", flush=True)
    model = proc = tmodel = None                              # loaded lazily only if a case is not cached

    raw_acc, rep_acc = {s: [] for s, _ in STRUCTS}, {s: [] for s, _ in STRUCTS}
    for k, cid in enumerate(cids):
        nii = nib.load(f"{CASES}/{cid}_label.nii.gz"); gt = np.asarray(nii.dataobj).astype(np.uint8)
        sp = nii.header.get_zooms()[:3]
        cached = os.path.join(RAW_CACHE, f"{cid}.nii.gz")
        if os.path.exists(cached):
            pred = np.asarray(nib.load(cached).dataobj).astype(np.uint8)
        else:
            if model is None:
                model, proc = IE.load_base(); tmodel = _load_sam3_ckpt(TUMOR_CKPT, "cuda")
            ct = nib.load(f"{CASES}/{cid}_ct.nii.gz").get_fdata()
            pred = np.zeros(ct.shape, np.uint8)
            for concept, lab in IE.ORGANS:                   # autonomous concept organs (base SAM3)
                m = IE.seg_concept(ct, model, proc, concept)
                pred[m > 0] = lab
            tm = IE.seg_concept(ct, tmodel, proc, "tumor")   # autonomous tumor
            pred[(tm > 0) & (pred > 0)] = TUMOR              # baseline pipeline: tumor kept inside an organ
        rep, _ = repair(pred, sp)                            # KG-guided repair
        row = []
        for s, lab in STRUCTS:
            g = gt == lab
            if g.sum() == 0:
                continue
            dr, dp = dice(pred == lab, g), dice(rep == lab, g)
            raw_acc[s].append(dr); rep_acc[s].append(dp); row.append(f"{s} {dr:.2f}->{dp:.2f}")
        print(f"[{k+1}/{len(cids)}] {cid}: " + "  ".join(row), flush=True)
        if model is not None:
            torch.cuda.empty_cache()

    summary = {}
    for s, _ in STRUCTS:
        if raw_acc[s]:
            summary[s] = {"n": len(raw_acc[s]),
                          "raw": round(float(np.mean(raw_acc[s])), 3),
                          "kg_repaired": round(float(np.mean(rep_acc[s])), 3)}
    json.dump({"summary": summary}, open(OUT, "w"), indent=2)
    print("\n=== KG-guided repair: mean Dice (raw -> repaired) ===", flush=True)
    for s, v in summary.items():
        print(f"  {s:13s} raw {v['raw']:.3f}  ->  KG-repaired {v['kg_repaired']:.3f}   (n={v['n']})", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
