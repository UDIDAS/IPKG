#!/usr/bin/env python3
"""Prompt-free 3-D TUMOR eval on held-out patients (IPKG round 2, the study lead 10-01): no box, no slice gate.

Every slice of the volume is prompted with the text "tumor" only; a slice is kept when the top query's confidence
>= 0.15 and the mask has >= 15 px (the auto_pair arm of rerun_final_cases.py, whose preprocessing, inference and
metric functions are imported unchanged). The reference mask is read ONLY for scoring.

Patients = the model's own held-out TEST list from results/audit/split_manifests_2026-10-01.json:
  sam3_tumor_generic (incremental stage 4, trained on all four tumor pools) -> 'global' tumor split
  sam3_tumor_ausam_{kits,flare} (per-dataset)                                -> 'filtered' tumor split
Volumes are the FULL public releases (LiTS = MSD Task03 NIfTI, not the liver-cropped training .npy slabs).

  VKG_DATA=... OUT_DIR=... python eval_tumor_promptfree.py --dataset lits|msd|kits|flare23 --ckpt <path> \
      --split global|filtered [--shard i/n]
-> $OUT_DIR/{model}/{dataset}/{case}.nii.gz (tumor mask, 1 = tumor) + per-case JSON lines in {dataset}.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time

import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
import sam3_autonomous_local as IE                                       # noqa: E402
from rerun_final_cases import load_model, predict, metrics, drop_small, to256, sl, sha256   # noqa: E402

DS = os.environ.get("VKG_DATASETS", "/path/to/cluster/projects/datasets")             # raw public releases
FLARE = os.environ.get("VKG_FLARE23", "/path/to/cluster/projects/VKG_data/flare23_pool")  # 576 FLARE23 CT + labels
OUT = os.environ.get("OUT_DIR", "/path/to/cluster/work/vkg_runs/promptfree")
MANIFEST = os.path.join(ROOT, "results", "audit", "split_manifests_2026-10-01.json")
# dataset -> (tumor-pool key in the manifest, GT tumor label(s), slice axis)
CFG = {"lits": ("lits", [2], 2), "msd": ("pancreas", [2], 2), "kits": ("kits", [2], 0), "flare23": ("flare", [14], 2)}


def load(ds, c):
    """-> ct, seg, affine, spacing, orient (callable that maps a working-grid mask back to the file's grid)."""
    ident = lambda m: m                                                    # noqa: E731
    if ds == "lits":      # FULL volume (MSD Task03 = LiTS train, liver_N == volume-N); the training .npy are
        v = c.replace("volume-", "")                     # liver-cropped z-slabs, i.e. a reference-derived slice gate
        ip, lp = f"{DS}/MSD/Task03_Liver/imagesTr/liver_{v}.nii.gz", f"{DS}/MSD/Task03_Liver/labelsTr/liver_{v}.nii.gz"
    elif ds == "msd":
        ip, lp = f"{DS}/MSD/Task07_Pancreas/imagesTr/{c}.nii.gz", f"{DS}/MSD/Task07_Pancreas/labelsTr/{c}.nii.gz"
    elif ds == "kits":
        ip, lp = f"{DS}/KiTS23/{c}/imaging.nii.gz", f"{DS}/KiTS23/{c}/segmentation.nii.gz"
    else:
        ip, lp = f"{FLARE}/images/{c}_0000.nii.gz", f"{FLARE}/labels/{c}.nii.gz"
    img = nib.load(ip)
    ct, seg = img.get_fdata(), np.asarray(nib.load(lp).dataobj)
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    if ds == "lits":      # in-plane orientation of the training .npy (image-only match: exact on volume-11)
        ct, seg = np.rot90(ct, -1, axes=(0, 1)).copy(), np.rot90(seg, -1, axes=(0, 1)).copy()
        sp = (sp[1], sp[0], sp[2])
        return ct, seg, img.affine, sp, lambda m: np.rot90(m, 1, axes=(0, 1))
    return ct, seg, img.affine, sp, ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CFG)); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", required=True, choices=["global", "filtered"]); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cases", nargs="+", default=None, help="explicit cases (e.g. the four showcase cases) instead of the test list")
    a = ap.parse_args()
    key, tlab, ax = CFG[a.dataset]
    cases = a.cases or json.load(open(MANIFEST))["tumor_pools"][a.split][key]["test"]
    i, n = map(int, a.shard.split("/"))
    cases = cases[i::n]
    mname = os.path.basename(a.ckpt).replace(".pth", "")
    od = f"{OUT}/{mname}/{a.dataset}"
    os.makedirs(od, exist_ok=True)
    log = f"{od}/{a.dataset}_{'showcase' if a.cases else 'shard%d' % i}.jsonl"
    done = {json.loads(l)["case"] for f in glob.glob(f"{od}/*.jsonl") for l in open(f)
            if "error" not in json.loads(l)}                   # any shard's finished cases
    done -= set(a.cases or [])                                   # explicit --cases always re-run

    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    model = load_model(a.ckpt)
    h = sha256(a.ckpt)
    for c in cases:
        if c in done:
            continue
        t0 = time.time()
        try:
            ct, seg, aff, sp, back = load(a.dataset, c)
        except Exception as e:                   # noqa: BLE001
            rec = {"case": c, "error": f"{type(e).__name__}: {str(e)[:120]}"}
            open(log, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True); continue
        gt = np.isin(seg, tlab)
        rgb = [IE.hu_to_rgb(to256(sl(ct, ax, z), 1), *IE.WIN).astype(np.uint8) for z in range(ct.shape[ax])]
        pred = predict(model, proc, rgb, seg, tlab, "tumor", ax, "auto", a.batch)   # 'auto': every slice, no box
        nib.save(nib.Nifti1Image(back(pred).astype(np.uint8), aff), f"{od}/{c}.nii.gz")
        vox_cm3 = float(np.prod(sp)) / 1000.0
        r = metrics(pred, gt, sp, vox_cm3)
        r_pp = metrics(drop_small(pred, max(1, int(round(0.1 / vox_cm3)))), gt, sp, vox_cm3)
        rec = {"case": c, "dataset": a.dataset, "model": mname, "model_sha256": h, "split": a.split,
               "protocol": "prompt-free: text 'tumor', every slice, no box, no slice gate (conf>=0.15, >=15 px)",
               "spacing_mm": list(sp), "gt_tumor_vox": int(gt.sum()), "seconds": round(time.time() - t0, 1),
               "raw": r, "drop_lt_0p1cm3": r_pp}
        open(log, "a").write(json.dumps(rec) + "\n")
        print(f"[{a.dataset} {c}] dice {r['dice']} hd95 {r['hd95_mm']} comps {r['components']} {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
