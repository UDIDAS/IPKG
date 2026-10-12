#!/usr/bin/env python3
"""TotalSegmentator release check: re-run the round-2 call on two showcase CTs and compare voxel for voxel with the
delivered round-2 masks, per organ (study lead 10-08, item 4).

Call (as round 2, run_totalseg_flare23.py): totalsegmentator(img, None, fast=True, ml=True,
roi_subset=[liver, kidney_left, kidney_right, pancreas, spleen, stomach, colon]).
Cases: pancreas_125 (single-organ TotalSegmentator batch) and FLARE23_0405 (the 10-01 FLARE23 batch).
The output is mapped into the delivered label scheme exactly as combine_ts_promptfree.py writes it
(MSD: TS pancreas 7 -> 1; FLARE23: TS 5->1 liver, 2->2 right kidney, 1->3 spleen, 7->4 pancreas, 3->13 left kidney),
and the delivered tumour voxels (2 / 14) are carried over, since the comparison concerns the organ labels.

Inputs ($TS_DATA): ct/pancreas_125_image.nii.gz, ct/ct_FLARE23_0405_0000.nii.gz (Drive final_cases_2026-09-19/ct/) and
shipped/pancreas_125.nii.gz, shipped/FLARE23_0405.nii.gz (Drive round2_2026-10-01/masks/showcase/).
Run once per TotalSegmentator release, each in its own environment:

  TS_DATA=... TOTALSEG_HOME_DIR=... <env>/bin/python ts_reproduce_showcase.py
-> results/audit/ts_version/rerun_<release>/<case>.nii.gz   (re-run masks, delivered label scheme)
   results/audit/ts_version/rerun_<release>/differences.json (per-organ voxel differences vs the delivered masks)
"""
import importlib.metadata as md
import json
import os

import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator

ROI = ["liver", "kidney_left", "kidney_right", "pancreas", "spleen", "stomach", "colon"]
DATA = os.environ.get("TS_DATA", "/path/to/ts_data")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
CASES = {"pancreas_125": ("ct/pancreas_125_image.nii.gz", {7: (1, "pancreas")}, 2),
         "FLARE23_0405": ("ct/ct_FLARE23_0405_0000.nii.gz",
                          {5: (1, "liver"), 2: (2, "right kidney"), 1: (3, "spleen"), 7: (4, "pancreas"), 3: (13, "left kidney")}, 14)}


def main():
    rel = md.version("TotalSegmentator")
    out_dir = os.path.join(ROOT, "results", "audit", "ts_version", f"rerun_{rel}")
    os.makedirs(out_dir, exist_ok=True)
    res = {"totalsegmentator": rel, "nnunetv2": md.version("nnunetv2"), "torch": md.version("torch"),
           "call": "fast=True, ml=True, roi_subset=" + ",".join(ROI), "cases": {}}
    for case, (ct, lut, tlab) in CASES.items():
        img = nib.load(os.path.join(DATA, ct))
        ts = np.asarray(totalsegmentator(img, None, fast=True, ml=True, roi_subset=ROI, quiet=True, verbose=False).dataobj).astype(np.uint8)
        sh_img = nib.load(os.path.join(DATA, "shipped", f"{case}.nii.gz"))
        sh = np.asarray(sh_img.dataobj).astype(np.uint8)
        tum = sh == tlab
        out = np.zeros_like(sh)
        for a, (b, _) in lut.items():
            out[ts == a] = b
        out[tum] = tlab
        nib.save(nib.Nifti1Image(out, sh_img.affine, sh_img.header), os.path.join(out_dir, f"{case}.nii.gz"))
        per = {}
        for _, (b, name) in lut.items():
            r, s = out == b, sh == b
            per[name] = {"label": b, "voxels_delivered": int(s.sum()), "voxels_rerun": int(r.sum()),
                         "voxels_differing": int((r ^ s).sum()),
                         "percent_of_delivered": round(100 * int((r ^ s).sum()) / max(1, int(s.sum())), 4)}
        org = np.isin(sh, [b for b, _ in lut.values()])
        tot = int((out != sh).sum())
        res["cases"][case] = {"organs": per, "organ_voxels_delivered": int(org.sum()), "voxels_differing_total": tot,
                              "percent_of_delivered_organ_voxels": round(100 * tot / int(org.sum()), 4)}
        print(rel, case, tot, "differing of", int(org.sum()), f"({res['cases'][case]['percent_of_delivered_organ_voxels']} %)",
              {k: v["voxels_differing"] for k, v in per.items()}, flush=True)
    json.dump(res, open(os.path.join(out_dir, "differences.json"), "w"), indent=1)
    print("->", out_dir)


if __name__ == "__main__":
    main()
