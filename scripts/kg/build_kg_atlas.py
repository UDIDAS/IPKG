#!/usr/bin/env python3
"""Rebuild kg_atlas.json — the organ-volume plausibility bands used by the validator
(kg_guided_segment.repair flags p1..p99; analyze_3d_kg.py uses p2.5..p97.5; run_pancreas_sam3.py 5..95).

Cohort: FLARE23 reference (13-organ) label masks under $VKG_DATA/flare23_labels_mc (605 cases, IDs >= 0140),
minus any case in the 40-patient closed-loop test set (--exclude), so the bands are test-independent.
Organs: liver 1, right_kidney 2, spleen 3, pancreas 4, left_kidney 13, and 'kidney' = both kidneys pooled
(the organ key used by the FLARE23 corpus / Table 3).  Output: $VKG_DATA/kg_atlas.json
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
LAB = {"liver": [1], "right_kidney": [2], "spleen": [3], "pancreas": [4], "left_kidney": [13], "kidney": [2, 13]}
PCT = [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99]


def vols(f):
    nii = nib.load(f); a = np.asarray(nii.dataobj); cm3 = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
    return os.path.basename(f)[:-7], {o: float(np.isin(a, labs).sum() * cm3) for o, labs in LAB.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(VKG_DATA, "flare23_labels_mc"))
    ap.add_argument("--exclude", default=None, help="file with case ids to leave out (the test set)")
    ap.add_argument("--out", default=os.path.join(VKG_DATA, "kg_atlas.json"))
    a = ap.parse_args()
    excl = set(l.strip() for l in open(a.exclude)) if a.exclude else set()
    files = [f for f in sorted(glob.glob(f"{a.labels}/*.nii.gz")) if os.path.basename(f)[:-7] not in excl]
    with ProcessPoolExecutor(16) as ex:
        res = dict(ex.map(vols, files))
    organs = {}
    for o in LAB:
        v = np.array([r[o] for r in res.values() if r[o] > 0])
        organs[o] = {"volume_cm3": {f"p{p:g}": round(float(np.percentile(v, p)), 1) for p in PCT}
                     | {"mean": round(float(v.mean()), 1), "sd": round(float(v.std()), 1), "n": int(len(v))}}
    atlas = {"source": "FLARE23 reference 13-organ label masks (labels_mc), test cases excluded",
             "n_cases": len(res), "excluded": sorted(excl & set(os.path.basename(f)[:-7] for f in glob.glob(f"{a.labels}/*.nii.gz"))),
             "organs": organs}
    json.dump(atlas, open(a.out, "w"), indent=1)
    for o, d in organs.items():
        b = d["volume_cm3"]; print(f"  {o:13s} n={b['n']:3d}  p1 {b['p1']:7.1f}  p50 {b['p50']:7.1f}  p99 {b['p99']:7.1f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
