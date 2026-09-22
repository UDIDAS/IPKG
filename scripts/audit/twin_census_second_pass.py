#!/usr/bin/env python3
"""Second pass of the twin census (2026-09-22): confirm the eleven KiTS23<->FLARE23 same-scan
pairs found by KS's CT-voxel-equality scan and fold them into the census artifact.

Provenance: KS's `scan_flare23_twins.py` (branch `abdomen/ipkg`, artifact
`paper/flare23_twin_scan_2026-09-20.json`) indexes KiTS23/Task03/Task07 CTs by canonical (RAS)
shape+spacing and compares candidate volumes VOXEL FOR VOXEL (`np.array_equal`) — an
annotation-independent test that is strictly stronger than the first pass's label-based
screen (organ <= 2 %, tumor <= 15 %): a same-scan pair whose kidney/tumor were re-annotated
beyond those tolerances is invisible to any label screen (FLARE23_0405 <-> case_00078 was the
first such case; these eleven are the rest that sit inside the 1,334 pool).  Of his 20
in-pool KiTS pairs, 9 were already removed by the first pass; the eleven here are the
remainder.  This script re-confirms each pair independently by GT-label mask overlap
(axis-normalized transposition search; kidney and tumor IoU) and rewrites
results/audit/axis_normalized_twin_census.json with a `second_pass_2026-09-22` section and
updated pool arithmetic: pool 1,334 -> 1,323, corpus 1,221 -> 1,210, twins 91 -> 102.

  V11_FLARE=/dev/shm/v11/flare V11_KITS=/dev/shm/v11/kits python twin_census_second_pass.py
"""
import itertools
import json
import os

import nibabel as nib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FLARE = os.environ.get("V11_FLARE", "/dev/shm/v11/flare")
KITS = os.environ.get("V11_KITS", "/dev/shm/v11/kits")
CENSUS = os.path.join(ROOT, "results", "audit", "axis_normalized_twin_census.json")

PAIRS = [("FLARE23_0126", "case_00090"), ("FLARE23_0444", "case_00165"),
         ("FLARE23_0617", "case_00092"), ("FLARE23_0802", "case_00024"),
         ("FLARE23_0883", "case_00008"), ("FLARE23_0886", "case_00039"),
         ("FLARE23_0910", "case_00075"), ("FLARE23_0922", "case_00043"),
         ("FLARE23_1274", "case_00021"), ("FLARE23_1512", "case_00083"),
         ("FLARE23_1630", "case_00060")]


def confirm(fid, kid):
    f = nib.load(f"{FLARE}/{fid}.nii.gz")
    k = nib.load(f"{KITS}/{kid}.nii.gz")
    fs = np.asarray(f.dataobj).astype(np.uint8)
    ks = np.asarray(k.dataobj).astype(np.uint8)
    fk, ft = np.isin(fs, [2, 13]), fs == 14
    kk, kt = ks == 1, ks == 2
    best, cfg = 0.0, None
    for perm in itertools.permutations(range(3)):
        if tuple(np.transpose(kk, perm).shape) != fk.shape:
            continue
        kp = np.transpose(kk, perm)
        for flips in itertools.product([False, True], repeat=3):
            m = kp
            for ax, fl in enumerate(flips):
                if fl:
                    m = np.flip(m, ax)
            u = np.logical_or(m, fk).sum()
            iou = float(np.logical_and(m, fk).sum() / u) if u else 0.0
            if iou > best:
                best, cfg = iou, (perm, flips)
    tiou = None
    if cfg is not None:
        m = np.transpose(kt, cfg[0])
        for ax, fl in enumerate(cfg[1]):
            if fl:
                m = np.flip(m, ax)
        u = np.logical_or(m, ft).sum()
        tiou = float(np.logical_and(m, ft).sum() / u) if u else 0.0
    return {"flare": f"flare23_{fid}", "query": kid,
            "same_sorted_grid": sorted(fs.shape) == sorted(ks.shape),
            "kidney_vox": [int(kk.sum()), int(fk.sum())],
            "tumor_vox": [int(kt.sum()), int(ft.sum())],
            "kidney_iou": round(best, 4), "tumor_iou": round(tiou, 4) if tiou is not None else None,
            "transposition": str(cfg), "verdict": "CONFIRMED_TWIN" if best >= 0.8 and (tiou or 0) >= 0.5 else "rejected"}


def main():
    rows = [confirm(fid, kid) for fid, kid in PAIRS]
    for r in rows:
        print(r["flare"], "=", r["query"], r["verdict"], "kidneyIoU", r["kidney_iou"], "tumorIoU", r["tumor_iou"])
    conf = [r for r in rows if r["verdict"] == "CONFIRMED_TWIN"]
    cen = json.load(open(CENSUS))
    cen["second_pass_2026-09-22"] = {
        "note": __doc__.strip(),
        "source": "KS CT-voxel-equality scan (abdomen/ipkg: paper/flare23_twin_scan_2026-09-20.json; "
                  "310 cross-collection same-scan pairs over the 576 UD-masked FLARE23 studies; 20 KiTS "
                  "pairs in-pool of which 9 were removed by the first pass)",
        "first_pass_blind_spot": "label-based screen (organ <= 2 %, tumor <= 15 %) cannot see same-scan "
                                 "pairs whose annotations were redone beyond those tolerances; CT voxel "
                                 "equality is annotation-independent",
        "pairs_confirmed": conf, "n_confirmed": len(conf),
        "residual": "KS's scan covers the 576 UD-masked FLARE23 studies; the 691 pool records without a "
                    "UD mask are label-screened only (first pass) - a lower bound, disclosed in the draft."}
    cen["pool_arithmetic"] = {"pool_before": 1347, "pool_after_first_pass": 1334,
                              "pool_after_second_pass": 1347 - 13 - len(conf),
                              "flare_corpus_after": 1234 - 13 - len(conf),
                              "total_twins_removed": 78 + 13 + len(conf)}
    json.dump(cen, open(CENSUS, "w"), indent=1)
    print(f"census updated: {len(conf)}/11 confirmed -> pool {1347-13-len(conf)}, twins {78+13+len(conf)}")


if __name__ == "__main__":
    main()
