#!/usr/bin/env python3
"""Containment re-extraction from the frozen FLARE23 predicted masks (queue item 4).

The frozen assemble (build_predicted_corpus_flare23.py) tests containment as
(tum & om) on EXCLUSIVE labels (a voxel is organ xor tumor), so the >=90 %-inside
rule can never fire and every tumor-bearing record reads 'boundary'.

Fix (docs/VKG_v9_ud_round_2026-09-15.md item 2): organ := organ UNION tumor-in-organ.
A tumor sitting inside the organ appears as a cavity in the exclusive organ mask; the
anatomic organ region is restored by filling those cavities, then the ORIGINAL rule
(>=90 % of the tumor within the 3-voxel dilation of the organ) is applied against the
restored region:

    O_anat = slice-wise (axial, axis 2 - the prediction axis) binary_fill_holes(om)
    frac   = |t_in  intersect  dilate(O_anat, 3)| / |t_in|
    containment = 'contained' if frac >= 0.90 else 'boundary'

Interior tumor voxels are exactly the filled cavities (the 'tumor-in-organ' union);
tumor extending beyond the restored organ surface by more than the 3-voxel tolerance
counts against containment - so subcapsular/exophytic components still read boundary.
A 3-D fill_holes variant is recorded per case as a sensitivity check (it is stricter:
a single-slice gap in the predicted shell 'leaks' the cavity in 3-D).

Tumor->organ attribution (winner-take-all over 3-voxel organ dilations) is copied
verbatim from assemble(); reproduced tumor_voxels are asserted against the frozen
corpus per (case, organ).

Outputs (frozen corpus untouched):
  results/kg/containment_reextraction_flare23.json          per-case fracs, old->new, summary
  corpora/containment_v2/corpus_predicted_flare23_containment_v2.json      corpus with only containment replaced
"""
import argparse
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
MASKS = "/path/to/staging/acm_data/flare23_masks/masks_predicted_flare23"
CORPUS_IN = os.path.join(ROOT, "corpora", "corpus_predicted_flare23.json")
CORPUS_OUT = os.path.join(ROOT, "corpora", "containment_v2", "corpus_predicted_flare23_containment_v2.json")
RESULT_OUT = os.path.join(ROOT, "results", "kg", "containment_reextraction_flare23.json")
POOLED = {"liver": [1], "kidney": [2, 13], "spleen": [3], "pancreas": [4]}
TUMOR = 14


def fill2d(om):
    out = np.zeros_like(om)
    for z in range(om.shape[2]):
        out[:, :, z] = ndimage.binary_fill_holes(om[:, :, z])
    return out


def one_case(path):
    cid = os.path.basename(path)[: -len(".nii.gz")]
    ml = np.asarray(nib.load(path).dataobj).astype(np.uint8)
    tum = ml == TUMOR
    # --- attribution: verbatim from build_predicted_corpus_flare23.assemble() ---
    dil = {o: ndimage.binary_dilation(np.isin(ml, labs), iterations=3) for o, labs in POOLED.items() if np.isin(ml, labs).any()}
    lbl, ncomp = ndimage.label(tum)
    assigned = {o: np.zeros(ml.shape, bool) for o in dil}
    for c in range(1, ncomp + 1):
        comp = lbl == c
        ov = {o: int((comp & d).sum()) for o, d in dil.items()}
        best = max(ov, key=ov.get) if ov else None
        if best is not None and ov[best] > 0:
            assigned[best] |= comp
    # --- containment v2 per tumor-bearing organ ---
    organs = {}
    for o, labs in POOLED.items():
        om = np.isin(ml, labs)
        if om.sum() == 0 or o not in assigned:
            continue
        t_in = assigned[o]
        tvox = int(t_in.sum())
        if tvox == 0:
            continue
        o2d = fill2d(om)
        o3d = ndimage.binary_fill_holes(om)
        d2 = ndimage.binary_dilation(o2d, iterations=3)
        d3 = ndimage.binary_dilation(o3d, iterations=3)
        f2 = float((t_in & d2).sum() / tvox)
        f3 = float((t_in & d3).sum() / tvox)
        organs[o] = {"tumor_voxels": tvox,
                     "frac_fill2d_dil3": round(f2, 4), "frac_fill3d_dil3": round(f3, 4),
                     "frac_fill2d_nodil": round(float((t_in & o2d).sum() / tvox), 4),
                     "containment_v2": "contained" if f2 >= 0.90 else "boundary",
                     "containment_v2_fill3d": "contained" if f3 >= 0.90 else "boundary"}
    return cid, organs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default=MASKS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.masks}/*.nii.gz"))[: a.limit]
    print(f"{len(files)} masks, {a.workers} workers", flush=True)
    with Pool(a.workers) as p:
        cases = dict(p.imap_unordered(one_case, files, chunksize=4))

    corpus = json.load(open(CORPUS_IN))
    mismatches, flips, rows = [], [], 0
    old_c, new_c, per_organ = Counter(), Counter(), {}
    for rec in corpus["records"]:
        cid = rec["case_id"].replace("flare23_", "", 1)
        for o, v in rec["organs"].items():
            if not v["has_tumor"]:
                continue
            rows += 1
            new = cases.get(cid, {}).get(o)
            if new is None or new["tumor_voxels"] != v["tumor_voxels"]:
                mismatches.append({"case": cid, "organ": o, "frozen_tvox": v["tumor_voxels"],
                                   "reextracted": new})
                continue
            old_c[v["containment"]] += 1
            new_c[new["containment_v2"]] += 1
            po = per_organ.setdefault(o, Counter())
            po[new["containment_v2"]] += 1
            if new["containment_v2"] != v["containment"]:
                flips.append({"case": cid, "organ": o, "old": v["containment"],
                              "new": new["containment_v2"], "frac": new["frac_fill2d_dil3"]})
            v["containment"] = new["containment_v2"]

    corpus["note"] = corpus.get("note", "") + (
        " CONTAINMENT v2 (2026-09-16): re-extracted from the frozen masks with organ := organ UNION "
        "tumor-in-organ (slice-wise axial fill_holes restores interior tumor cavities), then the original "
        ">=90 %-within-3-voxel-dilation rule; see results/kg/containment_reextraction_flare23.json. "
        "All other fields identical to corpus_predicted_flare23.json.")
    json.dump(corpus, open(CORPUS_OUT, "w"), indent=1)

    fill3d_c = Counter(v["containment_v2_fill3d"] for c in cases.values() for v in c.values())
    out = {
        "note": ("Containment re-extraction from the frozen 576 predicted FLARE23 masks (queue item 4). "
                 "Rule: organ := organ UNION tumor-in-organ, implemented as slice-wise (axial) fill_holes of the "
                 "exclusive organ mask, then contained iff >=90 % of the organ's attributed tumor lies within the "
                 "3-voxel dilation of the restored organ (the original extractor's tolerance). The frozen corpus is "
                 "degenerate (every tumor row 'boundary': exclusive labels make (tum & om)=0). Attribution "
                 "winner-take-all reproduced verbatim; tumor_voxels asserted against the frozen corpus."),
        "masks": a.masks,
        "n_masks": len(files),
        "n_tumor_rows": rows,
        "containment_old": dict(old_c),
        "containment_v2": dict(new_c),
        "containment_v2_by_organ": {o: dict(c) for o, c in sorted(per_organ.items())},
        "containment_v2_fill3d_sensitivity": dict(fill3d_c),
        "n_flips": len(flips),
        "flips": flips,
        "attribution_mismatches": mismatches,
        "per_case": {c: v for c, v in sorted(cases.items()) if v},
    }
    os.makedirs(os.path.dirname(RESULT_OUT), exist_ok=True)
    json.dump(out, open(RESULT_OUT, "w"), indent=1)
    print(f"tumor rows {rows} | old {dict(old_c)} -> v2 {dict(new_c)} (fill3d {dict(fill3d_c)}) | "
          f"flips {len(flips)} | attribution mismatches {len(mismatches)}", flush=True)
    print(f"-> {RESULT_OUT}\n-> {CORPUS_OUT}", flush=True)


if __name__ == "__main__":
    main()
