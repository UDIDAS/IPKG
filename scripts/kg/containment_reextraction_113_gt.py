#!/usr/bin/env python3
"""Containment v2 for the 113-cohort GT corpora (v11 item 2, the locally-runnable half).

Table 11's containment precision compares PREDICTED statements against GT statements — both
sides are degenerate ('boundary' everywhere, exclusive-label artifact). The predicted side
needs the frozen single-organ predicted masks (Delta); the GT side is computable now from:
  kits — staged native GT (/scratch .../kits_vol/<case>/segmentation.nii.gz; kidney 1, tumor 2,
         cyst 3 dropped; slice axis 0)
  msd  — MSD Pancreas labelsTr staged from Drive (pancreas 1, tumor 2; slice axis 2)
  lits — MSD Task03 labelsTr via the identity mapping (liver 1, tumor 2; slice axis 2 — the
         native grid, NOT the npy slab, so the lits preset's fill_axis 0 does not apply)

The rule is the DELIVERED extractor itself (app_handoff/extract_phenotypes.py, containment fix
2026-09-16): >=90 % of the attributed tumor within the 3-voxel dilation of the slice-wise
hole-filled organ. Only the containment field is taken; corpora are written as SIDE FILES
(corpus_gt_*_containment_v2.json) because retrieval relevance uses containment — the frozen
corpora and all frozen retrieval numbers stay untouched.
-> results/kg/containment_reextraction_113_gt.json + corpora/containment_v2/corpus_gt_{kits,lits,msd}_containment_v2.json
"""
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool

import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "app_handoff"))
from extract_phenotypes import extract                                  # noqa: E402

KITS = "/path/to/staging/acm_data/kits_vol"
MSD = "/path/to/staging/acm_data/msd_pancreas_labels"
TASK03 = "/path/to/staging/msd_task03/Task03_Liver/labelsTr"
OUT = os.path.join(ROOT, "results", "kg", "containment_reextraction_113_gt.json")
GMAP = json.load(open(f"{ROOT}/results/audit/lits_geometry_map.json"))["mapping"]


def one(job):
    ds, cid = job
    if ds == "kits":
        fp, kw = f"{KITS}/{cid}/segmentation.nii.gz", {"dataset": "kits"}
    elif ds == "msd":
        fp, kw = f"{MSD}/{cid}.nii.gz", {"dataset": "msd"}
    else:
        fp, kw = f"{TASK03}/{GMAP[cid]['liver']}.nii.gz", {"single_organ": "liver", "fill_axis": 2}
    nii = nib.load(fp)
    rec = extract(np.asarray(nii.dataobj).astype(np.uint8),
                  [float(z) for z in nii.header.get_zooms()[:3]], cid, **kw)
    organ = next(iter(rec["organs"]))
    return ds, cid, organ, rec["organs"][organ]["containment"], rec["organs"][organ]["has_tumor"]


def main():
    jobs, frozen = [], {}
    for ds in ("kits", "lits", "msd"):
        d = json.load(open(f"{ROOT}/corpora/corpus_gt_{ds}.json"))
        frozen[ds] = d
        jobs += [(ds, r["case_id"]) for r in d["records"]]
    with Pool(8) as p:
        res = {(ds, cid): (org, cont, ht) for ds, cid, org, cont, ht in p.imap_unordered(one, jobs)}

    summary, mismatches = {}, []
    for ds, d in frozen.items():
        old_c, new_c = Counter(), Counter()
        for r in d["records"]:
            org, cont, ht = res[(ds, r["case_id"])]
            p = r["organs"][org]
            if bool(p["has_tumor"]) != ht:
                mismatches.append({"ds": ds, "case": r["case_id"], "frozen_has_tumor": p["has_tumor"],
                                   "reextracted_has_tumor": ht})
            old_c[p["containment"]] += 1
            new_c[cont] += 1
            p["containment"] = cont
        d["note"] = d.get("note", "") + (
            " CONTAINMENT v2 (2026-09-16): GT-side re-extraction with the delivered extractor's fixed rule "
            "(organ := organ ∪ tumor-in-organ, slice-wise fill + 3-voxel dilation, >=90 %); native GT masks "
            "(kits staged, msd labelsTr, lits via MSD Task03 identity mapping). Only containment changed; "
            "see results/kg/containment_reextraction_113_gt.json. SIDE FILE - the frozen corpus stands "
            "(retrieval relevance uses containment).")
        json.dump(d, open(f"{ROOT}/corpora/containment_v2/corpus_gt_{ds}_containment_v2.json", "w"), indent=1)
        summary[ds] = {"old": dict(old_c), "v2": dict(new_c)}
        print(f"{ds}: {dict(old_c)} -> {dict(new_c)}", flush=True)

    out = {"note": __doc__.strip(), "summary": summary, "has_tumor_mismatches": mismatches,
           "predicted_side": "still Delta-gated (frozen single-organ predicted masks); Table 11's containment "
                             "precision recomputes when both sides are v2."}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"has_tumor mismatches: {len(mismatches)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
