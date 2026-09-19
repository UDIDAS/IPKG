#!/usr/bin/env python3
"""Containment v2 for the 113-cohort PREDICTED corpora — from the locally REGENERATED masks
(scripts/segmentation/regen_predicted_masks_113.py; the frozen masks are unreachable: KiTS/MSD
on Delta, LiTS gone). Completes the last open pipeline item: Table 11's containment cell.

Rule = the delivered extractor (app_handoff/extract_phenotypes.py, fixed 2026-09-16): >=90 % of
the attributed tumor within the 3-voxel dilation of the slice-wise hole-filled organ. Dataset
presets give the fill axis (kits 0, lits 0 — the npy slab grid the frozen corpus used, msd 2).

REGENERATED-MASK CAVEAT, and how it is contained: the weights differ from the per-dataset AUSAM
checkpoints (organ = base facebook/sam3, tumor = the FLARE23 expert, both GT-box semi-oracle),
so per-case fidelity vs the FROZEN corpora is enforced — the v2 containment is adopted ONLY
where the regenerated record agrees with the frozen record on has_tumor; disagreeing cases keep
their frozen value and are listed. Fidelity summary: results/segmentation/regen_masks_113_fidelity.json.

Also recomputes the Table 11 statement-precision containment row (summ_metrics.py method,
verbatim conditioning: containment counts only where the predicted record asserts a tumor)
against the v2 GT side (corpora/containment_v2/corpus_gt_*_containment_v2.json), giving the
draft's §5 containment-precision replacement for the frozen 0.844.

-> results/kg/containment_reextraction_113_predicted.json
-> corpora/containment_v2/corpus_predicted_{kits,lits,msd}_containment_v2.json
"""
import glob
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

MASKS = "/path/to/staging/acm_data/masks_predicted_113_regen"
OUT = os.path.join(ROOT, "results", "kg", "containment_reextraction_113_predicted.json")
ORGAN = {"kits": "kidney", "lits": "liver", "msd": "pancreas"}


def one(job):
    ds, cid = job
    nii = nib.load(f"{MASKS}/{ds}/{cid}.nii.gz")
    rec = extract(np.asarray(nii.dataobj).astype(np.uint8),
                  [float(z) for z in nii.header.get_zooms()[:3]], cid, dataset=ds)
    p = rec["organs"][ORGAN[ds]]
    return ds, cid, p["containment"], p["has_tumor"]


def main():
    jobs, frozen = [], {}
    for ds in ("kits", "lits", "msd"):
        d = json.load(open(f"{ROOT}/corpora/corpus_predicted_{ds}.json"))
        frozen[ds] = d
        jobs += [(ds, r["case_id"]) for r in d["records"]]
    with Pool(8) as p:
        res = {(ds, cid): (cont, ht) for ds, cid, cont, ht in p.imap_unordered(one, jobs)}

    summary, kept_frozen = {}, []
    for ds, d in frozen.items():
        org = ORGAN[ds]
        old_c, new_c = Counter(), Counter()
        for r in d["records"]:
            cont, ht = res[(ds, r["case_id"])]
            p = r["organs"][org]
            old_c[p["containment"]] += 1
            if bool(p["has_tumor"]) == ht:                    # fidelity gate: adopt only on has_tumor agreement
                p["containment"] = cont if p["has_tumor"] else "none"
            else:
                kept_frozen.append({"ds": ds, "case": r["case_id"], "frozen_has_tumor": p["has_tumor"],
                                    "regen_has_tumor": ht, "kept": p["containment"]})
            new_c[p["containment"]] += 1
        d["note"] = d.get("note", "") + (
            " CONTAINMENT v2 (2026-09-17): predicted-side re-extraction from REGENERATED semi-oracle masks "
            "(regen_predicted_masks_113.py — frozen masks unreachable; weights caveat + per-case fidelity in "
            "results/segmentation/regen_masks_113_fidelity.json) with the delivered extractor's fixed rule; adopted "
            "only where regen has_tumor == frozen has_tumor. Only containment changed. SIDE FILE — the frozen corpus "
            "stands (retrieval relevance uses containment).")
        json.dump(d, open(f"{ROOT}/corpora/containment_v2/corpus_predicted_{ds}_containment_v2.json", "w"), indent=1)
        summary[ds] = {"old": dict(old_c), "v2": dict(new_c)}
        print(f"{ds}: {dict(old_c)} -> {dict(new_c)}", flush=True)

    # ---- Table 11 containment row, v2 both sides (summ_metrics.py conditioning, verbatim) ----
    def load_side(kind):
        recs = []
        for f in sorted(glob.glob(f"{ROOT}/corpora/containment_v2/corpus_{kind}_*_containment_v2.json")):
            d = json.load(open(f))
            if d["dataset"] in ("kits", "lits", "msd"):
                recs += d["records"]
        return {r["case_id"]: r for r in recs}

    predv, gtv = load_side("predicted"), load_side("gt")
    ids = [i for i in predv if i in gtv]
    hits = tot = 0
    per_ds = {}
    for cid in ids:
        P, G = predv[cid], gtv[cid]
        for orgn in P.get("observed_organs", []):
            po, go = P["organs"].get(orgn, {}), G["organs"].get(orgn, {})
            if not go or not po.get("has_tumor"):
                continue
            tot += 1
            k = P["dataset"]
            per_ds.setdefault(k, [0, 0])
            per_ds[k][1] += 1
            if po.get("containment") == go.get("containment"):
                hits += 1
                per_ds[k][0] += 1
    fid = json.load(open(f"{ROOT}/results/segmentation/regen_masks_113_fidelity.json"))["summary"]
    out = {"note": __doc__.strip(), "summary_containment": summary,
           "kept_frozen_on_has_tumor_mismatch": kept_frozen,
           "regen_fidelity": fid,
           "table11_containment_precision_v2": {
               "precision": round(hits / tot, 3) if tot else None, "n_statements": tot,
               "per_dataset": {k: {"precision": round(h / n, 3), "n": n} for k, (h, n) in sorted(per_ds.items())},
               "frozen_value_replaced": 0.844,
               "method": "summ_metrics.py conditioning verbatim: containment statements counted only where the "
                         "predicted record asserts a tumor; scored vs the v2 GT record of the same case."}}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"Table 11 containment v2: {out['table11_containment_precision_v2']['precision']} "
          f"(n={tot}; was 0.844 degenerate) | kept-frozen cases: {len(kept_frozen)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
