#!/usr/bin/env python3
"""Extended duplicate audit over the 21 LiTS queries (queue item 2) — activates now that the
LiTS geometry is recovered (lits_geometry_map.py / MSD Task03 headers).

Tests per (LiTS query, FLARE23 candidate), same definitions as duplicate_scan_audit.py:
  T1 exact geometry: same native shape AND spacing (1e-3). Run against the shipped predicted
     FLARE23 mask headers (576 of 1,312 — the cases with masks in the release copy; the
     organ-voxel +-0.5 % clause needs GT labels, streamable from the Drive Metadata.zip for
     any hit). The known FLARE23 twin of the LiTS family, FLARE23_0102 (= liver_64 =
     volume-64), is already OUT of the de-duplicated pool and volume-64 is not a query.
  T2 physical: liver volume within 1 % AND tumor volume within 2 % (both physical), against
     ALL 1,312 GT-KG FLARE23 records (corpus_flare23_kg.json) using the REAL LiTS volumes.

Any pair passing either test is flagged for confirmation; none flagged -> the 1,366-case
de-duplicated pool stands for the LiTS queries.
-> results/audit/lits_extended_audit.json
"""
import glob
import json
import os

import nibabel as nib

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MASKS = "/path/to/staging/acm_data/flare23_masks/masks_predicted_flare23"
OUT = os.path.join(ROOT, "results", "audit", "lits_extended_audit.json")


def close(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def main():
    gmap = json.load(open(f"{ROOT}/results/audit/lits_geometry_map.json"))["mapping"]
    queries = {r["case_id"]: r for r in json.load(open(f"{ROOT}/corpora/corpus_gt_lits.json"))["records"]}
    flare = json.load(open(f"{ROOT}/corpora/corpus_flare23_kg.json"))["records"]
    dedup_ids = {r["case_id"] for r in json.load(open(f"{ROOT}/corpora/corpus_flare23_kg_dedup.json"))["records"]}

    headers = {}
    for f in sorted(glob.glob(f"{MASKS}/*.nii.gz")):
        nii = nib.load(f)
        headers[os.path.basename(f)[:-7]] = ([int(x) for x in nii.shape],
                                             [round(float(z), 4) for z in nii.header.get_zooms()[:3]])

    pairs = []
    for q, rec in queries.items():
        shape_q, sp_q = gmap[q]["shape_native"], gmap[q]["spacing"]
        ov_q = rec["organs"]["liver"]["organ_volume_cm3"]
        tv_q = rec["organs"]["liver"]["tumor_volume_cm3"]
        for cid, (shape_f, sp_f) in headers.items():                     # T1 vs shipped mask headers
            if shape_q == shape_f and all(abs(a - b) < 1e-3 for a, b in zip(sp_q, sp_f)):
                pairs.append({"query": q, "flare23": f"flare23_{cid}", "test": "T1_exact_geometry",
                              "shape": shape_q, "spacing": sp_q,
                              "in_dedup_pool": f"flare23_{cid}" in dedup_ids})
        for fr in flare:                                                 # T2 vs all 1,312 GT records
            liver = fr.get("organs", {}).get("liver")
            if not liver:
                continue
            if close(ov_q, liver["organ_volume_cm3"], 0.01) and \
               (close(tv_q, liver["tumor_volume_cm3"], 0.02) or (tv_q == 0 and liver["tumor_volume_cm3"] == 0)):
                pairs.append({"query": q, "flare23": fr["case_id"], "test": "T2_physical",
                              "q_organ_cm3": ov_q, "f_organ_cm3": liver["organ_volume_cm3"],
                              "q_tumor_cm3": tv_q, "f_tumor_cm3": liver["tumor_volume_cm3"],
                              "in_dedup_pool": fr["case_id"] in dedup_ids})

    out = {"note": __doc__.strip(), "n_queries": len(queries),
           "n_flare23_headers_checked": len(headers), "n_flare23_gt_records": len(flare),
           "known_prior_twin": "FLARE23_0102 = liver_64 = volume-64 (not a query; already out of the pool)",
           "pairs_flagged": pairs,
           "new_twins_in_dedup_pool": sorted({p["flare23"] for p in pairs if p["in_dedup_pool"]}),
           "verdict": ("no new LiTS twins — the 1,366-case de-duplicated pool stands"
                       if not any(p["in_dedup_pool"] for p in pairs) else "NEW TWINS FLAGGED — re-dedup required")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"flagged pairs: {len(pairs)}; new twins in dedup pool: {out['new_twins_in_dedup_pool']}")
    print(out["verdict"])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
