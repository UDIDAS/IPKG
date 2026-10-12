#!/usr/bin/env python3
"""Reference pack for verifying the corrected IPKG application (study lead 10-08, the pipeline team items 1-2): what each of the four
case pages must show, built from the final experimental artifacts and independent of the application's own code path.

Per case (lits_volume-76, kits_case_00067, msd_pancreas_125, flare23_FLARE23_0405):
  masks       md5 of the delivered semi-oracle mask (final_cases_2026-09-19/masks/); must be UNCHANGED
  record      the approved KG record = the round-2 re-issue of the delivered case record, with the 10-03 app-pool
              rule applied (an organ without tumour has no sub-site: 'unknown' -> 'na', status no_tumor)
              (records_reissue/final_cases_2026-09-19/<case>/record_predicted.json): observed organs (observability),
              per-organ has_tumor / tumour volume / burden / multiplicity / containment / sub-site (+ SNOMED, status,
              provenance) and the delivered word kept beside a corrected one
  cutpoints   burden tercile cut-points of the case's collection (burden_tercile_cutpoints.json)
  retrieval   top-10 of configuration (ii) on the 1,311-case pool, corrected arm (void sub-site axis dropped,
              precedence-resolved sub-site on both sides), ranked by the verified table code
              (build_retrieval_tables.Sim / arm_pool, branch abdomen/ipkg-v2), NOT by the application's
              ipkg_retrieval_corrected path, so the check is independent of the code under test
  pool        1,311 = 113 + 1,198

  APP_TREE=<abdomen/ipkg-v2 checkout> IPKG_ROUND2_DIR=<round2_2026-10-01> FINAL_CASES=<final_cases_2026-09-19> \
      python build_app_reference_2026-10-08.py
-> results/app_checks/app_reference_2026-10-08.json
"""
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
APP = os.environ.get("APP_TREE", "/path/to/ipkg-v2")
R2 = os.environ.get("IPKG_ROUND2_DIR", "/path/to/round2_2026-10-01")
FC = os.environ.get("FINAL_CASES", "/path/to/final_cases_2026-09-19")
OUT = os.path.join(ROOT, "results", "app_checks", "app_reference_2026-10-08.json")
CASES = {"lits_volume-76": ("volume-76", "lits"), "kits_case_00067": ("case_00067", "kits"),
         "msd_pancreas_125": ("pancreas_125", "msd"), "flare23_FLARE23_0405": ("FLARE23_0405", "flare23")}
FIELDS = ("has_tumor", "tumor_volume_cm3", "burden_cat", "multiplicity", "containment", "anatomic_location",
          "anatomic_location_snomed", "anatomic_location_status", "anatomic_location_delivered")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def tie_rule(scored):
    """The 10th score, the candidates strictly above it, and every candidate tied at it. A displayed top-10 is correct if
    it holds all candidates above the cutoff and fills the rest from the tied set (pool-order tie-break gives the list above)."""
    s10 = scored[9][3]
    return {"score_at_rank10": round(s10, 4),
            "above_cutoff": [c for _, _, c, s in scored if s > s10 + 1e-12],
            "tied_at_cutoff": [c for _, _, c, s in scored if abs(s - s10) < 1e-12],
            "tie_break": "pool assembly order (kits, lits, msd, then the 1,198 FLARE23 records), stable sort"}


def main():
    sys.argv = [sys.argv[0]]
    os.chdir(APP); sys.path.insert(0, APP)
    import build_retrieval_tables as B
    pool = B.pool_1311()
    corr = B.arm_pool("corrected", pool)
    sim = B.Sim(corr, B.ARMS["corrected"][1])
    order = list(pool)
    cut = json.load(open(os.path.join(FC, "burden_tercile_cutpoints.json")))
    out = {"note": __doc__.strip(), "pool_size": len(pool), "n_flare23_in_pool": sum(c.startswith("flare23_") for c in pool),
           "app_tree_commit": os.popen(f"git -C {APP} rev-parse HEAD").read().strip(), "cases": {}}
    for folder, (cid, ds) in CASES.items():
        rec = json.load(open(os.path.join(R2, "records_reissue", "final_cases_2026-09-19", folder, "record_predicted.json")))
        for o, p in (rec.get("organs") or {}).items():       # the 10-03 app-pool re-issue rule (app_pool_reissue_2026-10-03.json):
            if not p.get("has_tumor") and p.get("anatomic_location") == "unknown":   # no tumour -> no sub-site: 'na'
                p["anatomic_location"] = "na"; p["anatomic_location_status"] = "no_tumor"
        q = B.VOCAB["P"](rec)
        qid = rec.get("case_id")
        scored = []
        for i, c in enumerate(order):
            if c == qid or c.replace("flare23_", "") == cid:
                continue
            s = sim(q, corr[c], "proposed")
            if s is not None:
                scored.append((-s, i, c, s))
        scored.sort()
        mask = os.path.join(FC, "masks", f"{cid}.nii.gz")
        out["cases"][folder] = {
            "case_id": qid, "collection": ds,
            "mask_md5_must_be_unchanged": md5(mask) if os.path.exists(mask) else None,
            "observed_organs": rec.get("observed_organs"),
            "organs": {o: {k: p.get(k) for k in FIELDS if k in p} for o, p in (rec.get("organs") or {}).items()},
            "burden_cutpoints": cut.get(ds),
            "query_in_pool": qid in pool,
            "top10_ii_corrected_1311": [{"rank": r + 1, "case_id": c, "S": round(s, 4)} for r, (_, _, c, s) in enumerate(scored[:10])],
            "top10_tie_rule": tie_rule(scored),
            "n_ranked": len(scored)}
        print(folder, "| in pool:", qid in pool, "| top-10:", [c for _, _, c, _ in scored[:10]])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT)


if __name__ == "__main__":
    main()
