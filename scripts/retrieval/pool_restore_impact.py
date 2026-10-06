#!/usr/bin/env python3
"""Restore the two FLARE23 records the de-duplication removed in error, and measure what it changes (10-03).

The pool rule is: remove FLARE23 records that are the same scan as one of the 113 QUERY cases (a query must not
retrieve itself). Checked against the complete twin census (results/audit/flare23_twin_census_complete_2026-10-03.json),
114 of the 116 removals are query twins. Two are not:
  FLARE23_0042  removed as a twin of query pancreas_122 -- it is voxel-identical to pancreas_274 (a training case,
                not a query); pancreas_122's real twin FLARE23_1144 is removed separately.
  FLARE23_0102  removed 09-15 as the LiTS twin of liver_64 (= volume-64) when the LiTS geometry could not be
                tested; volume-64 is not one of the 21 LiTS queries.
Neither duplicates anything in the pool, so both return: pool 1,309 -> 1,311 (113 + 1,198).

Impact: every query re-ranked by the benchmark scorer (retrieval_core_local.similarity) on both pools, for the KG
configurations of export_rankings_113.py (the feature baselines need the cluster feature cache), same candidate
ordering rule (sorted ids; ties keep that order). Reported: queries whose top-10 SET or ORDER changes, and where the
two restored records land.

  python pool_restore_impact.py
-> corpora/reissue_2026-10-03/corpus_flare23_kg_dedup_1311.json (+ results/retrieval/pool_restore_impact_2026-10-03.json)
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
import export_rankings_113 as X                                          # noqa: E402

RESTORE = ["flare23_FLARE23_0042", "flare23_FLARE23_0102"]
OUTC = os.path.join(ROOT, "corpora", "reissue_2026-10-03", "corpus_flare23_kg_dedup_1311.json")
OUT = os.path.join(ROOT, "results", "retrieval", "pool_restore_impact_2026-10-03.json")


def main():
    X.core.ORGAN_UNIVERSE = X.T.UNIVERSE
    pred = X.load_corpus("predicted", datasets=["kits", "lits", "msd"])
    gt = X.load_corpus("gt", datasets=["kits", "lits", "msd"])
    q113 = sorted(i for i in pred if i in gt)
    full = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg.json")))["records"]}
    dd = json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg_dedup.json")))
    old = {r["case_id"]: r for r in dd["records"]}
    new = dict(old)
    for c in RESTORE:
        new[c] = full[c]
    res = {"note": __doc__.strip(), "pool_old": 113 + len(old), "pool_new": 113 + len(new), "configs": {}}
    for cfg, (side, mode) in X.KG_CFG.items():
        sims = {}
        for tag, fl in (("old", old), ("new", new)):
            sim = {**(gt if side == "gt" else pred), **fl}
            cids = q113 + sorted(fl)
            sims[tag] = {q: [c for _, c in X.rank_kg(sim, q, cids, mode, 30)] for q in q113}
        set_changed = [q for q in q113 if set(sims["old"][q][:10]) != set(sims["new"][q][:10])]
        order_changed = [q for q in q113 if sims["old"][q][:10] != sims["new"][q][:10]]
        landed = {c: sorted((q, sims["new"][q].index(c) + 1) for q in q113 if c in sims["new"][q][:10]) for c in RESTORE}
        res["configs"][cfg] = {"top10_set_changed": len(set_changed), "top10_order_changed": len(order_changed),
                               "queries_set_changed": set_changed, "restored_in_a_top10": landed}
        print(cfg, "set changed", len(set_changed), "order changed", len(order_changed),
              {c.split("_")[-1]: len(v) for c, v in landed.items()}, flush=True)
    os.makedirs(os.path.dirname(OUTC), exist_ok=True)
    nd = dict(dd)
    nd["records"] = [new[c] for c in sorted(new)]
    nd["note"] = (dd.get("note", "") + " | 2026-10-03: FLARE23_0042 and FLARE23_0102 RESTORED - neither is a twin of a "
                  "query (0042 = pancreas_274, 0102 = volume-64, both non-query; complete census "
                  "results/audit/flare23_twin_census_complete_2026-10-03.json). Pool 1,309 -> 1,311.")
    json.dump(nd, open(OUTC, "w"))
    json.dump(res, open(OUT, "w"), indent=1)
    print("->", OUTC, "\n->", OUT)


if __name__ == "__main__":
    main()
