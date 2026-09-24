#!/usr/bin/env python3
"""Per-query effect of removing the fourteen third-pass twins (12 KiTS + 2 LiTS) — the
companion to eleven_kits_effect_2026-09-22.json, same construction: stress pool,
configuration (ii), construction relevance, metrics for each affected query at pool 1,323
(twin present) vs pool 1,309 (twin removed), plus the all-113 means.

Each removed record was an S = 1.0 candidate of its own query (same scan, near-verbatim
extracted phenotype), so the expectation from the eleven-pair precedent is no metric change:
remaining relevant S = 1.0 ties back-fill the top-10.  This report verifies that expectation
per query instead of assuming it.

Runs AFTER twin_census_third_pass.py (the dedup corpus is the 1,196 state); the 1,323 pool is
reconstructed by adding the 14 back from the full corpus_flare23_kg.json.

-> results/retrieval/dedup/fourteen_twins_effect_2026-09-24.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import retrieval_core_local as core                      # noqa: E402
from retrieval_core_local import load_corpus             # noqa: E402
import retrieval_tables_789 as T                         # noqa: E402

OUT = os.path.join(ROOT, "results", "retrieval", "dedup", "fourteen_twins_effect_2026-09-24.json")


def main():
    core.ORGAN_UNIVERSE = T.UNIVERSE
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"])
    gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    q113 = sorted(i for i in pred if i in gt)
    full = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg.json")))["records"]}
    dedup = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg_dedup.json")))["records"]}
    third = json.load(open(os.path.join(ROOT, "results", "audit", "axis_normalized_twin_census.json")))["third_pass_2026-09-24"]
    pairs = {t["flare"]: t["query"] for t in third["pairs_confirmed"]}
    assert len(dedup) == 1196 and not (set(pairs) & set(dedup)), "run twin_census_third_pass.py first"
    flare_after = dedup
    flare_before = {**dedup, **{c: full[c] for c in pairs}}
    assert len(flare_before) == 1210

    affected_all = sorted(set(pairs.values()))
    res = {}
    for tag, flare in (("pool_1323", flare_before), ("pool_1309", flare_after)):
        pool = {**pred, **flare}
        cids = q113 + sorted(flare)
        pq = T.per_query(pool, pool, q113, cids, "proposed")
        res[tag] = pq
    keys = ("p5", "p10", "ap", "ndcg")
    affected = [q for q in affected_all if all(q in res[t] for t in res)]
    not_eval = [q for q in affected_all if q not in affected]

    def mean(pq, qs):
        return {k: round(sum(pq[q][k] for q in qs) / len(qs), 4) for k in keys}

    out = {"question": "per-query effect of removing the fourteen third-pass twins (each was an "
                       "S=1.0 candidate of its own query) - stress pool, configuration (ii), "
                       "construction relevance",
           "fourteen_pairs": "results/audit/axis_normalized_twin_census.json third_pass_2026-09-24",
           "per_query": {q: {"pool_1323": {k: round(res["pool_1323"][q][k], 4) for k in keys},
                             "pool_1309": {k: round(res["pool_1309"][q][k], 4) for k in keys}}
                         for q in affected},
           "summary": {"fourteen_queries_mean": {t: mean(res[t], affected) for t in res},
                       "all_113_mean": {t: mean(res[t], [q for q in q113 if q in res[t]]) for t in res}},
           "not_evaluable": not_eval,
           "note": "Removal deletes one RELEVANT S=1.0 candidate per query; remaining relevant "
                   "candidates back-fill the top-10, so P@k moves only where the twin was not "
                   "interchangeable with other S=1.0 ties."}
    json.dump(out, open(OUT, "w"), indent=1)
    for q in affected:
        b, a = out["per_query"][q]["pool_1323"], out["per_query"][q]["pool_1309"]
        delta = {k: round(a[k] - b[k], 4) for k in keys if abs(a[k] - b[k]) > 1e-9}
        print(q, "no change" if not delta else f"CHANGED {delta}")
    print(json.dumps(out["summary"], indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
