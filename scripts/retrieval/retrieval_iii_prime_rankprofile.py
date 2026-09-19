#!/usr/bin/env python3
"""Rank-level behaviour of configuration (iii') on the 1,425-case pool: why mAP falls (0.999 -> 0.757) while
P@10 / nDCG@10 barely move.  For each of the 113 queries, under (ii) and (iii'), we export the ranks of ALL
relevant candidates (construction-defined relevance, as Tab. 8), the rank of the first lesion-free FLARE23
candidate, and how many relevant candidates are displaced below it.  Output: results/retrieval/retrieval_iii_prime_rankprofile.json
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import retrieval_core_local as core                                    # noqa: E402
from retrieval_core_local import load_corpus, similarity, relevant     # noqa: E402
import retrieval_tables_789 as T                                       # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "retrieval", "retrieval_iii_prime_rankprofile.json")


def lesion_free(rec):
    return not any(d["has_tumor"] for d in rec["organs"].values())


def main():
    core.ORGAN_UNIVERSE = T.UNIVERSE
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"]); gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    flare = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg.json")))["records"]}
    pool = {**pred, **flare}; q113 = [i for i in pred if i in gt]; cids = q113 + sorted(flare)
    n_lf = sum(lesion_free(flare[c]) for c in flare)
    per = {}
    for q in q113:
        rel = {b for b in cids if b != q and relevant(pool[q], pool[b])}
        if not rel:
            continue
        row = {"n_relevant": len(rel)}
        for label, mode in (("ii", "proposed"), ("iii_prime", "imputed_absent")):
            cands = [(similarity(pool[q], pool[b], mode), b) for b in cids if b != q]
            cands = [(s, b) for s, b in cands if s is not None]; cands.sort(key=lambda x: -x[0])
            ranked = [b for _, b in cands]
            rr = sorted(i + 1 for i, b in enumerate(ranked) if b in rel)
            first_lf = next((i + 1 for i, b in enumerate(ranked) if b in flare and lesion_free(flare[b])), None)
            hit = ap = 0.0
            for i, b in enumerate(ranked):
                if b in rel:
                    hit += 1; ap += hit / (i + 1)
            row[label] = {"AP": ap / len(rel), "P@10": float(np.mean([b in rel for b in ranked[:10]])),
                          "relevant_ranks_median": float(np.median(rr)), "relevant_ranks_max": rr[-1],
                          "first_lesion_free_flare23_rank": first_lf,
                          "relevant_below_first_lesion_free": int(sum(r > first_lf for r in rr)) if first_lf else 0,
                          "lesion_free_in_top10": int(sum(1 for b in ranked[:10] if b in flare and lesion_free(flare[b])))}
        row["first_displaced_relevant_rank_iii_prime"] = next((r for r in sorted(i + 1 for i, b in enumerate(ranked) if b in rel)
                                                              if row["iii_prime"]["first_lesion_free_flare23_rank"] and r > row["iii_prime"]["first_lesion_free_flare23_rank"]), None)
        per[q] = row
    agg = {}
    for label in ("ii", "iii_prime"):
        agg[label] = {k: round(float(np.mean([v[label][k] for v in per.values() if v[label][k] is not None])), 3)
                      for k in ("AP", "P@10", "relevant_ranks_median", "relevant_ranks_max", "first_lesion_free_flare23_rank",
                                "relevant_below_first_lesion_free", "lesion_free_in_top10")}
    frac = [v["iii_prime"]["relevant_below_first_lesion_free"] / v["n_relevant"] for v in per.values()]
    fd = [v["first_displaced_relevant_rank_iii_prime"] for v in per.values() if v["first_displaced_relevant_rank_iii_prime"]]
    out = {"n_queries": len(per), "n_lesion_free_flare23_candidates": n_lf, "mean_relevant_per_query": round(float(np.mean([v["n_relevant"] for v in per.values()])), 1),
           "aggregate": agg,
           "iii_prime_fraction_of_relevant_displaced_below_first_lesion_free": {"mean": round(float(np.mean(frac)), 3), "median": round(float(np.median(frac)), 3)},
           "iii_prime_first_displaced_relevant_rank": {"n_queries_with_displacement": len(fd), "median": float(np.median(fd)) if fd else None,
                                                       "IQR": [float(np.percentile(fd, 25)), float(np.percentile(fd, 75))] if fd else None,
                                                       "min": min(fd) if fd else None},
           "per_query": per}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{len(per)} queries; {n_lf} lesion-free FLARE23 candidates in the pool; mean relevant/query {out['mean_relevant_per_query']}")
    for label in ("ii", "iii_prime"):
        print(f"  ({label}) " + "  ".join(f"{k}={v}" for k, v in agg[label].items()))
    print("  (iii') fraction of relevant displaced below the first lesion-free candidate:", out["iii_prime_fraction_of_relevant_displaced_below_first_lesion_free"])
    print("  (iii') first displaced relevant rank:", out["iii_prime_first_displaced_relevant_rank"])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
