#!/usr/bin/env python3
"""Draft Table 10 — rank of the intended same-organ FLARE23 target within the 1,425-case pool for three
single-organ queries, regenerated under the corrected organ universe with the Tab. 7-9 scorer
(retrieval_core_local / retrieval_tables_789).  The v6 definition (crossdataset_query_results.json,
'target_first_rank') is the rank at which the first candidate from the target dataset appears; we report that
and, more strictly, the rank of the first FLARE23 candidate that is RELEVANT to the query under the
construction-defined relevance of Tab. 8 — for (ii) proposed, (iii) coverage-blind and (iii') imputed-absent.
Output: results/retrieval/tab10_example_ranks.json
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import retrieval_core_local as core                                    # noqa: E402
from retrieval_core_local import load_corpus, similarity, relevant     # noqa: E402
import retrieval_tables_789 as T                                       # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(os.environ.get("VKG_RES_DIR", os.path.join(ROOT, "results", "retrieval", "dedup")), "tab10_example_ranks.json")
QUERIES = [("Pancreas -> FLARE23, contained tumor", "pancreas_041"),
           ("LiTS -> FLARE23, multifocal liver", "volume-101"),
           ("Pancreas -> FLARE23, high burden", "pancreas_015")]
MODES = {"proposed": "proposed", "coverage_blind": "coverage_blind", "imputed_absent": "imputed_absent"}


def main():
    core.ORGAN_UNIVERSE = T.UNIVERSE
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"]); gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    flare = {r["case_id"]: r for r in json.load(open(os.environ.get("VKG_FLARE_CORPUS", os.path.join(ROOT, "corpora", "corpus_flare23_kg.json"))))["records"]}
    pool = {**pred, **flare}; cids = [i for i in pred if i in gt] + sorted(flare)
    out = {"pool_size": len(cids), "universe": T.UNIVERSE, "queries": []}
    for title, q in QUERIES:
        qo = pred[q]["observed_organs"][0]; ph = pred[q]["organs"][qo]
        row = {"title": title, "query": q, "query_phenotype": {k: ph[k] for k in ("burden_cat", "multiplicity", "containment", "anatomic_location")}}
        for label, mode in MODES.items():
            cands = [(similarity(pool[q], pool[b], mode), b) for b in cids if b != q]
            cands = [(s, b) for s, b in cands if s is not None]
            cands.sort(key=lambda x: -x[0])                    # stable: ties keep candidate-list order, as in Tab. 7-9
            ranked = [b for _, b in cands]
            first_flare = next((i + 1 for i, b in enumerate(ranked) if b in flare), None)
            first_rel_flare = next((i + 1 for i, b in enumerate(ranked) if b in flare and relevant(pool[q], pool[b])), None)
            n_rel_flare = sum(1 for b in flare if relevant(pool[q], pool[b]))
            fl_only = [b for b in ranked if b in flare]                        # rank among FLARE23 candidates only
            first_rel_within_flare = next((i + 1 for i, b in enumerate(fl_only) if relevant(pool[q], pool[b])), None)
            n_single_above = (first_flare - 1) if first_flare else None       # single-organ candidates ranked above it
            row[label] = {"first_flare23_rank": first_flare, "first_relevant_flare23_rank": first_rel_flare,
                          "rank_within_flare23_only": first_rel_within_flare, "single_organ_candidates_above": n_single_above,
                          "n_relevant_flare23_candidates": n_rel_flare, "n_ranked": len(ranked)}
        out["queries"].append(row)
        print(f"{title:38s} {q:13s} " + "  ".join(f"{m}: pool rank #{row[m]['first_relevant_flare23_rank']} (within FLARE23 #{row[m]['rank_within_flare23_only']}, {row[m]['single_organ_candidates_above']} single-organ above)" for m in MODES))
    json.dump(out, open(OUT, "w"), indent=1); print(f"-> {OUT}")


if __name__ == "__main__":
    main()
