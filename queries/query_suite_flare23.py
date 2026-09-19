#!/usr/bin/env python3
"""Draft Table 6, FLARE23 rows: system-vs-reference P/R/F1 for every query family — in particular the
compositional / cross-organ families Q4a, Q4b, Q5 that are 100 % indeterminate on single-organ cases —
using the PREDICTED FLARE23 graph (corpora/corpus_predicted_flare23.json, semi-oracle AUSAM protocol as the
single-organ predicted corpora) against the reference FLARE23 graph (corpora/corpus_flare23_kg.json), on the
FLARE23 cases that have a CT in the release copy.  Verdict functions and scoring are those of
query_suite_eval.py (three-valued verdicts; P/R/F1 over determinate pairs).
Output: results/queries/query_suite_flare23.json
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import query_suite_eval as Q                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUT = os.path.join(ROOT, "results", "queries", "query_suite_flare23.json")
ORGANS = ["liver", "kidney", "pancreas", "spleen"]


def main():
    pred = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_predicted_flare23.json")))["records"]}
    gt = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg.json")))["records"]}
    ids = sorted(i for i in pred if i in gt)
    per_org = lambda f: [(o, (lambda r, o=o: f(r, o))) for o in ORGANS]
    rows = [
        Q.score("Q2 organ-specific lookup (pooled liver/kidney/pancreas/spleen)", per_org(Q.v_organ_lookup), pred, gt, ids),
        Q.score("Q3 phenotype retrieval: largest burden bin (pooled)", per_org(Q.v_high_bin), pred, gt, ids),
        Q.score("Q4a compositional: hepatic>5cm AND renal lesion", [("", lambda r: Q.v_compositional(r, "kidney"))], pred, gt, ids),
        Q.score("Q4b compositional: hepatic>5cm AND splenic lesion", [("", lambda r: Q.v_compositional(r, "spleen"))], pred, gt, ids),
        Q.score("Q5 cross-organ involvement (>=2 organs)", [("", Q.v_cross_organ)], pred, gt, ids),
        Q.score("Q7a high tumor burden (observed anatomy)", [("", Q.v_high_burden_any)], pred, gt, ids),
        Q.score("Q7b multifocal disease in one organ (pooled)", per_org(Q.v_multifocal), pred, gt, ids),
    ]
    # reference-side counts on the same cases (how many true instances the family actually has)
    ref_counts = {}
    for r, fns in zip(rows, [per_org(Q.v_organ_lookup), per_org(Q.v_high_bin), [("", lambda r: Q.v_compositional(r, "kidney"))],
                              [("", lambda r: Q.v_compositional(r, "spleen"))], [("", Q.v_cross_organ)], [("", Q.v_high_burden_any)],
                              per_org(Q.v_multifocal)]):
        vs = [f(gt[c]) for c in ids for _, f in fns]
        ref_counts[r["family"]] = {"T": vs.count("T"), "F": vs.count("F"), "U": vs.count("U")}
    det = [r for r in rows if r["F1"] is not None]
    macro = {"macro_P": round(sum(r["precision"] for r in det) / len(det), 3),
             "macro_R": round(sum(r["recall"] for r in det) / len(det), 3),
             "macro_F1": round(sum(r["F1"] for r in det) / len(det), 3), "n_evaluable_families": len(det)}
    print(f"Query suite on {len(ids)} FLARE23 cases (system=predicted semi-oracle graph, reference=GT graph):")
    for r in rows:
        rc = ref_counts[r["family"]]
        print(f"  {r['family']:60s} sys T/F/U={r['N_T']}/{r['N_F']}/{r['N_U']}  ref T={rc['T']}  P={r['precision']} R={r['recall']} F1={r['F1']} indet={r['indet_rate']}")
    print(f"  macro over evaluable families: P={macro['macro_P']} R={macro['macro_R']} F1={macro['macro_F1']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"n_cases": len(ids), "note": "FLARE23 cases with a CT in the release copy; predicted graph = semi-oracle AUSAM "
               "(build_predicted_corpus_flare23.py); reference graph = corpus_flare23_kg.json", "rows": rows,
               "reference_counts": ref_counts, "macro_over_evaluable": macro}, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
