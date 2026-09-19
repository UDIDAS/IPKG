#!/usr/bin/env python3
"""Configuration (iii') — the true Prop.-1 ablation proposed in docs/VKG_gamma_resolution_plan-0913.md §4:
impute "no finding" for every organ the query does not observe and score it against the candidate's
observed-absent (n term) and observed-present (p term) findings, instead of the flat one-sided penalty of the
coverage-blind ablation (iii).  Reported for the Tab. 7 pool, the Tab. 8 1,425-case pool and the Tab. 9 strata,
next to (ii) and (iii), with paired Δ vs (ii).  Supplementary — the v8.1 tables are not edited by this script.
Output: results/retrieval/retrieval_iii_prime.json
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import retrieval_core_local as core                      # noqa: E402
from retrieval_core_local import load_corpus             # noqa: E402
import retrieval_tables_789 as T                         # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(os.environ.get("VKG_RES_DIR", os.path.join(ROOT, "results", "retrieval")), "retrieval_iii_prime.json")
MODES = {"(ii) proposed": "proposed", "(iii) coverage-blind": "coverage_blind", "(iii') imputed-absent": "imputed_absent"}


def block(sim, rel, qids, cids, multi=False):
    pq = {k: T.per_query(sim, rel, qids, cids, m, multi_organ_queries=multi) for k, m in MODES.items()}
    rows = {k: T.agg(v) for k, v in pq.items()}
    for k in ("(iii) coverage-blind", "(iii') imputed-absent"):
        rows[k]["dmAP_vs_ii"] = T.paired(pq[k], pq["(ii) proposed"])
    ident = sorted(pq["(iii') imputed-absent"]) == sorted(pq["(ii) proposed"]) and \
        all(abs(pq["(iii') imputed-absent"][q]["ap"] - pq["(ii) proposed"][q]["ap"]) < 1e-12 for q in pq["(ii) proposed"])
    rows["ap_identical_iii_prime_vs_ii"] = ident
    return rows


def main():
    core.ORGAN_UNIVERSE = T.UNIVERSE
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"]); gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    flare = {r["case_id"]: r for r in json.load(open(os.environ.get("VKG_FLARE_CORPUS", os.path.join(ROOT, "corpora", "corpus_flare23_kg.json"))))["records"]}
    q113 = [i for i in pred if i in gt]
    out = {"tab7_pool": block(pred, gt, q113, q113)}
    pool = {**pred, **flare}; cids = q113 + sorted(flare)
    out["tab8_pool_1425"] = block(pool, pool, q113, cids)
    # Tab. 9 (a): within-dataset identity check — (iii') must tie (ii) exactly, per dataset
    out["tab9_a_within_dataset"] = {}
    for ds in ("kits", "lits", "msd"):
        dsq = [i for i in q113 if pred[i]["dataset"] == ds]
        pq = {k: T.per_query(pred, pred, dsq, dsq, m) for k, m in MODES.items()}
        rows = {k: T.agg(v) for k, v in pq.items()}
        rows["ap_identical_iii_prime_vs_ii"] = all(abs(pq["(iii') imputed-absent"][q]["ap"] - pq["(ii) proposed"][q]["ap"]) < 1e-12 for q in pq["(ii) proposed"])
        rows["ap_identical_iii_vs_ii"] = all(abs(pq["(iii) coverage-blind"][q]["ap"] - pq["(ii) proposed"][q]["ap"]) < 1e-12 for q in pq["(ii) proposed"])
        out["tab9_a_within_dataset"][ds] = rows
    out["tab9_b_single_to_flare23"] = block(pool, pool, q113, sorted(flare))
    out["tab9_c_flare23_to_single"] = block(pool, pool, sorted(flare), q113, multi=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for tab, rows in out.items():
        print(tab)
        if tab == "tab9_a_within_dataset":
            for ds, r in rows.items():
                print(f"  {ds}: (ii) mAP={r['(ii) proposed']['mAP']}  (iii') mAP={r[chr(40)+'iii'+chr(39)+') imputed-absent']['mAP']}  AP-identical (iii')={r['ap_identical_iii_prime_vs_ii']}  (iii)={r['ap_identical_iii_vs_ii']}")
            continue
        for k, v in rows.items():
            if isinstance(v, dict):
                d = v.get("dmAP_vs_ii")
                print(f"  {k:24s} n={v['n_queries']} P@10={v['P@10']} mAP={v['mAP']} nDCG={v['nDCG']} spur={v['spurious@10']} mism={v['mismatch@10']}"
                      + (f"  dmAP {d['delta']:+.3f} {d['ci95']} p={d['p_raw']:.2g}" if d else ""))
            else:
                print(f"  {k}: {v}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
