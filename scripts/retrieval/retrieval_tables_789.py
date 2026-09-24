#!/usr/bin/env python3
"""Fill the v7 draft's retrieval TBDs — Tables 7 (primary controlled benchmark), 8 (large-pool
stress test on the 1,425-case corpus) and 9 (stratified by observability) — from the shipped
corpora only (CPU; no CT data).

Definitions follow the draft (§4.5, §4.11, captions of Tabs 7-9):
  - Relevance, Tab 7: reference-derived (GT-mask phenotypes) for both query and candidate.
    Relevance, Tabs 8-9: construction-defined from each candidate's own recorded phenotype
    (predicted records for the 113; the FLARE23 KG records for the 1,312).
  - Mismatch@10 (outcome-based, all methods): top-10 candidate whose recorded/reference profile
    has the queried finding (>=2 informative feature agreements with the query's phenotype at its
    queried organ) in a DIFFERENT organ and not in the queried organ.
  - Spurious@10 (path-based, graph methods): top-10 candidate sharing NO observed organ with the
    query (reasoning path rests on anatomy the query does not observe).
  - Stats: paired query-level bootstrap 95% CI (B=5,000) + two-sided sign-flip permutation p
    (P=20,000), Holm within family F1 (comparators vs proposed, Tab 7) / F2 (the two
    heterogeneous-coverage strata, Tab 9).

Organ universe: the corrected 4-unit pooled universe [liver, spleen, pancreas, kidney]; with it
the Tab 9 within-dataset identity check passes for all three datasets. The v6 run's universe
carried right/left kidney but no pooled `kidney`, which made every KiTS query score 0 (all ties)
under coverage-blind — the stored rows 3-4 of retrieval_on_predicted.json reproduce exactly under
that legacy universe + msd,lits,kits file order (emitted here as `legacy_v6_reproduction`).

-> results/retrieval/retrieval_tab7_primary.json
-> results/retrieval/retrieval_tab8_stress.json
-> results/retrieval/retrieval_tab9_stratified.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import retrieval_core_local as core
from retrieval_core_local import load_corpus, relevant, _agree_over, _dcg, phen

RES = os.environ.get("VKG_RES_DIR", os.path.join(ROOT, "results", "retrieval", "dedup"))
B_BOOT, B_PERM = 5000, 20000
rng = np.random.default_rng(12345)

UNIVERSE = ["liver", "spleen", "pancreas", "kidney"]
LEGACY_UNIVERSE = ["liver", "spleen", "pancreas", "right_kidney", "left_kidney"]
CATS = ["burden_cat", "multiplicity", "containment", "anatomic_location"]


# ---------------- per-organ "queried finding" agreement (for Mismatch@10) ----------------
def organ_agree(qp, cp):
    """>=2 informative feature agreements between the query's phenotype at its queried organ
    and a candidate's phenotype at organ o — the same bar the relevance rule uses."""
    n = 0
    for k in CATS:
        va = qp.get(k)
        if va == cp.get(k) and va not in ("none", "na", "unknown", None):
            n += 1
    return n >= 2


def mismatch10(q_rel, cand_rels):
    """Outcome-based cross-organ mismatch over the top-10 candidate rel-records."""
    qo = q_rel["observed_organs"][0]
    qp = q_rel["organs"][qo]
    flags = []
    for c in cand_rels:
        at_qo = qo in c["organs"] and organ_agree(qp, c["organs"][qo])
        elsewhere = any(organ_agree(qp, p) for o, p in c["organs"].items() if o != qo)
        flags.append(1.0 if (not at_qo and elsewhere) else 0.0)
    return float(np.mean(flags)) if flags else 0.0


# ---------------- evaluation ----------------
def per_query(sim, rel, qids, cids, mode, multi_organ_queries=False):
    """Per-query metrics. sim/rel: case_id -> record for ranking / for relevance.
    Ties broken by candidate list order (stable sort), as in the original scripts."""
    out = {}
    for qi in qids:
        cands = []
        for bi in cids:
            if bi == qi:
                continue
            s = core.similarity(sim[qi], sim[bi], mode)
            if s is None:
                continue
            cands.append((s, bi))
        if not cands:
            continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if relevant(rel[qi], rel[bi]) else 0 for _, bi in cands]
        if sum(rels) == 0:
            continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        top10 = [bi for _, bi in cands[:10]]
        qobs = set(sim[qi]["observed_organs"])
        m = {"p5": float(np.mean(rels[:5])), "p10": float(np.mean(rels[:10])),
             "ap": ap / sum(rels),
             "ndcg": _dcg(rels[:10]) / (_dcg(sorted(rels, reverse=True)[:10]) or 1),
             "spur10": float(np.mean([0.0 if (qobs & set(sim[bi]["observed_organs"])) else 1.0
                                      for bi in top10]))}
        m["mism10"] = (mismatch10(rel[qi], [rel[bi] for bi in top10])
                       if not multi_organ_queries and len(rel[qi]["observed_organs"]) == 1 else None)
        out[qi] = m
    return out


def agg(pq):
    r = lambda k: round(float(np.mean([v[k] for v in pq.values() if v[k] is not None])), 3) \
        if any(v[k] is not None for v in pq.values()) else None
    return {"n_queries": len(pq), "P@5": r("p5"), "P@10": r("p10"), "mAP": r("ap"),
            "nDCG": r("ndcg"), "spurious@10": r("spur10"), "mismatch@10": r("mism10")}


def paired(dA, dB, key="ap"):
    keys = sorted(set(dA) & set(dB))
    dif = np.array([dA[k][key] - dB[k][key] for k in keys])
    if not len(dif):
        return None
    obs = float(dif.mean())
    boots = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(B_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(B_PERM, len(dif)))
    p = float((np.sum(np.abs((signs * dif).mean(axis=1)) >= abs(obs)) + 1) / (B_PERM + 1))
    return {"n_common": len(keys), "delta": round(obs, 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)], "p_raw": p}


def holm(pdict):
    items = sorted([(k, v) for k, v in pdict.items() if v is not None], key=lambda kv: kv[1]["p_raw"])
    m = len(items)
    for rank, (k, v) in enumerate(items):
        v["p_holm"] = round(min(1.0, (m - rank) * v["p_raw"]), 5)
    return pdict


def main():
    os.makedirs(RES, exist_ok=True)
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"])
    gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    flare = {r["case_id"]: r for r in
             json.load(open(os.environ.get("VKG_FLARE_CORPUS", os.path.join(ROOT, "corpora", "corpus_flare23_kg.json"))))["records"]}
    q113 = [i for i in pred if i in gt]
    core.ORGAN_UNIVERSE = UNIVERSE

    # ======================= Table 7: primary controlled benchmark =======================
    # 113 queries x 112 candidates, reference-derived relevance (GT for query AND candidate).
    print("Table 7: primary controlled benchmark (113-case reference-evaluable pool)")
    t7_pq = {
        "(i) Reference phenotypes + gamma (upper bound)": per_query(gt, gt, q113, q113, "proposed"),
        "(ii) Predicted phenotypes + gamma (proposed)":   per_query(pred, gt, q113, q113, "proposed"),
        "(iii) Predicted, coverage-blind (no gamma)":     per_query(pred, gt, q113, q113, "coverage_blind"),
        "(iv) Predicted, organ-agnostic baseline":        per_query(pred, gt, q113, q113, "base"),
    }
    t7 = {k: agg(v) for k, v in t7_pq.items()}
    ref = "(ii) Predicted phenotypes + gamma (proposed)"
    f1 = {k: paired(t7_pq[k], t7_pq[ref]) for k in t7_pq if k != ref}
    holm(f1)
    for k, v in f1.items():
        t7[k]["dmAP_vs_ii"] = v
    # baseline rows (v)-(vii): aggregates from baselines_retrieval.json (same 113 queries,
    # relevance from the same GT records => valid for Tab 7; per-query data would need a CT rerun).
    bl = json.load(open(os.path.join(RES, "baselines_retrieval.json")))
    t7_baselines = {f"(v-vii) {k}": {**v, "mismatch@10": None,
                                     "dmAP_vs_ii": "TBD - needs per-query rerun (CT data)"}
                    for k, v in bl.items() if isinstance(v, dict)}
    for k, v in t7.items():
        print(f"  {k:48s} P@10={v['P@10']} mAP={v['mAP']} nDCG={v['nDCG']} mism@10={v['mismatch@10']}")

    # legacy v6 reproduction (stored retrieval_on_predicted.json rows 3-4 exactly)
    core.ORGAN_UNIVERSE = LEGACY_UNIVERSE
    legacy_ids = []
    for ds in ("msd", "lits", "kits"):
        legacy_ids += [i for i in load_corpus("predicted", datasets=[ds]) if i in gt]
    legacy = {
        "3_PREDICTED_no_gamma (coverage_blind)": agg(per_query(pred, gt, legacy_ids, legacy_ids, "coverage_blind")),
        "4_PREDICTED_base (organ-agnostic)":     agg(per_query(pred, gt, legacy_ids, legacy_ids, "base")),
    }
    core.ORGAN_UNIVERSE = UNIVERSE

    json.dump({"n_queries": len(q113),
               "relevance": "reference-derived (GT-mask phenotypes), query and candidate",
               "universe": UNIVERSE, "stats": f"B_boot={B_BOOT}, B_perm={B_PERM}, Holm family F1",
               "rows": t7, "baseline_rows_aggregate_only": t7_baselines,
               "legacy_v6_reproduction": {
                   "note": "Stored retrieval_on_predicted.json rows 3-4 reproduce ONLY under the "
                           "legacy universe (right/left kidney, no pooled kidney) + msd,lits,kits "
                           "file order: KiTS queries then score 0 for every candidate under "
                           "coverage-blind/base and the metrics ride on tie order. Rows here are "
                           "that reproduction; the corrected-universe rows above supersede them.",
                   "universe": LEGACY_UNIVERSE, "rows": legacy}},
              open(os.path.join(RES, "retrieval_tab7_primary.json"), "w"), indent=1)

    # ======================= Table 8: large-pool stress test =======================
    # 113 queries vs the 1,425-case pool; construction-defined relevance (own records).
    print("\nTable 8: large-pool stress test (1,425-case corpus)")
    pool_pred = {**pred, **flare}
    pool_gt = {**gt, **flare}
    cids = q113 + sorted(flare)
    t8_pq = {
        "(i) Reference phenotypes + gamma (upper bound)": per_query(pool_gt, pool_gt, q113, cids, "proposed"),
        "(ii) Predicted phenotypes + gamma (proposed)":   per_query(pool_pred, pool_pred, q113, cids, "proposed"),
        "(iii) Predicted, coverage-blind (no gamma)":     per_query(pool_pred, pool_pred, q113, cids, "coverage_blind"),
        "(iv) Predicted, organ-agnostic baseline":        per_query(pool_pred, pool_pred, q113, cids, "base"),
    }
    t8 = {k: agg(v) for k, v in t8_pq.items()}
    f1s = {k: paired(t8_pq[k], t8_pq[ref]) for k in t8_pq if k != ref}
    holm(f1s)
    for k, v in f1s.items():
        t8[k]["dmAP_vs_ii"] = v
    for k, v in t8.items():
        print(f"  {k:48s} P@10={v['P@10']} mAP={v['mAP']} nDCG={v['nDCG']} "
              f"spur@10={v['spurious@10']} mism@10={v['mismatch@10']}")
    json.dump({"n_queries": len(q113), "n_pool": len(cids),
               "relevance": "construction-defined (candidate's own recorded phenotype); FLARE23 "
                            "candidates use the shipped GT-derived KG records",
               "universe": UNIVERSE, "stats": f"B_boot={B_BOOT}, B_perm={B_PERM}, Holm family F1",
               "rows": t8},
              open(os.path.join(RES, "retrieval_tab8_stress.json"), "w"), indent=1)

    # ======================= Table 9: stratified =======================
    print("\nTable 9: stratified by observability")
    t9 = {}
    # (a) within-dataset control — identity check, no p-value
    wa = {}
    for ds in ("kits", "lits", "msd"):
        dsq = [i for i in q113 if pred[i]["dataset"] == ds]
        pr = per_query(pred, pred, dsq, dsq, "proposed")
        cb = per_query(pred, pred, dsq, dsq, "coverage_blind")
        identical = all(pr[k]["ap"] == cb[k]["ap"] and pr[k]["ndcg"] == cb[k]["ndcg"]
                        for k in pr if k in cb) and set(pr) == set(cb)
        wa[ds] = {"proposed": agg(pr), "coverage_blind": agg(cb), "rankings_identical": identical}
    pooled_pr, pooled_cb = {}, {}
    for ds in ("kits", "lits", "msd"):
        dsq = [i for i in q113 if pred[i]["dataset"] == ds]
        pooled_pr.update(per_query(pred, pred, dsq, dsq, "proposed"))
        pooled_cb.update(per_query(pred, pred, dsq, dsq, "coverage_blind"))
    t9["(a) within-dataset (identity check)"] = {
        "per_dataset": wa, "proposed": agg(pooled_pr), "coverage_blind": agg(pooled_cb),
        "identity_check_passed": all(w["rankings_identical"] for w in wa.values()),
        "delta_obs": None, "note": "identity check, no p-value (family F2 excludes it)"}
    # (b) single-organ -> FLARE23
    fids = sorted(flare)
    b_pr = per_query({**pred, **flare}, {**pred, **flare}, q113, fids, "proposed")
    b_cb = per_query({**pred, **flare}, {**pred, **flare}, q113, fids, "coverage_blind")
    # (c) FLARE23 -> single-organ (FLARE removed from pool)
    c_pr = per_query({**flare, **pred}, {**flare, **pred}, fids, q113, "proposed", multi_organ_queries=True)
    c_cb = per_query({**flare, **pred}, {**flare, **pred}, fids, q113, "coverage_blind", multi_organ_queries=True)
    f2 = {"(b) single-organ -> FLARE23": paired(b_pr, b_cb),
          "(c) FLARE23 -> single-organ": paired(c_pr, c_cb)}
    holm(f2)
    t9["(b) single-organ -> FLARE23"] = {"proposed": agg(b_pr), "coverage_blind": agg(b_cb),
                                         "delta_obs_mAP": f2["(b) single-organ -> FLARE23"]}
    t9["(c) FLARE23 -> single-organ"] = {"proposed": agg(c_pr), "coverage_blind": agg(c_cb),
                                         "delta_obs_mAP": f2["(c) FLARE23 -> single-organ"]}
    for k, v in t9.items():
        pr, cb = v["proposed"], v["coverage_blind"]
        d = v.get("delta_obs_mAP") or v.get("delta_obs")
        print(f"  {k:36s} prop mAP={pr['mAP']} (n={pr['n_queries']})  cblind mAP={cb['mAP']}  "
              f"dObs={(d or {}).get('delta') if isinstance(d, dict) else d}")
    json.dump({"relevance": "construction-defined, as Table 8",
               "universe": UNIVERSE, "stats": f"B_boot={B_BOOT}, B_perm={B_PERM}, Holm family F2 "
                                              "over strata (b),(c)",
               "strata": t9},
              open(os.path.join(RES, "retrieval_tab9_stratified.json"), "w"), indent=1)
    print("\nSaved retrieval_tab7_primary.json / retrieval_tab8_stress.json / retrieval_tab9_stratified.json")


if __name__ == "__main__":
    main()
