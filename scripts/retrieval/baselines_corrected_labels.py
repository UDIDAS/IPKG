#!/usr/bin/env python3
"""Primary benchmark (draft Table 6), baseline rows (v)-(vii') re-scored under the CORRECTED relevance labels
(the study lead 10-05 item 1). The baselines do not read the sub-site, so their rankings are the published ones; only the gold
standard changes.

Rankings: baselines_tab7_tab8.py's radiomics / image-embedding retrieval, from the feature cache it builds
($VKG_DATA/baseline_features.npz; rebuilt 10-05 from the 113 CTs and reference masks).
Labels: the published (D) and corrected (P, precedence-resolved sub-site) relevance labels of the 113 x 112 pairs,
exported from the co-author's build_retrieval_tables.py (branch abdomen/ipkg-v2) together with its per-query graph rows and
its reference records in both vocabularies (results/retrieval/inputs/). The corrected labels flip 146 of 12,656
pairs, all to irrelevant.

Gates (the script refuses to write if any fails):
  1. under the published labels, rows (v)-(vii') reproduce the frozen baselines_tab7_tab8.json to the 3rd decimal;
  2. the exported labels equal retrieval_core_local.relevant() on the exported reference records, in both vocabularies;
  3. the exported graph rows reproduce the co-author's corrected table ((ii) mAP 0.967).
Statistics as in Table 6: paired query-level bootstrap 95% CI and sign-flip permutation p against (ii) under the
same labels; Holm over family F1 = the seven comparators (i), (iii), (iv), (v), (vi), (vii), (vii'); (iii') raw p.

  VKG_DATA=... python baselines_corrected_labels.py -> results/retrieval/dedup/baselines_corrected_labels_2026-10-05.json
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import retrieval_core_local as core                                     # noqa: E402
from retrieval_core_local import load_corpus, relevant, _dcg             # noqa: E402
import retrieval_tables_789 as T                                         # noqa: E402
from baselines_tab7_tab8 import cos                                      # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
INP = os.path.join(ROOT, "results", "retrieval", "inputs")
FEAT = os.path.join(os.environ.get("VKG_DATA", "/path/to/VKG_data"), "baseline_features.npz")
FROZEN = os.path.join(ROOT, "results", "retrieval", "dedup", "baselines_tab7_tab8.json")
OUT = os.path.join(ROOT, "results", "retrieval", "dedup", "baselines_corrected_labels_2026-10-05.json")
B_BOOT, B_PERM, SEED = 5000, 20000, 20261003
BASE = {"(v)": ("Radiomics-vector (organ-agnostic)", "R", False),
        "(vi)": ("Radiomics-vector (organ-conditioned)", "R", True),
        "(vii)": ("Image-embedding (organ-agnostic)", "E", False),
        "(vii')": ("Image-embedding (organ-conditioned)", "E", True)}
FROZEN_KEY = {"(v)": "(v) Radiomics-vector (organ-agnostic)", "(vi)": "(vi) Radiomics-vector (organ-conditioned)",
              "(vii)": "(vii) Image-embedding (organ-agnostic)", "(vii')": "(vii') Image-embedding (organ-conditioned)"}


def per_query(F, sim_recs, rel_recs, lab, qids, organ_conditioned, cids=None):
    """baselines_tab7_tab8.per_query_feat with the relevance read from a label matrix (candidates = cids or qids)."""
    out = {}
    cids = cids or qids
    for qi in qids:
        qo = set(sim_recs[qi]["observed_organs"]) & set(F[qi])
        cands = []
        for bi in cids:
            if bi == qi:
                continue
            if organ_conditioned:
                shared = qo & set(sim_recs[bi]["observed_organs"]) & set(F[bi])
                if not shared:
                    continue
                s = float(np.mean([cos(F[qi][o], F[bi][o]) for o in shared]))
            else:
                s = cos(F[qi]["__union__"], F[bi]["__union__"])
            cands.append((s, bi))
        cands.sort(key=lambda x: -x[0])
        rels = [lab[qi][bi] for _, bi in cands]
        if not cands or sum(rels) == 0:
            continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        top10 = [bi for _, bi in cands[:10]]
        qobs = set(sim_recs[qi]["observed_organs"])
        out[qi] = {"p5": float(np.mean(rels[:5])), "p10": float(np.mean(rels[:10])), "ap": ap / sum(rels),
                   "ndcg": _dcg(rels[:10]) / (_dcg(sorted(rels, reverse=True)[:10]) or 1),
                   "spur10": float(np.mean([0.0 if (qobs & set(sim_recs[bi]["observed_organs"])) else 1.0 for bi in top10])),
                   "mism10": T.mismatch10(rel_recs[qi], [rel_recs[bi] for bi in top10])
                   if len(rel_recs[qi]["observed_organs"]) == 1 else None}
    return out


def agg(pq):
    r = lambda k: round(float(np.mean([v[k] for v in pq.values() if v[k] is not None])), 3)
    return {"n_queries": len(pq), "P@5": r("p5"), "P@10": r("p10"), "mAP": r("ap"), "nDCG": r("ndcg"),
            "spurious@10": r("spur10"), "mismatch@10": r("mism10")}


def paired(dA, dB, rng):
    keys = sorted(set(dA) & set(dB))
    dif = np.array([dA[k]["ap"] - dB[k]["ap"] for k in keys])
    obs = float(dif.mean())
    boots = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(B_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(B_PERM, len(dif)))
    p = float((np.sum(np.abs((signs * dif).mean(axis=1)) >= abs(obs)) + 1) / (B_PERM + 1))
    return {"n_common": len(keys), "delta": round(obs, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "p_raw": p}


def main():
    core.ORGAN_UNIVERSE = T.UNIVERSE
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"])
    labels = json.load(open(os.path.join(INP, "relevance_labels_published_corrected_113.json")))
    q113 = labels["query_order"]
    recs = json.load(open(os.path.join(INP, "reference_records_published_corrected_113.json")))["records"]
    ks = json.load(open(os.path.join(INP, "graph_rows_perquery_published_corrected_2026-10-05.json")))
    z = np.load(FEAT, allow_pickle=True)
    R, E = z["R"].item(), z["E"].item()
    allR = np.array([R[c]["__union__"] for c in q113]); mu, sd = allR.mean(0), allR.std(0) + 1e-8
    feats = {"R": {c: {o: (v - mu) / sd for o, v in R[c].items()} for c in q113}, "E": E}

    gates = {}
    for v in ("D", "P"):                                          # gate 2: labels = relevant() on the records
        gates[f"labels_{v}_equal_relevant()"] = all(labels["relevance"][v][q][c] == int(bool(relevant(recs[v][q], recs[v][c])))
                                                    for q in q113 for c in q113 if c != q)
    gates["ks_corrected_(ii)_mAP_0.967"] = ks["corrected"]["rows"]["(ii)"]["mAP"] == 0.967   # gate 3
    frozen = json.load(open(FROZEN))["tab7"]["rows"]
    res = {"note": __doc__.strip(), "n_queries": len(q113), "n_candidates": len(q113) - 1,
           "labels_flipped": sum(labels["relevance"]["D"][q][c] != labels["relevance"]["P"][q][c] for q in q113 for c in q113 if c != q),
           "tables": {}}
    for v, arm in (("D", "published"), ("P", "corrected")):
        lab = labels["relevance"][v]
        pq = {t: per_query(feats[k], pred, recs[v], lab, q113, oc) for t, (_l, k, oc) in BASE.items()}
        graph = ks[arm]["per_query"]
        rows = {t: dict(ks[arm]["rows"][t]) for t in ("(i)", "(ii)", "(iii)", "(iii')", "(iv)")}
        for t in rows:
            rows[t].pop("dmAP_vs_ii", None)
        rows.update({t: dict(agg(p), label=BASE[t][0]) for t, p in pq.items()})
        rng = np.random.default_rng(SEED)
        allpq = {**graph, **pq}
        cmp_ = {t: paired(allpq[t], allpq["(ii)"], rng) for t in ("(i)", "(iii)", "(iii')", "(iv)", *BASE)}
        T.holm({t: c for t, c in cmp_.items() if t != "(iii')"})   # family F1; (iii') stays raw
        for t, c in cmp_.items():
            rows[t]["dmAP_vs_ii"] = c
        byds = {ds: {t: agg({q: m for q, m in pq[t].items() if pred[q]["dataset"] == ds}) for t in BASE}
                for ds in ("kits", "lits", "msd")}
        res["tables"][arm] = {"rows": rows, "baselines_by_dataset": byds}
        if v == "D":                                              # gate 1
            gates["published_labels_reproduce_frozen_(v)-(vii')"] = all(
                all(rows[t][m] == frozen[FROZEN_KEY[t]][m] for m in ("P@5", "P@10", "mAP", "nDCG", "spurious@10", "mismatch@10"))
                for t in BASE)
    res["gates"] = gates
    print(json.dumps(gates, indent=1))
    if not all(gates.values()):
        sys.exit("gate failed; nothing written")
    json.dump(res, open(OUT, "w"), indent=1)
    for arm in ("published", "corrected"):
        print(arm)
        for t, r in res["tables"][arm]["rows"].items():
            d = r.get("dmAP_vs_ii")
            print(f"  {t:6s} P@10 {r['P@10']} mAP {r['mAP']} nDCG {r['nDCG']} mism {r['mismatch@10']}"
                  + (f"  d {d['delta']} {d['ci95']} p_holm {d.get('p_holm', 'raw ' + format(d['p_raw'], '.1e'))}" if d else ""))
    print("->", OUT)


if __name__ == "__main__":
    main()
