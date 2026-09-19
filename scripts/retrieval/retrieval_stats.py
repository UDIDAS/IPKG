#!/usr/bin/env python3
"""Bootstrap 95% CIs + Holm-corrected paired permutation p-values for the retrieval Δs.
Reuses the exact retrieval scorer (kg_retrieval_v2.similarity/relevant/_dcg). Per-query AP/nDCG captured,
then paired on the COMMON query set for each comparison. -> results/retrieval_stats.json"""
import glob, json, sys
import numpy as np
sys.path.insert(0, "/home/user/SWOG/src/scripts")
from kg_retrieval_v2 import similarity, relevant, _dcg
RES = "/home/user/SWOG/results"
rng = np.random.default_rng(12345)

def load(kind):
    recs = []
    for f in glob.glob(f"{RES}/corpus_{kind}_*.json"):
        recs += json.load(open(f))["records"]
    return {r["case_id"]: r for r in recs}

def per_query(sim, rel, ids, mode, gamma_min=1):
    """Return {case_id: {'ap':, 'ndcg':}} for every query that has ≥1 relevant candidate."""
    out = {}
    for qi in ids:
        cands = []
        for bi in ids:
            if bi == qi: continue
            s = similarity(sim[qi], sim[bi], mode, gamma_min=gamma_min)
            if s is None: continue
            cands.append((s, bi))
        if not cands: continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if relevant(rel[qi], rel[bi]) else 0 for _, bi in cands]
        if sum(rels) == 0: continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r: hit += 1; ap += hit / (i + 1)
        ap /= sum(rels)
        nd = _dcg(rels[:10]) / (_dcg(sorted(rels, reverse=True)[:10]) or 1)
        out[qi] = {"ap": ap, "ndcg": nd}
    return out

def ci_mean(vals, B=5000):
    a = np.asarray(vals, float)
    boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(B)]
    return round(float(a.mean()), 3), round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)

def paired(dA, dB, B=5000, P=20000):
    """Paired Δ = A-B over common queries; bootstrap CI + two-sided sign-flip permutation p."""
    keys = sorted(set(dA) & set(dB))
    dif = np.array([dA[k] - dB[k] for k in keys])
    obs = float(dif.mean())
    boots = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(B)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(P, len(dif)))
    perm = (signs * dif).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (P + 1)
    return {"n_common": len(keys), "delta": round(obs, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "p_raw": float(p)}

def holm(pdict):
    items = sorted(pdict.items(), key=lambda kv: kv[1]["p_raw"])
    m = len(items); out = {}
    for rank, (k, v) in enumerate(items):
        adj = min(1.0, (m - rank) * v["p_raw"])
        out[k] = {**v, "p_holm": round(adj, 5)}
    return out

pred, gt = load("predicted"), load("gt")
ids = [i for i in pred if i in gt]
cfg = {
    "GT+gamma (upper bound)": per_query(gt, gt, ids, "proposed"),
    "PRED+gamma (primary)":   per_query(pred, gt, ids, "proposed"),
    "PRED no-gamma (blind)":  per_query(pred, gt, ids, "coverage_blind"),
    "PRED base":              per_query(pred, gt, ids, "base"),
}
per_cfg = {}
for name, d in cfg.items():
    m, l, h = ci_mean([v["ap"] for v in d.values()]); nm, nl, nh = ci_mean([v["ndcg"] for v in d.values()])
    per_cfg[name] = {"n": len(d), "mAP": m, "mAP_ci95": [l, h], "nDCG": nm, "nDCG_ci95": [nl, nh]}

comps = {
    "PRED+gamma − PRED no-gamma (γ ablation)": ("PRED+gamma (primary)", "PRED no-gamma (blind)"),
    "PRED+gamma − PRED base":                  ("PRED+gamma (primary)", "PRED base"),
    "GT+gamma − PRED+gamma (upper-bound gap)": ("GT+gamma (upper bound)", "PRED+gamma (primary)"),
}
dmap = {k: paired({q: cfg[a][q]["ap"] for q in cfg[a]}, {q: cfg[b][q]["ap"] for q in cfg[b]}) for k, (a, b) in comps.items()}
dnd  = {k: paired({q: cfg[a][q]["ndcg"] for q in cfg[a]}, {q: cfg[b][q]["ndcg"] for q in cfg[b]}) for k, (a, b) in comps.items()}
dmap, dnd = holm(dmap), holm(dnd)

out = {"n_aligned": len(ids), "method": "5000-sample query bootstrap 95% CI; 20000-perm two-sided sign-flip paired p; Holm across each family",
       "per_config": per_cfg, "delta_mAP": dmap, "delta_nDCG": dnd}
json.dump(out, open(f"{RES}/retrieval_stats.json", "w"), indent=2)
print(f"aligned {len(ids)} queries\n")
print("PER-CONFIG mAP / nDCG (95% CI):")
for k, v in per_cfg.items():
    print(f"  {k:32s} mAP {v['mAP']} {v['mAP_ci95']}  nDCG {v['nDCG']} {v['nDCG_ci95']}  (n={v['n']})")
print("\nΔ mAP (paired, common queries):")
for k, v in dmap.items():
    print(f"  {k:42s} Δ={v['delta']:+.3f} CI{v['ci95']} p_holm={v['p_holm']} (n={v['n_common']})")
print("\nΔ nDCG (paired, common queries):")
for k, v in dnd.items():
    print(f"  {k:42s} Δ={v['delta']:+.3f} CI{v['ci95']} p_holm={v['p_holm']} (n={v['n_common']})")
