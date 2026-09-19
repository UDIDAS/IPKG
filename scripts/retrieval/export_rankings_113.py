#!/usr/bin/env python3
"""Export per-query rankings for the 113 queries — the producer of
app_handoff/precomputed/rankings_113_queries.json (previously generated ad hoc on the cluster;
committed 2026-09-16 when the baseline re-rank moved local).

Depth 30 per query (was 10), on TWO pools:
  table8_pool113     — 113-case pool, reference relevance (GT query record vs GT candidate);
  table9_dedup_pool  — the de-duplicated pool (113 + the dedup-corpus FLARE23 cases whose CT+label
                       are staged, = the same file-defined sub-pool as baselines_tab7_tab8.py),
                       construction relevance (recorded records both sides).
The legacy key `table9_pool1425` is intentionally ABSENT: those lists were twin-contaminated
(the (vi) 87 / (vii') 66 twin entries) and truncated below 10 after filtering — superseded here.

Configs (same nine as the delivered file): i–iv, iii' via the benchmark scorer
(retrieval_core_local.similarity); v/vi/vii/vii' from the baseline feature cache
($VKG_DATA/baseline_features.npz, radiomics z-scored over the pool population / ResNet-50
embeddings; organ-conditioned = mean cosine over shared observed organs).

  VKG_DATA=... python export_rankings_113.py [--depth 30]
-> app_handoff/precomputed/rankings_113_queries.json
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
import retrieval_core_local as core                                      # noqa: E402
from retrieval_core_local import load_corpus, relevant                   # noqa: E402
import retrieval_tables_789 as T                                         # noqa: E402

VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
POOL = os.path.join(VKG_DATA, "flare23_pool")
FEAT = os.path.join(VKG_DATA, "baseline_features.npz")
OUT = os.path.join(ROOT, "app_handoff", "precomputed", "rankings_113_queries.json")

KG_CFG = {"i": ("gt", "proposed"), "ii": ("pred", "proposed"), "iii": ("pred", "coverage_blind"),
          "iii_prime": ("pred", "imputed_absent"), "iv": ("pred", "base")}
FT_CFG = {"v": ("R", False), "vi": ("R", True), "vii": ("E", False), "vii_prime": ("E", True)}


def cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def rank_feat(F, recs, qid, cids, organ_conditioned, depth):
    qo = set(recs[qid]["observed_organs"]) & set(F.get(qid, {}))
    scored = []
    for c in cids:
        if c == qid or c not in F:
            continue
        if organ_conditioned:
            shared = qo & set(recs[c]["observed_organs"]) & set(F[c])
            if not shared:
                continue
            s = float(np.mean([cos(F[qid][o], F[c][o]) for o in shared]))
        else:
            s = cos(F[qid]["__union__"], F[c]["__union__"])
        scored.append((s, c))
    scored.sort(key=lambda x: -x[0])
    return scored[:depth]


def rank_kg(sim_recs, qid, cids, mode, depth):
    scored = []
    for c in cids:
        if c == qid:
            continue
        s = core.similarity(sim_recs[qid], sim_recs[c], mode)
        if s is not None:
            scored.append((float(s), c))
    scored.sort(key=lambda x: -x[0])
    return scored[:depth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=30)
    a = ap.parse_args()
    core.ORGAN_UNIVERSE = T.UNIVERSE

    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"])
    gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    q113 = sorted(i for i in pred if i in gt)
    flare_all = {r["case_id"]: r for r in
                 json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg_dedup.json")))["records"]}
    pool_ids = sorted(c for c in flare_all
                      if os.path.exists(f"{POOL}/images/{c.replace('flare23_', '')}_0000.nii.gz")
                      and os.path.exists(f"{POOL}/labels/{c.replace('flare23_', '')}.nii.gz"))
    flare = {c: flare_all[c] for c in pool_ids}
    z = np.load(FEAT, allow_pickle=True)
    R, E = z["R"].item(), z["E"].item()

    def zscore(cases):
        allR = np.array([R[c]["__union__"] for c in cases if c in R])
        mu, sd = allR.mean(0), allR.std(0) + 1e-8
        return {c: {o: (v - mu) / sd for o, v in R[c].items()} for c in cases if c in R}

    def entries(scored, rel_q, rel_recs):
        return [{"rank": i + 1, "case_id": c, "dataset": rel_recs[c]["dataset"], "score": round(s, 4),
                 "relevant": bool(relevant(rel_q, rel_recs[c]))} for i, (s, c) in enumerate(scored)]

    out_q = {}
    pools = {
        "table8_pool113": {"cids": q113, "sim_pred": pred, "sim_gt": gt, "rel": gt, "rel_side": "reference (GT vs GT)"},
        "table9_dedup_pool": {"cids": q113 + pool_ids, "sim_pred": {**pred, **flare}, "sim_gt": {**gt, **flare},
                              "rel": {**pred, **flare}, "rel_side": "construction (recorded records)"},
    }
    feats_by_pool = {k: {"R": zscore(v["cids"]), "E": E} for k, v in pools.items()}
    for n, qid in enumerate(q113):
        out_q[qid] = {}
        for pname, p in pools.items():
            block = {}
            for cfg, (side, mode) in KG_CFG.items():
                sim = p["sim_gt"] if side == "gt" else p["sim_pred"]
                block[cfg] = entries(rank_kg(sim, qid, p["cids"], mode, a.depth), p["rel"][qid], p["rel"])
            for cfg, (fkey, oc) in FT_CFG.items():
                F = feats_by_pool[pname][fkey]
                block[cfg] = entries(rank_feat(F, p["sim_pred"], qid, p["cids"], oc, a.depth), p["rel"][qid], p["rel"])
            out_q[qid][pname] = block
        if (n + 1) % 20 == 0:
            print(f"  [{n+1}/113]", flush=True)

    out = {"note": (f"Top-{a.depth} per query, nine configurations, two pools. table8_pool113 = 113-case pool with "
                    f"reference relevance. table9_dedup_pool = the DE-DUPLICATED pool "
                    f"(113 + {len(pool_ids)} FLARE23 cases with staged CT+label from corpus_flare23_kg_dedup.json, "
                    f"1,347-pool universe) with construction relevance. The legacy table9_pool1425 lists "
                    f"(twin-contaminated, truncated after filtering) are superseded and removed. "
                    f"Producer: scripts/retrieval/export_rankings_113.py (2026-09-16)."),
           "configs": {"i": "Reference phenotypes (upper bound)", "ii": "Predicted phenotypes, proposed",
                       "iii": "Predicted, coverage-blind", "iii_prime": "Predicted, imputed-absent",
                       "iv": "Predicted, organ-agnostic", "v": "Radiomics-vector (organ-agnostic)",
                       "vi": "Radiomics-vector (organ-conditioned)", "vii": "Image-embedding (organ-agnostic)",
                       "vii_prime": "Image-embedding (organ-conditioned)"},
           "depth": a.depth, "n_flare23_candidates": len(pool_ids),
           "queries": out_q}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"{len(out_q)} queries x 2 pools x 9 configs at depth {a.depth} -> {OUT}")


if __name__ == "__main__":
    main()
