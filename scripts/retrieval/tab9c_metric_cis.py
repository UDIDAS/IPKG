#!/usr/bin/env python3
"""Per-metric bootstrap CIs for the Table 9 stratum (c) proposed row (YL 09-20 ask: "updated
nq, P@5, P@10, mAP, CIs, and the (iii') comparison" for FLARE23 -> single-organ after the
axis-normalized re-dedup).

The shipped retrieval_tab9_stratified.json carries point metrics and the PAIRED-delta CIs
(its convention); this adds plain per-metric bootstrap CIs for the (ii) row on the same
per-query values.  Standalone RNG (seed 20260920) so the frozen table JSONs stay byte-stable
- do NOT fold extra draws into retrieval_tables_789's shared rng.

  VKG_FLARE_CORPUS=corpora/corpus_flare23_kg_dedup.json python tab9c_metric_cis.py
-> results/retrieval/dedup/tab9c_metric_cis.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import retrieval_core_local as core                       # noqa: E402
from retrieval_core_local import load_corpus              # noqa: E402
import retrieval_tables_789 as T                          # noqa: E402

OUT = os.path.join(os.environ.get("VKG_RES_DIR", os.path.join(ROOT, "results", "retrieval", "dedup")),
                   "tab9c_metric_cis.json")
B = 5000


def main():
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"])
    gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    flare = {r["case_id"]: r for r in json.load(open(os.environ.get(
        "VKG_FLARE_CORPUS", os.path.join(ROOT, "corpora", "corpus_flare23_kg_dedup.json"))))["records"]}
    q113 = [i for i in pred if i in gt]
    core.ORGAN_UNIVERSE = T.UNIVERSE
    fids = sorted(flare)
    pq = T.per_query({**flare, **pred}, {**flare, **pred}, fids, q113, "proposed",
                     multi_organ_queries=True)
    rng = np.random.default_rng(20260920)
    qs = list(pq)
    n = len(qs)
    out = {"stratum": "(c) FLARE23 -> single-organ, configuration (ii) proposed",
           "n_queries": n, "n_flare_corpus": len(flare), "B_boot": B,
           "note": __doc__.strip(), "metrics": {}}
    for key, name in (("p5", "P@5"), ("p10", "P@10"), ("ap", "mAP"), ("ndcg", "nDCG")):
        vals = np.array([pq[q][key] for q in qs])
        boots = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(B)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["metrics"][name] = {"point": round(float(vals.mean()), 3),
                                "ci95": [round(float(lo), 3), round(float(hi), 3)]}
        print(f"{name}: {vals.mean():.3f} [{lo:.3f}, {hi:.3f}]")
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
