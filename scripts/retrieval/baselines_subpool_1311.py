#!/usr/bin/env python3
"""Large-pool stress test (draft Table 7), lower block: baselines (v)-(vii') on the CT sub-pool of the 1,311 pool
(the study lead 10-06 item 2), under the published and the corrected relevance labels.

Sub-pool: the 113 single-organ cases + the 525 FLARE23 candidates whose CT and label are in the release copy
(results/retrieval/dedup/ct_subpool_1311_2026-10-06.json) = 638 cases. Relevance is construction-defined, as in the
stress test: relevant() on each case's own record, taken from the co-author's arm pools (build_retrieval_tables.py,
abdomen/ipkg-v2): 'published' (delivered sub-site vocabulary) and 'corrected' (precedence-resolved + void fix).
the co-author's graph rows (ii)-(iv) on the same 638 cases are exported with them (inputs/graph_subpool_1311_2026-10-06.json) and
are the reference of the paired contrasts.

Rankings: baselines_tab7_tab8.py's features (cache $VKG_DATA/baseline_features.npz; radiomics z-scored over the
sub-pool population), ranked over the WHOLE sub-pool; metrics are on the full ranking. (Depth 30 is only the length
of the exported ranking lists in export_rankings_113.py, not a truncation of these metrics.)
Statistics: paired query-level bootstrap 95% CI + sign-flip permutation p vs (ii); Holm over F1 = the six
comparators (iii), (iv), (v), (vi), (vii), (vii'); (iii') raw p. (i) has no reference record for a FLARE23 candidate.

  VKG_DATA=... python baselines_subpool_1311.py -> results/retrieval/dedup/baselines_subpool_1311_2026-10-06.json
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import retrieval_core_local as core                                     # noqa: E402
from retrieval_core_local import relevant                                # noqa: E402
import retrieval_tables_789 as T                                         # noqa: E402
from baselines_corrected_labels import per_query, agg, paired, BASE, SEED   # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
FEAT = os.path.join(os.environ.get("VKG_DATA", "/path/to/VKG_data"), "baseline_features.npz")
GRAPH_EXPORT = os.path.join(ROOT, "results", "retrieval", "inputs", "graph_subpool_1311_2026-10-06.json")
OUT = os.path.join(ROOT, "results", "retrieval", "dedup", "baselines_subpool_1311_2026-10-06.json")


def main():
    core.ORGAN_UNIVERSE = T.UNIVERSE
    ks = json.load(open(GRAPH_EXPORT))
    cids = ks["cids_in_pool_order"]
    q113 = [c for c in cids if not c.startswith("flare23_")]
    z = np.load(FEAT, allow_pickle=True)
    R, E = z["R"].item(), z["E"].item()
    miss = [c for c in cids if c not in R or c not in E]
    if miss:
        sys.exit(f"features missing for {len(miss)} cases, e.g. {miss[:3]}")
    allR = np.array([R[c]["__union__"] for c in cids]); mu, sd = allR.mean(0), allR.std(0) + 1e-8
    feats = {"R": {c: {o: (v - mu) / sd for o, v in R[c].items()} for c in cids}, "E": E}
    res = {"note": __doc__.strip(), "n_queries": len(q113), "n_subpool": len(cids),
           "n_flare23_with_ct": len(cids) - len(q113), "arms": {}}
    for arm in ("published", "corrected"):
        recs = ks["arms"][arm]["records"]
        lab = {q: {c: int(bool(relevant(recs[q], recs[c]))) for c in cids if c != q} for q in q113}
        pq = {}
        for t, (_l, k, oc) in BASE.items():
            pq[t] = per_query_sub(feats[k], recs, lab, q113, cids, oc)
        graph = ks["arms"][arm]["per_query"]
        rows = {t: dict(ks["arms"][arm]["rows"][t]) for t in graph}
        rows.update({t: dict(agg(p), label=BASE[t][0]) for t, p in pq.items()})
        rng = np.random.default_rng(SEED)
        allpq = {**graph, **pq}
        cmp_ = {t: paired(allpq[t], allpq["(ii)"], rng) for t in ("(iii)", "(iii')", "(iv)", *BASE)}
        T.holm({t: c for t, c in cmp_.items() if t != "(iii')"})
        for t, c in cmp_.items():
            rows[t]["dmAP_vs_ii"] = c
        res["arms"][arm] = {"rows": rows}
        print(arm)
        for t, r in rows.items():
            d = r.get("dmAP_vs_ii")
            print(f"  {t:6s} n={r['n_queries']} P@10 {r['P@10']} mAP {r['mAP']} nDCG {r['nDCG']} mism {r['mismatch@10']}"
                  + (f"  d {d['delta']} {d['ci95']} p_holm {d.get('p_holm', 'raw %.1e' % d['p_raw'])}" if d else ""))
    json.dump(res, open(OUT, "w"), indent=1)
    print("->", OUT)


def per_query_sub(F, recs, lab, qids, cids, oc):
    """per_query over the sub-pool: candidates = all other sub-pool cases (labels as a matrix)."""
    full = {q: dict(lab[q]) for q in qids}
    for q in qids:                          # candidates that are not queries still need a label row entry
        for c in cids:
            full[q].setdefault(c, 0)
    return per_query(F, recs, recs, full, qids, oc, cids=cids)


if __name__ == "__main__":
    main()
