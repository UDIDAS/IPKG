#!/usr/bin/env python3
"""Sensitivity of the proposed configuration (ii) to the pancreatic sub-site convention: FLARE23 records carry
anatomic_location = 'unknown', which feat_agree() scores as a mismatch against an MSD query's head/body/tail.
Here 'unknown' on either side makes the sub-site feature not applicable (dropped from the per-organ mean) and
Tabs 7 / 8 / 9(b) are recomputed.  Output: results/retrieval/sensitivity_subsite_unknown.json
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import retrieval_core_local as core                          # noqa: E402
from retrieval_core_local import load_corpus                 # noqa: E402
import retrieval_tables_789 as T                             # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(os.environ.get("VKG_RES_DIR", os.path.join(ROOT, "results", "retrieval", "dedup")), "sensitivity_subsite_unknown.json")


def sim_na(A, B, mode, weights=None, gamma_min=1):
    oa, ob = set(A["observed_organs"]), set(B["observed_organs"]); shared = oa & ob
    if len(shared) < gamma_min:
        return None
    sims = []
    for o in shared:
        pa, pb = core.phen(A, o), core.phen(B, o)
        cats = [c for c in core.organ_cats(o) if not (c in core.LOC_CATS and "unknown" in (pa.get(c), pb.get(c)))]
        sims.append(sum(core.feat_agree(pa, pb, c, True) for c in cats) / len(cats))
    return float(np.mean(sims)) if sims else None


def main():
    core.ORGAN_UNIVERSE = T.UNIVERSE
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"]); gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    flare = {r["case_id"]: r for r in json.load(open(os.environ.get("VKG_FLARE_CORPUS", os.path.join(ROOT, "corpora", "corpus_flare23_kg.json"))))["records"]}
    pool = {**pred, **flare}; q113 = [i for i in pred if i in gt]; cids = q113 + sorted(flare)
    res = {}
    for name, fn in (("(ii) as published", core.similarity), ("(ii) sub-site n/a when unknown", sim_na)):
        keep = core.similarity; core.similarity = fn
        t7 = T.agg(T.per_query(pred, gt, q113, q113, "proposed")); t8 = T.agg(T.per_query(pool, pool, q113, cids, "proposed"))
        b = T.agg(T.per_query(pool, pool, q113, sorted(flare), "proposed")); core.similarity = keep
        res[name] = {"tab7_mAP": t7["mAP"], "tab8_mAP": t8["mAP"], "tab8_P@10": t8["P@10"], "tab9b_mAP": b["mAP"], "tab9b_P@10": b["P@10"]}
        print(name, res[name])
    json.dump({"note": __doc__.strip(), "rows": res}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
