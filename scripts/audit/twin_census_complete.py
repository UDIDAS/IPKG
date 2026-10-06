#!/usr/bin/env python3
"""Complete FLARE23 twin census + corrected round-2 twin exposure (the study lead 10-03, the pipeline team item 6; feeds item 7).

FLARE23 re-shares scans of KiTS23, MSD Pancreas and LiTS. Two censuses existed and they are complementary:
  * ours (waves 1-4, LiTS overlap census, 0405 = case_00078 check): built to de-duplicate the RETRIEVAL POOL, so it
    searched for twins of the 113 query cases and of pool records -- it never looked for twins of TRAINING cases;
  * the co-author's header scan of all 576 tumor-bearing FLARE23 cases (results/audit/inputs/flare23_twin_scan_ctvoxel_2026-09-20.json,
    from the app branch paper/): 310 twins (KiTS 133, MSD 127, LiTS 50), each matched to one public case.
Union = this census. Disagreements:
  * one conflict: wave 1 paired FLARE23_0042 with pancreas_122; the voxels say pancreas_274 (100 % voxel-equal,
    r = 1.0; pancreas_122 is on a different grid, 97 x 0.79 mm vs 115 x 0.70 mm). The wave-1 pair is dropped.
  * the co-author-only pairs (219) were spot-checked on the voxels: 9/9 sampled MSD pairs 100 % voxel-equal.
  * our 120 pairs absent from the co-author's scan all involve FLARE23 cases outside the co-author's 576 (non-tumor pool records).
No the co-author-only pair links a pool record to a query, so the pool de-duplication against the queries stands; but
FLARE23_0042 was removed from the pool as a twin of query pancreas_122, which it is not (restored 10-03, with FLARE23_0102).

  python twin_census_complete.py
-> results/audit/flare23_twin_census_complete_2026-10-03.json, results/audit/promptfree_twin_exposure_2026-10-03.json
"""
import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
import promptfree_twin_exposure as T                                     # noqa: E402

A = os.path.join(ROOT, "results", "audit")
SCAN = os.path.join(A, "inputs", "flare23_twin_scan_ctvoxel_2026-09-20.json")
OUTC = os.path.join(A, "flare23_twin_census_complete_2026-10-03.json")
OUTE = os.path.join(A, "promptfree_twin_exposure_2026-10-03.json")
DROP = {("pancreas_122", "FLARE23_0042")}                                # voxel-refuted wave-1 pair
ADD_EVIDENCE = {"FLARE23_0042": {"twin": "pancreas_274", "voxel_equal_frac": 1.0, "corr": 1.0,
                                 "refuted": {"pancreas_122": "different grid: (512,512,97) @ 0.791 mm vs (512,512,115) @ 0.703 mm"}}}
SPOT = {"FLARE23_1289": "pancreas_264", "FLARE23_2009": "pancreas_109", "FLARE23_1363": "pancreas_299", "FLARE23_0262": "pancreas_180",
        "FLARE23_1028": "pancreas_304", "FLARE23_1556": "pancreas_069", "FLARE23_1482": "pancreas_302", "FLARE23_1344": "pancreas_212",
        "FLARE23_2073": "pancreas_342"}                                  # 100 % voxel-equal (10-03 spot check)


def norm(c):
    return "volume-" + c.split("_")[1] if c.startswith("liver_") else c


def main():
    ours = {(q, f, w) for q, f, w in T.pairs() if (q, f) not in DROP}
    ks = {}
    for f, r in json.load(open(SCAN))["results"].items():
        if r.get("twin"):
            ks[f] = norm(r["twin"]["case"])
    P = set(ours) | {(c, f, "ks_header_scan") for f, c in ks.items()}
    by_f = defaultdict(set)
    for q, f, w in P:
        by_f[f].add(q)
    multi = {f: sorted(v) for f, v in by_f.items() if len(v) > 1}
    census = {"note": __doc__.strip(), "n_pairs": len({(q, f) for q, f, _ in P}), "n_flare23_with_twin": len(by_f),
              "by_collection": dict(Counter("KiTS23" if q.startswith("case_") else "MSD Pancreas" if q.startswith("pancreas_") else "LiTS"
                                            for q, f in {(q, f) for q, f, _ in P})),
              "sources": dict(Counter(w for _, _, w in P)), "dropped_pairs": [{"query": q, "flare23": f, "why": "voxel-refuted"} for q, f in DROP],
              "conflict_resolution": ADD_EVIDENCE, "ks_pairs_voxel_spot_check": {"n": len(SPOT), "all_identical": True, "pairs": SPOT},
              "flare23_with_more_than_one_twin": multi,
              "pool_consequence": "FLARE23_0042 is not a twin of any query (pancreas_274 is a training case, not one of the 113); "
                                  "it was removed from the de-duplicated pool on the refuted pancreas_122 pairing. FLARE23_0102 (= volume-64, "
                                  "not a query) was likewise removed although it duplicates nothing in the pool. Both restored 10-03 "
                                  "(scripts/retrieval/pool_restore_impact.py) -> pool 1,311. No the co-author-only pair links a pool record to a query.",
              "pairs": sorted([{"public_case": q, "flare23": f, "source": w} for q, f, w in P], key=lambda x: (x["flare23"], x["public_case"]))}
    json.dump(census, open(OUTC, "w"), indent=1)
    # exposure: re-run the round-2 computation on the complete census
    T.pairs = lambda: P
    T.OUT = OUTE
    T.main()
    e = json.load(open(OUTE))
    e["note"] = "Round-2 twin exposure recomputed on the COMPLETE census (flare23_twin_census_complete_2026-10-03.json); " \
                "supersedes promptfree_twin_exposure_2026-10-01.json, whose partial census missed twins of training cases.\n\n" + e["note"]
    old = json.load(open(os.path.join(A, "promptfree_twin_exposure_2026-10-01.json")))["datasets"]
    e["change_vs_2026-10-01"] = {ds: {"n_with_twin": [old[ds]["n_with_twin"], v["n_with_twin"]],
                                      "n_exposed_via_twin": [old[ds]["n_exposed_via_twin"], v["n_exposed_via_twin"]]} for ds, v in e["datasets"].items()}
    json.dump(e, open(OUTE, "w"), indent=1)
    print(json.dumps(e["change_vs_2026-10-01"], indent=1))
    print("pairs", census["n_pairs"], census["by_collection"], "multi-twin FLARE23:", len(multi))


if __name__ == "__main__":
    main()
