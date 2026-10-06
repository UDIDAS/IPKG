#!/usr/bin/env python3
"""Prompt-free tumor arm re-scored on the shared 157-scan cohort with the complete twin census (the study lead 10-03, the pipeline team item 7).

Cohort: the round-2 held-out TEST lists of sam3_tumor_generic (global split) -- LiTS 21, MSD 56, KiTS 36, FLARE23 54
= 167 scored volumes -- minus the 10 FLARE23 test volumes that ARE a LiTS/MSD/KiTS test volume (same CT, voxel- or
header-matched in the complete census), which were scored twice. Each such scan is counted once, under its
single-organ collection (the volume the collection's own annotation was drawn on) -> 157 unique scans.

Exposure (complete census, results/audit/promptfree_twin_exposure_2026-10-03.json): a scan is EXPOSED when any twin
of it sits in the generic model's TRAIN or VAL split. PRIMARY = twin-clean scans; SECONDARY = all 157.
Metrics per scan come unchanged from results/segmentation/promptfree_round2_2026-10-01.json (raw = every kept slice;
gated = components touching the TotalSegmentator host organ dilated 5 mm). Means with percentile-bootstrap 95 % CIs
over scans (2,000 resamples, seed 0), per collection and pooled.

  python rescore_cohort157.py -> results/segmentation/cohort157_rescore_2026-10-03.json
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
R2 = os.path.join(ROOT, "results", "segmentation", "promptfree_round2_2026-10-01.json")
EXPO = os.path.join(ROOT, "results", "audit", "promptfree_twin_exposure_2026-10-03.json")
OUT = os.path.join(ROOT, "results", "segmentation", "cohort157_rescore_2026-10-03.json")
METRICS = ("dice", "nsd_2mm", "hd95_mm", "components")


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return None
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def summarize(rows):
    out = {"n": len(rows)}
    for arm in ("tumor_raw", "tumor_gated"):
        s = {}
        for k in METRICS:
            v = [r[arm][k] for r in rows if arm in r and r[arm].get(k) is not None]
            s[k] = round(float(np.mean(v)), 4) if v else None
            if k == "dice":
                s["dice_ci95"] = boot(v)
                s["dice_median"] = round(float(np.median(v)), 4) if v else None
        s["n_scored"] = sum(arm in r for r in rows)
        out[arm] = s
    return out


def main():
    r2 = json.load(open(R2))["models"]["sam3_tumor_generic"]["datasets"]
    ex = json.load(open(EXPO))["datasets"]
    exposed = {r["case"]: r["exposed"] for v in ex.values() for r in v["cases"]}
    single = {r["case"] for ds in ("lits", "msd", "kits") for r in ex[ds]["cases"]}
    dup = sorted({r["case"]: t["twin"] for r in ex["flare23"]["cases"] for t in r["twins"] if t["twin"] in single}.items())
    dup_f = {f for f, _ in dup}
    cohort = {}
    for ds in ("lits", "msd", "kits", "flare23"):
        rows = [r for r in r2[ds]["cases"] if not (ds == "flare23" and r["case"] in dup_f)]
        for r in rows:
            # a deduplicated single-organ scan is exposed if either copy's twins are in TRAIN/VAL
            twin_f = [f for f, s in dup if s == r["case"]]
            r["_exposed"] = exposed.get(r["case"], False) or any(exposed.get(f, False) for f in twin_f)
        cohort[ds] = rows
    allrows = [r for v in cohort.values() for r in v]
    res = {"note": __doc__.strip(), "n_scans": len(allrows), "dropped_duplicate_flare23_scans": [{"flare23": f, "same_scan_as": s} for f, s in dup],
           "per_collection": {}, "pooled": {}}
    for ds, rows in list(cohort.items()) + [("pooled", allrows)]:
        clean = [r for r in rows if not r["_exposed"]]
        block = {"primary_twin_clean": summarize(clean), "secondary_all": summarize(rows), "n_exposed": len(rows) - len(clean)}
        (res["pooled"] if ds == "pooled" else res["per_collection"]).update(block if ds == "pooled" else {ds: block})
    # FLARE23 on its own 54-scan test table (no cross-collection de-duplication), complete census (the study lead 10-03 list, item 5)
    f54 = r2["flare23"]["cases"]
    res["flare23_54_scan_table"] = {"note": "FLARE23 global-split test list, all 54 scans; twin-clean = no twin in the generic "
                                            "model's TRAIN/VAL under the complete census",
                                    "primary_twin_clean": summarize([r for r in f54 if not exposed.get(r["case"], False)]),
                                    "secondary_all": summarize(f54),
                                    "twin_clean_cases": sorted(r["case"] for r in f54 if not exposed.get(r["case"], False))}
    res["cases"] = [{"case": r["case"], "dataset": r["dataset"], "exposed_via_twin": r["_exposed"],
                     "tumor_raw_dice": r["tumor_raw"]["dice"], "tumor_gated_dice": r.get("tumor_gated", {}).get("dice")} for r in allrows]
    json.dump(res, open(OUT, "w"), indent=1)
    for ds, b in list(res["per_collection"].items()) + [("pooled", res["pooled"])]:
        p, s = b["primary_twin_clean"], b["secondary_all"]
        print(f"{ds:8s} clean n={p['n']:3d} raw {p['tumor_raw']['dice']} {p['tumor_raw']['dice_ci95']} gated {p['tumor_gated']['dice']} {p['tumor_gated']['dice_ci95']}"
              f" | all n={s['n']:3d} raw {s['tumor_raw']['dice']} gated {s['tumor_gated']['dice']}")
    f = res["flare23_54_scan_table"]
    print(f"FLARE23 54-scan table: twin-clean n={f['primary_twin_clean']['n']} raw {f['primary_twin_clean']['tumor_raw']['dice']} "
          f"{f['primary_twin_clean']['tumor_raw']['dice_ci95']} gated {f['primary_twin_clean']['tumor_gated']['dice']} "
          f"{f['primary_twin_clean']['tumor_gated']['dice_ci95']} | all n={f['secondary_all']['n']}")
    print("->", OUT)


if __name__ == "__main__":
    main()
