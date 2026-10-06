#!/usr/bin/env python3
"""Twin exposure of the round-2 prompt-free tumor test sets (IPKG round 2).

sam3_tumor_generic was trained on all four tumor pools (global seed-42 split). FLARE23 re-shares scans of KiTS,
MSD Pancreas and LiTS patients, so a LiTS/MSD/KiTS TEST patient whose FLARE23 twin sits in the FLARE tumor
TRAIN/VAL split was seen in training under another name (and vice versa for FLARE23 test patients).
Twin pairs = every confirmed pair in the committed censuses (wave 1 duplicate_scan_audit, wave 2 axis-normalized,
wave 3 the co-author CT-voxel second pass, wave 4 grid census, LiTS<->FLARE23 overlap census, the the study lead-confirmed 0102 pair and the 0405 = case_00078 check).

  python promptfree_twin_exposure.py -> results/audit/promptfree_twin_exposure_2026-10-01.json
"""
import json
import os
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
A = os.path.join(ROOT, "results", "audit")
OUT = os.path.join(A, "promptfree_twin_exposure_2026-10-01.json")


def fid(x):
    return x.replace("flare23_", "")


def pairs():
    P = set()
    d = json.load(open(f"{A}/duplicate_scan_audit.json"))
    conf = {fid(x) for x in d["twins_confirmed"]}
    P |= {(p["query"], fid(p["flare23"]), "wave1") for p in d["pairs_flagged"] if fid(p["flare23"]) in conf}
    for m in d["manually_confirmed_twins"]:
        P.add(("volume-64", fid(m["flare23"]), "wave1_manual"))
    a = json.load(open(f"{A}/axis_normalized_twin_census.json"))
    P |= {(p["query"], fid(p["flare"]), "wave2") for p in a["new_twin_pairs"]}
    for w, k in (("wave3", "second_pass_2026-09-22"), ("wave4", "third_pass_2026-09-24")):
        P |= {(p["query"], fid(p["flare"]), w) for p in a[k]["pairs_confirmed"]}
    lc = json.load(open(f"{A}/lits_flare23_overlap_census.json"))["confirmed"]
    P |= {(v["volume"], fid(f), "lits_census") for f, v in lc.items()}
    if json.load(open(f"{A}/flare0405_kits00078_twin_check.json")).get("same_scan"):
        P.add(("case_00078", "FLARE23_0405", "flare0405_kits00078_twin_check"))
    return P


def main():
    tp = json.load(open(f"{A}/split_manifests_2026-10-01.json"))["tumor_pools"]["global"]
    role = {}
    for ds, sp in tp.items():
        for r in ("train", "val", "test"):
            for c in sp[r]:
                role[c] = r
    P = pairs()
    tw = defaultdict(set)
    for q, f, w in P:
        tw[q].add((f, w)); tw[f].add((q, w))
    out = {"note": __doc__.strip(), "model": "sam3_tumor_generic", "split": "global", "n_pairs": len(P), "datasets": {}}
    for ds, key in (("lits", "lits"), ("msd", "pancreas"), ("kits", "kits"), ("flare23", "flare")):
        rows = []
        for c in tp[key]["test"]:
            t = sorted(tw.get(c, ()))
            seen = [{"twin": x, "wave": w, "twin_role": role.get(x, "not in tumor pools")} for x, w in t]
            rows.append({"case": c, "twins": seen, "exposed": any(s["twin_role"] in ("train", "val") for s in seen)})
        out["datasets"][ds] = {"n_test": len(rows), "n_with_twin": sum(bool(r["twins"]) for r in rows),
                               "n_exposed_via_twin": sum(r["exposed"] for r in rows),
                               "exposed_cases": [r["case"] for r in rows if r["exposed"]], "cases": rows}
    json.dump(out, open(OUT, "w"), indent=1)
    for ds, v in out["datasets"].items():
        print(ds, {k: v[k] for k in ("n_test", "n_with_twin", "n_exposed_via_twin")})
    print("->", OUT)


if __name__ == "__main__":
    main()
