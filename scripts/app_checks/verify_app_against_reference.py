#!/usr/bin/env python3
"""Verify an IPKG application build against the reference pack (study lead 10-08, the pipeline team items 1-2).

Reads each case package of the build (<app_cases>/<case>/{record_predicted.json, retrieval_top10_ii.json} and
masks/<case_id>.nii.gz) and compares with results/app_checks/app_reference_2026-10-08.json:
  masks      md5 identical (masks must be unchanged)
  record     observed organs; per organ has_tumor, tumour volume, burden, multiplicity, containment, sub-site and its
             SNOMED code (corrected values; the delivered word may be shown beside them)
  retrieval  the displayed top-10 holds every candidate scored above the 10th score and fills the rest from the
             candidates tied at it; exact order is reported separately (it depends only on the tie-break)
  pool       stated pool size 1,311

  python verify_app_against_reference.py <app_cases_dir> [--masks <dir>] [--label <build tag>]
-> results/app_checks/verify_<label>.json, exit 1 on any failure
"""
import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
REF = os.path.join(ROOT, "results", "app_checks", "app_reference_2026-10-08.json")


def ids_of(top):
    L = top.get("top10") or top.get("paper_frozen_top10") or top.get("results") or []
    return [(h.get("case_id") or h.get("id")) if isinstance(h, dict) else h for h in L]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cases"); ap.add_argument("--masks"); ap.add_argument("--label", default="build")
    a = ap.parse_args()
    ref = json.load(open(REF)); rep = {"build": a.label, "cases": {}}; fails = 0
    for folder, exp in ref["cases"].items():
        d = os.path.join(a.cases, folder); r = {"checks": []}
        def check(name, ok, detail=""):
            nonlocal fails
            r["checks"].append({"check": name, "pass": bool(ok), "detail": detail}); fails += 0 if ok else 1
        rec = json.load(open(os.path.join(d, "record_predicted.json")))
        check("observed organs", sorted(rec.get("observed_organs", [])) == sorted(exp["observed_organs"] or []),
              f"{rec.get('observed_organs')} vs {exp['observed_organs']}")
        for o, fe in exp["organs"].items():
            fa = (rec.get("organs") or {}).get(o, {})
            for k, v in fe.items():
                if k in ("anatomic_location_delivered", "anatomic_location_status"):   # bookkeeping fields, not displayed
                    continue
                check(f"{o}.{k}", fa.get(k) == v, f"app {fa.get(k)!r} vs reference {v!r}")
        top = json.load(open(os.path.join(d, "retrieval_top10_ii.json")))
        shown = ids_of(top); tr = exp["top10_tie_rule"]
        above, tied = set(tr["above_cutoff"]), set(tr["tied_at_cutoff"])
        check("top-10 has every candidate above the cutoff", above <= set(shown), f"missing {sorted(above - set(shown))}")
        check("top-10 fills only from the tied set", set(shown) - above <= tied, f"outside {sorted(set(shown) - above - tied)}")
        check("top-10 has 10 entries", len(shown) == 10, str(len(shown)))
        r["top10_exact_order_matches_pool_order_tiebreak"] = shown == [x["case_id"] for x in exp["top10_ii_corrected_1311"]]
        pool_txt = json.dumps(top.get("pool", ""))
        check("pool stated as 1,311", bool(re.search(r"1,?311", pool_txt)), pool_txt[:120])
        if a.masks:
            mp = os.path.join(a.masks, f"{exp['case_id'].replace('flare23_', '')}.nii.gz")
            got = hashlib.md5(open(mp, "rb").read()).hexdigest() if os.path.exists(mp) else None
            check("mask unchanged", got == exp["mask_md5_must_be_unchanged"], f"{got} vs {exp['mask_md5_must_be_unchanged']}")
        rep["cases"][folder] = r
        print(f"{folder}: {sum(c['pass'] for c in r['checks'])}/{len(r['checks'])} pass" + "".join(
            f"\n   FAIL {c['check']}: {c['detail']}" for c in r["checks"] if not c["pass"]))
    out = os.path.join(ROOT, "results", "app_checks", f"verify_{a.label}.json")
    json.dump(rep, open(out, "w"), indent=1); print("->", out, "| failures:", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
