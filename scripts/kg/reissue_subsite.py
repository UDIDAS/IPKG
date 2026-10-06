#!/usr/bin/env python3
"""Sub-site re-issue of the delivered records (IPKG round 2; the co-author v2 report §2.2 'pending the pipeline team re-issue').

Rule = scripts/kg/pancreas_subsite.py (patient left-right in world mm, thirds, PCA cross-check, fail-closed guards).
  * liver / kidney / spleen entries: anatomic_location -> "na" (no head/body/tail; their own sub-sites would be
    Couinaud segments / poles, which the pipeline does not compute).
  * MSD pancreas, REFERENCE records: the corrected site from results/kg/pancreas_subsite_msd_gt.json.
  * MSD pancreas, PREDICTED records: the masks behind the frozen predicted corpus were not retained (and the tumor
    checkpoint that drew them was deleted 08-22), so the corrected rule cannot be applied -> "unknown"; the round-2
    record (TS pancreas + prompt-free tumor, results/segmentation/promptfree_round2_2026-10-01.json) carries the
    recomputed site. Exception: the shipped final-case mask of pancreas_125 exists and is re-scored directly.
  * FLARE23 records: already "unknown" for pancreas (never computed); other organs -> "na".
Every changed entry keeps the delivered word in anatomic_location_delivered. The frozen corpora are not modified:
re-issued copies go to corpora/reissue_2026-10-01/ and app_handoff/final_cases_2026-09-19_reissue/.

  python reissue_subsite.py
"""
import glob
import json
import os
import shutil
import sys

import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from pancreas_subsite import pancreas_subsite, SNOMED                    # noqa: E402

GT = {r["case"]: r for r in json.load(open(os.path.join(ROOT, "results", "kg", "pancreas_subsite_msd_gt.json")))["records"]}
OUTC = os.path.join(ROOT, "corpora", "reissue_2026-10-01")
FC = os.path.join(ROOT, "app_handoff", "final_cases_2026-09-19")
OUTF = os.path.join(ROOT, "app_handoff", "final_cases_2026-09-19_reissue")
DISPLAYED_125 = os.path.join(FC, "masks", "pancreas_125.nii.gz")


def fix(rec, predicted, counts, override=None):
    for organ, od in (rec.get("organs") or {}).items():
        if not isinstance(od, dict) or "anatomic_location" not in od:
            continue
        old = od["anatomic_location"]
        if organ != "pancreas":
            new, why = "na", "not a pancreas entry: head/body/tail does not apply"
        elif override is not None:
            new, why = override
        elif rec.get("dataset") != "msd":
            new, why = old, "unchanged (FLARE23: never computed)"
        elif predicted:
            new, why = "unknown", "source mask not retained: corrected rule cannot be applied (see round-2 record)"
        else:
            g = GT.get(rec["case_id"])
            new, why = (g["site"], f"corrected rule on reference label ({g.get('reason')})") if g else ("unknown", "no reference audit row")
        if new != old:
            od["anatomic_location_delivered"] = old
        od["anatomic_location"] = new
        od["anatomic_location_rule"] = why
        if new in SNOMED and organ == "pancreas":
            od["anatomic_location_snomed"] = SNOMED[new]
        counts[(organ, old, new)] = counts.get((organ, old, new), 0) + 1


def main():
    os.makedirs(os.path.join(OUTC, "containment_v2"), exist_ok=True)
    report = {"note": __doc__.strip(), "corpora": {}, "final_cases": {}}
    for f in sorted(glob.glob(os.path.join(ROOT, "corpora", "*.json")) + glob.glob(os.path.join(ROOT, "corpora", "containment_v2", "*.json"))):
        d = json.load(open(f))
        recs = d["records"] if isinstance(d, dict) and "records" in d else None
        if recs is None:
            continue
        counts = {}
        for r in recs:
            fix(r, "predicted" in os.path.basename(f), counts)
        if isinstance(d, dict):
            d["subsite_reissue"] = "2026-10-01: scripts/kg/reissue_subsite.py (see results/kg/subsite_reissue_2026-10-01.json)"
        rel = os.path.relpath(f, os.path.join(ROOT, "corpora"))
        json.dump(d, open(os.path.join(OUTC, rel), "w"))
        report["corpora"][rel] = [{"organ": o, "delivered": a, "reissued": b, "n": n} for (o, a, b), n in sorted(counts.items())]
    # final cases: the four shipped packages
    if os.path.isdir(FC):
        if os.path.exists(OUTF):
            shutil.rmtree(OUTF)
        shutil.copytree(FC, OUTF, ignore=shutil.ignore_patterns("ct", "masks"))
        n = nib.load(DISPLAYED_125)
        m = np.asarray(n.dataobj)
        s125 = pancreas_subsite(m >= 1, m == 2, n.affine)
        report["final_cases"]["pancreas_125_displayed_mask"] = s125
        for p in sorted(glob.glob(os.path.join(OUTF, "*", "record_*.json"))):
            r = json.load(open(p))
            if "organs" not in r:
                continue
            counts = {}
            pred = "reference" not in os.path.basename(p)
            ov = (s125["site"], f"corrected rule on the shipped displayed mask ({s125['reason']})") if (pred and r.get("case_id") == "pancreas_125") else None
            fix(r, pred, counts, ov)
            json.dump(r, open(p, "w"), indent=1)
            report["final_cases"][os.path.relpath(p, OUTF)] = [{"organ": o, "delivered": a, "reissued": b} for (o, a, b), _ in counts.items()]
    # ontology: head -> the 'Structure of' concept, same family as body / tail
    om_p = os.path.join(ROOT, "kg_graphs", "ontology_mappings.json")
    om = json.load(open(om_p))
    h = om["mappings"]["AnatomicSite::Head of pancreas"]["SNOMED CT"]
    if h["code"] != SNOMED["head"]:
        report["ontology_change"] = {"concept": "AnatomicSite::Head of pancreas", "from": dict(h)}
        h.update(code=SNOMED["head"], display="Structure of head of pancreas")
        json.dump(om, open(om_p, "w"), indent=1)
    out = os.path.join(ROOT, "results", "kg", "subsite_reissue_2026-10-01.json")
    json.dump(report, open(out, "w"), indent=1)
    for k, v in report["corpora"].items():
        print(k, [(x["organ"], x["delivered"], x["reissued"], x["n"]) for x in v if x["delivered"] != x["reissued"]])
    print("pancreas_125 displayed:", s125.get("site"), s125.get("f_lr"), s125.get("reason"))
    print("->", out)


if __name__ == "__main__":
    main()
