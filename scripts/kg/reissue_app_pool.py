#!/usr/bin/env python3
"""Re-issue of the application's record files with corrected sub-sites and ONE void spelling (the study lead 10-03, the pipeline team item 5).

Outputs (app_handoff/reissue_2026-10-03/):
  corpus_single_organ_113.json   replaces corpus_eval_pool_1425.json's 113 single-organ PREDICTED records (the
                                 1,312 FLARE23 records leave this file: the app swaps in the de-duplicated
                                 1,198-record FLARE23 corpus, re-issued alongside as corpus_flare23_kg_dedup.json; 0042 and 0102 restored -> pool 1,311)
  corpus_reference_113.json      the same 113 cases' REFERENCE (GT) records
  corpus_flare23_kg_dedup.json   the 1,198 FLARE23 pool records (two wrongly removed records restored), same rule
  flare23_FLARE23_0286/          the archived (superseded) 0286 final-case records, same rule
Records are the CURRENT journal corpora (real LiTS volumes since 09-16; the app pool still carried voxel/1000
pseudo-volumes) with FROZEN containment (rankings are computed on the frozen record; containment v2 is display-only).

Sub-site rule (scripts/kg/pancreas_subsite.py): patient left-right fraction of the tumor along the gland, thirds,
PCA cross-check, fail-closed guards.
  * pancreas, MSD reference records : the rule on the reference label (results/kg/pancreas_subsite_msd_gt.json)
  * pancreas, MSD predicted records : the rule on the REGENERATED semi-oracle mask (results/kg/
                                      msd_predicted_subsite_regen_2026-10-03.json) -- one rule for all 56, so
                                      pancreas_125 carries the same kind of value as every other MSD record
  * everything else                 : void
VOID = "na" everywhere (no 'unknown', 'none' or pancreatic words on liver/kidney); WHY it is void is kept in
anatomic_location_status: not_applicable (not a pancreas entry) | no_tumor | guard_failed (rule's orientation or
LR/PCA guard failed; site withheld for review) | not_computed (FLARE23 records: never derived).
The delivered word survives in anatomic_location_delivered whenever it changed.

  python reissue_app_pool.py
"""
import copy
import glob
import json
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
C = os.path.join(ROOT, "corpora")
OUT = os.path.join(ROOT, "app_handoff", "reissue_2026-10-03")
GT = {r["case"]: r for r in json.load(open(os.path.join(ROOT, "results", "kg", "pancreas_subsite_msd_gt.json")))["records"]}
PR = json.load(open(os.path.join(ROOT, "results", "kg", "msd_predicted_subsite_regen_2026-10-03.json")))["cases"]
SNOMED = {"head": "64163001", "body": "40133006", "tail": "73239005"}
ARCHIVE_0286 = os.environ.get("ARCHIVE_0286", "/path/to/staging/a0286")      # Drive archive/flare23_FLARE23_0286 copy


def site_for(rec, organ, od, kind):
    """-> (anatomic_location, status, provenance)"""
    if organ != "pancreas":
        return "na", "not_applicable", "head/body/tail applies to the pancreas only"
    if not od.get("has_tumor"):
        return "na", "no_tumor", "no tumor in this organ"
    if rec.get("dataset") != "msd":
        return "na", "not_computed", "FLARE23 records: sub-site never derived"
    src = GT.get(rec["case_id"]) if kind == "reference" else PR.get(rec["case_id"])
    if not src:
        return "na", "not_computed", "no rule output for this case"
    s = src["site"]
    if s in SNOMED:
        prov = ("rule on the reference label" if kind == "reference" else
                "rule on the regenerated semi-oracle mask (ausam_msd organ + flare_only tumor)")
        return s, "computed", f"{prov}; f_lr={src.get('f_lr')} f_pca={src.get('f_pca')}"
    return "na", "guard_failed", f"rule withheld the site: {src.get('reason')}"


def fix(rec, kind, log):
    for organ, od in (rec.get("organs") or {}).items():
        if not isinstance(od, dict) or "anatomic_location" not in od:
            continue
        delivered = od.get("anatomic_location_delivered", od["anatomic_location"])
        new, status, prov = site_for(rec, organ, od, kind)
        for k in ("anatomic_location_rule", "anatomic_location_snomed"):
            od.pop(k, None)
        od["anatomic_location"] = new
        od["anatomic_location_status"] = status
        od["anatomic_location_provenance"] = prov
        if new in SNOMED:
            od["anatomic_location_snomed"] = SNOMED[new]
        if new != delivered:
            od["anatomic_location_delivered"] = delivered
        else:
            od.pop("anatomic_location_delivered", None)
        k = (kind, rec.get("dataset"), organ, delivered, new, status)
        log[k] = log.get(k, 0) + 1


def load(p):
    return json.load(open(p))["records"]


def main():
    os.makedirs(OUT, exist_ok=True)
    log = {}
    pred = [r for ds in ("kits", "lits", "msd") for r in load(f"{C}/corpus_predicted_{ds}.json")]
    ref = [r for ds in ("kits", "lits", "msd") for r in load(f"{C}/corpus_gt_{ds}.json")]
    fl = load(f"{C}/reissue_2026-10-03/corpus_flare23_kg_dedup_1311.json")   # 1,198: FLARE23_0042 + 0102 restored
    for r in pred:
        fix(r, "predicted", log)
    for r in ref:
        fix(r, "reference", log)
    for r in fl:
        fix(r, "reference", log)
    meta = lambda n, what: {"n": n, "issued": "2026-10-03", "note": __doc__.strip(), "what": what}   # noqa: E731
    json.dump({**meta(len(pred), "113 single-organ PREDICTED records (KiTS 36, LiTS 21, MSD 56); replaces the 113 in corpus_eval_pool_1425.json"),
               "records": pred}, open(f"{OUT}/corpus_single_organ_113.json", "w"))
    json.dump({**meta(len(ref), "the same 113 cases, REFERENCE (GT) records"), "records": ref}, open(f"{OUT}/corpus_reference_113.json", "w"))
    json.dump({**meta(len(fl), "1,198 de-duplicated FLARE23 pool records (reference-derived; FLARE23_0042 and FLARE23_0102 restored 10-03, pool = 1,311)"), "records": fl}, open(f"{OUT}/corpus_flare23_kg_dedup.json", "w"))
    if os.path.isdir(ARCHIVE_0286):
        dst = f"{OUT}/flare23_FLARE23_0286"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(ARCHIVE_0286, dst, ignore=shutil.ignore_patterns("*.nii*"))
        for p in sorted(glob.glob(f"{dst}/record_*.json")):
            r = json.load(open(p))
            fix(r, "reference" if "reference" in os.path.basename(p) else "predicted", log)
            json.dump(r, open(p, "w"), indent=1)
    # checks: one void spelling, pancreas_125 consistent everywhere
    vals = {od["anatomic_location"] for r in pred + ref + fl for od in r["organs"].values() if isinstance(od, dict) and "anatomic_location" in od}
    p125 = [r for r in pred if r["case_id"] == "pancreas_125"][0]["organs"]["pancreas"]
    summary = {"values_present": sorted(vals), "pancreas_125_predicted": {k: p125[k] for k in p125 if k.startswith("anatomic_location")},
               "changes": [{"kind": a, "dataset": b, "organ": c, "delivered": d, "reissued": e, "status": f, "n": n}
                           for (a, b, c, d, e, f), n in sorted(log.items(), key=lambda x: (str(x[0])))]}
    json.dump(summary, open(os.path.join(ROOT, "results", "kg", "app_pool_reissue_2026-10-03.json"), "w"), indent=1)
    print("values:", sorted(vals))
    print("pancreas_125 predicted:", summary["pancreas_125_predicted"])
    for x in summary["changes"]:
        if x["organ"] == "pancreas" and x["dataset"] == "msd":
            print(x)


if __name__ == "__main__":
    main()
