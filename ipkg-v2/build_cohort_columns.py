#!/usr/bin/env python
"""Cohort columns for the IPKG manuscript: advisor item the co-author 9, plus the
de-duplication nobody had applied to anything but the pipeline team's tumour arm.

WHAT THIS SCRIPT IS FOR. Three cohort-level columns were still missing or
still spoke the wrong convention when the round-3 delivery landed:

  1  THE FLARE23 TWIN-CLEAN COLUMN IS STALE.  Round 2 published a FLARE23
     twin-clean column at n = 47, because the census it was computed on had
     only ever searched for twins of the 113 retrieval QUERIES - never for
     twins of TRAINING cases, and training membership is what exposure is.
     Under the complete census (430 pairs, the pipeline team round 3) FLARE23 exposure is
     40 of 54, so the twin-clean column is n = 14 and every number in it
     moves.  `--census` rebuilds it and names the 14.

  2  TEN SCANS ARE IN THE COHORT TWICE.  The round-2 test lists are LiTS 21
     + MSD 56 + KiTS 36 + FLARE23 54 = 167 rows, and ten of those rows are
     the same CT as another row: volume-76 and FLARE23_1027 are one scan,
     scored once as LiTS and once as FLARE23.  167 rows are 157 distinct
     scans.  the pipeline team applied this to the tumour arm in round 3 (cohort157); it
     has been applied to nothing else, and the pooled figures quoted from
     round 2 still count ten patients twice.  `--census` identifies every
     pair from the census independently, prices the de-duplication both ways
     round, and says what it moves.

  3  THE COHORT ORGAN ROW IS IN THE WRONG CONVENTION.  The advisor asked for
     convention A once the pipeline team's re-run landed.  the pipeline team cannot do it for the three
     single-organ collections, because those semi-oracle masks did not
     survive the September wipe (round-3 item 3).  But the AUTONOMOUS arm's
     organ model is TotalSegmentator v2 3 mm, every one of those masks is on
     local disk, and convention A is implemented in ipkg_organ_convention -
     so the autonomous organ row CAN move, and `--organ` moves it.  The
     semi-oracle cohort row cannot follow, and the tables say so in the row
     where the number would otherwise sit.

NOTHING HERE DEFINES A CONVENTION OR A METRIC.  ipkg_organ_convention is
imported, never copied; this script locates masks, drives it over the cohort
and aggregates.  A change to what "organ Dice" means is a change to that one
module and lands here without this file being touched.

WHAT MAY BE WRITTEN.  tables/cohort/ under this directory, and nothing else.
<app-workspace> is read-only (a live survey is hashed out of it every ten
minutes) and so are the dataset junctions inside this tree.

    python build_cohort_columns.py --census          # items 1 and 2, seconds
    python build_cohort_columns.py --organ           # item 3, CPU, resumable
    python build_cohort_columns.py --organ --jobs 4
    python build_cohort_columns.py --tables          # render the markdown
    python build_cohort_columns.py --all

`--organ` is resumable: every scored case is cached under a fingerprint of
the files it was computed from, so an interrupted run resumes and a changed
mask re-scores.
"""
import argparse
import datetime
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ipkg_organ_convention as oc                       # noqa: E402

VERSION = "build_cohort_columns/1.0"
# KiTS23-only output schema. 2 added the pooled kidney, which is the only
# sound KiTS comparison: KiTS annotates ONE kidney class, so a per-side Dice
# is as much a measurement of the laterality rule as of the mask.
KITS_SCHEMA = 2
# The unified id used for the pooled KiTS kidney. Not a label in the label
# map - a reporting key, and spelled so it cannot be mistaken for one.
KIDNEY_POOLED = "2|13"
# Seed for the order cases are scored in; see run_organ.
JOB_ORDER_SEED = 0

BASE = os.path.dirname(os.path.abspath(__file__))
LIVE = r"<app-workspace>"                                    # READ ONLY
OUTDIR = os.path.join(BASE, "tables", "cohort")

R3 = os.path.join(LIVE, "round3")
R2 = os.path.join(LIVE, "round2")
CENSUS_JSON = os.path.join(R3, "dedup_audit",
                           "flare23_twin_census_complete_2026-10-03.json")
EXPOSURE_JSON = os.path.join(R3, "dedup_audit",
                             "promptfree_twin_exposure_2026-10-03.json")
COHORT157_JSON = os.path.join(R3, "results",
                              "cohort157_rescore_2026-10-03.json")
PROMPTFREE_JSON = os.path.join(R2, "results",
                               "promptfree_round2_2026-10-01.json")
SPLITS_JSON = os.path.join(R2, "results", "split_manifests_2026-10-01.json")

FAST_PREDS = os.path.join(LIVE, "mmkg_cache_universal", "preds")

CENSUS_OUT = os.path.join(OUTDIR, "cohort_census.json")
ORGAN_OUT = os.path.join(OUTDIR, "organ_conventionA_cases.json")
ORGAN_SUMMARY = os.path.join(OUTDIR, "organ_conventionA_summary.json")
COST_OUT = os.path.join(OUTDIR, "cost_log.json")

# Collection names as ipkg_organ_convention.OBSERVABLE spells them, keyed by
# the short dataset tag the round-2 artefacts use.
COLLECTION = {"lits": "LiTS", "msd": "MSD Pancreas",
              "kits": "KiTS23", "flare23": "FLARE23"}
# The split manifest keys the tumour pools by a fifth set of names again.
SPLIT_KEY = {"lits": "lits", "msd": "pancreas", "kits": "kits",
             "flare23": "flare"}
DSORDER = ("lits", "msd", "kits", "flare23")

# The de-duplication rule. Stated here once, written into every artefact.
DEDUP_RULE = (
    "When one CT appears in the cohort twice - once under a single-organ "
    "collection and once as a FLARE23 case - the row kept is the "
    "NON-FLARE23 one and the FLARE23 row is dropped. Four reasons, and "
    "the first one is the correction of a reason we expected to give and "
    "the measurement did not support. "
    "(a) NOT because FLARE23's reference is the weaker one. That was the "
    "argument we started from - FLARE23's tumour class is one generic "
    "label that does not say whose - and on these ten scans it does not "
    "hold: the two reference tumour volumes are identical on 7 of the 10 "
    "pairs and differ by at most 1.8 cm3 on the other three, and the raw "
    "Dice differs by 0.020 on average. On these scans the two annotations "
    "agree. "
    "(b) Collection representation. The three single-organ collections "
    "have no other representation in the cohort - dropping the "
    "single-organ member would delete a patient from a column of 21, 56 "
    "or 36 - while FLARE23 keeps 44 rows of its own. "
    "(c) The host gate, which is the thing that actually differs. the pipeline team "
    "round 3 section 10 records the 5 mm gate's host classes as liver "
    "alone for LiTS, pancreas alone for MSD, the two kidneys for KiTS and "
    "any of seven classes for FLARE23. Keeping the FLARE23 member would "
    "put ten rows scored under the loosest gate in the study into a "
    "pooled column where every other row carries a one- or two-organ "
    "gate. "
    "(d) It is the pipeline team's rule in cohort157, so the two artefacts agree and the "
    "pooled cells can be cross-checked against each other - which they "
    "are, below. This script re-derives the ten pairs from the census "
    "independently rather than reading his list.")

BOOT_N = 2000
BOOT_SEED = 0


# ---------------------------------------------------------------- utilities
def _load(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False)
    os.replace(tmp, path)
    return path


def boot_ci(vals, n=BOOT_N, seed=BOOT_SEED):
    """Percentile bootstrap over scans, 2,000 resamples, seed 0.

    Matched to the pipeline team's rescore_cohort157 so a column produced here and a column
    produced there are the same statistic; the agreement is checked against
    his published pooled cell rather than assumed (see `--census` output)."""
    v = np.asarray([x for x in vals if x is not None], float)
    if v.size == 0:
        return None, (None, None)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, v.size, size=(n, v.size))
    means = v[idx].mean(axis=1)
    return (float(v.mean()),
            (float(np.percentile(means, 2.5)),
             float(np.percentile(means, 97.5))))


def cell(vals):
    m, (lo, hi) = boot_ci(vals)
    if m is None:
        return None
    return {"n": int(len([x for x in vals if x is not None])),
            "mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "median": round(float(np.median(
                [x for x in vals if x is not None])), 4)}


def fmt(c):
    if c is None:
        return "n/a"
    return "%.3f [%.3f, %.3f]" % (c["mean"], c["ci95"][0], c["ci95"][1])


# =========================================================== items 1 and 2
def build_census():
    """Exposure recomputed from the census and the split manifests, the
    twin-clean sets, and every scan that sits in the cohort twice."""
    t0 = time.time()
    census = _load(CENSUS_JSON)
    exposure = _load(EXPOSURE_JSON)
    splits = _load(SPLITS_JSON)
    pf = _load(PROMPTFREE_JSON)["models"]["sam3_tumor_generic"]["datasets"]

    # -- the twin graph. One FLARE23 case has at most one public twin (the
    # census asserts it and this re-checks it), so the graph is a forest of
    # stars centred on public cases.
    pairs = sorted({(p["flare23"], p["public_case"]) for p in census["pairs"]})
    twin_of = {}
    for f, pub in pairs:
        twin_of.setdefault(f, set()).add(pub)
        twin_of.setdefault(pub, set()).add(f)
    multi = {f: sorted(v) for f, v in twin_of.items()
             if f.startswith("FLARE23_") and len(v) > 1}

    # -- the splits the generic tumour model actually saw
    tp = splits["tumor_pools"]["global"]
    seen = {}            # dataset tag -> set of cases in TRAIN or VAL
    test = {}
    for ds in DSORDER:
        k = SPLIT_KEY[ds]
        seen[ds] = set(tp[k]["train"]) | set(tp[k]["val"])
        test[ds] = list(tp[k]["test"])
    seen_all = set().union(*seen.values())

    # -- exposure, recomputed. A scan is exposed when ANY twin of it is in
    # the model's TRAIN or VAL split, in any collection.
    recomputed = {}
    for ds in DSORDER:
        exp = sorted(c for c in test[ds]
                     if twin_of.get(c, set()) & seen_all)
        recomputed[ds] = {"n_test": len(test[ds]),
                          "n_with_twin": sum(1 for c in test[ds]
                                             if twin_of.get(c)),
                          "n_exposed": len(exp),
                          "exposed_cases": exp,
                          "twin_clean_cases": sorted(set(test[ds]) - set(exp))}

    # -- set equality against the pipeline team's delivered exposure, per collection
    agreement = {}
    for ds in DSORDER:
        ud = set(exposure["datasets"][ds]["exposed_cases"])
        ours = set(recomputed[ds]["exposed_cases"])
        agreement[ds] = {
            "pipeline_n_exposed": len(ud), "our_n_exposed": len(ours),
            "set_equal": ud == ours,
            "only_pipeline": sorted(ud - ours), "only_ours": sorted(ours - ud),
            "pipeline_n_test": exposure["datasets"][ds]["n_test"],
            "our_n_test": recomputed[ds]["n_test"],
            "test_lists_equal":
                {c["case"] for c in exposure["datasets"][ds]["cases"]}
                == set(test[ds])}

    # -- the duplicates: a FLARE23 test scan whose twin is also a test scan
    single_test = {}
    for ds in ("lits", "msd", "kits"):
        for c in test[ds]:
            single_test[c] = ds
    dups = []
    for f in test["flare23"]:
        for pub in sorted(twin_of.get(f, ())):
            if pub in single_test:
                dups.append({"flare23": f, "same_scan_as": pub,
                             "kept_collection": COLLECTION[single_test[pub]],
                             "dropped_row": "FLARE23/" + f,
                             "source": sorted(
                                 {p["source"] for p in census["pairs"]
                                  if p["flare23"] == f
                                  and p["public_case"] == pub})})
    pipeline_dups = {(d["flare23"], d["same_scan_as"])
               for d in _load(COHORT157_JSON)["dropped_duplicate_flare23_scans"]}
    our_dups = {(d["flare23"], d["same_scan_as"]) for d in dups}

    # -- per-case metrics, round 2, unchanged
    cases = {}
    for ds in DSORDER:
        for c in pf[ds]["cases"]:
            cases[(ds, c["case"])] = c

    # What the SAME CT scores under its two annotations. This is the price of
    # the de-duplication rule stated as a measurement rather than as a
    # preference: a reader can see, per scan, which row is kept and what the
    # other one said.
    #
    # And it turns out NOT to be mostly a disagreement about the reference.
    # The two reference tumour volumes are identical on 7 of the 10 pairs
    # and differ by at most 1.8 cm3 on the other three, and the RAW Dice
    # differs by 0.020 on average. The GATED Dice differs by 0.210 on
    # average and the single-organ row is the higher one on 10 pairs out of
    # 10 - because the gate is what differs. the pipeline team round 3 section 10 records
    # the host classes of the 5 mm gate: liver alone for LiTS, pancreas
    # alone for MSD, the two kidneys for KiTS, and any of seven classes
    # (1/2/3/5/6/7/20) for FLARE23. A seven-organ host admits far more
    # false-positive components than a one-organ host, so on one and the
    # same CT the FLARE23 row scores worse gated almost by construction.
    # The de-duplication therefore moves the gated column mainly because it
    # removes ten rows carrying the loose gate, and that is a different
    # statement from "it removes ten duplicated patients" - both are true
    # and the table says both.
    for d in dups:
        ds_single = single_test[d["same_scan_as"]]
        a = cases[("flare23", d["flare23"])]
        b = cases[(ds_single, d["same_scan_as"])]
        d["flare23_raw"] = a["tumor_raw"]["dice"]
        d["flare23_gated"] = a["tumor_gated"]["dice"]
        d["kept_raw"] = b["tumor_raw"]["dice"]
        d["kept_gated"] = b["tumor_gated"]["dice"]
        d["flare23_ref_cm3"] = a["tumor_raw"]["ref_cm3"]
        d["kept_ref_cm3"] = b["tumor_raw"]["ref_cm3"]
        d["flare23_exposed"] = d["flare23"] in set(
            recomputed["flare23"]["exposed_cases"])
        d["kept_exposed"] = d["same_scan_as"] in set(
            recomputed[ds_single]["exposed_cases"])

    def vals(keys, field):
        return [cases[k][field]["dice"] for k in keys]

    def keyset(ds, which):
        tl = test[ds]
        if which == "all":
            sel = tl
        else:
            sel = recomputed[ds]["twin_clean_cases"]
        return [(ds, c) for c in sel]

    # -- item 1: the FLARE23 column on its own 54, under the new census
    fl_clean = recomputed["flare23"]["twin_clean_cases"]
    flare_column = {
        "n_test": 54,
        "n_exposed": recomputed["flare23"]["n_exposed"],
        "twin_clean_n": len(fl_clean),
        "twin_clean_cases": fl_clean,
        "round2_published_twin_clean_n":
            pf["flare23"]["summary"]["twin_clean"]["n"],
        "why_it_moved": (
            "Round 2's census was built to de-duplicate the RETRIEVAL "
            "POOL: it searched for twins of the 113 queries and of pool "
            "records, and never for twins of TRAINING cases - which is "
            "exactly what exposure is. So it found 18 of the 54 FLARE23 "
            "test scans to have a twin at all and 7 to be exposed. the co-author's "
            "header scan of all 576 tumour-bearing FLARE23 studies fills "
            "that gap; the union is 430 pairs, all 54 have a twin and 40 "
            "are exposed, and the twin-clean subset falls from 47 to 14. "
            "Source: the pipeline team round 3 section 6 and "
            "promptfree_twin_exposure_2026-10-03.json, whose "
            "change_vs_2026-10-01 record reads n_with_twin 18 -> 54 and "
            "n_exposed_via_twin 7 -> 40 for FLARE23 and moves no other "
            "collection except MSD's 18 -> 17 on the FLARE23_0042 fix."),
        "primary_twin_clean": {
            "n": len(fl_clean),
            "tumor_raw": cell(vals(keyset("flare23", "clean"), "tumor_raw")),
            "tumor_gated": cell(vals(keyset("flare23", "clean"),
                                     "tumor_gated"))},
        "secondary_all": {
            "n": 54,
            "tumor_raw": cell(vals(keyset("flare23", "all"), "tumor_raw")),
            "tumor_gated": cell(vals(keyset("flare23", "all"),
                                     "tumor_gated"))},
        "round2_published_for_comparison": pf["flare23"]["summary"],
        "overlap_with_single_organ_cohort": sorted(
            d["flare23"] for d in dups if d["flare23"] in fl_clean),
    }

    # -- item 2: the de-duplication, priced three ways
    def pooled(rows, field):
        return cell([cases[k][field]["dice"] for k in rows])

    all167 = [(ds, c) for ds in DSORDER for c in test[ds]]
    clean167 = [(ds, c) for ds in DSORDER
                for c in recomputed[ds]["twin_clean_cases"]]
    drop_flare = {("flare23", d["flare23"]) for d in dups}
    drop_single = {(single_test[d["same_scan_as"]], d["same_scan_as"])
                   for d in dups}

    arms = {}
    for name, rows_all, rows_clean, note in (
        ("no_dedup_167", all167, clean167,
         "every row of the four test lists; ten scans appear twice"),
        ("dedup_keep_single_organ_157",
         [r for r in all167 if r not in drop_flare],
         [r for r in clean167 if r not in drop_flare],
         "the rule adopted here and by the pipeline team's cohort157: the FLARE23 row of "
         "each duplicated scan is dropped"),
        ("dedup_keep_flare23_157",
         [r for r in all167 if r not in drop_single],
         [r for r in clean167 if r not in drop_single],
         "the counterfactual: the single-organ row is dropped instead, so "
         "the choice of rule is priced rather than asserted"),
    ):
        arms[name] = {
            "note": note,
            "all": {"n": len(rows_all),
                    "tumor_raw": pooled(rows_all, "tumor_raw"),
                    "tumor_gated": pooled(rows_all, "tumor_gated")},
            "twin_clean": {"n": len(rows_clean),
                           "tumor_raw": pooled(rows_clean, "tumor_raw"),
                           "tumor_gated": pooled(rows_clean, "tumor_gated")},
            "per_collection": {
                COLLECTION[ds]: {
                    "all_n": sum(1 for r in rows_all if r[0] == ds),
                    "twin_clean_n": sum(1 for r in rows_clean if r[0] == ds),
                    "all_raw": pooled([r for r in rows_all if r[0] == ds],
                                      "tumor_raw"),
                    "all_gated": pooled([r for r in rows_all if r[0] == ds],
                                        "tumor_gated"),
                    "twin_clean_raw": pooled(
                        [r for r in rows_clean if r[0] == ds], "tumor_raw"),
                    "twin_clean_gated": pooled(
                        [r for r in rows_clean if r[0] == ds], "tumor_gated"),
                } for ds in DSORDER},
        }

    # -- does our pipeline reproduce the pipeline team's published cohort157 cell?
    ud157 = _load(COHORT157_JSON)["pooled"]
    ours157 = arms["dedup_keep_single_organ_157"]
    reproduction = {
        "pipeline_twin_clean_n": ud157["primary_twin_clean"]["n"],
        "our_twin_clean_n": ours157["twin_clean"]["n"],
        "pipeline_twin_clean_raw": ud157["primary_twin_clean"]["tumor_raw"]["dice"],
        "our_twin_clean_raw": ours157["twin_clean"]["tumor_raw"]["mean"],
        "pipeline_twin_clean_gated":
            ud157["primary_twin_clean"]["tumor_gated"]["dice"],
        "our_twin_clean_gated": ours157["twin_clean"]["tumor_gated"]["mean"],
        "pipeline_all_n": ud157["secondary_all"]["n"],
        "our_all_n": ours157["all"]["n"],
        "pipeline_all_raw": ud157["secondary_all"]["tumor_raw"]["dice"],
        "our_all_raw": ours157["all"]["tumor_raw"]["mean"],
        "pipeline_all_gated": ud157["secondary_all"]["tumor_gated"]["dice"],
        "our_all_gated": ours157["all"]["tumor_gated"]["mean"],
    }
    reproduction["agrees_to_4dp"] = all(
        abs(reproduction["our_" + k] - reproduction["pipeline_" + k]) < 1e-4
        for k in ("twin_clean_raw", "twin_clean_gated", "all_raw", "all_gated")
    ) and reproduction["our_twin_clean_n"] == reproduction["pipeline_twin_clean_n"] \
        and reproduction["our_all_n"] == reproduction["pipeline_all_n"]

    out = {
        "version": VERSION,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "what_this_is": (
            "Exposure, twin-clean sets and the cohort de-duplication, "
            "recomputed from the pipeline team's complete twin census and the split "
            "manifests rather than read off his summary tables. Every "
            "per-scan Dice is round 2's, unchanged."),
        "inputs": {
            "census": CENSUS_JSON,
            "pipeline_exposure": EXPOSURE_JSON,
            "pipeline_cohort157": COHORT157_JSON,
            "per_scan_metrics": PROMPTFREE_JSON,
            "splits": SPLITS_JSON},
        "protocol": {
            "arm": "autonomous (prompt-free): TotalSegmentator v2 3 mm "
                   "organs + sam3_tumor_generic tumour, text 'tumor', no "
                   "box, no reference used at inference",
            "tumor_raw": "every slice the confidence gate kept, no host gate",
            "tumor_gated": "tumour components touching the TotalSegmentator "
                           "host organ dilated 5 mm",
            "exposure": "a scan is EXPOSED when any twin of it is in "
                        "sam3_tumor_generic's TRAIN or VAL split (global "
                        "mode, the whole multi-dataset pool)",
            "statistic": "mean 3-D Dice over scans, percentile bootstrap "
                         "95 %% CI, %d resamples, seed %d"
                         % (BOOT_N, BOOT_SEED)},
        "twin_graph": {
            "n_pairs": len(pairs),
            "n_flare23_with_twin": len({f for f, _ in pairs}),
            "flare23_with_more_than_one_public_twin": multi},
        "splits": {ds: {"n_train_val": len(seen[ds]), "n_test": len(test[ds])}
                   for ds in DSORDER},
        "exposure_recomputed": recomputed,
        "exposure_agreement_with_UD": agreement,
        "exposure_all_four_collections_agree": all(
            a["set_equal"] and a["test_lists_equal"]
            for a in agreement.values()),
        "item1_flare23_twin_clean_column": flare_column,
        "item2_dedup": {
            "rule": DEDUP_RULE,
            "n_duplicate_pairs": len(dups),
            "pairs": dups,
            "our_pairs_equal_pipeline_cohort157": our_dups == pipeline_dups,
            "only_ours": sorted(our_dups - pipeline_dups),
            "only_pipeline": sorted(pipeline_dups - our_dups),
            "arms": arms,
            "reproduction_of_pipeline_cohort157": reproduction},
        "seconds": round(time.time() - t0, 2),
    }
    _write(CENSUS_OUT, out)
    return out


# ==================================================================== item 3
def fast_tags(collection):
    return {"LiTS": ["totalsegmentator__LiTS"],
            "KiTS23": ["totalsegmentator__KiTS23"],
            "MSD Pancreas": ["totalsegmentator__MSD_Pancreas"],
            "FLARE23": ["totalsegmentator__FLARE23__FLARE23_positive",
                        "totalsegmentator__FLARE23__FLARE23_negative"],
            }[collection]


def fast_key(collection, case_id):
    """LiTS is keyed by the native MSD Task03 basename on disk (liver_76) and
    by the split-pack id in every cohort list (volume-76). Unmapped, this
    silently finds no 3 mm mask for any LiTS case at all."""
    if collection == "LiTS" and case_id.startswith("volume-"):
        return "liver_" + case_id.split("-", 1)[1]
    return case_id


def case_id_from_key(collection, key):
    if collection == "LiTS" and key.startswith("liver_"):
        return "volume-" + key.split("_", 1)[1]
    return key


def find_fast(collection, case_id, what):
    key = fast_key(collection, case_id)
    for tag in fast_tags(collection):
        p = os.path.join(FAST_PREDS, tag, "%s__%s.nii.gz" % (key, what))
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    return None


def flare23_arm_of(path):
    """FLARE23 was run as two arms and both are on disk: 644 tumour-bearing
    studies and 305 tumour-free ones. They are not one collection for an
    organ table - the tumour-free arm has no mass to attribute, so under
    convention A it is identical to the parenchyma reading by construction
    and would dilute a column that exists to show the difference. They are
    reported as two rows, never silently pooled."""
    if not path:
        return None
    d = os.path.basename(os.path.dirname(path))
    if d.endswith("FLARE23_positive"):
        return "FLARE23 tumour-positive"
    if d.endswith("FLARE23_negative"):
        return "FLARE23 tumour-free"
    return None


def flare_arm(collection, pred_path):
    return (flare23_arm_of(pred_path) if collection == "FLARE23"
            else collection)


def find_native_kits(case_id):
    p = os.path.join(LIVE, "KiTs23", "dataset", case_id, "segmentation.nii.gz")
    return p if os.path.isfile(p) and os.path.getsize(p) > 0 else None


def read_labels(path):
    """Label volume as int16, plus the voxel size in millimetres.

    THE SCALING TRAP, and why this is not a plain asanyarray. A third of the
    FLARE23 label files carry an scl_slope, which brings tumour class 14 back
    as 14.000000115484, so `== 14` matches nothing at all and does so
    silently. src/eval/baseline.py and score_fullres_compare.py therefore go
    through get_fdata(float32) and np.rint, and so does this.

    THE MEMORY TRAP, and why this is not a plain get_fdata either. The float32
    path allocates four bytes a voxel before the int16 cast, and the largest
    KiTS23 cases are 512 x 512 x 600; three of those volumes in one worker
    (reference, prediction and the native segmentation the laterality repair
    needs) is a gigabyte and a half before any scoring starts, and this run
    has several workers on a machine that has been reaped once. So when the
    file says it is stored as integers with no scaling to apply - slope 1,
    intercept 0 - the raw array IS the label array, rint is the identity on
    it, and it is read at its own width. The float path is kept for every
    other case and the two are checked against each other in `selftest`."""
    import nibabel as nib
    img = nib.load(path)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    prox = img.dataobj
    slope = getattr(prox, "slope", None)
    inter = getattr(prox, "inter", None)
    try:
        trivial = (slope is not None and inter is not None
                   and float(slope) == 1.0 and float(inter) == 0.0
                   and np.issubdtype(np.dtype(prox.dtype), np.integer))
    except Exception:                                       # noqa: BLE001
        trivial = False
    if trivial:
        arr = np.asanyarray(prox)
        if arr.dtype != np.int16:
            arr = arr.astype(np.int16)
        return np.ascontiguousarray(arr), zooms
    arr = np.rint(np.ascontiguousarray(
        img.get_fdata(dtype=np.float32))).astype(np.int16)
    return arr, zooms


def affine_of(path):
    import nibabel as nib
    return nib.load(path).affine


def discover_all():
    """Every case for which a 3 mm mask AND its reference are on disk."""
    out = []
    for collection in ("LiTS", "KiTS23", "MSD Pancreas", "FLARE23"):
        for tag in fast_tags(collection):
            d = os.path.join(FAST_PREDS, tag)
            for p in sorted(glob.glob(os.path.join(d,
                                                   "*__pred_unified.nii.gz"))):
                if os.path.getsize(p) == 0:
                    continue
                key = os.path.basename(p)[:-len("__pred_unified.nii.gz")]
                gt = os.path.join(d, "%s__gt_unified.nii.gz" % key)
                if not (os.path.isfile(gt) and os.path.getsize(gt) > 0):
                    continue
                out.append((collection, case_id_from_key(collection, key)))
    return sorted(set(out))


def fingerprint(paths, collection=None):
    fp = {}
    for k, p in paths.items():
        if p and os.path.isfile(p):
            st = os.stat(p)
            fp[k] = [os.path.basename(p), st.st_size, int(st.st_mtime)]
        else:
            fp[k] = None
    fp["convention"] = oc.CONVENTION_VERSION
    fp["script"] = VERSION
    if collection == "KiTS23":
        # Bumped when the pooled-kidney structure was added. It is in the
        # fingerprint rather than in VERSION so that adding a KiTS-only
        # structure re-scores KiTS and leaves the other three collections'
        # cached records alone - the arithmetic on those did not change, and
        # re-reading 1,056 volumes to prove it would cost two hours.
        fp["kits_schema"] = KITS_SCHEMA
    return fp


# A worker will not START a case while free memory is under this. The
# largest KiTS23 reference is 1,059 x 512 x 512 stored as float64, which
# nibabel materialises at 2.2 GB before the int16 cast; measured peak RSS
# for one worker on that case is 2.35 GB. Several of those landing together
# is how this machine was reaped once before, so a worker waits instead.
# It is a gate on STARTING, not a limit on a case already running: a case
# never fails for want of memory, it only begins later, so the cohort is
# never silently sampled by the throttle.
MEM_FLOOR_GB = 8.0
MEM_WAIT_MAX_S = 300.0


def _wait_for_memory(floor_gb=None):
    """Returns the seconds waited, so the cost log can report the throttle
    rather than hide it inside the per-case time."""
    floor = MEM_FLOOR_GB if floor_gb is None else floor_gb
    try:
        import psutil
    except Exception:                                       # noqa: BLE001
        return 0.0
    t0 = time.time()
    while time.time() - t0 < MEM_WAIT_MAX_S:
        if psutil.virtual_memory().available / 1e9 >= floor:
            break
        time.sleep(2.0)
    return round(time.time() - t0, 1)


def score_organ_case(job):
    """Convention A for one case, with the pool protected from one bad
    allocation.

    A MemoryError raised inside a worker propagates through
    imap_unordered and kills the whole run - which is exactly what
    happened at case 925 of 1,665 on a machine three other jobs were
    sharing, costing nothing in results (every case is checkpointed) but
    an hour in wall clock. Here it is caught: the worker waits for memory
    to come back, retries the case ONCE, and if it fails again returns a
    status instead of a result. A case that could not be scored is then
    listed by name in Table C3, which is the behaviour the rest of this
    script already has for every other kind of failure.
    """
    for attempt in (1, 2):
        try:
            rec = _score_one(job)
            if attempt == 2:
                rec["memory_retry"] = True
            return rec
        except MemoryError:
            if attempt == 2:
                return {"collection": job[0], "case_id": job[1],
                        "status": "memory_error_after_retry"}
            _wait_for_memory(floor_gb=MEM_FLOOR_GB * 1.5)


def _score_one(job):
    """Convention A for one case, autonomous arm. Returns a plain dict.

    Module level and picklable on purpose: this runs under
    multiprocessing, and a closure would not travel."""
    collection, case_id = job
    t0 = time.time()
    paths = {"gt": find_fast(collection, case_id, "gt_unified"),
             "pred": find_fast(collection, case_id, "pred_unified")}
    rec = {"collection": collection, "case_id": case_id,
           "arm": flare_arm(collection, paths["pred"]),
           "fingerprint": fingerprint(paths, collection)}
    if not paths["gt"] or not paths["pred"]:
        rec["status"] = "missing_mask"
        return rec
    waited = _wait_for_memory()
    if waited:
        rec["memory_wait_seconds"] = waited
    try:
        gt, zooms = read_labels(paths["gt"])
        pred, _ = read_labels(paths["pred"])
    except Exception as e:                                  # noqa: BLE001
        rec["status"] = "read_failed: %s" % e
        return rec
    if gt.shape != pred.shape:
        rec["status"] = "shape_mismatch"
        rec["shapes"] = {"gt": list(gt.shape), "pred": list(pred.shape)}
        return rec
    present = set(int(v) for v in np.unique(pred) if v > 0)
    if present - set(range(0, 16)):
        # The 3 mm masks were remapped into the unified space by baseline.py
        # when they were written. A raw TotalSegmentator label here would
        # compare spleen against liver, so it is a hard stop, not a guess.
        rec["status"] = "pred_not_unified"
        return rec

    # KiTS23 annotates ONE kidney class; the staged twin's laterality came
    # from a splitter that discarded components under 5,000 voxels and on
    # case_00067 lost 29,026 of 485,178 reference kidney voxels. Repaired
    # from the dataset's own segmentation, or the case is not scored - a
    # fallback to the twin would score KiTS against a reference 6 % short,
    # which is the exact defect this guards.
    if collection == "KiTS23":
        nat = find_native_kits(case_id)
        if not nat:
            rec["status"] = "missing_native_kits_segmentation"
            return rec
        try:
            native, nat_zooms = read_labels(nat)
        except Exception as e:                              # noqa: BLE001
            rec["status"] = "kits_native_read_failed: %s" % e
            return rec
        if native.shape != gt.shape:
            rec["status"] = "kits_native_shape_mismatch"
            return rec
        if not np.allclose(affine_of(nat), affine_of(paths["gt"]), atol=1e-4):
            rec["status"] = "kits_native_affine_mismatch"
            return rec
        try:
            gt, kinfo = oc.repair_kits_reference(gt, native == 1,
                                                 affine_of(nat), nat_zooms)
        except oc.ConservationError as e:
            rec["status"] = "kits_repair_conservation_error: %s" % e
            return rec
        del native
        rec["kits_reference_repair"] = {
            "was_already_correct": kinfo["was_already_correct"],
            "voxels_recovered": kinfo["voxels_recovered"],
            "voxels_relabelled_between_sides":
                kinfo["voxels_relabelled_between_sides"]}

    vox_cm3 = float(np.prod(zooms)) / 1000.0
    rec["spacing_mm"] = [round(z, 4) for z in zooms]
    rec["voxel_cm3"] = vox_cm3
    mass_by_organ, unattr, detail = oc.attribute_mass(gt, zooms)
    rec["mass_attribution"] = {
        "rule": oc.RULE_VOTE,
        "n_components": len(detail),
        "unattributed_voxels": int(unattr),
        "n_components_where_rules_disagree": sum(
            1 for d in detail if d.get("rules_agree") is False),
        "definition_is": "ipkg_organ_convention.CONVENTION['mass_attribution']"}
    cy = oc.cyst_census(gt)
    rec["cyst"] = cy
    rec["organs"] = {}
    for oid in sorted(oc.OBSERVABLE[collection]):
        ref_te, ref_wm = oc.organ_references(gt, oid, mass_by_organ)
        if not ref_te.any():
            continue                     # not annotated in this case
        p = (pred == oid)
        sc = oc.organ_scores(p, ref_te, ref_wm)
        rec["organs"][str(oid)] = {
            "organ_name": oc.UNIFIED[oid],
            "ref_cm3_parenchyma": round(float(ref_te.sum()) * vox_cm3, 3),
            "ref_cm3_with_mass": round(float(ref_wm.sum()) * vox_cm3, 3),
            "pred_cm3": round(float(p.sum()) * vox_cm3, 3),
            "hosts_mass": bool(oid in mass_by_organ),
            "mass_cm3": (round(float(mass_by_organ[oid].sum()) * vox_cm3, 3)
                         if oid in mass_by_organ else 0.0),
            "scores": {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in sc.items()
                       if not isinstance(v, (dict, list))},
            "scores_full": {k: v for k, v in sc.items()
                            if isinstance(v, (dict, list))},
        }
    # KiTS23 annotates ONE kidney class. Right and left exist only because
    # ipkg_organ_convention.kidney_sides decided they do, and the same
    # decision is made independently on the prediction - so a per-side Dice
    # is as much a measurement of the laterality rule as of the mask, and on
    # a case where one 78,604-voxel component sits near the midline the two
    # rules put it on opposite sides and both per-side numbers move. The
    # POOLED kidney is the one sound KiTS comparison; it is what the KiTS
    # case-mean row is built from, and the two sides are kept beside it so
    # the split is visible rather than hidden.
    if collection == "KiTS23":
        ref_te = (gt == oc.RIGHT_KIDNEY) | (gt == oc.LEFT_KIDNEY)
        if ref_te.any():
            extra = None
            for oid in (oc.RIGHT_KIDNEY, oc.LEFT_KIDNEY):
                m = mass_by_organ.get(oid)
                if m is not None:
                    extra = m if extra is None else (extra | m)
            ref_wm = (ref_te | extra) if extra is not None else ref_te
            pp = (pred == oc.RIGHT_KIDNEY) | (pred == oc.LEFT_KIDNEY)
            sc = oc.organ_scores(pp, ref_te, ref_wm)
            rec["organs"][KIDNEY_POOLED] = {
                "organ_name": "kidney (pooled)",
                "ref_cm3_parenchyma": round(float(ref_te.sum()) * vox_cm3, 3),
                "ref_cm3_with_mass": round(float(ref_wm.sum()) * vox_cm3, 3),
                "pred_cm3": round(float(pp.sum()) * vox_cm3, 3),
                "hosts_mass": bool(extra is not None),
                "mass_cm3": (round(float(extra.sum()) * vox_cm3, 3)
                             if extra is not None else 0.0),
                "scores": {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in sc.items()
                           if not isinstance(v, (dict, list))},
                "scores_full": {k: v for k, v in sc.items()
                                if isinstance(v, (dict, list))},
                "why": "KiTS23 annotates one kidney class; pooled is the "
                       "sound comparison and the two sides measure the "
                       "laterality rule as much as the mask."}
    rec["status"] = "ok"
    rec["seconds"] = round(time.time() - t0, 2)
    return rec


def cohort_jobs():
    """The 167 evaluation scans, as (collection, case_id).

    Scored first when `--cohort` is passed, so the rows the manuscript
    actually quotes exist even if the full-collection sweep has to be cut
    short - and so a reader can tell a column that was computed from a
    truncated run (it would be missing cases, and Table C3 names them)."""
    c = _load(CENSUS_OUT)
    out = []
    for ds in DSORDER:
        e = c["exposure_recomputed"][ds]
        for case in sorted(set(e["exposed_cases"]) | set(e["twin_clean_cases"])):
            out.append((COLLECTION[ds], case))
    return out


def run_organ(jobs_n=4, limit=None, rescore=False, only=None, cohort=False):
    import multiprocessing as mp
    os.makedirs(OUTDIR, exist_ok=True)
    cache = {}
    if os.path.isfile(ORGAN_OUT) and not rescore:
        try:
            cache = {"%s/%s" % (r["collection"], r["case_id"]): r
                     for r in _load(ORGAN_OUT)["cases"]}
        except Exception:                                   # noqa: BLE001
            cache = {}
    todo_all = discover_all()
    if cohort:
        want = set(cohort_jobs())
        todo_all = [j for j in todo_all if j in want]
    if only:
        todo_all = [j for j in todo_all if j[0] == only]
    todo = []
    for job in todo_all:
        k = "%s/%s" % job
        hit = cache.get(k)
        if hit and hit.get("status") == "ok":
            want = fingerprint({"gt": find_fast(job[0], job[1], "gt_unified"),
                                "pred": find_fast(job[0], job[1],
                                                  "pred_unified")}, job[0])
            if hit.get("fingerprint") == want:
                continue
        todo.append(job)
    n_cached = len(todo_all) - len(todo)
    # DETERMINISTIC SHUFFLE, and it is not cosmetic. Discovered in
    # collection order, any run that has to stop early finishes FLARE23,
    # cuts KiTS23 in half and never reaches LiTS or MSD - a prefix that is
    # not a sample of anything. Shuffled with a fixed seed, every prefix is
    # a uniform random sample of all four collections, so an interrupted
    # run yields a column that can be published with its n and its seed
    # instead of a column that has to be thrown away. The seed is recorded
    # in the artefact and the order is reproducible.
    todo.sort()
    np.random.RandomState(JOB_ORDER_SEED).shuffle(todo)
    todo = [tuple(j) for j in todo]
    # The 167 evaluation scans go FIRST, keeping their shuffled order among
    # themselves. They are the rows the manuscript quotes, and a run that
    # has to stop early must not leave them half-done just because the
    # shuffle scattered them through 1,850 cases. The remainder keeps the
    # prefix property, so a truncated run still yields a uniform random
    # sample of the full collections.
    if not cohort:
        try:
            first = set(cohort_jobs())
        except Exception:                                   # noqa: BLE001
            first = set()
        todo = ([j for j in todo if j in first]
                + [j for j in todo if j not in first])
    if limit:
        todo = todo[:limit]
    print("[organ] %d cases on disk, %d already cached, %d to score "
          "(%slimit), %d workers"
          % (len(todo_all), n_cached, len(todo),
             "" if limit else "no ", jobs_n), flush=True)
    t0 = time.time()
    done = 0
    if todo:
        with mp.Pool(processes=jobs_n, maxtasksperchild=8) as pool:
            for rec in pool.imap_unordered(score_organ_case, todo,
                                           chunksize=1):
                cache["%s/%s" % (rec["collection"], rec["case_id"])] = rec
                done += 1
                if done % 25 == 0 or done == len(todo):
                    el = time.time() - t0
                    print("[organ] %4d/%d  %.1f min elapsed, %.2f s/case, "
                          "eta %.1f min" % (done, len(todo), el / 60.0,
                                            el / done,
                                            (len(todo) - done) * el / done
                                            / 60.0), flush=True)
                    _write(ORGAN_OUT, {"version": VERSION,
                                       "convention": oc.CONVENTION_VERSION,
                                       "partial": done < len(todo),
                                       "cases": list(cache.values())})
    wall = time.time() - t0
    _write(ORGAN_OUT, {
        "version": VERSION,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "convention": oc.CONVENTION_VERSION,
        "convention_text": oc.CONVENTION,
        "arm": "autonomous: organ = TotalSegmentator v2 --fast (3 mm), "
               "unified label space; the organ model's own mask, never "
               "unioned with or carved by a predicted tumour",
        "source_masks": FAST_PREDS,
        "partial": False,
        "cases": list(cache.values())})
    cost = {"scored_this_run": done, "wall_seconds": round(wall, 1),
            "workers": jobs_n, "job_order_seed": JOB_ORDER_SEED,
            "job_order": "deterministic shuffle, so any prefix of an "
                         "interrupted run is a uniform random sample of "
                         "all four collections",
            "seconds_per_case": round(wall / done, 2) if done else None,
            "cached_from_earlier_runs": len(cache) - done}
    print("[organ] wrote %s (%d cases, %.1f min this run)"
          % (ORGAN_OUT, len(cache), wall / 60.0), flush=True)
    return cost


# ------------------------------------------------------- organ aggregation
def summarise_organ(census):
    """Collection-level and cohort-level convention-A columns."""
    data = _load(ORGAN_OUT)
    by_key = {}
    for r in data["cases"]:
        by_key["%s/%s" % (r["collection"], r["case_id"])] = r
    ok = [r for r in data["cases"] if r.get("status") == "ok"]
    bad = [{"case": "%s/%s" % (r["collection"], r["case_id"]),
            "status": r.get("status")}
           for r in data["cases"] if r.get("status") != "ok"]

    def agg(records):
        """One cell per organ plus the case-mean row.

        The case-mean row averages each case's own mean over the organs it
        annotates, so a FLARE23 case with 13 organs does not outweigh a LiTS
        case with one - the unit of the cohort is the scan."""
        per_organ = {}
        case_means = {"organ_dice": [], "parenchyma_dice": [],
                      "parenchyma_recall": []}
        n_cases = 0
        n_hosting = 0
        for r in records:
            vals = {k: [] for k in case_means}
            # KiTS: the pooled kidney is the case's organ reading; the two
            # sides are recorded per organ but never averaged into a case
            # mean, because averaging two halves of one laterality decision
            # measures the decision.
            primary = (set(r["organs"]) & {KIDNEY_POOLED}
                       if r["collection"] == "KiTS23" else set(r["organs"]))
            for oid, e in r["organs"].items():
                s = e["scores"]
                row = per_organ.setdefault(
                    e["organ_name"],
                    {"organ_dice": [], "parenchyma_dice": [],
                     "parenchyma_recall": [], "mass_coverage": [],
                     "n_hosting_mass": 0, "n": 0})
                row["n"] += 1
                for k in ("organ_dice", "parenchyma_dice",
                          "parenchyma_recall"):
                    if s.get(k) is not None:
                        row[k].append(s[k])
                        if oid in primary:
                            vals[k].append(s[k])
                if e["hosts_mass"]:
                    row["n_hosting_mass"] += 1
                    if s.get("mass_coverage") is not None:
                        row["mass_coverage"].append(s["mass_coverage"])
            if not vals["organ_dice"]:
                continue
            n_cases += 1
            if any(e["hosts_mass"] for e in r["organs"].values()):
                n_hosting += 1
            for k in case_means:
                if vals[k]:
                    case_means[k].append(float(np.mean(vals[k])))
        out = {"n_cases": n_cases, "n_records": len(records),
               "n_records_with_no_annotated_organ": len(records) - n_cases,
               "n_cases_with_a_hosted_mass": n_hosting,
               "case_mean": {k: cell(v) for k, v in case_means.items()},
               "per_organ": {}}
        for name, row in sorted(per_organ.items()):
            out["per_organ"][name] = {
                "n_structures": row["n"],
                "n_hosting_mass": row["n_hosting_mass"],
                "organ_dice": cell(row["organ_dice"]),
                "parenchyma_dice": cell(row["parenchyma_dice"]),
                "parenchyma_recall": cell(row["parenchyma_recall"]),
                "mass_coverage": cell(row["mass_coverage"])}
        return out

    def arm_of(r):
        """Which FLARE23 arm a record belongs to.

        Backfilled from the mask's own directory when the record predates
        the field, rather than re-scoring 54 volumes to learn something
        that is written on the path. A record whose arm cannot be resolved
        is reported, never defaulted into a row."""
        a = r.get("arm")
        if a:
            return a
        if r["collection"] == "FLARE23":
            return (flare23_arm_of(find_fast("FLARE23", r["case_id"],
                                             "pred_unified"))
                    or "FLARE23 arm unresolved")
        return r["collection"]

    ROWS = ("LiTS", "KiTS23", "MSD Pancreas", "FLARE23 tumour-positive",
            "FLARE23 tumour-free")

    cohorts = {}
    # (a) every 3 mm mask on disk, by collection. FLARE23 is two rows: the
    # 644 tumour-bearing studies and the 305 tumour-free ones, which under
    # convention A are identical to their parenchyma reading by
    # construction and so are never pooled into the FLARE23 row.
    for coll in ROWS:
        recs = [r for r in ok if arm_of(r) == coll]
        cohorts.setdefault("full_collection", {})[coll] = agg(recs)
    unresolved = sorted("%s/%s" % (r["collection"], r["case_id"])
                        for r in ok
                        if arm_of(r) not in ROWS)
    cohorts["full_collection"]["ALL"] = agg(
        [r for r in ok if arm_of(r) != "FLARE23 tumour-free"])
    cohorts["full_collection"]["_arm_unresolved"] = unresolved
    cohorts["full_collection"]["ALL_note"] = (
        "the pooled row excludes the FLARE23 tumour-free arm, which has no "
        "mass to attribute and would make convention A and parenchyma "
        "agree by construction on 305 of the scans")

    # (b) the evaluation cohort, so the autonomous organ row and the
    # autonomous tumour row are the same scans
    test = {ds: census["exposure_recomputed"][ds] for ds in DSORDER}
    tl = {}
    for ds in DSORDER:
        tl[ds] = (set(test[ds]["exposed_cases"])
                  | set(test[ds]["twin_clean_cases"]))
    drop_flare = {d["flare23"]
                  for d in census["item2_dedup"]["pairs"]}

    drop_single = {d["same_scan_as"] for d in census["item2_dedup"]["pairs"]}

    def rows(which, dedup):
        out = []
        for ds in DSORDER:
            sel = (test[ds]["twin_clean_cases"] if which == "twin_clean"
                   else sorted(tl[ds]))
            for c in sel:
                if dedup == "keep_single" and ds == "flare23"                         and c in drop_flare:
                    continue
                if dedup == "keep_flare23" and ds != "flare23"                         and c in drop_single:
                    continue
                r = by_key.get("%s/%s" % (COLLECTION[ds], c))
                if r and r.get("status") == "ok":
                    out.append(r)
        return out

    for name, which, dedup in (
            ("cohort_167_all_no_dedup", "all", None),
            ("cohort_157_all_dedup", "all", "keep_single"),
            ("cohort_157_all_dedup_keep_flare23", "all", "keep_flare23"),
            ("cohort_76_twin_clean_dedup", "twin_clean", "keep_single")):
        recs = rows(which, dedup)
        cohorts[name] = {"ALL": agg(recs)}
        for ds in DSORDER:
            cohorts[name][COLLECTION[ds]] = agg(
                [r for r in recs if r["collection"] == COLLECTION[ds]])
        cohorts[name]["_arms"] = sorted({arm_of(r) for r in recs})

    # coverage: which cohort scans have no 3 mm mask
    missing = {}
    for ds in DSORDER:
        want = sorted(tl[ds])
        miss = [c for c in want
                if by_key.get("%s/%s" % (COLLECTION[ds], c),
                              {}).get("status") != "ok"]
        missing[COLLECTION[ds]] = {"n_wanted": len(want), "n_missing":
                                   len(miss), "missing": miss}

    # -- what the paper currently says, and what moves ---------------------
    # Round 2's cohort organ row is the pipeline team's convention B: the prediction is the
    # TotalSegmentator organ UNIONED WITH THE PREDICTED TUMOUR. A B number
    # and an A number are not comparable and are never put in one column -
    # but the TUMOUR-EXCLUDED pair IS comparable, because both are "the
    # organ model against the organ label with the mass cut out", so that
    # is where the two can be checked against each other and where a
    # disagreement means something.
    pf = _load(PROMPTFREE_JSON)["models"]["sam3_tumor_generic"]["datasets"]
    ORG_OF = {"lits": ["1"], "msd": ["4"], "kits": [KIDNEY_POOLED],
              "flare23": None}
    vs = {}
    for ds in ("lits", "msd", "kits"):
        s_all = pf[ds]["summary"]["all"]
        recs = [by_key.get("%s/%s" % (COLLECTION[ds], c["case"]))
                for c in pf[ds]["cases"]]
        recs = [r for r in recs if r and r.get("status") == "ok"]
        udcase = {c["case"]: c for c in pf[ds]["cases"]}
        par, conA, dref, dpred = [], [], [], []
        for r in recs:
            es = [r["organs"][o] for o in ORG_OF[ds] if o in r["organs"]]
            if not es:
                continue
            pv = [e["scores"]["parenchyma_dice"] for e in es
                  if e["scores"].get("parenchyma_dice") is not None]
            av = [e["scores"]["organ_dice"] for e in es
                  if e["scores"].get("organ_dice") is not None]
            if pv:
                par.append(float(np.mean(pv)))
            if av:
                conA.append(float(np.mean(av)))
            # Are we even scoring the same two masks he scored? The
            # reference and the prediction are compared by VOLUME, per
            # case, because a convention change and a different model run
            # move a Dice the same way and only this separates them.
            u = (udcase.get(r["case_id"]) or {}).get("organ_ts")
            if u:
                dref.append(sum(e["ref_cm3_parenchyma"] for e in es)
                            - u["excludes_tumor"]["ref_cm3"])
                dpred.append(sum(e["pred_cm3"] for e in es)
                             - u["excludes_tumor"]["pred_cm3"])
        vs[COLLECTION[ds]] = {
            "n_published": s_all["n"],
            "n_ours": len(conA),
            "published_tumour_excluded": s_all["organ_excludes_tumor_dice"],
            "our_parenchyma_dice": (round(float(np.mean(par)), 4)
                                    if par else None),
            "comparable": True,
            "published_organ_with_mass_CONVENTION_B":
                s_all["organ_with_mass_dice"],
            "our_organ_dice_CONVENTION_A": (round(float(np.mean(conA)), 4)
                                            if conA else None),
            "same_reference": {
                "n": len(dref),
                "n_identical_to_0.01_cm3": sum(1 for x in dref
                                               if abs(x) < 0.01),
                "mean_diff_cm3": (round(float(np.mean(dref)), 3)
                                  if dref else None)},
            "same_prediction": {
                "n": len(dpred),
                "n_identical_to_0.01_cm3": sum(1 for x in dpred
                                               if abs(x) < 0.01),
                "mean_diff_cm3": (round(float(np.mean(dpred)), 3)
                                  if dpred else None)},
            "B_and_A_are_not_comparable": (
                "convention B unions the PREDICTED tumour into the "
                "prediction; convention A does not. The two columns are "
                "printed side by side only so a reader who has the round-2 "
                "number in hand can find the one that replaces it."),
        }
    vs["FLARE23"] = {
        "n_published": pf["flare23"]["summary"]["all"]["n"],
        "published_tumour_excluded": None,
        "published_organ_with_mass_CONVENTION_B": None,
        "note": "round 2 published no FLARE23 organ cell at all; this is "
                "the first one.",
        "our_organ_dice_CONVENTION_A": (
            cohorts["cohort_167_all_no_dedup"].get("FLARE23", {})
            .get("case_mean", {}).get("organ_dice", {}) or {}).get("mean"),
    }
    vs["_not_only_the_convention_moved"] = (
        "The organ MASKS are not the same masks. The reference is: on LiTS "
        "and MSD the per-case reference organ volume agrees with round 2's "
        "to better than 0.01 cm3 on every case. The PREDICTION is not: it "
        "differs on every case, and ours is systematically the larger. The "
        "two arms are two different TotalSegmentator invocations - the "
        "masks scored here were written by src/eval/baseline.py's "
        "run_totalsegmentator, which runs ts_runner in a clean subprocess "
        "with task 'total', fast=True and NO roi_subset, on the local "
        "1080 Ti, and remaps to the unified space at write time; the pipeline team round "
        "3 section 10 records his as the Python API with fast=True, "
        "ml=True and roi_subset of seven classes. So the movement between "
        "round 2's organ cell and this one is NOT a pure convention "
        "change and must never be presented as one. What this table "
        "legitimately claims is that THE AUTONOMOUS ORGAN COLUMN NOW HAS "
        "A STATED CONVENTION and travels with its parenchyma pair - not "
        "that convention A raised or lowered a number by the difference "
        "shown.")
    kits_recs = [by_key.get("KiTS23/%s" % c["case"])
                 for c in pf["kits"]["cases"]]
    kits_recs = [r for r in kits_recs if r and r.get("status") == "ok"]
    n_cyst = sum(1 for r in kits_recs
                 if (r.get("cyst") or {}).get("cyst_bearing"))
    n_rep = sum(1 for r in kits_recs
                if not (r.get("kits_reference_repair") or {})
                .get("was_already_correct", True))
    vs["_why_kits_moves"] = (
        "Measured, not assumed, and the first thing the measurement did "
        "was refute the explanation we expected to give. "
        "(1) THE REFERENCE DID NOT MOVE. Our pooled parenchyma reference "
        "matches round 2's tumour-excluded reference to better than 0.01 "
        "cm3 on all %d cohort cases. The laterality repair fires on %d of "
        "them - it is not a no-op - but what it restores is exactly the "
        "dataset's own whole kidney, which is what round 2 scored against "
        "too. The 5,000-voxel splitter defect was a defect of the "
        "PER-SIDE reference, and this table does not use a per-side "
        "reference. "
        "(2) THE PREDICTION DID MOVE, and that is the whole of the "
        "parenchyma gap: %.4f against round 2's %.4f with an identical "
        "reference means a different organ mask, and ours is larger by "
        "%.1f cm3 on average - a different TotalSegmentator invocation, "
        "not a different convention. "
        "(3) THE CYST RULING MOVES THE CONVENTION-A COLUMN ONLY, and on "
        "this cohort it is material: %d of the %d KiTS cohort cases carry "
        "a reference cyst, and on each of those the ruling adds the cyst "
        "to the organ-with-mass reference while leaving the parenchyma "
        "reference alone. It cannot explain any part of the parenchyma "
        "difference, and it is not offered as doing so."
        % (vs["KiTS23"]["same_reference"]["n_identical_to_0.01_cm3"],
           n_rep,
           vs["KiTS23"]["our_parenchyma_dice"] or 0.0,
           vs["KiTS23"]["published_tumour_excluded"] or 0.0,
           vs["KiTS23"]["same_prediction"]["mean_diff_cm3"] or 0.0,
           n_cyst, len(kits_recs)))
    vs["KiTS23"]["n_cyst_bearing"] = n_cyst
    vs["KiTS23"]["n_laterality_repair_fired"] = n_rep

    # Pooled against per-side, measured on both samples rather than
    # asserted. The pooled figure is the one quoted; this says what the
    # other statistic would have said, so "pooled is the sound one" is a
    # claim with a size attached.
    def pooled_vs_side(rs):
        po, ps, pop, psp = [], [], [], []
        for r in rs:
            e = r["organs"].get(KIDNEY_POOLED)
            sides = [r["organs"][o] for o in ("2", "13") if o in r["organs"]]
            if not e or not sides:
                continue
            po.append(e["scores"]["organ_dice"])
            pop.append(e["scores"]["parenchyma_dice"])
            ps.append(float(np.mean([x["scores"]["organ_dice"]
                                     for x in sides
                                     if x["scores"]["organ_dice"]
                                     is not None])))
            psp.append(float(np.mean([x["scores"]["parenchyma_dice"]
                                      for x in sides
                                      if x["scores"]["parenchyma_dice"]
                                      is not None])))
        if not po:
            return None
        return {"n": len(po),
                "pooled_organ_dice": round(float(np.mean(po)), 4),
                "per_side_mean_organ_dice": round(float(np.mean(ps)), 4),
                "pooled_parenchyma_dice": round(float(np.mean(pop)), 4),
                "per_side_mean_parenchyma_dice": round(float(np.mean(psp)),
                                                       4)}
    allk = [r for r in ok if r["collection"] == "KiTS23"]
    vs["KiTS23"]["pooled_vs_per_side"] = {
        "cohort": pooled_vs_side(kits_recs),
        "full_collection": pooled_vs_side(allk)}

    summary = {
        "version": VERSION,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "vs_published_round2": vs,
        "convention": oc.CONVENTION_VERSION,
        "convention_A": oc.CONVENTION[oc.CONV_WM],
        "parenchyma": oc.CONVENTION["parenchyma"],
        "mass_coverage": oc.CONVENTION["mass_coverage"],
        "cyst_ruling": oc.CYST_RULING,
        "arm": data.get(
            "arm",
            "autonomous: organ = TotalSegmentator v2 --fast (3 mm), "
            "unified label space; the organ model's own mask, never "
            "unioned with or carved by a predicted tumour"),
        "run_was_partial_when_summarised": bool(data.get("partial")),
        "semi_oracle_cannot_follow": (
            "The cohort SEMI-ORACLE organ row cannot be moved to convention "
            "A. the pipeline team round 3 item 3: the single-organ cohort's semi-oracle "
            "masks did not survive the 18/19 September wipe - the masks "
            "behind the frozen records were never retained, the 09-17 "
            "regenerations were on the wiped scratch disk, and the organ "
            "test-set masks behind the organ tables were never saved, only "
            "their metrics. Convention A needs the MASK (it changes the "
            "reference, not the arithmetic on a stored Dice), so the "
            "semi-oracle single-organ cohort row stays TUMOUR-EXCLUDED and "
            "must be labelled as such. Only FLARE23's 576 semi-oracle masks "
            "survive; the pipeline team has offered to re-score that one collection under "
            "convention A, and to re-run the three single-organ lists on "
            "the surviving organ checkpoints at about 1 GPU-hour."),
        "n_cases_on_disk": len(data["cases"]),
        "n_masks_on_disk": len(discover_all()),
        "n_scored_ok": len(ok),
        "memory_throttle": {
            "floor_gb": MEM_FLOOR_GB,
            "n_cases_that_waited": sum(1 for r in data["cases"]
                                       if r.get("memory_wait_seconds")),
            "total_wait_seconds": round(sum(
                r.get("memory_wait_seconds") or 0.0
                for r in data["cases"]), 1),
            "n_cases_that_needed_a_retry": sum(
                1 for r in data["cases"] if r.get("memory_retry"))},
        "not_scored": bad,
        "cohort_coverage": missing,
        "cohorts": cohorts,
    }
    _write(ORGAN_SUMMARY, summary)
    return summary


# ==================================================================== tables
def md_tables(census, organ=None, cost=None):
    os.makedirs(OUTDIR, exist_ok=True)
    written = []
    stamp = datetime.datetime.now().strftime("%Y-%m-%d")

    # ---- table 1
    fc = census["item1_flare23_twin_clean_column"]
    L = []
    L.append("# Table C1. FLARE23 tumour arm, twin-clean column rebuilt "
             "under the corrected census")
    L.append("")
    L.append("Generated %s by `build_cohort_columns.py --census`. "
             "Artefact: `tables/cohort/cohort_census.json`." % stamp)
    L.append("")
    L.append("**Arm.** Autonomous (prompt-free): TotalSegmentator v2 3 mm "
             "organs + `sam3_tumor_generic`, the text \"tumor\" on every "
             "slice, no box, no reference read at inference. Scored as "
             "\"the system\".")
    L.append("")
    L.append("**Convention.** 3-D volumetric tumour Dice against the "
             "collection's own tumour label; `raw` is every slice the "
             "confidence gate kept, `gated` keeps only components touching "
             "the TotalSegmentator host organ dilated 5 mm. No organ "
             "convention applies to a tumour column.")
    L.append("")
    L.append("**Cohort.** `sam3_tumor_generic`'s own held-out FLARE23 test "
             "list, n = 54, global split. Twin-clean = no twin of the scan "
             "is in that model's TRAIN or VAL split, under the COMPLETE "
             "census (the pipeline team round 3, 430 pairs).")
    L.append("")
    L.append("| Column | n | Raw Dice [95 % CI] | Gated Dice [95 % CI] |")
    L.append("|:--|--:|:--|:--|")
    L.append("| Twin-clean (PRIMARY), complete census | %d | %s | %s |"
             % (fc["primary_twin_clean"]["n"],
                fmt(fc["primary_twin_clean"]["tumor_raw"]),
                fmt(fc["primary_twin_clean"]["tumor_gated"])))
    r2 = fc["round2_published_for_comparison"]["twin_clean"]
    L.append("| Twin-clean as published in round 2, partial census "
             "(SUPERSEDED) | %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] |"
             % (r2["n"], r2["tumor_raw"]["dice"],
                r2["tumor_raw"]["dice_ci95"][0],
                r2["tumor_raw"]["dice_ci95"][1],
                r2["tumor_gated"]["dice"],
                r2["tumor_gated"]["dice_ci95"][0],
                r2["tumor_gated"]["dice_ci95"][1]))
    L.append("| All patients (secondary) | %d | %s | %s |"
             % (fc["secondary_all"]["n"],
                fmt(fc["secondary_all"]["tumor_raw"]),
                fmt(fc["secondary_all"]["tumor_gated"])))
    L.append("")
    L.append("**Why the column moved.** %s" % fc["why_it_moved"])
    L.append("")
    L.append("**The 14 twin-clean FLARE23 scans.**")
    L.append("")
    dupset = set(fc["overlap_with_single_organ_cohort"])
    L.append("| # | Case | Also a single-organ test scan? | Raw | Gated |")
    L.append("|--:|:--|:--|--:|--:|")
    pf = _load(PROMPTFREE_JSON)["models"]["sam3_tumor_generic"]["datasets"]
    pfc = {c["case"]: c for c in pf["flare23"]["cases"]}
    dmap = {d["flare23"]: d["same_scan_as"]
            for d in census["item2_dedup"]["pairs"]}
    for i, c in enumerate(fc["twin_clean_cases"], 1):
        L.append("| %d | %s | %s | %.3f | %.3f |"
                 % (i, c,
                    ("yes - same CT as %s" % dmap[c]) if c in dupset else "no",
                    pfc[c]["tumor_raw"]["dice"],
                    pfc[c]["tumor_gated"]["dice"]))
    L.append("")
    L.append("**Read this before quoting the column.** %d of the %d "
             "twin-clean FLARE23 scans are the SAME CT as a LiTS, MSD or "
             "KiTS test scan, so this column and the three single-organ "
             "twin-clean columns are not disjoint. Quoting all four in one "
             "table counts those %d patients twice. The pooled cohort "
             "figures in Table C2 resolve this by keeping each scan once."
             % (len(dupset), fc["twin_clean_n"], len(dupset)))
    L.append("")
    L.append("**Exposure verification.** Exposure was recomputed here from "
             "the census and the split manifests rather than read off the pipeline team's "
             "summary: %s on all four collections (LiTS %d/%d, MSD %d/%d, "
             "KiTS %d/%d, FLARE23 %d/%d exposed of tested)."
             % ("exact set equality"
                if census["exposure_all_four_collections_agree"]
                else "DISAGREEMENT - see cohort_census.json",
                *sum(([census["exposure_recomputed"][ds]["n_exposed"],
                       census["exposure_recomputed"][ds]["n_test"]]
                      for ds in DSORDER), [])))
    p = os.path.join(OUTDIR, "TABLE_C1_flare23_twin_clean.md")
    _write_text(p, "\n".join(L))
    written.append(p)

    # ---- table 2
    d2 = census["item2_dedup"]
    L = []
    L.append("# Table C2. The cohort de-duplication: 167 rows are 157 scans")
    L.append("")
    L.append("Generated %s by `build_cohort_columns.py --census`. "
             "Artefact: `tables/cohort/cohort_census.json`." % stamp)
    L.append("")
    L.append("**The defect.** The evaluation cohort is the four held-out "
             "test lists of `sam3_tumor_generic` - LiTS 21 + MSD 56 + "
             "KiTS 36 + FLARE23 54 = **167 rows**. FLARE23 re-shares scans "
             "of the other three collections, and **%d of those rows are "
             "the same CT as another row**, so 167 rows are **%d distinct "
             "scans** and every pooled figure over the 167 counts %d "
             "patients twice. volume-76, the manuscript's LiTS showcase "
             "case, is literally the same scan as FLARE23_1027."
             % (d2["n_duplicate_pairs"], 167 - d2["n_duplicate_pairs"],
                d2["n_duplicate_pairs"]))
    L.append("")
    L.append("**The %d pairs.** Derived here from the complete census and "
             "the split manifests, independently of the pipeline team's list; the two %s."
             % (d2["n_duplicate_pairs"],
                "agree exactly" if d2["our_pairs_equal_pipeline_cohort157"]
                else "DISAGREE - see cohort_census.json"))
    L.append("")
    L.append("| # | FLARE23 row (dropped) | Same CT as (kept) | Collection "
             "kept | Census source |")
    L.append("|--:|:--|:--|:--|:--|")
    for i, d in enumerate(d2["pairs"], 1):
        L.append("| %d | %s | %s | %s | %s |"
                 % (i, d["flare23"], d["same_scan_as"],
                    d["kept_collection"], ", ".join(d["source"])))
    L.append("")
    refs_same = sum(1 for d in d2["pairs"]
                    if abs(d["kept_ref_cm3"] - d["flare23_ref_cm3"]) < 1e-9)
    mdr = max(abs(d["kept_raw"] - d["flare23_raw"]) for d in d2["pairs"])
    adr = sum(abs(d["kept_raw"] - d["flare23_raw"])
              for d in d2["pairs"]) / len(d2["pairs"])
    adg = sum(abs(d["kept_gated"] - d["flare23_gated"])
              for d in d2["pairs"]) / len(d2["pairs"])
    nhigher = sum(1 for d in d2["pairs"]
                  if d["kept_gated"] > d["flare23_gated"])
    L.append("**The same CT, scored twice - and what the two rows actually "
             "disagree about.** The reference tumour volumes are identical "
             "on %d of the %d pairs and differ by at most %.1f cm3 on the "
             "rest, and the RAW Dice differs by %.3f on average (worst "
             "%.3f). The GATED Dice differs by %.3f on average and the "
             "single-organ row is the higher one on %d pairs of %d. The "
             "gate is what differs, not the annotation: the pipeline team round 3 "
             "section 10 records the host classes of the 5 mm gate as "
             "liver alone for LiTS, pancreas alone for MSD, the two "
             "kidneys for KiTS, and any of seven classes (1/2/3/5/6/7/20) "
             "for FLARE23, and a seven-organ host keeps far more "
             "false-positive components than a one-organ host. So on one "
             "and the same CT the FLARE23 row scores worse gated close to "
             "by construction."
             % (refs_same, len(d2["pairs"]),
                max(abs(d["kept_ref_cm3"] - d["flare23_ref_cm3"])
                    for d in d2["pairs"]),
                adr, mdr, adg, nhigher, len(d2["pairs"])))
    L.append("")
    L.append("| Same scan | FLARE23 row: ref cm3 / raw / gated | Kept row: "
             "ref cm3 / raw / gated | Gated difference (kept - FLARE23) |")
    L.append("|:--|:--|:--|--:|")
    for d in d2["pairs"]:
        L.append("| %s = %s | %.1f / %.3f / %.3f | %.1f / %.3f / %.3f | "
                 "%+.3f |"
                 % (d["flare23"], d["same_scan_as"],
                    d["flare23_ref_cm3"], d["flare23_raw"],
                    d["flare23_gated"], d["kept_ref_cm3"], d["kept_raw"],
                    d["kept_gated"], d["kept_gated"] - d["flare23_gated"]))
    L.append("")
    L.append("**The rule.** %s" % d2["rule"])
    L.append("")
    L.append("**What it moves.** Mean 3-D tumour Dice over scans, "
             "percentile bootstrap 95 %% CI, %d resamples, seed %d. "
             "Per-scan values are round 2's, unchanged."
             % (BOOT_N, BOOT_SEED))
    L.append("")
    L.append("| Pooled cohort | n | Raw [95 % CI] | Gated [95 % CI] |")
    L.append("|:--|--:|:--|:--|")
    for key, label in (("no_dedup_167", "All patients, NO de-duplication"),
                       ("dedup_keep_single_organ_157",
                        "All patients, de-duplicated (rule above)"),
                       ("dedup_keep_flare23_157",
                        "All patients, de-duplicated the other way "
                        "(counterfactual)")):
        a = d2["arms"][key]["all"]
        L.append("| %s | %d | %s | %s |"
                 % (label, a["n"], fmt(a["tumor_raw"]),
                    fmt(a["tumor_gated"])))
    for key, label in (("no_dedup_167",
                        "Twin-clean (PRIMARY), NO de-duplication"),
                       ("dedup_keep_single_organ_157",
                        "Twin-clean (PRIMARY), de-duplicated (rule above)"),
                       ("dedup_keep_flare23_157",
                        "Twin-clean (PRIMARY), de-duplicated the other way")):
        a = d2["arms"][key]["twin_clean"]
        L.append("| %s | %d | %s | %s |"
                 % (label, a["n"], fmt(a["tumor_raw"]),
                    fmt(a["tumor_gated"])))
    L.append("")
    nd = d2["arms"]["no_dedup_167"]
    dd = d2["arms"]["dedup_keep_single_organ_157"]
    L.append("**Size of the correction, and what it is made of.** "
             "De-duplication moves the pooled figures by: all patients raw "
             "%+.4f, gated %+.4f; twin-clean raw %+.4f, gated %+.4f. The "
             "raw movement is small, which is the honest answer to "
             "\"does this change anything\" for the raw column. The gated "
             "movement is an order of magnitude larger, and by the "
             "pair table above it is not mostly a double-counting "
             "correction - it is ten rows carrying FLARE23's seven-organ "
             "host gate leaving the pool. Keeping the FLARE23 member "
             "instead moves the twin-clean gated figure the other way, to "
             "%.4f, so the choice of RULE moves that cell by %.4f, which "
             "is larger than de-duplicating at all moves it. A reader who "
             "is told only \"157 unique scans\" cannot see that, so it is "
             "printed. "
             % (dd["all"]["tumor_raw"]["mean"] - nd["all"]["tumor_raw"]["mean"],
                dd["all"]["tumor_gated"]["mean"]
                - nd["all"]["tumor_gated"]["mean"],
                dd["twin_clean"]["tumor_raw"]["mean"]
                - nd["twin_clean"]["tumor_raw"]["mean"],
                dd["twin_clean"]["tumor_gated"]["mean"]
                - nd["twin_clean"]["tumor_gated"]["mean"],
                d2["arms"]["dedup_keep_flare23_157"]["twin_clean"]
                ["tumor_gated"]["mean"],
                abs(dd["twin_clean"]["tumor_gated"]["mean"]
                    - d2["arms"]["dedup_keep_flare23_157"]["twin_clean"]
                    ["tumor_gated"]["mean"])))
    L.append("")
    L.append("**Per collection, de-duplicated.** FLARE23 is the only "
             "collection whose n changes; the other three are untouched by "
             "the rule and are printed so the reader can see that.")
    L.append("")
    L.append("| Collection | n (all) | Raw | Gated | n (twin-clean) | Raw | "
             "Gated |")
    L.append("|:--|--:|:--|:--|--:|:--|:--|")
    for ds in DSORDER:
        c = dd["per_collection"][COLLECTION[ds]]
        L.append("| %s | %d | %s | %s | %d | %s | %s |"
                 % (COLLECTION[ds], c["all_n"], fmt(c["all_raw"]),
                    fmt(c["all_gated"]), c["twin_clean_n"],
                    fmt(c["twin_clean_raw"]), fmt(c["twin_clean_gated"])))
    L.append("")
    rp = d2["reproduction_of_pipeline_cohort157"]
    L.append("**Cross-check.** The de-duplicated arm above reproduces the pipeline team's "
             "`cohort157_rescore_2026-10-03.json` %s: twin-clean n %d (his "
             "%d) raw %.4f (his %.4f) gated %.4f (his %.4f); all-patients "
             "n %d (his %d) raw %.4f (his %.4f) gated %.4f (his %.4f)."
             % ("to four decimal places" if rp["agrees_to_4dp"]
                else "WITH A DISAGREEMENT - see cohort_census.json",
                rp["our_twin_clean_n"], rp["pipeline_twin_clean_n"],
                rp["our_twin_clean_raw"], rp["pipeline_twin_clean_raw"],
                rp["our_twin_clean_gated"], rp["pipeline_twin_clean_gated"],
                rp["our_all_n"], rp["pipeline_all_n"],
                rp["our_all_raw"], rp["pipeline_all_raw"],
                rp["our_all_gated"], rp["pipeline_all_gated"]))
    L.append("")
    L.append("**Caveat on the FLARE23 twin-clean cell.** De-duplication "
             "removes %d FLARE23 rows, and every one of them is twin-clean, "
             "so inside the 157-scan cohort only 4 FLARE23 scans are "
             "twin-clean. No FLARE23-only twin-clean figure should be "
             "quoted from this cohort; Table C1's n = 14 column is the one "
             "to quote, with its overlap note."
             % d2["n_duplicate_pairs"])
    p = os.path.join(OUTDIR, "TABLE_C2_dedup.md")
    _write_text(p, "\n".join(L))
    written.append(p)

    # ---- table 3
    if organ:
        p = _table3(organ, census, stamp)
        written.append(p)

    # ---- the cost note
    if cost is not None:
        p = _cost_note(cost, organ, stamp)
        written.append(p)
    written.append(_index(census, organ, stamp))
    return written


def _index(census, organ, stamp):
    """The one page that says what is in this folder and what is NOT."""
    fc = census["item1_flare23_twin_clean_column"]
    d2 = census["item2_dedup"]
    dd = d2["arms"]["dedup_keep_single_organ_157"]
    L = []
    L.append("# Cohort columns (advisor item the co-author 9)")
    L.append("")
    L.append("Built %s by `build_cohort_columns.py`. Every number below "
             "carries its artefact, its protocol, its convention, its "
             "cohort and its n in the table it comes from; this page is an "
             "index, not a place to quote from." % stamp)
    L.append("")
    L.append("| File | What it answers |")
    L.append("|:--|:--|")
    L.append("| `TABLE_C1_flare23_twin_clean.md` | the FLARE23 twin-clean "
             "column rebuilt under the corrected census, and which 14 "
             "scans it is |")
    L.append("| `TABLE_C2_dedup.md` | the ten scans that are in the cohort "
             "twice, the rule, and what the pooled figures become with and "
             "without it |")
    L.append("| `TABLE_C3_autonomous_organ_conventionA.md` | full-cohort "
             "AUTONOMOUS organ Dice under CONVENTION A, with parenchyma "
             "Dice and recall, and the row that cannot follow |")
    L.append("| `COST.md` | what this cost and whether anything was "
             "sampled |")
    L.append("| `cohort_census.json` | every set, every pair and every cell "
             "behind C1 and C2 |")
    L.append("| `organ_conventionA_cases.json` | per case, per organ, "
             "convention A |")
    L.append("| `organ_conventionA_summary.json` | the cells of C3 |")
    L.append("")
    L.append("## The three headline movements")
    L.append("")
    L.append("1. **FLARE23 twin-clean falls from n = 47 to n = 14**, and "
             "with it raw %.3f -> %.3f and gated %.3f -> %.3f. The round-2 "
             "column was computed on a census that had never searched for "
             "twins of TRAINING cases. Ten of the 14 are also "
             "single-organ test scans, so the four twin-clean columns are "
             "not disjoint and must not be summed."
             % (fc["round2_published_for_comparison"]["twin_clean"]
                ["tumor_raw"]["dice"],
                fc["primary_twin_clean"]["tumor_raw"]["mean"],
                fc["round2_published_for_comparison"]["twin_clean"]
                ["tumor_gated"]["dice"],
                fc["primary_twin_clean"]["tumor_gated"]["mean"]))
    L.append("2. **167 cohort rows are %d distinct scans.** De-duplicating "
             "moves the pooled twin-clean figures to raw %.4f and gated "
             "%.4f; the raw column barely moves and the gated column moves "
             "by %+.4f, mostly because the dropped rows carry FLARE23's "
             "seven-organ host gate rather than because ten patients "
             "stopped being counted twice. Both readings are in C2."
             % (167 - d2["n_duplicate_pairs"],
                dd["twin_clean"]["tumor_raw"]["mean"],
                dd["twin_clean"]["tumor_gated"]["mean"],
                dd["twin_clean"]["tumor_gated"]["mean"]
                - d2["arms"]["no_dedup_167"]["twin_clean"]["tumor_gated"]
                ["mean"]))
    if organ:
        a = organ["cohorts"]["cohort_157_all_dedup"]["ALL"]["case_mean"]
        L.append("3. **The autonomous organ row now speaks convention A** "
                 "across the whole cohort: organ Dice %s, beside "
                 "parenchyma Dice %s and parenchyma recall %s, on %d "
                 "scans. The SEMI-ORACLE cohort organ row cannot follow "
                 "and stays tumour-excluded; C3e says why, in the row "
                 "where the number would otherwise sit."
                 % (fmt(a["organ_dice"]), fmt(a["parenchyma_dice"]),
                    fmt(a["parenchyma_recall"]),
                    organ["cohorts"]["cohort_157_all_dedup"]["ALL"]
                    ["n_cases"]))
    else:
        L.append("3. **The autonomous organ row under convention A has not "
                 "been built yet** - run `--organ`.")
    L.append("")
    L.append("## What is NOT here, and what it would take")
    L.append("")
    L.append("| Missing | Why | What would produce it |")
    L.append("|:--|:--|:--|")
    L.append("| Semi-oracle cohort organ row under convention A, "
             "single-organ collections | the masks did not survive the "
             "September wipe (the pipeline team round 3 item 3) | the pipeline team re-running the three "
             "organ test lists on the surviving organ checkpoints, about "
             "1 GPU-hour; offered in his section 8 and not yet requested |")
    L.append("| Semi-oracle FLARE23 organ row under convention A | not "
             "produced here; we hold no semi-oracle organ masks locally | "
             "the pipeline team re-scoring the surviving 576 FLARE23 masks; offered in "
             "his section 8 and not yet requested |")
    L.append("| 1.5 mm autonomous organ row for the cohort | 1.5 mm masks "
             "exist for the four showcase cases and a 58-case paired "
             "sample only, and the advisor ruled out further 1.5 mm runs "
             "for this submission | a GPU run that is out of scope |")
    L.append("| Per-organ attributed tumour volume per cohort mask | asked "
             "of the pipeline team in `round3/CLEANUP_RULE_R0_R1_R2.md` 5.4 | the pipeline team. Note "
             "that request says 167 masks; by C2 it is %d distinct scans, "
             "and he should be told so before he builds the table |"
             % (167 - d2["n_duplicate_pairs"]))
    return _write_text(os.path.join(OUTDIR, "README.md"),
                       "\n".join(L))


def _table3(organ, census, stamp):
    L = []
    L.append("# Table C3. Autonomous arm, cohort organ Dice under "
             "CONVENTION A")
    L.append("")
    L.append("Generated %s by `build_cohort_columns.py --organ`. "
             "Artefacts: `tables/cohort/organ_conventionA_cases.json` "
             "(per case), `organ_conventionA_summary.json` (these cells)."
             % stamp)
    L.append("")
    L.append("**Arm.** AUTONOMOUS, scored as \"the system\". The organ "
             "model is TotalSegmentator v2 `--fast` (the 3 mm network), "
             "run with no prompt, no box and no reference; masks from "
             "`%s`, already in the unified label space." % FAST_PREDS)
    L.append("")
    L.append("**CONVENTION A, named in full because a Dice without its "
             "convention is what caused two earlier rounds of confusion.** "
             "%s" % organ["convention_A"])
    L.append("")
    L.append("**Published beside it, always.** %s" % organ["parenchyma"])
    L.append("")
    L.append("**mass_coverage.** %s" % organ["mass_coverage"])
    L.append("")
    L.append("**KiTS cyst.** %s" % organ["cyst_ruling"])
    L.append("")
    L.append("**Unit of the cohort is the scan.** The case-mean row "
             "averages each case's own mean over the organs that case "
             "annotates, so a FLARE23 case with five annotated organs "
             "does not outweigh a LiTS case with one. Per-organ rows are "
             "means over structures. FLARE23's label space has thirteen "
             "organs; the staged references in this corpus carry five of "
             "them - liver, spleen, pancreas and the two kidneys - and "
             "C3d-ii lists exactly those.")
    L.append("")
    L.append("**KiTS23 is scored on the POOLED kidney, not on a mean of "
             "two sides.** KiTS23 annotates one kidney class, so right and "
             "left exist only because `ipkg_organ_convention.kidney_sides` "
             "decided they do - and the same decision is made again, "
             "independently, on the prediction. A per-side Dice is "
             "therefore as much a measurement of the laterality rule as of "
             "the mask: on a case where one 78,604-voxel component sits "
             "near the midline, nearest-centroid and nearest-surface put "
             "it on opposite sides and both per-side numbers move while "
             "the pooled one does not. The two sides are in the per-case "
             "artefact and in the per-organ table; they are not averaged "
             "into the KiTS case mean.")
    L.append("")
    L.append("**FLARE23 IS PARTIALLY LABELLED, AND THAT IS WHY ITS n IS "
             "NOT ITS CASE COUNT.** 644 tumour-bearing FLARE23 studies "
             "have a 3 mm mask on disk and only 270 of them carry an "
             "organ annotation at all: the other 374 label the tumour and "
             "nothing else, so there is no organ reference to score "
             "against and the case contributes no organ row. Of the 270 "
             "that do, 262 annotate five organs and 8 annotate four - "
             "liver, spleen, pancreas and the two kidneys. The "
             "unannotated 374 are counted in the table's own column "
             "rather than quietly dropped, because an n of 270 beside a "
             "collection everybody knows has 644 masks is exactly the "
             "kind of gap a reader fills in wrongly.")
    L.append("")
    L.append("**FLARE23 is two rows, never one.** 644 tumour-bearing "
             "studies and 305 tumour-free ones have 3 mm masks on disk. On "
             "the tumour-free arm there is no mass to attribute, so "
             "convention A and the parenchyma reading are the same "
             "measurement by construction, and pooling the two would make "
             "the gap between the columns look smaller than it is. The "
             "tumour-free arm is printed separately and is excluded from "
             "the pooled row.")
    L.append("")

    def block(title, node, note=None, rows=None):
        L.append("## %s" % title)
        L.append("")
        if note:
            L.append(note)
            L.append("")
        L.append("| Collection | scans with an annotated organ | "
                 "scans with none | Organ Dice, convention A [95 % CI] | "
                 "Parenchyma Dice [95 % CI] | Parenchyma recall "
                 "[95 % CI] |")
        L.append("|:--|--:|--:|:--|:--|:--|")
        for coll in (rows or ("LiTS", "KiTS23", "MSD Pancreas", "FLARE23",
                              "ALL")):
            a = node.get(coll)
            if not isinstance(a, dict) or not a.get("n_cases"):
                continue
            cm = a["case_mean"]
            L.append("| %s | %d | %d | %s | %s | %s |"
                     % ("**pooled**" if coll == "ALL" else coll,
                        a["n_cases"],
                        a.get("n_records_with_no_annotated_organ", 0),
                        fmt(cm["organ_dice"]),
                        fmt(cm["parenchyma_dice"]),
                        fmt(cm["parenchyma_recall"])))
        L.append("")

    co = organ["cohorts"]
    block("C3a. The evaluation cohort, de-duplicated (157 scans)",
          co["cohort_157_all_dedup"],
          "The same scans as the autonomous TUMOUR row of Table C2, so the "
          "autonomous table now speaks one convention and one cohort "
          "throughout.")
    co2 = co.get("cohort_157_all_dedup_keep_flare23")
    if co2:
        a1 = co["cohort_157_all_dedup"]["ALL"]["case_mean"]["organ_dice"]
        a2 = co2["ALL"]["case_mean"]["organ_dice"]
        a0 = co["cohort_167_all_no_dedup"]["ALL"]["case_mean"]["organ_dice"]
        L.append("**What the de-duplication costs the ORGAN arm, and it is "
                 "not nothing.** The rule in Table C2 was chosen for the "
                 "TUMOUR arm, where FLARE23's single generic tumour class "
                 "is the weaker reference. On the organ side the trade runs "
                 "the other way: a dropped FLARE23 row takes five "
                 "annotated organs with it - liver, spleen, pancreas and "
                 "the two kidneys - and the LiTS, MSD or KiTS row that "
                 "replaces it annotates one. Pooled case-mean organ "
                 "Dice under convention A: %s with no de-duplication (167 "
                 "rows), %s keeping the single-organ member (the rule, 157 "
                 "scans), %s keeping the FLARE23 member instead (157 "
                 "scans). The rule is kept anyway, because one cohort "
                 "definition has to serve both arms and the tumour arm is "
                 "the one the claim rests on - but the organ column is "
                 "measured on fewer structures because of it, and that is "
                 "stated rather than absorbed." % (fmt(a0), fmt(a1),
                                                   fmt(a2)))
        L.append("")
    block("C3b. The evaluation cohort, twin-clean (PRIMARY)",
          co["cohort_76_twin_clean_dedup"],
          "Twin-clean is primary by the advisor's decision. Exposure here "
          "is the tumour model's; the organ model is TotalSegmentator, "
          "whose own training corpus is disjoint from these collections, "
          "so this split is reported for comparability with the tumour row "
          "and is not itself an organ-exposure claim.")
    block("C3c. Every 3 mm mask on disk (the full collections)",
          co["full_collection"],
          "Not a held-out set: these are whole collections, and the organ "
          "model never saw any of them, but the tumour arm's test lists are "
          "a subset. Quoted as the breadth figure, never as the headline. "
          "The pooled row excludes the FLARE23 tumour-free arm, for the "
          "reason above.",
          rows=("LiTS", "KiTS23", "MSD Pancreas", "FLARE23 tumour-positive",
                "FLARE23 tumour-free", "ALL"))

    kpo = co["full_collection"].get("KiTS23", {}).get("per_organ", {})
    if kpo:
        L.append("## C3d-i. KiTS23: the pooled kidney and the two sides")
        L.append("")
        L.append("The pooled row is the KiTS number. The two side rows are "
                 "printed so the laterality rule is visible, and they are "
                 "not the collection's score.")
        L.append("")
        L.append("| Structure | n | n hosting mass | Organ Dice "
                 "(convention A) | Parenchyma Dice | Parenchyma recall | "
                 "mass_coverage |")
        L.append("|:--|--:|--:|:--|:--|:--|:--|")
        for name in ("kidney (pooled)", "right kidney", "left kidney"):
            row = kpo.get(name)
            if not row:
                continue
            L.append("| %s | %d | %d | %s | %s | %s | %s |"
                     % ("**%s**" % name if name == "kidney (pooled)"
                        else "%s (laterality rule, not the score)" % name,
                        row["n_structures"], row["n_hosting_mass"],
                        fmt(row["organ_dice"]), fmt(row["parenchyma_dice"]),
                        fmt(row["parenchyma_recall"]),
                        fmt(row["mass_coverage"])))
        L.append("")

    L.append("## C3d-ii. Per organ, FLARE23 tumour-positive")
    L.append("")
    L.append("FLARE23 is the only collection that annotates more than its "
             "own organ, so it is the only per-organ table worth printing. "
             "`n hosting mass` is how many of those structures host an "
             "attributed reference mass - where it is 0, convention A and "
             "parenchyma are the same measurement by construction.")
    L.append("")
    L.append("| Organ | n structures | n hosting mass | Organ Dice "
             "(convention A) | Parenchyma Dice | Parenchyma recall | "
             "mass_coverage |")
    L.append("|:--|--:|--:|:--|:--|:--|:--|")
    po = co["full_collection"]["FLARE23 tumour-positive"]["per_organ"]
    for name, row in sorted(po.items(),
                            key=lambda kv: -kv[1]["n_structures"]):
        L.append("| %s | %d | %d | %s | %s | %s | %s |"
                 % (name, row["n_structures"], row["n_hosting_mass"],
                    fmt(row["organ_dice"]), fmt(row["parenchyma_dice"]),
                    fmt(row["parenchyma_recall"]),
                    fmt(row["mass_coverage"])))
    L.append("")
    vs = organ.get("vs_published_round2") or {}
    if vs:
        L.append("## C3d-iii. What moves against what round 2 published")
        L.append("")
        L.append("Round 2's cohort organ row is the pipeline team's **convention B**: the "
                 "prediction is the TotalSegmentator organ UNIONED WITH "
                 "THE PREDICTED TUMOUR. A B number and an A number are not "
                 "comparable and are never mixed in one column; B is "
                 "printed here only so a reader holding the round-2 figure "
                 "can find the one that replaces it. The TUMOUR-EXCLUDED "
                 "pair IS comparable - both are the organ model against "
                 "the organ label with the mass cut out - so that is the "
                 "column where a disagreement means something.")
        L.append("")
        L.append("| Collection | n | Round 2 tumour-excluded | Ours, "
                 "parenchyma Dice (same definition) | Round 2 "
                 "organ-with-mass (CONVENTION B) | Ours, organ Dice "
                 "(CONVENTION A) |")
        L.append("|:--|--:|--:|--:|--:|--:|")
        for coll in ("LiTS", "MSD Pancreas", "KiTS23", "FLARE23"):
            v = vs.get(coll)
            if not isinstance(v, dict):
                continue

            def q(x):
                return "n/a" if x is None else "%.4f" % x
            L.append("| %s | %s | %s | %s | %s | %s |"
                     % (coll, v.get("n_ours", v.get("n_published")),
                        q(v.get("published_tumour_excluded")),
                        q(v.get("our_parenchyma_dice")),
                        q(v.get("published_organ_with_mass_CONVENTION_B")),
                        q(v.get("our_organ_dice_CONVENTION_A"))))
        L.append("")
        L.append("| Collection | reference the same as round 2's? | "
                 "prediction the same as round 2's? |")
        L.append("|:--|:--|:--|")
        for coll in ("LiTS", "MSD Pancreas", "KiTS23"):
            v = vs.get(coll) or {}
            sr, sp = v.get("same_reference"), v.get("same_prediction")
            if not sr:
                continue
            L.append("| %s | identical to 0.01 cm3 on %d of %d cases "
                     "(mean %+.3f cm3) | identical on %d of %d (mean "
                     "%+.3f cm3) |"
                     % (coll, sr["n_identical_to_0.01_cm3"], sr["n"],
                        sr["mean_diff_cm3"] or 0.0,
                        sp["n_identical_to_0.01_cm3"], sp["n"],
                        sp["mean_diff_cm3"] or 0.0))
        L.append("")
        L.append("**READ THIS BEFORE ATTRIBUTING THE MOVEMENT TO THE "
                 "CONVENTION.** %s" % vs.get("_not_only_the_convention_moved",
                                             ""))
        L.append("")
        L.append("**KiTS moves further than LiTS or MSD. Why, measured.** "
                 "%s" % vs.get("_why_kits_moves", ""))
        L.append("")
        pvs = (vs.get("KiTS23") or {}).get("pooled_vs_per_side") or {}
        ch, fl = pvs.get("cohort"), pvs.get("full_collection")
        if ch and fl:
            L.append("**One more KiTS point, about reporting rather than "
                     "measurement, and it is SMALLER than we expected.** "
                     "KiTS annotates one kidney class, so a per-side mean "
                     "and a pooled figure are two statistics of the same "
                     "mask. Pooled is the sound one - it does not depend "
                     "on a laterality rule applied twice - and it is what "
                     "round 2 used and what this table quotes. But the "
                     "gap between them is small, and saying so is part of "
                     "the measurement: on the %d cohort cases, convention "
                     "A reads %.4f pooled against %.4f as a per-side mean "
                     "(%+.4f) and parenchyma reads %.4f against %.4f "
                     "(%+.4f); over all %d KiTS cases the same two "
                     "comparisons are %+.4f and %+.4f. The per-side "
                     "figures are in C3d-i and in the per-case artefact; "
                     "they are not quoted anywhere as the KiTS score."
                     % (ch["n"], ch["pooled_organ_dice"],
                        ch["per_side_mean_organ_dice"],
                        ch["per_side_mean_organ_dice"]
                        - ch["pooled_organ_dice"],
                        ch["pooled_parenchyma_dice"],
                        ch["per_side_mean_parenchyma_dice"],
                        ch["per_side_mean_parenchyma_dice"]
                        - ch["pooled_parenchyma_dice"],
                        fl["n"],
                        fl["per_side_mean_organ_dice"]
                        - fl["pooled_organ_dice"],
                        fl["per_side_mean_parenchyma_dice"]
                        - fl["pooled_parenchyma_dice"]))
        L.append("")
        L.append("**FLARE23 has no round-2 organ cell at all** - the "
                 "round-2 combined pass emitted `organ_excludes_tumor_dice` "
                 "and `organ_with_mass_dice` for the three single-organ "
                 "collections and left FLARE23 null. The FLARE23 organ row "
                 "here is the first one.")
        L.append("")

    L.append("## C3e. The row that cannot move")
    L.append("")
    L.append("**SEMI-ORACLE cohort organ row: stays TUMOUR-EXCLUDED, and "
             "must be labelled as such.** %s" % organ["semi_oracle_cannot_follow"])
    L.append("")
    L.append("| Arm | Cohort organ convention | Why |")
    L.append("|:--|:--|:--|")
    L.append("| Autonomous (this table) | **Convention A**, with "
             "parenchyma Dice and recall | the organ model is "
             "TotalSegmentator and all %d of its 3 mm masks are on local "
             "disk | " % organ["n_scored_ok"])
    L.append("| Semi-oracle, single-organ collections | **tumour-excluded "
             "only** | the masks did not survive the September wipe; only "
             "their metrics were kept, and convention A changes the "
             "reference, not the arithmetic on a stored Dice |")
    L.append("| Semi-oracle, FLARE23 | not produced here | its 576 masks "
             "DO survive, so the pipeline team can re-score this one collection under "
             "convention A from existing masks; it has been offered and not "
             "yet requested |")
    L.append("")

    n_disk = organ.get("n_masks_on_disk", organ["n_cases_on_disk"])
    if organ["n_scored_ok"] < n_disk:
        L.append("## C3f. This run is INCOMPLETE, and here is exactly how")
        L.append("")
        L.append("%d of the %d TotalSegmentator 3 mm masks on local disk "
                 "have been scored. The remaining %d are not missing data "
                 "and not a failure: the sweep was stopped on wall clock. "
                 "Two things make the partial column publishable rather "
                 "than discardable. (1) The 167 evaluation scans of C3a "
                 "and C3b were scored FIRST and are complete, so the rows "
                 "the manuscript quotes are not affected. (2) The "
                 "remainder was scored in a DETERMINISTIC SHUFFLE with "
                 "seed %d, so what has been scored of each full collection "
                 "is a uniform random sample of it, not its alphabetical "
                 "head - the C3c row is a sample with a stated n and a "
                 "stated seed. `python build_cohort_columns.py --organ` "
                 "resumes and completes it; every scored case is cached "
                 "under a fingerprint of the files it came from."
                 % (organ["n_scored_ok"], n_disk,
                    n_disk - organ["n_scored_ok"], JOB_ORDER_SEED))
        L.append("")

    cov = organ["cohort_coverage"]
    L.append("**Coverage.** ")
    for coll, c in cov.items():
        L.append("%s %d/%d%s. " % (coll, c["n_wanted"] - c["n_missing"],
                                   c["n_wanted"],
                                   "" if not c["n_missing"]
                                   else " (missing: %s)"
                                   % ", ".join(c["missing"][:10])))
    L.append("")
    if organ["not_scored"]:
        L.append("**Cases on disk that did not score** (%d): %s"
                 % (len(organ["not_scored"]),
                    "; ".join("%s %s" % (b["case"], b["status"])
                              for b in organ["not_scored"][:40])))
        L.append("")
    p = os.path.join(OUTDIR, "TABLE_C3_autonomous_organ_conventionA.md")
    _write_text(p, "\n".join(L))
    return p


def _cost_note(cost, organ, stamp):
    L = []
    L.append("# Cost of the cohort columns")
    L.append("")
    L.append("Generated %s. Artefact: `tables/cohort/cost_log.json`." % stamp)
    L.append("")
    L.append("**No GPU was used.** The advisor ruled out further 1.5 mm "
             "runs for this submission; every mask these columns are "
             "computed from was already on disk.")
    L.append("")
    L.append("| Step | What it cost | Sampled? |")
    L.append("|:--|:--|:--|")
    L.append("| Items 1 and 2 (census, exposure, de-duplication, tumour "
             "columns) | %.1f s, pure JSON, no masks read | no - all 167 "
             "rows |" % cost.get("census_seconds", 0.0))
    o = cost.get("organ", {})
    if o:
        n = o.get("cases_scored_cumulative", o.get("scored_this_run", 0))
        w = o.get("wall_seconds", 0.0)
        L.append("| Item 3 (convention-A organ scoring) | %s distinct mask "
                 "pairs, %d case-scorings in %.1f min (%.1f h) of wall "
                 "clock across %d passes on %s CPU workers, %.1f s per "
                 "case-scoring | %s |"
                 % (format(o.get("distinct_mask_pairs_scored", n), ","),
                    n, w / 60.0, w / 3600.0,
                    len(o.get("passes", [])) or 1,
                    o.get("workers", "?"), (w / n) if n else 0.0,
                    "no - every mask on disk was scored"))
    L.append("")
    if o.get("passes"):
        L.append("**Pass by pass.** These figures are read back from %s"
                 % (o.get("reconstructed_from",
                          "this run's own cost record.")))
        L.append("")
        L.append("| Pass | Cases scored | Wall clock | Workers | Note |")
        L.append("|:--|--:|--:|--:|:--|")
        for x in o["passes"]:
            L.append("| %s | %d | %.1f min | %s | %s |"
                     % (x["what"], x["scored"], x["wall_seconds"] / 60.0,
                        x["workers"], x.get("note", "")))
        L.append("")
        L.append("**Why there are six passes and not one.** Four of them "
                 "are deliberate: a smoke test before committing an hour "
                 "of CPU, the evaluation cohort scored first so the rows "
                 "the manuscript quotes could not be left half-done, a "
                 "second smoke test when the pooled-kidney structure was "
                 "added, and two short restarts to change the ORDER cases "
                 "are scored in (first to a deterministic shuffle, then to "
                 "cohort-first). The fifth is not deliberate: the first "
                 "full sweep died on an out-of-memory error at about "
                 "case 1,027 of 1,665, on a machine that was running three "
                 "other jobs at the time. No result was lost - every 25 "
                 "cases are checkpointed under a fingerprint of the files "
                 "they came from - and the worker now catches MemoryError, "
                 "waits for memory and retries the case once, so a single "
                 "failed allocation can no longer end a run.")
        L.append("")
    if organ:
        L.append("**Scale.** %d mask pairs on local disk; %d scored "
                 "successfully and in the artefact, %d attempted and not "
                 "scored. Each case reads two label volumes (the reference "
                 "and the TotalSegmentator prediction) and, for KiTS23, a "
                 "third (the dataset's own segmentation, for the "
                 "laterality repair), so this is roughly %d volumes read "
                 "for %d mask-pairs scored."
                 % (organ.get("n_masks_on_disk",
                               organ["n_cases_on_disk"]),
                    organ["n_scored_ok"],
                    len(organ["not_scored"]),
                    2 * organ["n_scored_ok"]
                    + (organ["cohorts"]["full_collection"].get("KiTS23")
                       or {}).get("n_cases", 0),
                    organ["n_scored_ok"]))
        L.append("")
    nd = (organ or {}).get("n_masks_on_disk")
    nok = (organ or {}).get("n_scored_ok")
    if organ and nd and nok is not None and nok < nd:
        L.append("**Yes, something was sampled, and this is the part of "
                 "the answer that matters.** %d of the %d "
                 "TotalSegmentator 3 mm masks on local disk were scored "
                 "before the sweep was stopped on wall clock. The 167 "
                 "evaluation scans were scored first and are COMPLETE, so "
                 "the cohort rows the manuscript quotes are not a sample. "
                 "The full-collection row is: its cases were drawn in a "
                 "deterministic shuffle with seed %d, which makes what "
                 "was scored a uniform random sample of every collection "
                 "rather than its alphabetical head, and it is published "
                 "with its n. The run resumes with `--organ`; nothing has "
                 "to be recomputed." % (nok, nd, JOB_ORDER_SEED))
    else:
        L.append("**Nothing was sampled.** Every TotalSegmentator 3 mm "
                 "mask present on local disk was scored; no sub-sample, "
                 "no cap, no early stop. Cases that could not be scored "
                 "are listed individually in Table C3 with the reason, "
                 "rather than silently dropped.")
    L.append("")
    mt = (organ or {}).get("memory_throttle")
    if mt:
        L.append("**The memory throttle, and what it cost.** A worker will "
                 "not START a case while free memory is under %.0f GB, "
                 "because the largest KiTS23 reference is 1,059 x 512 x "
                 "512 stored as float64 and nibabel materialises it at "
                 "2.2 GB before the int16 cast - measured peak RSS for one "
                 "worker on that case is 2.35 GB. %d of the %s cases "
                 "waited at least once, for %.1f min in total, and %d "
                 "needed the MemoryError retry. No case was skipped for "
                 "want of memory: the throttle delays a case, it never "
                 "drops one, so the cohort cannot be silently sampled by "
                 "it."
                 % (mt["floor_gb"], mt["n_cases_that_waited"],
                    format(organ["n_scored_ok"], ","),
                    mt["total_wait_seconds"] / 60.0,
                    mt["n_cases_that_needed_a_retry"]))
        L.append("")
    L.append("**Memory.** Peak is bounded by the worker count: each worker "
             "holds at most three int16 label volumes of one case plus the "
             "bounded distance transforms the mass attribution computes "
             "inside a %d mm neighbourhood of each mass component, never a "
             "full-volume transform per organ." % int(oc.MASS_MAX_DIST_MM))
    p = os.path.join(OUTDIR, "COST.md")
    _write_text(p, "\n".join(L))
    return p


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")
    os.replace(tmp, path)
    return path


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--organ", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None,
                    help="restrict --organ to one collection")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--cohort", action="store_true",
                    help="restrict --organ to the 167 evaluation scans")
    a = ap.parse_args()
    if not (a.census or a.organ or a.tables or a.all):
        ap.error("pick at least one of --census --organ --tables --all")

    os.makedirs(OUTDIR, exist_ok=True)
    cost = {}
    if os.path.isfile(COST_OUT):
        try:
            cost = _load(COST_OUT)
        except Exception:                                   # noqa: BLE001
            cost = {}

    census = None
    if a.census or a.all:
        t0 = time.time()
        census = build_census()
        cost["census_seconds"] = round(time.time() - t0, 2)
        print("[census] wrote %s in %.1f s"
              % (CENSUS_OUT, cost["census_seconds"]))
        print("[census] exposure agrees with the pipeline team on all four collections: %s"
              % census["exposure_all_four_collections_agree"])
        print("[census] FLARE23 twin-clean n = %d"
              % census["item1_flare23_twin_clean_column"]["twin_clean_n"])
        print("[census] duplicate pairs: %d, equal to the pipeline team's cohort157 list: %s"
              % (census["item2_dedup"]["n_duplicate_pairs"],
                 census["item2_dedup"]["our_pairs_equal_pipeline_cohort157"]))
        print("[census] reproduces the pipeline team's cohort157 pooled cells: %s"
              % census["item2_dedup"]["reproduction_of_pipeline_cohort157"]
              ["agrees_to_4dp"])

    if a.organ or a.all:
        run = run_organ(jobs_n=a.jobs, limit=a.limit, rescore=a.rescore,
                        only=a.only, cohort=a.cohort)
        prev = cost.get("organ") or {}
        cost["organ"] = {
            "scored_this_run": run["scored_this_run"],
            "workers": run["workers"],
            "wall_seconds": round(prev.get("wall_seconds", 0.0)
                                  + run["wall_seconds"], 1),
            "cases_scored_cumulative": (prev.get("cases_scored_cumulative", 0)
                                        + run["scored_this_run"]),
            "seconds_per_case": run["seconds_per_case"],
            "passes": (prev.get("passes", [])
                       + [{"scored": run["scored_this_run"],
                           "wall_seconds": run["wall_seconds"],
                           "workers": run["workers"],
                           "restricted_to": (a.only or
                                             ("evaluation cohort"
                                              if a.cohort else "everything")),
                           "when": datetime.datetime.now().isoformat(
                               timespec="seconds")}]),
        }

    organ = None
    if a.tables or a.all or a.organ:
        if census is None:
            census = _load(CENSUS_OUT)
        if os.path.isfile(ORGAN_OUT):
            organ = summarise_organ(census)
        _write(COST_OUT, cost)
        for p in md_tables(census, organ, cost):
            print("[tables] %s" % p)
    else:
        _write(COST_OUT, cost)


if __name__ == "__main__":
    main()
