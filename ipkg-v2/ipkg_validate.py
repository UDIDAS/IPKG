#!/usr/bin/env python3
"""The validation suite that should have existed before a clinician saw the app.

WHY THIS FILE EXISTS
--------------------
Everything the repository checked before today passed, and everything it
checked was plumbing: that the packaged files resolve, that the retrieval pool
is 1,334, that rank() reproduces the frozen top-10, that every case folder has
its sixteen verdicts. None of it looks at a voxel. So a mask could be measured
against the wrong reference, a printed volume could belong to a different run
of the network, and a pancreatic head could be displayed as a tail with a
SNOMED code attached to it, and the whole suite would stay green. That is
exactly what happened: a radiation oncologist found the defects by looking at
the pictures, which is the most expensive validator we have.

These checks are written against the defects that were actually found, in the
form "what measurement, taken on what artefact, would have made this loud":

  V1  identity and geometry. The mask, the CT and the reference share a shape,
      a spacing and an affine, and the mask on screen is the file we think it
      is - hashed, not assumed from a path.
  V2  predicted-versus-reference agreement per structure, with the convention
      spelled out in the output, plus the two invariants the semi-oracle
      protocol claims: no prediction on a reference-absent slice, and no
      missing prediction on a slice the 50 px gate admitted.
  V3  claim-by-claim. Every number the UI prints for a case is recomputed from
      the artefact the record claims to describe, using the pipeline team's own delivered
      extractor so the convention cannot drift. When a number does not
      reproduce, the check searches the other candidate artefacts and names
      the one that DOES - which is how "this figure belongs to a different
      mask" becomes a sentence instead of a suspicion.
  V4  anatomical sanity in world coordinates. This is the check that would
      have caught head/tail with no per-case work at all: laterality against
      an independent anchor, organ position against a body midline measured
      from the CT, the pancreatic sub-site against a structure our axis never
      touched, and the population-level test - a corpus whose pancreatic
      lesions are 43 per cent tail and 20 per cent head is not a corpus, it is
      a bug.
  V9  scan-quality assertions: label exclusivity and integrality, the lesion
      inside the organ the record says hosts it, and attenuation sanity, which
      is the cheapest way to notice that a mask and a CT have come apart.
  V10 one runner, one report per case (markdown and JSON) under validation/.
  V11 the DISPLAY CONTRACT, added 2026-10-04. V1 to V9 ask whether a number
      is true. V11 asks whether a true number is readable, which is the half
      that went wrong this week: an organ Dice printed without its
      convention is one of three different measurements (the same kidney
      prediction reads 0.4919 under convention A and 0.9701 under B); a pool
      size printed from a cached graph is a denominator nobody can check;
      and a claim that was WITHHELD, rendered as a negative, is a clinical
      assertion nobody made. Three rows, one per failure mode, population
      and artefact level rather than per case.

WHAT THIS FILE IS NOT
---------------------
It is not a fixer. Decision 1 of the 2026-09-29 audit is that corrected values
are SHOWN and never silently substituted, and nothing here writes to a
delivered file, a mask, a record or a sidecar. The suite owns
ipkg_validate.py and the validation/ directory and touches nothing else.

It is also not an accuracy panel for the rater page. Decision 2 is that raters
see no Dice, so every agreement figure below is marked
audience=internal-evaluation. The SEMI-ORACLE DISCLOSURE is method rather than
score and does belong on the rater page; it is carried in the report header for
that use, and it is the one thing here that a rater should see.

A FAIL is a finding, not a defect in the suite. Several of the FAILs on the
first run are conditions the audit already documented, and those carry a
`known` field that quotes where they were written down; a FAIL with no `known`
field is new. A suite that passes everything on its first run is a suite that
is not looking.

    python ipkg_validate.py                 # all four cases, all checks
    python ipkg_validate.py --case case_00067
    python ipkg_validate.py --check V4 --check V9
    python ipkg_validate.py --corpus-only   # the population test alone
    python ipkg_validate.py --selftest      # the suite's own comparators

It exits non-zero when anything FAILs, so it can sit in front of a demo. A
filtered run overwrites the summary with a filtered one, and the summary says
so in its own header.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import nibabel as nib
from scipy import ndimage

BASE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(BASE, "src"), BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ipkg_final_cases as FC                                    # noqa: E402
import build_case_metrics as BCM                                 # noqa: E402
import ipkg_subsite_fix as SF                                    # noqa: E402
from common.label_map import UNIFIED, MAPS                       # noqa: E402

OUT_DIR = os.path.join(BASE, "validation")
LOCK_JSON = os.path.join(OUT_DIR, "mask_fingerprints.json")
REPORT_JSON = os.path.join(OUT_DIR, "validation_report.json")
SUMMARY_MD = os.path.join(OUT_DIR, "SUMMARY.md")
SCHEMA = "ipkg_validation/1"
AUDIT = "IPKG audit 2026-09-29"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"

AUDIENCE_INTERNAL = "internal-evaluation"
AUDIENCE_RATER = "rater-safe"

# The one thing in this file a rater should see. It is quoted rather than
# derived so that the sentence on the rater page and the sentence in the
# validation report cannot drift apart.
SEMI_ORACLE_DISCLOSURE = (
    "Every displayed mask is semi-oracle: SAM 3 was given a bounding box "
    "computed from the ground-truth contour of that structure on that slice "
    "(pad 3 px) and was run only on slices where the reference covers at "
    "least 50 px at 256x256, so there are zero predicted voxels on any "
    "reference-absent slice. The agreement figures in this report are "
    "interactive ceilings, not deployment numbers.")

# ---------------------------------------------------------------------------
# tolerances, each with the reason it is the number it is
# ---------------------------------------------------------------------------
# The records round cm3 to two decimals, so a reproduction can never be closer
# than half of the last printed digit; 0.05 cm3 is that half-digit with room
# for one float32 conversion of the spacing.
VOL_TOL_CM3 = 0.05
# Above a few hundred cm3 the rounding tolerance is tighter than the
# arithmetic, so the volume check also accepts a relative match. 0.2 per cent
# is one part in five hundred - far smaller than any real difference between
# two runs of a network, and far larger than a spacing printed at float32.
VOL_TOL_FRAC = 0.002
# Voxel counts are integers on both sides. Nothing legitimate moves them.
VOXEL_TOL = 0
# the pipeline team publishes Dice to three decimals; 0.002 admits their rounding and nothing
# else. Recomputing with a different convention moves Dice by far more.
DICE_TOL = 0.002
# A twentieth of the smallest voxel edge in the four cases (0.76 mm), which is
# below anything a header can legitimately disagree by once both files are
# written from the same series.
AFFINE_TOL_MM = 0.05
ZOOM_TOL_MM = 1e-4
# The semi-oracle slice gate, in pixels of a 256x256 slice, verbatim from the
# delivered protocol.
SLICE_GATE_PX_AT_256 = 50

# Attenuation bands. These are deliberately wide: the four cases are in
# different contrast phases and a band tight enough to separate portal-venous
# liver from non-contrast liver would fire on the phase rather than on a
# defect. What they DO catch is a mask that has come apart from its CT, which
# puts a structure's median into fat, air or bone.
HU_ORGAN_MEDIAN_BAND = (-20.0, 260.0)
HU_LESION_MEDIAN_BAND = (-40.0, 300.0)
HU_AIR = -200.0          # below this is gas, not parenchyma
HU_BONE = 400.0          # at or above this is bone or metal
HU_AIR_FRAC_MAX = 0.05
HU_BONE_FRAC_MAX = 0.02

# PDAC sub-site epidemiology, as the accepted clinical distribution: the head
# carries roughly 60-70 per cent of pancreatic ductal adenocarcinomas, the
# body and the tail roughly 15 per cent each. The gate below is deliberately
# far looser than that band - it only asks that the head be the modal sub-site
# and that it beat the tail - because the point is not to enforce a textbook
# histogram on a research corpus, it is to fail a corpus that has the two
# words swapped.
PDAC_HEAD_BAND = (0.55, 0.75)
PDAC_BODY_BAND = (0.10, 0.25)
PDAC_TAIL_BAND = (0.05, 0.25)
PDAC_HEAD_FLOOR = 0.45

# ---------------------------------------------------------------------------
# THE TWO KINDS OF "na", WHICH ARE NOT THE SAME FINDING
#
# the pipeline team's round-3 re-issue settled on ONE spelling for a void - every void is the
# word "na" - and put the REASON in a separate field beside it,
# anatomic_location_status. That was the right move and it is the whole reason
# this block exists, because the four reasons it distinguishes are four
# different statements about a patient and only one of them is an absence:
#
#   not_applicable  the organ has no head, body or tail AT ALL. A kidney has
#                   poles, a liver has Couinaud segments. Nothing can ever be
#                   measured here and the void is permanent and correct.
#   not_computed    the organ HAS the axis and nobody ran the rule on it. the pipeline team
#                   says so in README_ROUND3.md section 5: "Optional, not
#                   done: 223 FLARE23 pool records have a pancreatic tumor
#                   whose sub-site was never derived." That is a GAP, and a
#                   gap is not an absence.
#   guard_failed    the rule RAN and its orientation guard refused to answer.
#                   A refusal is not an absence either.
#   computed        a measurement.
#
# Until 2026-10-04 this suite skipped on the VALUE - `if word in (None,
# "unknown", "na", "none", "not_computed"): continue` - which cannot tell the
# four apart, because the pipeline team spells all four "na". The consequence is reported in
# V4.6/<corpus>/<dataset>/<organ>/reach: the population test, the one check in
# this file written to catch a swapped head and tail at scale, read ZERO
# lesion rows out of the 1,198-record FLARE23 corpus and said nothing about
# it. A check that reports PASS because it has stopped looking is worse than
# one that fails.
SUBSITE_STATUS_ABSENT = ("not_applicable",)
SUBSITE_STATUS_UNMEASURED = ("not_computed", "guard_failed")
SUBSITE_STATUS_MEASURED = ("computed",)
SUBSITE_STATUS_KNOWN = (SUBSITE_STATUS_ABSENT + SUBSITE_STATUS_UNMEASURED
                        + SUBSITE_STATUS_MEASURED)
# The value-level void tokens. Used ONLY where a record carries no status
# field at all, and the row says so when it has had to fall back to them.
SUBSITE_VOID_WORDS = ("", "unknown", "na", "none", "not_computed",
                      "not computed")


def _subsite_state(block):
    """("measured"|"absent"|"unmeasured"|"unstated", status, word).

    `status` is the pipeline team's own anatomic_location_status where the record carries
    one, and None where it does not; the fourth state says the record is
    older than the field and was classified on its value alone.
    """
    word = block.get("anatomic_location")
    word = None if word is None else str(word).strip().lower()
    raw = block.get("anatomic_location_status")
    status = None if raw is None else str(raw).strip().lower()
    if status in SUBSITE_STATUS_ABSENT:
        return "absent", status, word
    if status in SUBSITE_STATUS_UNMEASURED:
        return "unmeasured", status, word
    if status in SUBSITE_STATUS_MEASURED:
        return "measured", status, word
    if status:
        # A status this suite has never seen. It is NOT quietly treated as a
        # void: an unknown reason for an absence is a finding of its own.
        return "unstated", status, word
    if word is None or word in SUBSITE_VOID_WORDS:
        return "unstated", None, word
    return "measured", None, word


_S6 = ndimage.generate_binary_structure(3, 1)
_S26 = ndimage.generate_binary_structure(3, 3)

# Which array axis the pipeline team's delivered extractor fills holes along, per dataset,
# verbatim from ipkg_delivered/code/extract_phenotypes.py DATASETS. It is
# reproduced here rather than imported because the presets also remap the
# labels, and our masks are already in the unified label space. flare23 has
# no preset there; the extractor's own default for a multi-label mask is 2.
DELIVERED_FILL_AXIS = {"kits": 0, "lits": 0, "msd": 2, "flare23": 2}

# The record field names the UI prints, and how each one is compared. "exact"
# means string, boolean or integer equality; "cm3" uses the volume tolerances.
CLAIM_FIELDS = (("present", "exact"),
                ("has_tumor", "exact"),
                ("organ_volume_cm3", "cm3"),
                ("tumor_volume_cm3", "cm3"),
                ("tumor_voxels", "voxels"),
                ("burden_cat", "exact"),
                ("multiplicity", "exact"),
                ("containment", "exact"))

# Which structure of the unified mask each record organ is measured on. The
# records carry ONE kidney per case (right/left is an app-side geometric
# split), so the kidney row is the pooled structure.
RECORD_ORGAN_LABELS = {"liver": (1,), "kidney": (2, 13), "spleen": (3,),
                       "pancreas": (4,)}


# ---------------------------------------------------------------------------
# a result row
# ---------------------------------------------------------------------------
def _r(cid, check, title, status, measured, expected, note=None, known=None,
       audience=AUDIENCE_INTERNAL, **extra):
    """One check outcome.

    `measured` is always populated, including on a PASS, because the point of
    the report is the number and not the verdict: a reader who disagrees with
    a threshold can still see what was measured. `known` quotes where a FAIL
    was already written down, so a new finding is the row with no `known`.
    """
    row = {"id": check, "title": title, "status": status,
           "measured": measured, "expected": expected,
           "audience": audience}
    if note:
        row["note"] = note
    if known:
        row["known"] = known
    row.update(extra)
    row["case"] = cid
    return row


def _num(v, nd=4):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not np.isfinite(f):
        return None
    return round(f, nd)


# ---------------------------------------------------------------------------
# hashing and the fingerprint lock
# ---------------------------------------------------------------------------
def _rel(path):
    """A repo-relative path where one exists, the absolute path otherwise.

    os.path.relpath raises across Windows drive letters, and some of the
    things this suite reads legitimately live on another mount (the
    TotalSegmentator anchor set, a scratch file in a self-test), so a report
    must not die on a path it can still name."""
    try:
        return os.path.relpath(path, BASE).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _load_lock():
    """The fingerprint lock, or an empty one.

    A hash on its own proves nothing - it has to be compared with a hash taken
    earlier. The first run therefore RECORDS rather than asserts, and says so
    (SKIP); from the second run on, a file whose content moved under the app is
    a FAIL with both hashes printed. The lock lives beside the reports so that
    it travels with them.
    """
    if os.path.isfile(LOCK_JSON):
        try:
            with open(LOCK_JSON, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and isinstance(d.get("files"), dict):
                return d
        except Exception:                                     # noqa: BLE001
            pass
    return {"schema": "ipkg_validation/fingerprints/1",
            "note": "sha256 of every file the four case pages are drawn from, "
                    "recorded the first time ipkg_validate.py saw it. A "
                    "changed hash is V1.1's FAIL.",
            "files": {}}


def _lock_check(lock, cid, role, path):
    """Hash `path` and compare it with the lock. Returns (status, measured)."""
    rel = _rel(path)
    key = "%s::%s" % (cid, role)
    dig = sha256(path)
    st = os.stat(path)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    prev = lock["files"].get(key)
    measured = {"role": role, "path": rel, "bytes": int(st.st_size),
                "sha256": dig}
    if prev is None:
        lock["files"][key] = {"path": rel, "bytes": int(st.st_size),
                              "sha256": dig, "first_seen": now,
                              "last_seen": now}
        return SKIP, dict(measured, lock="recorded now (first run)")
    measured["sha256_recorded"] = prev.get("sha256")
    measured["recorded_on"] = prev.get("first_seen")
    if prev.get("sha256") == dig:
        prev["last_seen"] = now
        prev["path"] = rel
        return PASS, dict(measured, lock="unchanged since %s"
                                         % prev.get("first_seen"))
    measured["bytes_recorded"] = prev.get("bytes")
    return FAIL, dict(measured, lock="CONTENT CHANGED since %s"
                                     % prev.get("first_seen"))


def _save_lock(lock):
    _ensure_out()
    with open(LOCK_JSON, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=1, sort_keys=True)


def _ensure_out():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)


# ---------------------------------------------------------------------------
# world geometry on an axis-aligned grid
#
# Every one of the four cases is stored on an axis-aligned grid (a permutation
# and a scale), but not in one orientation: KiTS is IPL and the other three are
# RAS. So a world coordinate can be read off a one-dimensional profile of the
# mask along the single array axis that carries it, which costs a sum instead
# of an argwhere over eighty-five million voxels. The alignment is CHECKED, not
# assumed; a rotated grid returns None and the caller says so rather than
# quietly measuring along the wrong line.
# ---------------------------------------------------------------------------
def _axis_for_world(affine, comp):
    """(array axis, aligned?) for world component `comp` of an affine."""
    row = np.abs(np.asarray(affine, float)[comp, :3])
    ax = int(np.argmax(row))
    rest = float(np.sort(row)[:2].sum())
    return ax, bool(rest <= 1e-6 * max(1.0, float(row[ax])))


def _world_profile(mask, affine, comp):
    """(world values, voxel counts) along world component `comp`.

    One entry per index of the array axis that carries that component: the
    world coordinate of every voxel in that slab, and how many mask voxels it
    holds. Weighted means and percentiles of a world coordinate follow from
    this without ever materialising a coordinate array.
    """
    ax, aligned = _axis_for_world(affine, comp)
    if not aligned:
        return None
    other = tuple(a for a in range(3) if a != ax)
    n = mask.sum(axis=other).astype(np.int64)
    idx = np.arange(mask.shape[ax], dtype=float)
    A = np.asarray(affine, float)
    vals = A[comp, ax] * idx + A[comp, 3]
    return vals, n


def _wmean(prof):
    if prof is None:
        return None
    vals, n = prof
    tot = float(n.sum())
    if tot <= 0:
        return None
    return float((vals * n).sum() / tot)


def _wpct(prof, q):
    if prof is None:
        return None
    vals, n = prof
    o = np.argsort(vals)
    v, w = vals[o], n[o].astype(float)
    tot = w.sum()
    if tot <= 0:
        return None
    c = np.cumsum(w) - 0.5 * w
    return float(np.interp(q / 100.0 * tot, c, v))


def _wspan(prof):
    """(min, max) world coordinate actually occupied by the mask."""
    if prof is None:
        return None, None
    vals, n = prof
    hit = np.where(n > 0)[0]
    if not hit.size:
        return None, None
    lo = float(min(vals[hit]))
    hi = float(max(vals[hit]))
    return lo, hi


def _world_centroid(mask, affine):
    """Centroid in world millimetres, or None on a rotated grid."""
    out = []
    for comp in range(3):
        m = _wmean(_world_profile(mask, affine, comp))
        if m is None:
            return None
        out.append(m)
    return out


def _axis_of_code(affine, codes):
    """The array axis whose orientation code is one of `codes` (e.g. S/I for
    the axial stacking axis), or None."""
    try:
        ac = nib.aff2axcodes(np.asarray(affine, float))
    except Exception:                                         # noqa: BLE001
        return None
    for a, c in enumerate(ac):
        if c in codes:
            return a
    return None


# ---------------------------------------------------------------------------
# small measurement helpers
# ---------------------------------------------------------------------------
def _dice(p, r):
    """2|P and R| / (|P| + |R|). Two empty masks are 1.0 by convention, which
    is stated in the output rather than left for the reader to guess."""
    np_, nr = int(p.sum()), int(r.sum())
    if np_ == 0 and nr == 0:
        return 1.0, 0, 0, 0
    inter = int(np.logical_and(p, r).sum())
    return (2.0 * inter / (np_ + nr)), inter, np_, nr


def _cm3(nvox, spacing):
    return float(nvox) * float(np.prod(spacing)) / 1000.0


def _hu_stats(ct, mask):
    """Attenuation profile of one structure. Returns None for an empty mask."""
    n = int(mask.sum())
    if n == 0:
        return None
    hu = ct[mask]
    return {"voxels": n,
            "median_hu": _num(np.median(hu), 1),
            "mean_hu": _num(hu.mean(), 1),
            "p05_hu": _num(np.percentile(hu, 5), 1),
            "p95_hu": _num(np.percentile(hu, 95), 1),
            "air_frac": _num(float((hu < HU_AIR).mean()), 5),
            "bone_frac": _num(float((hu >= HU_BONE).mean()), 5)}


def _components(mask, structure=_S26):
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return 0, 0, 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return int(n), int(sizes.max()), int((sizes >= 10).sum())


def _mask_of(labels_arr, ids):
    if len(ids) == 1:
        return labels_arr == ids[0]
    return np.isin(labels_arr, list(ids))


# ---------------------------------------------------------------------------
# the case under test
#
# Path resolution, the reference rule per dataset and the reduction of the
# LiTS label onto the pipeline team's display slab are NOT re-implemented here. They live in
# build_case_metrics.case_inputs, they were verified against the delivered
# files during the audit, and a second copy of them would only give the suite
# a way to disagree with the numbers the app shows for a reason that is not a
# defect. What this module adds is everything case_inputs does not do: the
# hashes, the world-coordinate anatomy, the claim reproduction and the report.
# ---------------------------------------------------------------------------
def load_case(entry, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    cid = entry["display_id"]
    say("  loading %s ..." % cid)
    t0 = time.time()
    inp = BCM.case_inputs(entry)
    pred_im = inp["pred_image"]
    ct_im = nib.load(inp["ct_path"])
    ref = inp["variants"][0]
    ctx = {"entry": entry, "cid": cid, "dataset": entry["dataset"],
           "collection": entry["collection"], "organ": entry["organ"],
           "pred": inp["pred"], "pred_path": inp["pred_path"],
           "pred_affine": np.asarray(pred_im.affine, float),
           "pred_zooms": BCM._spacing(pred_im),
           "pred_axcodes": "".join(nib.aff2axcodes(pred_im.affine)),
           "ct": inp["ct"], "ct_path": inp["ct_path"],
           "ct_affine": np.asarray(ct_im.affine, float),
           "ct_zooms": BCM._spacing(ct_im),
           "variants": inp["variants"], "ref": ref["labels"],
           "ref_variant": ref, "spacing": inp["spacing"],
           "grid": inp["grid"], "inherited_notes": inp["notes"],
           "structures": BCM._structures_for(inp["pred"], ref["labels"]),
           "load_seconds": round(time.time() - t0, 1)}
    ctx["axial_axis"] = _axis_of_code(ctx["pred_affine"], ("S", "I"))
    say("    %s  grid %s  %s  %.0fs"
        % (cid, "x".join(str(v) for v in ctx["grid"]["shape"]),
           ctx["pred_axcodes"], ctx["load_seconds"]))
    return ctx


def _ref_source_path(variant):
    return os.path.join(BASE, variant["source"]["path"].replace("/", os.sep))


def _published_fingerprints(cid):
    """What ipkg_case_metrics.json says the app's own numbers were measured
    from. This is the independent expectation V1.2 compares against: if the
    file on disk is not the file the published metrics were taken on, the
    panel is describing something else."""
    try:
        doc = BCM.load()
    except Exception:                                         # noqa: BLE001
        return None
    c = (doc.get("cases") or {}).get(cid)
    if not c:
        return None
    out = {"prediction": c.get("prediction", {}).get("source"),
           "ct": c.get("ct", {}).get("source"),
           "reference": (c.get("reference") or {}).get("source")}
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------------------
# V1  identity and geometry
# ---------------------------------------------------------------------------
def v1_identity(ctx, lock):
    cid = ctx["cid"]
    out = []
    roles = [("prediction", ctx["pred_path"]), ("ct", ctx["ct_path"])]
    for i, v in enumerate(ctx["variants"]):
        roles.append(("reference" if i == 0 else "reference_alt%d" % i,
                      _ref_source_path(v)))

    # V1.1  the mask on screen is the file we saw last time.
    for role, path in roles:
        status, measured = _lock_check(lock, cid, role, path)
        out.append(_r(cid, "V1.1/%s" % role,
                      "the %s file is byte-identical to the one this suite "
                      "recorded" % role, status, measured,
                      "sha256 unchanged since it was first recorded",
                      note=("nothing to compare against yet - this run "
                            "records the hash; a changed file is a FAIL from "
                            "the next run on")
                           if status == SKIP else None))

    # V1.2  the file on disk is the file the published metrics describe.
    pub = _published_fingerprints(cid)
    for role, path in roles:
        want = (pub or {}).get(role)
        if not want:
            out.append(_r(cid, "V1.2/%s" % role,
                          "the %s file is the one ipkg_case_metrics.json was "
                          "measured from" % role, SKIP,
                          {"role": role, "published_fingerprint": None},
                          "a published fingerprint to compare with",
                          note="ipkg_case_metrics.json records no "
                               "fingerprint for this role"))
            continue
        st = os.stat(path)
        same_bytes = int(st.st_size) == int(want.get("bytes", -1))
        out.append(_r(
            cid, "V1.2/%s" % role,
            "the %s file is the one ipkg_case_metrics.json was measured from"
            % role, PASS if same_bytes else FAIL,
            {"role": role,
             "path_on_disk": _rel(path),
             "bytes_on_disk": int(st.st_size),
             "bytes_published": want.get("bytes"),
             "mtime_on_disk": int(st.st_mtime),
             "mtime_published": want.get("mtime"),
             "mtime_agrees": int(st.st_mtime) == int(want.get("mtime", -1))},
            "byte size equal to the published fingerprint",
            note="the gate is the byte size; a differing mtime alone is a "
                 "copy, not a new file, and is reported rather than failed"))

    # V1.3  shape. Nothing downstream means anything if the three volumes are
    # not the same array.
    shp = tuple(int(v) for v in ctx["pred"].shape)
    rows = [("ct", tuple(int(v) for v in ctx["ct"].shape))]
    for i, v in enumerate(ctx["variants"]):
        rows.append(("reference" if i == 0 else "reference_alt%d" % i,
                     tuple(int(x) for x in v["labels"].shape)))
    bad = [r for r, s in rows if s != shp]
    out.append(_r(cid, "V1.3", "mask, CT and reference share one array shape",
                  PASS if not bad else FAIL,
                  {"prediction": list(shp),
                   **{r: list(s) for r, s in rows}},
                  "every volume on the drawn mask's shape",
                  note=None if not bad else "disagreeing: %s" % ", ".join(bad)))

    # V1.4  spacing.
    zp = ctx["pred_zooms"]
    zc = ctx["ct_zooms"]
    ok_z = bool(np.allclose(zp, zc, atol=ZOOM_TOL_MM))
    out.append(_r(cid, "V1.4", "mask and CT share one voxel spacing",
                  PASS if ok_z else FAIL,
                  {"prediction_mm": [_num(v, 6) for v in zp],
                   "ct_mm": [_num(v, 6) for v in zc],
                   "max_abs_diff_mm": _num(float(np.abs(zp - zc).max()), 8)},
                  "equal to within %g mm" % ZOOM_TOL_MM))

    # V1.5  affine. Split from the spacing check on purpose: a zeroed world
    # origin leaves the shape and the zooms intact and still destroys every
    # absolute world position, which is exactly the FLARE23 label condition.
    A, C = ctx["pred_affine"], ctx["ct_affine"]
    d_dir = float(np.abs(A[:3, :3] - C[:3, :3]).max())
    d_org = float(np.abs(A[:3, 3] - C[:3, 3]).max())
    out.append(_r(cid, "V1.5/ct", "mask and CT share one affine",
                  PASS if max(d_dir, d_org) <= AFFINE_TOL_MM else FAIL,
                  {"max_abs_direction_diff": _num(d_dir, 6),
                   "max_abs_origin_diff_mm": _num(d_org, 3),
                   "mask_axcodes": ctx["pred_axcodes"],
                   "ct_axcodes": "".join(nib.aff2axcodes(C)),
                   "mask_origin_mm": [_num(v, 3) for v in A[:3, 3]],
                   "ct_origin_mm": [_num(v, 3) for v in C[:3, 3]]},
                  "equal to within %g mm" % AFFINE_TOL_MM))
    for i, v in enumerate(ctx["variants"]):
        p = _ref_source_path(v)
        R = np.asarray(nib.load(p).affine, float)
        rd = float(np.abs(A[:3, :3] - R[:3, :3]).max())
        ro = float(np.abs(A[:3, 3] - R[:3, 3]).max())
        role = "reference" if i == 0 else "reference_alt%d" % i
        zeroed = bool(np.allclose(R[:3, 3], 0.0)) and \
            not bool(np.allclose(A[:3, 3], 0.0))
        red = v.get("reduction")
        measured = {"max_abs_direction_diff": _num(rd, 6),
                    "max_abs_origin_diff_mm": _num(ro, 3),
                    "reference_axcodes": "".join(nib.aff2axcodes(R)),
                    "reference_origin_mm": [_num(x, 3) for x in R[:3, 3]],
                    "reference_origin_is_zeroed": zeroed,
                    "reference_path":
                        _rel(p),
                    "note_on_grid": v.get("provenance")}
        if red:
            # A reduced reference is not a file on the drawn grid, it is a
            # derivation onto it, so raw affine equality is the wrong
            # question: the native label file SHOULD have a different affine.
            # What has to hold is that the reduction is derivable from the two
            # affines rather than typed, which is what these parameters are -
            # a slab re-cut somewhere else would change them instead of being
            # silently mis-measured.
            measured["reduction"] = red
            in_plane_ok = abs(float(ctx["pred_zooms"][0])
                              / float(BCM._spacing(nib.load(p))[0])
                              - red["in_plane_factor"]) < 1e-3
            measured["in_plane_factor_follows_from_the_two_spacings"] = \
                bool(in_plane_ok)
            out.append(_r(
                cid, "V1.5/%s" % role,
                "the %s reduction onto the drawn grid is derived from the two "
                "affines" % role, PASS if in_plane_ok else FAIL, measured,
                "an in-plane factor and a z offset that follow from the two "
                "affines",
                note="this reference is a reduction of the dataset's own "
                     "label onto the pipeline team's display slab, so its source file is on "
                     "a different grid by construction and affine equality "
                     "is not the test; the derivation is"))
            continue
        known = None
        if zeroed:
            known = ("the FLARE23 Box label files carry a zeroed world "
                     "origin; ipkg_case_metrics.json records the same "
                     "condition and compares the two volumes array-wise "
                     "after checking shape and zooms")
        out.append(_r(
            cid, "V1.5/%s" % role,
            "mask and %s share one affine" % role,
            PASS if max(rd, ro) <= AFFINE_TOL_MM else FAIL, measured,
            "equal to within %g mm" % AFFINE_TOL_MM, known=known))
    # (V1.6 follows)

    # V1.6  the label values are integers. FLARE23's own label files store
    # integers as floats - a tumour label read as 14.000000115 and tested with
    # == 14 yields an empty mask in silence, which is the one failure mode that
    # produces a perfectly plausible screen with nothing on it.
    for role, path in roles:
        if role == "ct":
            continue
        raw = np.asanyarray(nib.load(path).dataobj)
        dev = float(np.abs(raw - np.rint(raw)).max()) if raw.size else 0.0
        ids = sorted(int(v) for v in np.unique(np.rint(raw)))
        unknown = [v for v in ids if v != 0 and v not in UNIFIED]
        native = role != "prediction" and ctx["dataset"] != "flare23"
        out.append(_r(
            cid, "V1.6/%s" % role,
            "the %s label values are integers, and in a known label space"
            % role,
            PASS if dev <= 1e-4 and (native or not unknown) else FAIL,
            {"dtype": str(raw.dtype),
             "max_deviation_from_integer": _num(dev, 9),
             "label_ids": ids,
             "unified_names": {str(v): UNIFIED.get(v) for v in ids if v},
             "ids_outside_the_unified_space": unknown,
             "space": "the dataset's own label space" if native
                      else "the unified label space"},
            "integral values, and for a unified-space volume no label id "
            "outside UNIFIED",
            note=None if not native else
                 "this is the dataset's own file, so its ids are the "
                 "dataset's; the mapping into the unified space is checked "
                 "where the reference is used"))
    return out


# ---------------------------------------------------------------------------
# V2  predicted versus reference agreement
# ---------------------------------------------------------------------------
def _needs_geometric_split(ds):
    """True when the dataset annotates ONE kidney class, so right and left are
    assigned by us rather than by the annotator. label_map marks it with a
    negative target, which is how the delivered code itself says a lookup
    table is not enough."""
    name = {"kits": "KiTS23", "lits": "LiTS", "msd": "MSD Pancreas",
            "flare23": "FLARE23"}.get(ds)
    return any(int(v) < 0 for v in MAPS.get(name, {}).values())


def _verdict_structures(ctx):
    """The structures a per-slice verdict may ride on for this case."""
    names = [n for n, _, _ in ctx["structures"]]
    if _needs_geometric_split(ctx["dataset"]):
        drop = {"right_kidney", "left_kidney"}
    else:
        drop = {"kidney_pooled"}
    return [n for n in names if n not in drop]


def v2_agreement(ctx):
    cid = ctx["cid"]
    pred, ref, sp = ctx["pred"], ctx["ref"], ctx["spacing"]
    out = []
    conv = BCM.CONVENTIONS["dice"]
    per = {}
    for name, ids, kind in ctx["structures"]:
        p, r = _mask_of(pred, ids), _mask_of(ref, ids)
        d, inter, npv, nrv = _dice(p, r)
        per[name] = {
            "kind": kind, "label_ids": list(ids),
            "dice": _num(d, 4),
            "predicted_voxels": npv, "reference_voxels": nrv,
            "intersection_voxels": inter,
            "predicted_cm3": _num(_cm3(npv, sp), 2),
            "reference_cm3": _num(_cm3(nrv, sp), 2),
            "predicted_minus_reference_cm3": _num(_cm3(npv - nrv, sp), 2),
            "outside_reference_cm3": _num(_cm3(npv - inter, sp), 2),
            "missed_reference_cm3": _num(_cm3(nrv - inter, sp), 2)}
    out.append(_r(cid, "V2.1",
                  "predicted-versus-reference agreement per structure, on the "
                  "grid the app draws",
                  PASS, {"convention": conv,
                         "reference": ctx["ref_variant"]["label"],
                         "reference_provenance":
                             ctx["ref_variant"]["provenance"],
                         "grid": ctx["grid"], "structures": per},
                  "measured and reported; this row states the numbers and the "
                  "convention rather than judging them - the judgements are "
                  "V3 (does the printed figure reproduce) and V2.2-V2.4 (does "
                  "the protocol hold)",
                  note="Dice of two empty masks is 1.0 by convention. "
                       "Kidneys are also reported pooled because the records "
                       "carry one kidney per case."))

    # V2.2 / V2.3  the two invariants the semi-oracle protocol claims. They
    # are the only part of the protocol that is checkable from the delivered
    # files, and they are worth checking precisely because they are the reason
    # the Dice values are ceilings: a prediction that never runs where the
    # reference is absent can never produce a false positive slice.
    ax = ctx["axial_axis"]
    if ax is None:
        out.append(_r(cid, "V2.2", "no prediction on a reference-absent slice",
                      SKIP, {"axial_axis": None},
                      "the axial stacking axis",
                      note="no S/I axis in the affine's codes, so 'slice' is "
                           "not defined for this grid"))
    else:
        other = tuple(a for a in range(3) if a != ax)
        inplane = int(np.prod([ctx["pred"].shape[a] for a in other]))
        scale = (256.0 * 256.0) / max(1, inplane)
        # Which structures the VERDICT rides on. Where the dataset annotates
        # one kidney class and the sides were assigned by us, a per-side row
        # can miss a slice purely because the two assignment rules put a
        # component on opposite sides - so the per-side rows are reported and
        # the pooled row is judged. Where the dataset annotates both, the
        # pooled row is the redundant one.
        judged = set(_verdict_structures(ctx))
        ghosts, missed, gate_rows = {}, {}, {}
        for name, ids, kind in ctx["structures"]:
            p = _mask_of(pred, ids).sum(axis=other)
            r = _mask_of(ref, ids).sum(axis=other)
            r256 = r * scale
            gh = np.where((p > 0) & (r == 0))[0]
            ms = np.where((p == 0) & (r256 >= SLICE_GATE_PX_AT_256))[0]
            # A slice sitting AT the gate is indeterminate rather than
            # missing: the pipeline team's gate was applied to a 256x256 resampling, and a
            # rescale of a 200-voxel contour on a 512 grid lands on 50.0 px,
            # which is the threshold itself. So the verdict rides on the
            # slices that clear the gate by more than a pixel, and the ones
            # at the edge are reported beside them.
            clear = ms[r256[ms] >= SLICE_GATE_PX_AT_256 + 1.0]
            edge = ms[r256[ms] < SLICE_GATE_PX_AT_256 + 1.0]
            ghosts[name] = {"slices": [int(v) for v in gh[:40]],
                            "n_slices": int(gh.size),
                            "voxels_on_those_slices": int(p[gh].sum())}
            missed[name] = {
                "slices": [int(v) for v in clear[:40]],
                "n_slices": int(clear.size),
                "reference_voxels_missed": int(r[clear].sum()),
                "per_slice_reference_px_at_256":
                    [[int(z), int(r[z]), _num(float(r256[z]), 1)]
                     for z in clear[:40]],
                "slices_at_the_gate_within_1_px": [int(v) for v in edge[:40]],
                "n_slices_at_the_gate_within_1_px": int(edge.size),
                "per_slice_at_the_gate":
                    [[int(z), int(r[z]), _num(float(r256[z]), 1)]
                     for z in edge[:40]]}
            below = np.where((r256 > 0) & (r256 < SLICE_GATE_PX_AT_256))[0]
            gate_rows[name] = {
                "reference_slices": int((r > 0).sum()),
                "reference_slices_at_or_above_gate":
                    int((r256 >= SLICE_GATE_PX_AT_256).sum()),
                "reference_slices_below_gate": int(below.size),
                "predicted_slices": int((p > 0).sum()),
                "predicted_on_a_below_gate_slice":
                    int(((p > 0) & (r256 > 0) &
                         (r256 < SLICE_GATE_PX_AT_256)).sum())}
        n_ghost = sum(v["n_slices"] for k, v in ghosts.items()
                      if k in judged)
        out.append(_r(
            cid, "V2.2", "no prediction on a reference-absent slice",
            PASS if n_ghost == 0 else FAIL,
            {"axial_axis": ax, "in_plane_voxels": inplane,
             "structures_judged": sorted(judged),
             "per_structure": ghosts,
             "total_slices_on_the_judged_structures": n_ghost,
             "total_slices_all_structures":
                 sum(v["n_slices"] for v in ghosts.values())},
            "zero, which is what the semi-oracle protocol claims",
            note="a non-zero count here would mean the box was not taken "
                 "from this reference on this grid",
            audience=AUDIENCE_INTERNAL))
        n_missed = sum(v["n_slices"] for k, v in missed.items()
                       if k in judged)
        out.append(_r(
            cid, "V2.3",
            "a prediction on every slice the 50 px gate admitted",
            PASS if n_missed == 0 else FAIL,
            {"gate_px_at_256": SLICE_GATE_PX_AT_256,
             "area_scale_to_256": _num(scale, 4),
             "structures_judged": sorted(judged),
             "per_structure": missed,
             "total_slices_on_the_judged_structures": n_missed,
             "total_slices_all_structures":
                 sum(v["n_slices"] for v in missed.values())},
            "zero admitted slices without a prediction",
            note="the gate is stated at 256x256, so the reference area per "
                 "slice is scaled by (256*256)/in-plane voxels before it is "
                 "compared; a non-zero count is a slice the box was computed "
                 "for and the network returned nothing on. Per-side kidney "
                 "rows are reported and not judged where the sides are a "
                 "computed split - a slice can 'miss' there only because the "
                 "two split rules disagree, which is a laterality question "
                 "and not a gate question."))
        out.append(_r(
            cid, "V2.4", "the slice gate, as it fell on this case", PASS,
            {"per_structure": gate_rows}, "reported",
            note="reported rather than judged: how many reference slices sat "
                 "below the gate is a property of the protocol, and it is "
                 "the number that says how much of the structure the "
                 "network was never asked about"))

    # V2.5  where a case has two candidate references, how far apart they are.
    # A Dice quoted against one of them is only as meaningful as their
    # agreement, and on LiTS the two are different reductions of one label.
    if len(ctx["variants"]) > 1:
        agree = {}
        for name, ids, kind in ctx["structures"]:
            a = _mask_of(ctx["variants"][0]["labels"], ids)
            b = _mask_of(ctx["variants"][1]["labels"], ids)
            d, inter, na, nb = _dice(a, b)
            agree[name] = {"dice": _num(d, 4), "primary_voxels": na,
                           "alternate_voxels": nb,
                           "difference_voxels": int(na - nb)}
        worst = min(v["dice"] for v in agree.values()) if agree else 1.0
        out.append(_r(
            cid, "V2.5", "the case's two candidate references agree",
            PASS if worst >= 0.99 else FAIL,
            {"primary": ctx["variants"][0]["label"],
             "alternate": ctx["variants"][1]["label"],
             "per_structure": agree, "worst_dice": _num(worst, 4)},
            "Dice at or above 0.99 between the two reductions",
            note="two reductions of one dataset label onto one display slab "
                 "should differ only where a block vote is tied; a larger "
                 "gap means the figure a reader sees depends on which "
                 "reference was used"))
    return out


# ---------------------------------------------------------------------------
# V3  claim by claim
#
# The rule is that a number on screen has to be recomputable from the artefact
# it claims to describe. Two things make this check say something rather than
# merely agree with itself:
#
#   1. The recomputation runs the pipeline team's OWN delivered extractor
#      (ipkg_delivered/code/extract_phenotypes.py), so a disagreement cannot
#      be a difference of convention between their definition of containment
#      or multiplicity and ours.
#   2. When a number does not reproduce on the artefact it claims, the check
#      does not stop at "mismatch": it re-runs on every other candidate
#      artefact of the case and names the one that DOES reproduce it. That is
#      what turns "the volume looks wrong" into "this figure is the earlier
#      prediction, not the mask the page draws".
# ---------------------------------------------------------------------------
def _extract_on(mask, spacing, cid, ds):
    """One candidate artefact through the pipeline team's delivered extractor.

    dataset is passed as "new" on purpose. The delivered presets for kits /
    lits / msd also REMAP the labels (organ 1, tumour 2), and our masks are
    already in the unified space, so a preset would silently read the unified
    right kidney (2) as a tumour. What the presets otherwise supply - the
    cyst drop and the hole-filling axis - is supplied explicitly instead, and
    recorded in the returned settings so the report says what was run.
    """
    EP = FC.pipeline_module("extract_phenotypes")
    m = np.asarray(mask, np.uint8)
    dropped = None
    if ds == "kits":
        # The delivered kits preset drops native label 3, the cyst, because a
        # cyst is not a lesion in the evaluated pipeline. In the unified space
        # the cyst is 15.
        n = int((m == 15).sum())
        if n:
            m = np.where(m == 15, 0, m)
        dropped = {"label": 15, "name": "cyst", "voxels": n}
    cps = FC.cutpoints().get(ds) or {}
    terc = None
    if cps.get("t1_vox") is not None:
        terc = [float(cps["t1_vox"]), float(cps["t2_vox"])]
    fill = DELIVERED_FILL_AXIS.get(ds, 2)
    rec = EP.extract(m, np.asarray(spacing, float), cid, dataset="new",
                     terciles_vox=terc, single_organ=None, fill_axis=fill)
    settings = {"extractor": "ipkg_delivered/code/extract_phenotypes.py",
                "terciles_vox": terc,
                "terciles_cm3": None if terc else [EP.T1_CM3, EP.T2_CM3],
                "fill_axis": fill,
                "spacing_mm": [_num(v, 6) for v in spacing],
                "dropped_label": dropped}
    return rec, settings


def _lits_native_reference(ctx):
    """The LiTS liver label on its NATIVE grid.

    The delivered LiTS record does not reproduce on the display slab, and the
    reason matters: a figure computed on a 0.742 mm grid and shown beside a
    1.484 mm slab is not wrong, it is a different measurement of the same
    patient. Offering the native grid as a candidate lets V3 say which of the
    two a printed number came from instead of only that it missed.
    """
    p = os.path.join(BASE, "MSD Task 3 Liver", "Task03_Liver", "labelsTr",
                     "liver_76.nii.gz")
    if not os.path.isfile(p):
        return None
    im, arr = BCM._labels(p)
    return ("the LiTS label on its own native grid",
            BCM._remap(arr, MAPS["LiTS"]), BCM._spacing(im))


def _candidates(ctx):
    """Every artefact a printed number could plausibly have been measured on."""
    out = [("the prediction the page draws, on the grid it draws it",
            ctx["pred"], ctx["spacing"])]
    for i, v in enumerate(ctx["variants"]):
        out.append(("%s%s" % (v["label"], "" if i == 0 else " (alternate)"),
                    v["labels"], ctx["spacing"]))
    if ctx["dataset"] == "lits":
        nat = _lits_native_reference(ctx)
        if nat:
            out.append(nat)
    return out


def _cmp(kind, claimed, got):
    """(agrees, difference) for one printed field under its comparison rule."""
    if claimed is None or got is None:
        return (claimed is None and got is None), None
    if kind == "exact":
        return bool(claimed == got), None
    if kind == "voxels":
        return abs(int(claimed) - int(got)) <= VOXEL_TOL, int(got) - int(claimed)
    a, b = float(claimed), float(got)
    d = b - a
    tol = max(VOL_TOL_CM3, VOL_TOL_FRAC * max(abs(a), abs(b)))
    return bool(abs(d) <= tol), d


# Fields whose disagreement the audit's own cross-check already measured, and
# whose conclusion is therefore quotable as `known`.
VOLUME_FIELDS = frozenset(("organ_volume_cm3", "tumor_volume_cm3",
                           "tumor_voxels", "organ_volume_cm3_pseudo",
                           "tumor_volume_cm3_pseudo"))

# record_containment_v2's containment is an editorial override rather than a
# measurement, and the pipeline team said so when they delivered it. A containment that does
# not reproduce from the mask IS that override; the check still reports it,
# because the number on screen is still not the number the mask gives, but it
# is not a new finding.
CONTAINMENT_V2_KNOWN = (
    "record_containment_v2 is the pipeline team's display record and is defined as "
    "record_predicted with the containment flag overridden - 'identical to "
    "predicted except containment flips boundary -> contained on LiTS and "
    "MSD' (ipkg_final_cases.record) - so a containment that does not "
    "reproduce from the mask is that override and not a measurement")

CONTAINMENT_SUPERSEDED_KNOWN = (
    "record_predicted and record_reference carry the containment from before "
    "the 2026-09-16 fix, and record_containment_v2 - the record the app "
    "displays - carries the corrected value, which is the one the mask gives. "
    "So this row says the SUPERSEDED record disagrees with the mask, which is "
    "what a superseded record is for")

_XC_CACHE = {}
_DICE_CMP_CACHE = {}


def _audit_crosscheck(cid, kind):
    """What ipkg_case_metrics.json's own volume cross-check concluded for this
    record. The containment_v2 record carries the same volumes as predicted,
    so it reads the predicted row."""
    if cid not in _XC_CACHE:
        try:
            doc = BCM.load()
        except Exception:                                     # noqa: BLE001
            doc = {}
        _XC_CACHE[cid] = (((doc.get("cases") or {}).get(cid) or {})
                          .get("record_volume_crosscheck") or {})
    xc = _XC_CACHE[cid]
    key = "predicted" if kind == "containment_v2" else kind
    rows = (xc.get("records") or {}).get(key)
    if not rows:
        return None
    misses = [k for k, v in rows.items()
              if v.get("reproduces_on_this_grid") is False]
    if not misses:
        return None
    return ("the 2026-09-29 audit measured the same disagreement: %s. Its "
            "conclusion: %s" % (", ".join(sorted(misses)),
                                xc.get("conclusion") or "(none recorded)"))


def _audit_dice_comparison(cid):
    """The audit's own recomputed-versus-published tumour Dice for the case,
    so a V3.2 miss that only reproduces a figure already measured on
    2026-09-29 is not reported as new."""
    if cid not in _DICE_CMP_CACHE:
        try:
            doc = BCM.load()
        except Exception:                                     # noqa: BLE001
            doc = {}
        _DICE_CMP_CACHE[cid] = (((doc.get("cases") or {}).get(cid) or {})
                                .get("tumour_dice_comparison") or {})
    return _DICE_CMP_CACHE[cid]


def _claim_row(record_organ, claimed, got, extra_pairs=()):
    """Field-by-field comparison of one organ block."""
    fields = {}
    for name, kind in CLAIM_FIELDS:
        if name not in claimed:
            continue
        ok, diff = _cmp(kind, claimed.get(name), got.get(name))
        fields[name] = {"printed": claimed.get(name),
                        "recomputed": got.get(name),
                        "agrees": ok,
                        "difference": _num(diff, 4) if diff is not None
                        else None,
                        "rule": kind}
    for name, kind, value in extra_pairs:
        if name not in claimed:
            continue
        ok, diff = _cmp(kind, claimed.get(name), value)
        fields[name] = {"printed": claimed.get(name), "recomputed": value,
                        "agrees": ok,
                        "difference": _num(diff, 4) if diff is not None
                        else None,
                        "rule": kind}
    return fields


def v3_claims(ctx, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    cid, ds = ctx["cid"], ctx["dataset"]
    out = []
    cands = _candidates(ctx)
    say("    V3: running the delivered extractor on %d candidate artefacts"
        % len(cands))
    runs = []
    for label, mask, sp in cands:
        rec, settings = _extract_on(mask, sp, cid, ds)
        runs.append({"artefact": label, "organs": rec.get("organs") or {},
                     "observed_organs": rec.get("observed_organs") or [],
                     "settings": settings,
                     "voxels": {o: int(_mask_of(mask, ids).sum())
                                for o, ids in RECORD_ORGAN_LABELS.items()},
                     "tumour_voxels": int((mask == 14).sum())})

    # Which run is the artefact each record kind CLAIMS to describe. The
    # predicted and containment_v2 records are the drawn prediction; the
    # reference record is the dataset's own label.
    claimed_run = {"predicted": 0, "containment_v2": 0, "reference": 1}

    for kind in FC.RECORD_KINDS:
        rec = FC.record(cid, kind) or {}
        organs = rec.get("organs") or {}
        run = runs[claimed_run[kind]]
        per_organ, fails = {}, []
        for organ, block in sorted(organs.items()):
            got = (run["organs"].get(organ) or {})
            nvox = run["voxels"].get(organ)
            extra = []
            # organ_volume_cm3_pseudo / tumor_volume_cm3_pseudo are the record's
            # own voxel counts divided by 1000 (see ipkg_provenance), so they
            # are a DIRECT claim about a voxel count and the sharpest test in
            # the record: a spacing cannot explain a difference in them.
            if nvox is not None:
                extra.append(("organ_volume_cm3_pseudo", "cm3", nvox / 1000.0))
            tv = (got.get("tumor_voxels") if got else None)
            if tv is not None:
                extra.append(("tumor_volume_cm3_pseudo", "cm3", tv / 1000.0))
            fields = _claim_row(organ, block, got, extra)
            missed = [f for f, v in fields.items() if not v["agrees"]]
            # For every field that missed, ask which other artefact of this
            # case reproduces it. This is the part that names the mask.
            repro = {}
            for f in missed:
                hits = []
                for j, alt in enumerate(runs):
                    if j == claimed_run[kind]:
                        continue
                    ag = alt["organs"].get(organ) or {}
                    if f.endswith("_pseudo"):
                        base = "organ" if f.startswith("organ") else "tumour"
                        nv = (alt["voxels"].get(organ) if base == "organ"
                              else (ag.get("tumor_voxels")))
                        val = None if nv is None else nv / 1000.0
                    else:
                        val = ag.get(f.replace("_pseudo", ""))
                    kindf = dict(CLAIM_FIELDS).get(f, "cm3")
                    ok, _ = _cmp(kindf, block.get(f), val)
                    if ok:
                        hits.append({"artefact": alt["artefact"],
                                     "value": val})
                repro[f] = hits
            per_organ[organ] = {"fields": fields,
                                "fields_that_missed": missed,
                                "reproduced_by": repro,
                                "recomputed_organ_voxels": nvox,
                                "recomputed_tumour_voxels_in_organ":
                                    got.get("tumor_voxels") if got else None}
            if missed:
                fails.append("%s: %s" % (organ, ", ".join(missed)))
        obs_ok = sorted(organs) == sorted(run["observed_organs"])
        # `known` is attached only when EVERY field that missed was already
        # documented. One undocumented miss and the row stays NEW, which is
        # the whole point of the flag.
        missed_all = sorted({f for v in per_organ.values()
                             for f in v["fields_that_missed"]})
        bits, covered = [], set()
        xc = _audit_crosscheck(cid, kind)
        if xc and (VOLUME_FIELDS & set(missed_all)):
            bits.append(xc)
            covered |= VOLUME_FIELDS
        if "containment" in missed_all:
            if kind == "containment_v2":
                bits.append(CONTAINMENT_V2_KNOWN)
                covered.add("containment")
            else:
                # The superseded records only count as documented when the
                # DISPLAY record carries the value the mask actually gives;
                # otherwise the disagreement is nobody's stated intention.
                disp = (FC.record(cid, "containment_v2") or {}).get("organs") \
                    or {}
                if all(disp.get(o, {}).get("containment")
                       == (run["organs"].get(o) or {}).get("containment")
                       for o, v in per_organ.items()
                       if "containment" in v["fields_that_missed"]):
                    bits.append(CONTAINMENT_SUPERSEDED_KNOWN)
                    covered.add("containment")
        known = (" | ".join(bits)
                 if bits and set(missed_all) <= covered and obs_ok else None)
        out.append(_r(
            cid, "V3.1/%s" % kind,
            "every number record_%s prints is recomputable from the artefact "
            "it claims to describe" % kind,
            PASS if not fails and obs_ok else FAIL,
            {"record": "record_%s.json" % kind,
             "claimed_artefact": run["artefact"],
             "extractor_settings": run["settings"],
             "observed_organs_printed": sorted(organs),
             "observed_organs_recomputed": sorted(run["observed_organs"]),
             "fields_that_missed_anywhere": missed_all,
             "per_organ": per_organ},
            "every printed field equal to the recomputation, within the "
            "rounding the record itself prints",
            note=None if not fails else
                 "fields that did not reproduce - %s" % "; ".join(fails),
            known=known))

    # V3.2  the published tumour Dice against the same Dice recomputed from
    # the drawn mask. A Dice is a number the internal view prints, so it is a
    # claim like any other; the rater page shows none of this.
    rep = FC.dice(cid) or {}
    if rep.get("tumor_dice") is None:
        out.append(_r(cid, "V3.2", "the published tumour Dice reproduces",
                      SKIP, {"published": None}, "a published figure",
                      note="the pipeline team reported no tumour Dice for this case"))
    else:
        # Every candidate reference, not only the primary one. A published
        # Dice is a statement about a PAIR, and on LiTS the two reductions of
        # the dataset label give different third decimals - so the useful
        # output is not "it missed" but "it reproduces against this one".
        pub = float(rep["tumor_dice"])
        tried = []
        for i, v in enumerate(ctx["variants"]):
            d, inter, npv, nrv = _dice(ctx["pred"] == 14, v["labels"] == 14)
            tried.append({"reference": v["label"],
                          "primary": bool(i == 0),
                          "recomputed": _num(d, 4),
                          "delta_recomputed_minus_published": _num(d - pub, 4),
                          "reproduces": bool(abs(d - pub) <= DICE_TOL),
                          "predicted_voxels": npv, "reference_voxels": nrv,
                          "intersection_voxels": inter})
        hit = [t for t in tried if t["reproduces"]]
        best = min(tried, key=lambda t: abs(t["delta_recomputed_minus_published"]))
        adc = _audit_dice_comparison(cid)
        d_known = None
        if not hit and adc.get("delta_recomputed_minus_published") is not None:
            same = abs(float(adc["delta_recomputed_minus_published"])
                       - float(best["delta_recomputed_minus_published"])) \
                <= 0.001
            if same:
                d_known = ("the audit measured the same delta on 2026-09-29 "
                           "(recomputed %.4f against published %s, %+.4f) and "
                           "recorded its reading: %s"
                           % (float(adc.get("recomputed_from_drawn_mask", 0)),
                              adc.get("published"),
                              float(adc["delta_recomputed_minus_published"]),
                              adc.get("note", "")))
        out.append(_r(
            cid, "V3.2", "the published tumour Dice reproduces from the drawn "
            "mask against the dataset's own label",
            PASS if hit else FAIL,
            {"published": rep.get("tumor_dice"),
             "published_source": rep.get("source"),
             "against_each_candidate_reference": tried,
             "reproduced_by": [t["reference"] for t in hit],
             "closest": best["reference"],
             "closest_delta": best["delta_recomputed_minus_published"],
             "drawn_mask_tumour_voxels": int((ctx["pred"] == 14).sum()),
             "record_predicted_tumour_voxels": sum(
                 int(b.get("tumor_voxels") or 0) for b in
                 ((FC.record(cid, "predicted") or {}).get("organs")
                  or {}).values()),
             "why_those_two_are_here":
                 "if the drawn mask and the predicted record disagree on the "
                 "lesion's voxel count, then a published Dice that misses "
                 "here and a volume that misses in V3.1 are one finding and "
                 "not two: the figure was computed before the mask the page "
                 "draws was written",
             "convention": BCM.CONVENTIONS["dice"]},
            "equal to within %g against at least one candidate reference"
            % DICE_TOL,
            note="a miss against every candidate means the published figure "
                 "was computed on a different mask or a different convention; "
                 "a hit against a NON-primary reference means the figure is "
                 "sound but is not the pair the panel names",
            known=d_known))
        if rep.get("organ_dice") is not None:
            organ_ids = RECORD_ORGAN_LABELS.get(
                str(rep.get("organ", "")).split()[0].lower())
            if organ_ids:
                od, oi, opv, orv = _dice(_mask_of(ctx["pred"], organ_ids),
                                         _mask_of(ctx["ref"], organ_ids))
                odelta = od - float(rep["organ_dice"])
                out.append(_r(
                    cid, "V3.3", "the published organ Dice reproduces",
                    PASS if abs(odelta) <= DICE_TOL else FAIL,
                    {"organ_as_published": rep.get("organ"),
                     "label_ids": list(organ_ids),
                     "published": rep.get("organ_dice"),
                     "recomputed": _num(od, 4),
                     "delta_recomputed_minus_published": _num(odelta, 4),
                     "predicted_voxels": opv, "reference_voxels": orv},
                    "equal to within %g" % DICE_TOL))

    # V3.4  the record's internal consistency, which needs no mask at all. A
    # record that says a lesion is present, gives it a volume and then calls
    # its containment "none" is inconsistent with itself whatever the voxels
    # say, and "none" is the value the extractor writes only when there is no
    # lesion.
    for kind in FC.RECORD_KINDS:
        rec = FC.record(cid, kind) or {}
        bad = []
        detail = {}
        for organ, b in sorted((rec.get("organs") or {}).items()):
            ht = bool(b.get("has_tumor"))
            tv = b.get("tumor_volume_cm3")
            tx = b.get("tumor_voxels")
            row = {"has_tumor": ht, "tumor_volume_cm3": tv,
                   "tumor_voxels": tx, "burden_cat": b.get("burden_cat"),
                   "multiplicity": b.get("multiplicity"),
                   "containment": b.get("containment")}
            if ht:
                if b.get("containment") in (None, "none"):
                    bad.append("%s: has_tumor is true but containment is %r"
                               % (organ, b.get("containment")))
                if b.get("burden_cat") in (None, "none"):
                    bad.append("%s: has_tumor is true but burden_cat is %r"
                               % (organ, b.get("burden_cat")))
                if b.get("multiplicity") in (None, "none"):
                    bad.append("%s: has_tumor is true but multiplicity is %r"
                               % (organ, b.get("multiplicity")))
                if tv is not None and float(tv) <= 0:
                    bad.append("%s: has_tumor is true but tumor_volume_cm3 "
                               "is %s" % (organ, tv))
                if tx is None:
                    row["tumor_voxels_absent"] = True
            else:
                for f in ("containment", "burden_cat", "multiplicity"):
                    if b.get(f) not in (None, "none"):
                        bad.append("%s: has_tumor is false but %s is %r"
                                   % (organ, f, b.get(f)))
            detail[organ] = row
        out.append(_r(
            cid, "V3.4/%s" % kind,
            "record_%s is consistent with itself" % kind,
            PASS if not bad else FAIL,
            {"record": "record_%s.json" % kind, "per_organ": detail,
             "inconsistencies": bad},
            "has_tumor true implies a burden, a multiplicity and a "
            "containment other than none; has_tumor false implies all three "
            "are none"))

    # V3.5  the burden cut-point must not be the case's own value. A tercile
    # edge computed over a corpus that CONTAINS the case, landing exactly on
    # that case's own voxel count, makes the displayed burden category an
    # artefact of the case being in its own reference population.
    cp = FC.cutpoints().get(ds) or {}
    rec = FC.record(cid, "containment_v2") or {}
    own = None
    for organ, b in (rec.get("organs") or {}).items():
        if b.get("tumor_voxels") is not None:
            own = int(b["tumor_voxels"])
            own_organ = organ
            break
    if own is None or cp.get("t2_vox") is None:
        out.append(_r(cid, "V3.5",
                      "the burden cut-point is not the case's own value",
                      SKIP,
                      {"case_tumor_voxels": own, "cutpoints": cp},
                      "a voxel cut-point and a printed voxel count",
                      note="this dataset's cut-points are in cm3, or the "
                           "record prints no voxel count, so the identity "
                           "cannot be tested this way"))
    else:
        hit = [k for k in ("t1_vox", "t2_vox")
               if int(cp.get(k) or -1) == own]
        out.append(_r(
            cid, "V3.5", "the burden cut-point is not the case's own value",
            PASS if not hit else FAIL,
            {"case": cid, "organ": own_organ, "case_tumor_voxels": own,
             "t1_vox": cp.get("t1_vox"), "t2_vox": cp.get("t2_vox"),
             "n_tumor_cases_in_the_tercile_population":
                 cp.get("n_tumor_cases"),
             "cut_points_equal_to_the_case": hit,
             "rule": FC.cutpoints().get("rule")},
            "neither tercile edge equal to this case's own tumour voxel "
            "count",
            note="an edge that IS the case's own value means the case sits "
                 "exactly on its own threshold: the displayed burden "
                 "category is then a property of the corpus membership, not "
                 "of the lesion"))
    return out


# ---------------------------------------------------------------------------
# V4  anatomical sanity in world coordinates
#
# This is the check the audit wishes had existed, because it needs no
# knowledge of any particular case. Head and tail, right and left, above and
# below are statements about the patient, so they have to be made in the
# patient's frame - the affine - and not on the axis the scanner happened to
# stack slices along. The sub-site defect is precisely the difference between
# those two, and so is every laterality swap that has not happened yet.
#
# A detail worth stating: KiTS23's grid carries a zeroed world origin, so an
# ABSOLUTE world position on that case means nothing. Every test below is a
# comparison between two things measured through the SAME affine - an organ
# against the body midline, one organ against another - which is unaffected by
# where the origin sits. Only the direction matrix has to be right, and V4.1
# tests that directly wherever an annotated pair exists.
# ---------------------------------------------------------------------------
SUBSITE_WORDS = ("head", "body", "tail")

# What sub-site vocabulary each organ actually has. A kidney has poles, a
# liver has Couinaud segments, and neither has a head or a tail; a record that
# prints one is describing a pancreas it does not have.
ORGAN_SUBSITE_VOCABULARY = {
    "pancreas": {"allowed": ("head", "body", "tail", "unknown"),
                 "why": "the pancreas is described head / body / tail along "
                        "the gland's right-to-left long axis"},
    "kidney": {"allowed": ("upper pole", "interpolar", "lower pole",
                           "unknown"),
               "why": "the kidney has no head, body or tail; its regions are "
                      "the upper pole, the interpolar region and the lower "
                      "pole"},
    "liver": {"allowed": ("unknown",),
              "why": "the liver's sub-sites are the Couinaud segments I-VIII, "
                     "defined by portal and hepatic venous anatomy, and "
                     "nothing in this pipeline computes them"},
    "spleen": {"allowed": ("unknown",),
               "why": "the spleen is not sub-divided in this schema"},
}


def _side_of(mask, affine, mid):
    """Where a structure sits relative to the body midline, in world x."""
    prof = _world_profile(mask, affine, 0)
    if prof is None or int(prof[1].sum()) == 0:
        return None
    lo, hi = _wspan(prof)
    c = _wmean(prof)
    return {"voxels": int(prof[1].sum()),
            "centroid_x_mm": _num(c, 1),
            "offset_from_midline_mm": _num(c - mid, 1),
            "min_x_mm": _num(lo, 1), "max_x_mm": _num(hi, 1),
            "straddles_midline": bool(lo is not None and lo < mid < hi),
            "side": "right" if c > mid else "left"}


def v4_anatomy(ctx, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    cid, ds = ctx["cid"], ctx["dataset"]
    A, ref, ct = ctx["pred_affine"], ctx["ref"], ctx["ct"]
    out = []

    # V4.0  the anchor everything else is measured against: the body midline,
    # taken from the CT rather than from world x = 0, because a zeroed origin
    # would otherwise put the whole patient on one side of "the midline".
    body = ct > -500.0
    bprof = _world_profile(body, A, 0)
    if bprof is None or int(bprof[1].sum()) == 0:
        out.append(_r(cid, "V4.0", "a body midline can be measured from the "
                      "CT", SKIP, {"axcodes": ctx["pred_axcodes"]},
                      "an axis-aligned grid with tissue in it",
                      note="the grid is not axis-aligned, so a world "
                           "coordinate cannot be read off a one-dimensional "
                           "profile and V4 does not run"))
        return out
    mid = _wpct(bprof, 50)
    blo, bhi = _wspan(bprof)
    out.append(_r(
        cid, "V4.0", "the body midline, measured from the CT", PASS,
        {"rule": "median world x of every voxel above -500 HU; the couch and "
                 "the arms are left in because both are near-symmetric in x "
                 "and neither moves a median by a centimetre",
         "midline_x_mm": _num(mid, 1),
         "body_x_range_mm": [_num(blo, 1), _num(bhi, 1)],
         "body_width_mm": _num((bhi - blo) if None not in (blo, bhi) else None,
                               1),
         "axcodes": ctx["pred_axcodes"],
         "world_origin_mm": [_num(v, 1) for v in A[:3, 3]],
         "origin_is_zeroed": bool(np.allclose(A[:3, 3], 0.0)),
         "why_this_is_still_valid_with_a_zeroed_origin":
             "every V4 test compares two things measured through this same "
             "affine, so a shifted origin cancels; only the direction matrix "
             "has to be right, and V4.1 tests that where a pair exists"},
        "a midline inside the body's own x range"))

    # V4.1  laterality against an independent anchor. The strong form of this
    # test needs a pair of structures the DATASET labelled as left and right;
    # where the sides were computed by us from the same affine the test is
    # circular and says so instead of passing.
    geometric_split = any(int(v) < 0 for v in MAPS.get(
        {"kits": "KiTS23", "lits": "LiTS", "msd": "MSD Pancreas",
         "flare23": "FLARE23"}[ds], {}).values())
    rk, lk = ref == 2, ref == 13
    if not (rk.any() and lk.any()):
        out.append(_r(cid, "V4.1", "the right kidney is at greater world x "
                      "than the left", SKIP,
                      {"right_kidney_voxels": int(rk.sum()),
                       "left_kidney_voxels": int(lk.sum())},
                      "both kidneys annotated in the reference",
                      note="this case's reference carries no annotated "
                           "left/right pair"))
    elif geometric_split:
        rs, ls = _side_of(rk, A, mid), _side_of(lk, A, mid)
        straddle = bool(rs and ls and rs["side"] == "right"
                        and ls["side"] == "left")
        out.append(_r(
            cid, "V4.1", "the kidney pair straddles the body midline",
            PASS if straddle else FAIL,
            {"right_kidney": rs, "left_kidney": ls,
             "midline_x_mm": _num(mid, 1),
             "why_not_the_stronger_test":
                 "this dataset annotates ONE kidney class, so right and left "
                 "were assigned by us from this same affine. Asking whether "
                 "the right one has the larger x would only re-read our own "
                 "rule. What is NOT circular is whether the two masses fall "
                 "on opposite sides of a midline measured from the CT, which "
                 "is what a split of one organ into two would fail."},
            "the two kidney masses on opposite sides of the CT midline"))
    else:
        rs, ls = _side_of(rk, A, mid), _side_of(lk, A, mid)
        gap = (rs["centroid_x_mm"] - ls["centroid_x_mm"]) if rs and ls else None
        out.append(_r(
            cid, "V4.1", "the right kidney is at greater world x than the "
            "left", PASS if (gap is not None and gap > 0) else FAIL,
            {"right_kidney": rs, "left_kidney": ls,
             "right_minus_left_x_mm": _num(gap, 1),
             "anchor": "the dataset's own right/left kidney annotations, "
                       "which our pipeline never computed",
             "convention": "NIfTI RAS+, so greater world x is further toward "
                           "the patient's right"},
            "a positive right-minus-left offset in world x",
            note="a negative offset means the affine's assertion about which "
                 "way is right disagrees with the dataset's own annotation, "
                 "and every laterality word in the app is then mirrored"))

    # V4.2  organ position. Each annotated organ on the side of the midline
    # that organ has in every human being.
    expect = {"liver": "right", "spleen": "left", "right_kidney": "right",
              "left_kidney": "left"}
    rows, bad = {}, []
    for name, ids, kind in ctx["structures"]:
        if name not in expect and name != "pancreas":
            continue
        m = _mask_of(ref, ids)
        if not m.any():
            continue
        s = _side_of(m, A, mid)
        if name == "pancreas":
            # The gland crosses the midline, so the test is that it does, and
            # that its centroid is not out at the flank.
            ok = bool(s["straddles_midline"]
                      and abs(s["offset_from_midline_mm"]) <= 60.0)
            s["expected"] = "straddling the midline, centroid within 60 mm"
        else:
            ok = s["side"] == expect[name]
            s["expected"] = expect[name]
        s["agrees"] = ok
        rows[name] = s
        if not ok:
            bad.append(name)
    out.append(_r(
        cid, "V4.2", "every annotated organ is on the side of the body it "
        "belongs on", PASS if not bad else FAIL,
        {"midline_x_mm": _num(mid, 1), "per_structure": rows,
         "reference": ctx["ref_variant"]["label"]},
        "liver and right kidney right of the midline, spleen and left kidney "
        "left of it, pancreas straddling it",
        note=None if not bad else "disagreeing: %s" % ", ".join(bad)))

    # V4.3  the two orderings that do not depend on a midline at all: the
    # liver dome sits above the kidney, and the kidneys sit behind the liver.
    # Both are comparisons between two organs through one affine, so they
    # survive a zeroed origin and an unusual patient position.
    def _cen(ids):
        m = _mask_of(ref, ids)
        return (_world_centroid(m, A) if m.any() else None)

    liver = _cen((1,))
    rkid = _cen((2,))
    lkid = _cen((13,))
    spleen = _cen((3,))
    pairs, bad3 = {}, []
    def _order(nm, a, b, comp, why, want_positive=True):
        if a is None or b is None:
            return
        d = a[comp] - b[comp]
        ok = (d > 0) if want_positive else (d < 0)
        pairs[nm] = {"difference_mm": _num(d, 1), "agrees": bool(ok),
                     "component": "xyz"[comp], "expected": why}
        if not ok:
            bad3.append(nm)
    _order("liver_above_right_kidney", liver, rkid, 2,
           "the liver centroid superior to the right kidney centroid")
    _order("spleen_above_left_kidney", spleen, lkid, 2,
           "the spleen centroid superior to the left kidney centroid")
    _order("right_kidney_behind_liver", rkid, liver, 1,
           "the retroperitoneal kidney posterior to the liver centroid",
           want_positive=False)
    if pairs:
        out.append(_r(
            cid, "V4.3", "the organ pairs are in the anatomical order",
            PASS if not bad3 else FAIL,
            {"per_pair": pairs,
             "centroids_world_mm": {
                 "liver": [_num(v, 1) for v in liver] if liver else None,
                 "right_kidney": [_num(v, 1) for v in rkid] if rkid else None,
                 "left_kidney": [_num(v, 1) for v in lkid] if lkid else None,
                 "spleen": [_num(v, 1) for v in spleen] if spleen else None},
             "convention": "world y increases anteriorly and world z "
                           "superiorly in NIfTI RAS+"},
            "every pair in the order anatomy puts it in",
            note=None if not bad3 else "disagreeing: %s" % ", ".join(bad3)))
    else:
        out.append(_r(cid, "V4.3", "the organ pairs are in the anatomical "
                      "order", SKIP, {"available_pairs": 0},
                      "two annotated organs to order",
                      note="this case's reference annotates one organ and its "
                           "lesion, so there is no pair to order"))

    # V4.4  the printed sub-site word is a word the organ actually has. This
    # is not a measurement, it is a vocabulary check, and it is the one that
    # fires on a kidney called "head" and a liver called "body" without
    # looking at a single voxel.
    rec = FC.record(cid, "containment_v2") or {}
    vocab_rows, vocab_bad = {}, []
    for organ, b in sorted((rec.get("organs") or {}).items()):
        word = b.get("anatomic_location")
        voc = ORGAN_SUBSITE_VOCABULARY.get(organ)
        allowed = voc["allowed"] if voc else ("unknown",)
        ok = word in allowed
        vocab_rows[organ] = {"printed": word, "allowed": list(allowed),
                             "agrees": bool(ok),
                             "why": (voc or {}).get("why")}
        if not ok:
            vocab_bad.append("%s: %r" % (organ, word))
    out.append(_r(
        cid, "V4.4", "the printed sub-site is a sub-site the organ has",
        PASS if not vocab_bad else FAIL,
        {"record": "record_containment_v2.json", "per_organ": vocab_rows},
        "every anatomic_location inside its organ's own vocabulary",
        note=None if not vocab_bad else
             "outside the organ's vocabulary: %s" % "; ".join(vocab_bad),
        known=None if not vocab_bad else
              "ipkg_corrections.json voids the liver and kidney sub-site "
              "records for exactly this reason (void_reasons); the app shows "
              "the correction as a labelled overlay and the delivered record "
              "is unchanged on disk"))

    # V4.5  the pancreatic sub-site against a structure our axis never
    # touched. Both rules are computed on the same masks so that a difference
    # cannot be blamed on the mask, and the duodenum decides which end of the
    # gland is the head.
    panc = (rec.get("organs") or {}).get("pancreas") or {}
    if not (panc.get("has_tumor") and (ref == 4).any() and (ref == 14).any()):
        out.append(_r(
            cid, "V4.5", "the printed pancreatic sub-site survives an "
            "independent anchor", SKIP,
            {"record_pancreas_has_tumor": panc.get("has_tumor"),
             "record_pancreas_anatomic_location":
                 panc.get("anatomic_location"),
             "reference_pancreas_voxels": int((ref == 4).sum()),
             "reference_tumour_voxels": int((ref == 14).sum())},
            "a pancreatic lesion to place",
            note="head / body / tail is a statement about where a lesion sits "
                 "in the gland, so it is only testable where the record gives "
                 "the PANCREAS a lesion. This case's lesion is in another "
                 "organ, and V4.4 has already checked that the word printed "
                 "for that organ is a word the organ has."))
    else:
        gland = (ref == 4) | (ref == 14)
        gidx = np.argwhere(gland)
        tidx = np.argwhere(ref == 14)
        pidx = np.argwhere(ref == 4)
        off_d, word_d = SF.delivered_rule(gland, ref == 14,
                                          SF.DELIVERED_AX.get(ds, 2))
        corr = SF.corrected_rule(gidx, tidx, A, pidx)
        anchor = SF.duodenum_anchor(gidx, A, SF.ts_pred_path(cid))
        printed = ((rec.get("organs") or {}).get("pancreas")
                   or {}).get("anatomic_location")
        anchored = corr.get("label") if (
            anchor.get("orientation") == "confirmed"
            and corr.get("guard") == "passed") else None
        ok = bool(anchored and printed == anchored)
        out.append(_r(
            cid, "V4.5", "the printed pancreatic sub-site survives an "
            "independent anchor", PASS if ok else FAIL,
            {"printed": printed,
             "delivered_rule": {
                 "axis": SF.DELIVERED_AX.get(ds, 2),
                 "what_that_axis_is": "the acquisition slice axis of this "
                                      "dataset, which is inferior-superior "
                                      "here and not the gland's long axis",
                 "offset": _num(off_d, 4), "word": word_d},
             "corrected_rule": corr,
             "independent_anchor": anchor,
             "anchored_word": anchored,
             "anchor_description":
                 "TotalSegmentator v2 duodenum (label 19): the pancreatic "
                 "head is the end of the gland the duodenum cradles, and the "
                 "kidney pair in the same prediction doubles as a mirror test "
                 "on the affine itself"},
            "the printed word equal to the word the anchored world-x rule "
            "gives",
            known=None if ok else
                  "ipkg_corrections.json records the same correction for this "
                  "case (build_predicted_corpus.py:128-135 measures the "
                  "sub-site along the axial slice axis); the app shows it as "
                  "a labelled overlay"))
    return out


# ---------------------------------------------------------------------------
# V9  scan-quality assertions
# ---------------------------------------------------------------------------
def v9_scan_quality(ctx, verbose=True):
    cid, ds = ctx["cid"], ctx["dataset"]
    pred, ref, ct, sp = ctx["pred"], ctx["ref"], ctx["ct"], ctx["spacing"]
    out = []

    # V9.1  label exclusivity. A single-valued integer volume gives one label
    # per voxel for free, so the content of this check is the pair of
    # invariants the semi-oracle protocol implies: the network was prompted
    # from the reference, so it cannot draw a structure the reference does not
    # carry, and a reference structure with no prediction at all is a
    # structure the network was never asked about.
    pids = set(int(v) for v in np.unique(pred)) - {0}
    rids = set(int(v) for v in np.unique(ref)) - {0}
    extra = sorted(pids - rids)
    absent = sorted(rids - pids)
    out.append(_r(
        cid, "V9.1", "the drawn labels are one per voxel and none of them is "
        "a structure the reference does not carry",
        PASS if (pred.ndim == 3 and not extra) else FAIL,
        {"prediction_ndim": int(pred.ndim),
         "prediction_label_ids": sorted(pids),
         "reference_label_ids": sorted(rids),
         "predicted_but_not_in_the_reference": extra,
         "predicted_but_not_in_the_reference_names":
             [UNIFIED.get(v) for v in extra],
         "in_the_reference_but_never_predicted": absent,
         "in_the_reference_but_never_predicted_names":
             [UNIFIED.get(v) for v in absent],
         "in_the_reference_but_never_predicted_voxels":
             {str(v): int((ref == v).sum()) for v in absent}},
        "a three-dimensional single-valued volume whose label set is a subset "
        "of the reference's",
        note="a structure in the reference and never predicted is reported "
             "rather than failed: it means the network was not run on that "
             "structure, which is a scope statement, not a defect"))

    # V9.2  the lesion is inside the organ the record says hosts it. The
    # attribution rule is the pipeline team's own - each lesion component goes to the ONE
    # organ whose three-voxel dilation it overlaps most - so a disagreement
    # here is a disagreement about anatomy and not about the rule.
    tum = ref == 14
    if not tum.any():
        out.append(_r(cid, "V9.2", "every lesion sits in the organ the record "
                      "names", SKIP, {"reference_tumour_voxels": 0},
                      "a lesion in the reference",
                      note="this case's reference carries no lesion"))
    else:
        present = {o: _mask_of(ref, ids)
                   for o, ids in RECORD_ORGAN_LABELS.items()
                   if _mask_of(ref, ids).any()}
        dil = {o: ndimage.binary_dilation(m, iterations=3)
               for o, m in present.items()}
        lab, ncomp = ndimage.label(tum, structure=_S26)
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        order = np.argsort(sizes)[::-1]
        comps = []
        for ci in order[:12]:
            ci = int(ci)
            if sizes[ci] <= 0:
                continue
            comp = lab == ci
            ov = {o: int((comp & d).sum()) for o, d in dil.items()}
            host = max(ov, key=ov.get) if ov else None
            if host is not None and ov[host] == 0:
                host = None
            touch = None
            if host is not None:
                shell = ndimage.binary_dilation(comp, structure=_S6) & ~comp
                touch = _num(float((shell & present[host]).sum())
                             / max(1, int(shell.sum())), 4)
            comps.append({
                "voxels": int(sizes[ci]),
                "cm3": _num(_cm3(int(sizes[ci]), sp), 2),
                "overlap_with_each_organ_dilated_3vox": ov,
                "attributed_host": host,
                "fraction_of_the_lesion_shell_that_is_host_parenchyma": touch,
                "fraction_of_the_component_inside_the_host_dilation":
                    _num(float(ov[host]) / max(1, int(sizes[ci])), 4)
                    if host else None})
        rec_organs = [o for o, b in ((FC.record(cid, "containment_v2") or {})
                                     .get("organs") or {}).items()
                      if b.get("has_tumor")]
        hosts = sorted({c["attributed_host"] for c in comps
                        if c["attributed_host"]})
        orphan = [c for c in comps if not c["attributed_host"]]
        agrees = bool(hosts and (not rec_organs or set(hosts) <= set(rec_organs)))
        out.append(_r(
            cid, "V9.2", "every lesion sits in the organ the record names",
            PASS if (agrees and not orphan) else FAIL,
            {"reference_tumour_voxels": int(tum.sum()),
             "components_26_connected": int(ncomp),
             "largest_components": comps,
             "attributed_hosts": hosts,
             "organs_the_record_gives_a_lesion": sorted(rec_organs),
             "components_attributed_to_no_organ": len(orphan),
             "voxels_attributed_to_no_organ":
                 int(sum(c["voxels"] for c in orphan)),
             "why_an_orphan_matters":
                 "a lesion component that touches no organ is still counted "
                 "when a record decides between solitary and multifocal, so "
                 "a two-voxel speck can be the whole of the difference "
                 "between those two words on screen; V3.1 says whether it "
                 "was",
             "rule": "each lesion component to the ONE organ whose 3-voxel "
                     "dilation it overlaps most, verbatim from "
                     "ipkg_delivered/code/extract_phenotypes.py"},
            "every component attributed to an organ, and every attributed "
            "organ one the record gives a lesion to",
            note="the shell fraction is reported beside the verdict because a "
                 "large exophytic lesion legitimately has most of its surface "
                 "outside the parenchyma; what would be a defect is a lesion "
                 "that touches no host at all"))

    # V9.3  attenuation sanity. The cheapest possible test that a mask and a
    # CT still belong together: a structure whose median attenuation has moved
    # into fat, gas or bone is not a structure, it is a misalignment.
    rows, bad, shifts = {}, [], {}
    for name, ids, kind in ctx["structures"]:
        pair = {}
        for which, arr in (("reference", ref), ("prediction", pred)):
            m = _mask_of(arr, ids)
            st = _hu_stats(ct, m)
            if st is None:
                continue
            band = (HU_LESION_MEDIAN_BAND if kind in ("tumour", "cyst")
                    else HU_ORGAN_MEDIAN_BAND)
            ok = (band[0] <= st["median_hu"] <= band[1]
                  and st["air_frac"] <= HU_AIR_FRAC_MAX
                  and st["bone_frac"] <= HU_BONE_FRAC_MAX)
            st["median_band_hu"] = list(band)
            st["agrees"] = bool(ok)
            rows["%s/%s" % (name, which)] = st
            pair[which] = st
            if not ok:
                bad.append("%s/%s median %s HU, air %.3f, bone %.3f"
                           % (name, which, st["median_hu"], st["air_frac"],
                              st["bone_frac"]))
        if "reference" in pair and "prediction" in pair:
            shifts[name] = {
                "reference_median_hu": pair["reference"]["median_hu"],
                "prediction_median_hu": pair["prediction"]["median_hu"],
                "shift_hu": _num(pair["prediction"]["median_hu"]
                                 - pair["reference"]["median_hu"], 1)}
    out.append(_r(
        cid, "V9.3", "every structure's attenuation is plausible for that "
        "structure on this CT", PASS if not bad else FAIL,
        {"ct": _rel(ctx["ct_path"]),
         "organ_median_band_hu": list(HU_ORGAN_MEDIAN_BAND),
         "lesion_median_band_hu": list(HU_LESION_MEDIAN_BAND),
         "max_air_fraction": HU_AIR_FRAC_MAX,
         "max_bone_fraction": HU_BONE_FRAC_MAX,
         "air_below_hu": HU_AIR, "bone_at_or_above_hu": HU_BONE,
         "per_structure": rows},
        "a soft-tissue median, with almost no gas and almost no bone inside "
        "any structure",
        note="the bands are wide on purpose: the four cases are in different "
             "contrast phases, and a band tight enough to separate "
             "portal-venous liver from non-contrast liver would fire on the "
             "phase instead of on a defect",
        **({"failing": bad} if bad else {})))

    # V9.4  the same structure, drawn and referenced, on the same CT. This is
    # not a restatement of Dice: a structure can have a poor Dice and still be
    # drawing the right tissue, because a boundary error moves the outline
    # without changing what is inside it. A median attenuation that MOVES
    # between the reference and the prediction of one structure says something
    # different - that the two outlines contain different tissue.
    judged = set(_verdict_structures(ctx))
    off = [n for n, v in shifts.items()
           if n in judged and v["shift_hu"] is not None
           and abs(v["shift_hu"]) > 40.0]
    off_all = [n for n, v in shifts.items()
               if v["shift_hu"] is not None and abs(v["shift_hu"]) > 40.0]
    out.append(_r(
        cid, "V9.4", "the drawn structure and the reference structure contain "
        "the same tissue", PASS if not off else FAIL,
        {"per_structure": shifts, "max_shift_hu": 40.0,
         "structures_judged": sorted(judged),
         "structures_over_the_threshold": off_all,
         "structures_over_the_threshold_and_judged": off,
         "why_this_is_not_dice":
             "a boundary error moves an outline without changing what is "
             "inside it, so a low Dice with a flat median is a shape "
             "disagreement; a moved median is a tissue disagreement, which is "
             "what a mis-assigned component or a shifted mask looks like"},
        "a median attenuation within 40 HU of the reference structure's",
        note="per-side kidney rows are reported and not judged where the "
             "sides are a computed split"))
    return out


# ---------------------------------------------------------------------------
# V4.6  the population-level test
#
# The per-case sub-site check needs a per-case anchor, and there are 56 MSD
# records and only one TotalSegmentator set. The corpus test needs nothing at
# all: a pancreatic-lesion corpus in which the tail outnumbers the head is
# not a corpus of pancreatic cancer, whatever the individual cases say. It
# would have fired on the very first corpus that was built, before any case
# was ever pinned, and it costs one pass over a JSON file.
# ---------------------------------------------------------------------------
def _corpus_files():
    out = []
    cdir = getattr(FC, "CORPORA_DIR", None)
    if cdir and os.path.isdir(cdir):
        for fn in sorted(os.listdir(cdir)):
            if fn.endswith(".json"):
                out.append(os.path.join(cdir, fn))
    return out


def v4_corpus(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    out, seen = [], {}
    for path in _corpus_files():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as exc:                              # noqa: BLE001
            out.append(_r("corpus", "V4.6/%s" % os.path.basename(path),
                          "the corpus sub-site distribution is "
                          "epidemiologically possible", SKIP,
                          {"error": str(exc)}, "a readable corpus file"))
            continue
        recs = doc.get("records") or []
        tally = {}
        # Every lesion row is accounted for, in one of four states, and the
        # three that do not reach the distribution test are COUNTED rather
        # than skipped. See SUBSITE_STATUS_* above: "na" is one spelling for
        # four different statements and the reason is in a field beside it.
        reach = {}
        for rec in recs:
            ds = rec.get("dataset")
            for organ, b in (rec.get("organs") or {}).items():
                if not b.get("has_tumor"):
                    continue
                key = (ds, organ)
                state, status, word = _subsite_state(b)
                rk = reach.setdefault(key, {"measured": 0, "absent": 0,
                                            "unmeasured": 0, "unstated": 0,
                                            "statuses": {}})
                rk[state] += 1
                rk["statuses"][str(status)] = \
                    rk["statuses"].get(str(status), 0) + 1
                if state != "measured":
                    continue
                tally.setdefault(key, {})
                tally[key][word] = tally[key].get(word, 0) + 1
        base = os.path.basename(path)
        seen[base] = {
            "records": len(recs),
            "lesion_rows_with_a_sub_site_word":
                {"%s/%s" % k: sum(v.values()) for k, v in sorted(tally.items())},
            "lesion_rows_by_state":
                {"%s/%s" % k: {s: v[s] for s in
                               ("measured", "absent", "unmeasured", "unstated")}
                 for k, v in sorted(reach.items())}}

        # ------------------------------------------------------------------
        # V4.6/.../reach - WHAT THE DISTRIBUTION TEST COULD SEE.
        #
        # The test below only ever reads rows in state "measured". That is
        # correct - you cannot test a distribution of words against rows that
        # carry no word - but it is only HALF a statement, and the other half
        # is the one that went missing: how many lesion rows it could not see,
        # and why. An organ that HAS head / body / tail and whose lesion rows
        # are all unmeasured contributes nothing, and before this row existed
        # it contributed nothing SILENTLY.
        # ------------------------------------------------------------------
        for (ds, organ), rk in sorted(reach.items()):
            voc = ORGAN_SUBSITE_VOCABULARY.get(organ, {})
            has_axis = bool(set(voc.get("allowed", ())) & {"head", "body",
                                                           "tail"})
            total = sum(rk[s] for s in ("measured", "absent", "unmeasured",
                                        "unstated"))
            blind = rk["unmeasured"] + rk["unstated"]
            # An organ with no sub-site axis is SUPPOSED to contribute
            # nothing, and saying so is the point: that is the void working.
            ok = (not has_axis) or (rk["measured"] > 0 and blind == 0)
            out.append(_r(
                "corpus", "V4.6/%s/%s/%s/reach" % (base, ds, organ),
                "the population test can see every lesion row on an organ "
                "that has a sub-site",
                PASS if ok else FAIL,
                {"corpus": base, "dataset": ds, "organ": organ,
                 "lesion_rows": total,
                 "organ_has_a_sub_site_axis": has_axis,
                 "measured_and_tested": rk["measured"],
                 "absent_not_applicable": rk["absent"],
                 "unmeasured_never_derived_or_guard_refused":
                     rk["unmeasured"],
                 "unstated_no_status_field": rk["unstated"],
                 "anatomic_location_status_values": rk["statuses"],
                 "why_this_row_exists":
                     "the pipeline team spells all four reasons for a void 'na' and puts "
                     "the reason in anatomic_location_status. A test that "
                     "skips on the VALUE cannot tell 'this organ has no "
                     "sub-site' from 'nobody measured this organ's "
                     "sub-site', and the second one is a gap that a "
                     "population test must declare rather than absorb."},
                "every lesion row on an organ with a sub-site axis either "
                "measured, or declared here",
                known=None if ok else
                      "the pipeline team README_ROUND3.md section 5: 'Optional, not "
                      "done: 223 FLARE23 pool records have a pancreatic "
                      "tumor whose sub-site was never derived. The rule "
                      "could be run on their reference labels if the "
                      "advisor wants FLARE23 candidates to carry a site.' "
                      "The GAP is the pipeline team's and is documented; what was not "
                      "documented is that this suite was silently skipping "
                      "the rows, which is fixed by this row existing."))

        for (ds, organ), counts in sorted(tally.items()):
            n = sum(counts.values())
            pct = {k: counts[k] / n for k in counts}
            voc = ORGAN_SUBSITE_VOCABULARY.get(organ, {})
            allowed = voc.get("allowed", ("unknown",))
            wrong_vocab = sorted(set(counts) - set(allowed))
            if wrong_vocab:
                out.append(_r(
                    "corpus", "V4.6/%s/%s/%s/vocabulary" % (base, ds, organ),
                    "the corpus uses only sub-site words the organ has",
                    FAIL,
                    {"corpus": base, "dataset": ds, "organ": organ,
                     "rows": n, "counts": counts,
                     "words_outside_the_vocabulary": wrong_vocab,
                     "allowed": list(allowed), "why": voc.get("why")},
                    "no head / body / tail on an organ that has none",
                    known="ipkg_corrections.json voids these rows "
                          "(void_reasons); this row is the population-scale "
                          "form of the same defect"))
                continue
            if organ != "pancreas":
                continue
            head = pct.get("head", 0.0)
            body = pct.get("body", 0.0)
            tail = pct.get("tail", 0.0)
            modal = max(pct, key=pct.get)
            ok = bool(modal == "head" and head >= PDAC_HEAD_FLOOR
                      and head > tail)
            out.append(_r(
                "corpus", "V4.6/%s/%s/%s" % (base, ds, organ),
                "the corpus pancreatic sub-site distribution is "
                "epidemiologically possible", PASS if ok else FAIL,
                {"corpus": base, "dataset": ds, "organ": organ,
                 "lesion_rows": n, "counts": counts,
                 "head_pct": _num(100.0 * head, 1),
                 "body_pct": _num(100.0 * body, 1),
                 "tail_pct": _num(100.0 * tail, 1),
                 "modal_sub_site": modal,
                 "accepted_distribution": {
                     "head": [_num(100 * PDAC_HEAD_BAND[0], 0),
                              _num(100 * PDAC_HEAD_BAND[1], 0)],
                     "body": [_num(100 * PDAC_BODY_BAND[0], 0),
                              _num(100 * PDAC_BODY_BAND[1], 0)],
                     "tail": [_num(100 * PDAC_TAIL_BAND[0], 0),
                              _num(100 * PDAC_TAIL_BAND[1], 0)],
                     "source": "the accepted clinical distribution of "
                               "pancreatic ductal adenocarcinoma by "
                               "sub-site: roughly 60-70 per cent head, 15 "
                               "per cent body, 15 per cent tail"},
                 "gate": "the head must be the modal sub-site, must hold at "
                         "least %d per cent of the lesions and must beat the "
                         "tail. The gate is far looser than the accepted "
                         "band on purpose: the point is not to enforce a "
                         "textbook histogram on a research corpus, it is to "
                         "fail a corpus that has the two words swapped."
                         % int(100 * PDAC_HEAD_FLOOR)},
                "the head modal and above %d per cent"
                % int(100 * PDAC_HEAD_FLOOR),
                known=None if ok else
                      "ipkg_corrections.json records the corrected "
                      "distribution for the same rows (head 66-68 per cent "
                      "after the world-x rule), with 47/47 TotalSegmentator "
                      "duodenum agreement"))
    if seen:
        out.append(_r("corpus", "V4.6/inventory",
                      "what the population test looked at", PASS, seen,
                      "reported"))
    return out


# ---------------------------------------------------------------------------
# V11  THE DISPLAY CONTRACT
#
# V1-V9 ask whether a number is true. V11 asks the three questions that were
# missed this week, all of which are about whether a true number is
# READABLE - because a correct figure whose definition, denominator or
# epistemic status is not beside it is not a finding, it is a guess the
# reader has to make.
#
#   V11.1  every displayed organ Dice names its convention. Two conventions
#          were in play until 2026-10-02 and they differ by up to 0.48 on one
#          of our own cases (case_00067 kidney: 0.4919 under A, 0.9701 under
#          B - see ipkg_organ_convention.py). A bare "kidney 0.8109" is not
#          a measurement, it is one of several.
#   V11.2  no surface states a pool size other than the live one. The pool
#          has been re-counted six times in three weeks and the number is
#          drawn INSIDE the figure images.
#   V11.3  a withheld claim never renders as a negative finding. "We did not
#          measure this organ" and "this organ is clear" are different
#          sentences and only one of them is a clinical claim.
#
# These are population-level and artefact-level, not per-case, so V11 runs
# once beside V4.6 rather than inside run_case.
# ---------------------------------------------------------------------------

# The three readings an organ figure can be, plus the two names the project
# publishes them under. A display NAMES its convention when one of these
# appears in the cell, or in the section the figure sits in.
_CONV_RE = re.compile(
    r"conv(?:ention)?\.?\s*[ab]\b|organ[-\s]with[-\s]mass|"
    r"tum(?:ou)?r[-\s]excluded|parenchyma|lesion[-\s]as[-\s]ignore[-\s]region|"
    r"ipkg_organ_convention", re.I)
_ORGAN_W = r"(?:organ|liver|kidney|spleen|pancreas|parenchyma)"
_ORGAN_RE = re.compile(_ORGAN_W, re.I)
_DICE_RE = re.compile(r"\bdice\b", re.I)
_FIG_RE = re.compile(r"(?<![\d.])0\.\d{3,4}\b")
# A CELL THAT NAMES A STRUCTURE, as opposed to a cell that merely contains
# an organ's letters. "MSD Pancreas" is a collection and "pancreas_125" is a
# case id, and both sat in TUMOUR-Dice tables that the row rule reported as
# unnamed organ figures on its first run. A check that cries about a table
# it has misread is a check that gets switched off, so the cell has to BE a
# structure name and not contain one.
_IS_STRUCTURE = re.compile(
    r"^\**\s*(?:the\s+)?(?:left|right|pooled)?\s*"
    r"(?:liver|kidney|spleen|pancreas|organ|parenchyma)"
    r"(?:\s+(?:pooled|parenchyma|dice))?\s*\**$", re.I)
# A table whose every Dice column is about the LESION is not an organ-Dice
# table, whatever words appear in its rows.
_TUMOUR_W = re.compile(r"tum(?:ou)?r|lesion", re.I)
# A figure in a table body with the structure named IN THE SAME CELL, which
# is how the prose surfaces write it: "liver **0.9449**". Bounded by | so it
# cannot reach across a column.
_ORGAN_FIG = re.compile(_ORGAN_W + r"[^|]{0,40}?\**\s*0\.\d{3,4}", re.I)
# The same thing in prose: "<organ> Dice ... 0.xxxx".
_ORGAN_PROSE = re.compile(_ORGAN_W + r"\s+dice[^.\n]{0,120}?0\.\d{3,4}", re.I)
_MD_SEP = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")

# WHICH SURFACES ARE SWEPT. Everything a reader of this repository opens:
# the manuscript table sets, the paper memos and the top-level READMEs.
# validation/ is excluded because it is this suite's own output and quoting
# a figure back is what a report does.
_SURFACE_GLOBS = ("tables/**/*.md", "paper/**/*.md", "*.md")

# A DOCUMENT OLDER THAN THE CONVENTION CANNOT NAME IT. Convention A was
# adopted on the date in ipkg_organ_convention.CONVENTION["adopted"]; a memo
# dated before that in its own filename is a record of what was known then,
# and holding it to a rule that did not exist would turn this check into a
# nag nobody acts on. The exemption is DERIVED from the filename date and
# the adoption date rather than listed, so it cannot quietly grow into an
# allowlist, and every exempted file is still reported by name.
_DATED = re.compile(r"(20\d\d)-(\d\d)-(\d\d)")
CONVENTION_ADOPTED = "2026-10-02"


def _surface_files():
    import glob                                               # noqa: PLC0415
    out = []
    for pat in _SURFACE_GLOBS:
        out += glob.glob(os.path.join(BASE, pat), recursive=True)
    keep = []
    for p in sorted(set(out)):
        rel = os.path.relpath(p, BASE).replace("\\", "/")
        if rel.startswith("validation/") or "/superseded/" in rel:
            continue
        keep.append((rel, p))
    return keep


def _predates_the_convention(rel):
    m = _DATED.search(os.path.basename(rel))
    return bool(m) and m.group(0) < CONVENTION_ADOPTED


def _organ_dice_displays(lines):
    """Every place an organ Dice figure is shown, with its section text.

    Yields (line_number, snippet, section_text). A table is read as a unit -
    header plus body - because the convention is usually named in the header
    cell or in the paragraph above, and a body cell read on its own would
    report every well-captioned table as a defect.
    """
    sec_start, sec_of = 0, {}
    for i, ln in enumerate(lines):
        if ln.startswith("#"):
            sec_start = i
        sec_of[i] = sec_start
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if (s.startswith("|") and i + 1 < len(lines)
                and _MD_SEP.match(lines[i + 1])):
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            # Does this table have an ORGAN Dice column at all? A header
            # whose only Dice cells say tumour or lesion is a lesion table
            # and the row rule below must not walk its rows.
            hcells = [c.strip() for c in s.strip().strip("|").split("|")]
            dice_cells = [c for c in hcells if _DICE_RE.search(c)]
            organ_dice_col = any(_ORGAN_RE.search(c) or
                                 not _TUMOUR_W.search(c)
                                 for c in dice_cells)
            if _DICE_RE.search(s):
                section = "\n".join(lines[sec_of[i]:j])
                for k in range(i, j):
                    m = _ORGAN_FIG.search(lines[k])
                    if m:
                        yield k + 1, m.group(0).strip()[:80], section
                        continue
                    # THE ROW FORM, which is how the manuscript tables write
                    # it: "| FLARE23 | liver | 270 | ... | **0.9649** |".
                    # The structure and the figure are in different CELLS,
                    # so the same-cell pattern above walks straight past the
                    # whole of tables/ - and a check that silently declines
                    # to look at the surfaces that matter most is the exact
                    # defect this block was called to find elsewhere.
                    if (k > i + 1 and organ_dice_col
                            and _FIG_RE.search(lines[k])):
                        cells = [c.strip() for c in
                                 lines[k].strip().strip("|").split("|")]
                        org = next((c for c in cells
                                    if _IS_STRUCTURE.match(c)), None)
                        if org is None:
                            continue
                        fig = _FIG_RE.search(lines[k]).group(0)
                        yield k + 1, ("%s ... %s" % (org[:40], fig)), section
            i = j
            continue
        m = _ORGAN_PROSE.search(s)
        if m:
            yield i + 1, m.group(0).strip()[:110], \
                "\n".join(lines[sec_of[i]:i + 1])
        i += 1


def v11_1_organ_dice_convention():
    rows, bad, exempt, ok_n = [], [], [], 0
    for rel, path in _surface_files():
        try:
            lines = open(path, encoding="utf-8").read().splitlines()
        except Exception as exc:                              # noqa: BLE001
            rows.append(_r("surfaces", "V11.1/%s" % rel,
                           "every displayed organ Dice names its convention",
                           SKIP, {"error": str(exc)}, "a readable surface"))
            continue
        hits = list(_organ_dice_displays(lines))
        if not hits:
            continue
        old = _predates_the_convention(rel)
        for ln, snip, section in hits:
            if _CONV_RE.search(section):
                ok_n += 1
            elif old:
                exempt.append({"surface": rel, "line": ln, "figure": snip})
            else:
                bad.append({"surface": rel, "line": ln, "figure": snip})
    measured = {
        "surfaces_swept": len(_surface_files()),
        "organ_dice_displays_found": ok_n + len(bad) + len(exempt),
        "named_their_convention": ok_n,
        "unnamed_on_a_live_surface": len(bad),
        "unnamed_in_a_record_that_predates_the_convention": len(exempt),
        "convention_adopted": CONVENTION_ADOPTED,
        "what_counts_as_naming_it":
            "convention A or B, organ-with-mass, tumour-excluded, "
            "parenchyma, lesion-as-ignore-region, or ipkg_organ_convention, "
            "in the figure's own cell or in the section it sits in",
        "unnamed": bad,
        "exempt_because_the_file_is_dated_before_the_convention": exempt,
        "why_it_matters":
            "the same organ prediction against the same reference reads "
            "0.4919 under convention A and 0.9701 under convention B on "
            "case_00067 kidney (ipkg_organ_convention.py). A figure printed "
            "without its convention is not a measurement a reader can use."}
    rows.append(_r(
        "surfaces", "V11.1",
        "every displayed organ Dice names its convention",
        PASS if not bad else FAIL, measured,
        "every organ Dice figure on a live surface inside a section that "
        "names one of the three readings"))
    return rows


def v11_2_pool_on_every_surface():
    """One pool size, everywhere, and it is the one the files compute.

    check_pool_statements.py already owns what a pool STATEMENT looks like
    and which historical ones are declared. It is imported rather than
    re-implemented: two definitions of "the pool is stated here" would drift
    within a week, which is the defect this whole block is about.
    """
    try:
        import check_pool_statements as CPS                   # noqa: PLC0415
    except Exception as exc:                                  # noqa: BLE001
        return [_r("surfaces", "V11.2",
                   "no surface states a pool size other than the live one",
                   SKIP, {"error": str(exc)},
                   "check_pool_statements.py importable")]
    # live_pool() returns the size, the assembly and - the part that matters
    # most here - any complaint that the corpus FILES and the frozen
    # POOL_EXPECTED have drifted apart. If those two disagree, every
    # statement in the tree is being measured against a size nobody froze,
    # and no verdict below means anything, so it is reported first.
    live, assembly, drift = CPS.live_pool()
    found = CPS.sweep(live)
    stale = [f for f in found if f.get("verdict") == "stale"]
    hand = [f for f in found if f.get("verdict") == "handoff"]
    pk = CPS.packaged(live)
    fz = CPS.frozen_state(live)
    pk_bad = [r for r in pk if not r.get("ok", True)]
    fz_bad = [r for r in fz if not r.get("ok", True)]
    bad = list(drift) + stale + pk_bad + fz_bad
    measured = {
        "live_pool": live,
        "the_files_and_the_freeze_agree": not drift,
        "drift": list(drift),
        "how_the_live_pool_is_obtained":
            "check_pool_statements.live_pool(), which counts the corpus "
            "FILES rather than reading any constant",
        "assembly": assembly,
        "statements_that_are_not_the_live_size_and_nobody_owns": len(stale),
        "statements_owned_by_another_workstream": len(hand),
        "packaged_statements_that_do_not_match_their_declaration":
            len(pk_bad),
        "frozen_state_rows_that_do_not_match": len(fz_bad),
        "stale": [{"surface": f["path"], "line": f["line"],
                   "states": f["stated"], "text": (f.get("text") or "")[:120]}
                  for f in stale],
        "handoff_owners": sorted({f.get("owner") or "?" for f in hand}),
        "note_on_the_rendered_page":
            "this row reads FILES. The pool size as RENDERED is asserted by "
            "ipkg_ui_walk A7, which reads the DOM of the running build, "
            "because segx_testcase caches the graph per process and a "
            "server started before a corpus swap serves the old pool for "
            "the life of the process with a correct footer beside it."}
    return [_r("surfaces", "V11.2",
               "no surface states a pool size other than the live one",
               PASS if not bad else FAIL, measured,
               "zero unowned stale statements, zero packaged or frozen rows "
               "off their declaration")]


# V11.3 ----------------------------------------------------------------------
# A WITHHELD CLAIM IS NOT A NEGATIVE FINDING. The application's own verdict
# legend says this in as many words - "N not asserted ... withheld is not
# negative, and N never says the organ is clear" - so the contract is already
# written down and already shown to clinicians. This row asks whether the
# artefacts behind that legend keep it. There are two places they can break
# it and both are checked:
#
#   (a) the lesion-claim clean-up. R2 WITHHOLDS an organ claim it cannot
#       support. If the withholding is implemented by writing has_tumor
#       false into the record, the query suite reads it as a measurement and
#       answers F, and every withheld claim becomes a clinical negative.
#   (b) the sub-site. the pipeline team ships the REASON for every void in
#       anatomic_location_status, and a reason of not_computed means the
#       axis was never measured on an organ that has one. If the resolver
#       flattens that into the same status a kidney gets, the two arrive at
#       the display surfaces indistinguishable, and the surface words it the
#       way it words a kidney: "does not apply to this organ".
_NEGATIVE_FIELDS = ("has_tumor", "tumor_volume_cm3", "tumor_voxels",
                    "burden_cat", "multiplicity", "containment")

# WHICH QUESTIONS REST ON A WITHHELD ORGAN CLAIM. The query suite writes an
# organ three ways and the first version of this matcher knew one of them:
# it looked for the noun, so it found "Q2 lesion in spleen" and walked past
# "Q4b hepatic tumor volume > 65.4 cm3 AND splenic lesion" - "spleen" is not
# a substring of "splenic". It also walked past "Q5 lesions in >= 2 organs",
# which names no organ and counts all of them, and which the clean-up flips
# from T to F on the strength of the three withheld claims alone. An
# under-count here understates the finding, so all three forms are matched
# and the cross-organ question is attributed to every withheld organ.
_ORGAN_ADJECTIVE = {"liver": ("liver", "hepatic"),
                    "kidney": ("kidney", "renal"),
                    "spleen": ("spleen", "splenic"),
                    "pancreas": ("pancreas", "pancreatic")}
# A question that counts organs rather than naming one. It rests on every
# claim the rule withholds.
_COUNTS_ORGANS = re.compile(r"\b(?:\d+\s*)?organs\b|in\s*>=?\s*\d|"
                            r"in\s*≥\s*\d", re.I)


def _question_rests_on(question, withheld):
    """The withheld organ a question's answer depends on, or None."""
    q = str(question).lower()
    for organ in withheld:
        if not organ:
            continue
        for word in _ORGAN_ADJECTIVE.get(organ, (organ,)):
            if word in q:
                return organ
    if withheld and _COUNTS_ORGANS.search(q):
        return "+".join(o for o in withheld if o) + " (counted, not named)"
    return None

# WHAT COUNTS AS READING the pipeline team'S REASON FIELD. The field name has to be QUOTED
# and next to a subscript, a .get(), an equality or a membership test. The
# first version of this left the quote optional after `in\s`, which also
# matches the English word "in" - and the pack that documents this finding
# says "...put the reason in anatomic_location_status: not_applicable..."
# So writing the finding down created a reader, and the check that reports
# the finding switched itself off. See the selftest cases named for it.
_READS_STATUS = re.compile(r"""(?:\[|\.get\(|==|!=|\bin\s+)\s*["']"""
                           r"""anatomic_location_status["']""")


def v11_3_withheld_is_not_negative():
    rows = []

    # (a) the lesion-claim clean-up -----------------------------------------
    path = os.path.join(BASE, "mmkg_cache_universal",
                        "ipkg_lesion_cleanup.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cu = json.load(fh)
    except Exception as exc:                                  # noqa: BLE001
        rows.append(_r("surfaces", "V11.3/claims",
                       "a withheld lesion claim never renders as a negative "
                       "finding", SKIP, {"error": str(exc)},
                       "ipkg_lesion_cleanup.json readable"))
        cu = None
    if cu is not None:
        blk = cu.get("verdicts_FLARE23_0405") or {}
        dropped = [d.get("organ") for d in (blk.get("organs_dropped") or [])]
        verdicts = blk.get("verdicts") or {}
        offenders, kept = [], []
        for q, v in sorted(verdicts.items()):
            organ = _question_rests_on(q, dropped)
            if organ is None:
                continue
            if str(v.get("cleaned")).upper() == "F":
                # A question that was ALREADY F before the rule ran is not a
                # conversion, but it is still answered on a claim that no
                # longer exists, so it is reported with the distinction
                # rather than folded into one number.
                offenders.append(
                    {"question": q, "organ": organ,
                     "delivered": v.get("delivered"),
                     "cleaned": v.get("cleaned"),
                     "reference": v.get("reference"),
                     "turned_from_a_positive_into_a_negative":
                         str(v.get("delivered")).upper() == "T"})
            else:
                kept.append(q)
        rec = (blk.get("cleaned_record") or {}).get("organs") or {}
        written = {o: {f: rec[o].get(f) for f in _NEGATIVE_FIELDS
                       if f in rec.get(o, {})}
                   for o in dropped if o in rec}
        agree = blk.get("agreement_with_reference") or {}
        measured = {
            "artefact": "mmkg_cache_universal/ipkg_lesion_cleanup.json "
                        "(ipkg_lesion_cleanup.py verdict_effect)",
            "case": "FLARE23_0405",
            "organ_claims_the_rule_withholds": dropped,
            "questions_resting_on_a_withheld_claim_answered_F":
                len(offenders),
            "of_those_turned_from_a_positive_into_a_negative":
                sum(1 for o in offenders
                    if o["turned_from_a_positive_into_a_negative"]),
            "questions_resting_on_a_withheld_claim_not_answered_F":
                len(kept),
            "offenders": offenders,
            "what_the_cleaned_record_writes_for_a_withheld_organ": written,
            "the_headline_this_buys": agree,
            "the_contract_this_breaks":
                "the application's own verdict legend, on all four case "
                "pages: 'N not asserted - the lesion claim the question "
                "rests on was withheld by the claim clean-up, so the answer "
                "is withdrawn rather than answered; withheld is not "
                "negative, and N never says the organ is clear.'",
            "why_the_headline_is_the_problem":
                "agreement_with_reference counts a manufactured F as "
                "agreeing with a reference that happens to be F. On this "
                "case the reference is negative for all three withheld "
                "organs, so the rule scores %s and the substitution is "
                "invisible; against a reference that was positive the same "
                "rule would be CREDITED for a negative it never measured."
                % (agree.get("cleaned") is not None
                   and "%s of %s" % (agree.get("cleaned"), agree.get("n"))
                   or "its headline")}
        rows.append(_r(
            "surfaces", "V11.3/claims",
            "a withheld lesion claim never renders as a negative finding",
            PASS if not offenders else FAIL, measured,
            "every question resting on a withheld organ claim answered N, "
            "never F"))

    # (b) the sub-site ------------------------------------------------------
    try:
        import ipkg_scorer_fixed as SFX                       # noqa: PLC0415
        pool = FC.pool_records()
    except Exception as exc:                                  # noqa: BLE001
        rows.append(_r("surfaces", "V11.3/subsite",
                       "a sub-site that was never measured is not presented "
                       "as a sub-site the organ does not have",
                       SKIP, {"error": str(exc)}, "the pool resolvable"))
        return rows
    by_state = {}
    for rec in pool:
        ds, cid = rec.get("dataset"), rec.get("case_id")
        for organ, b in (rec.get("organs") or {}).items():
            if not b.get("has_tumor"):
                continue
            state, status, word = _subsite_state(b)
            got = SFX.resolve_site(ds, cid, organ, b.get("anatomic_location"))
            key = (state, str(status), str((got or {}).get("status")))
            e = by_state.setdefault(key, {"n": 0, "example": None})
            e["n"] += 1
            if e["example"] is None:
                e["example"] = {"dataset": ds, "case": cid, "organ": organ,
                                "resolved_note": (got or {}).get("note")}
    absent_states = {k[2] for k in by_state if k[0] == "absent"}
    unmeas_states = {k[2] for k in by_state if k[0] == "unmeasured"}
    collide = sorted(absent_states & unmeas_states)
    n_unmeasured = sum(v["n"] for k, v in by_state.items()
                       if k[0] == "unmeasured")
    # The second half of the proof, and the stronger one: a distinction that
    # no consumer reads is not a distinction the surfaces can honour.
    #
    # THIS SCAN DISABLED ITSELF ONCE, ON ITS FIRST RUN, AND THE INCIDENT IS
    # WHY IT IS WRITTEN THIS WAY. The first version matched
    #     (\[|\.get\(|==|in\s)\s*["']?anatomic_location_status
    # which looks like "a subscript, a .get(), a comparison or a membership
    # test". It is not: the `in\s` arm with the quote optional also matches
    # the ENGLISH word "in". The pack that documents this very finding
    # contains the sentence "...and put the reason in
    # anatomic_location_status: not_applicable, not_computed..." - so
    # writing the finding down produced a "reader", the reader count went
    # from 0 to 1, and the row flipped from FAIL to PASS. A check that is
    # switched off by its own documentation is the exact shape of defect
    # this block was called to find, and it was caught by reading the
    # output rather than by trusting the run.
    #
    # Two changes. The field name must be QUOTED - prose does not quote it -
    # and validation/ is pruned outright, because this suite's own files
    # and its own report are not consumers: a validator noticing a field
    # does not put it on a screen.
    readers = []
    _NOT_A_CONSUMER = ("ipkg_validate.py",)
    _NOT_A_CONSUMER_DIR = ("validation", ".git", "__pycache__", "node_modules")
    for dirpath, dirnames, filenames in os.walk(BASE):
        dirnames[:] = [d for d in dirnames if d not in _NOT_A_CONSUMER_DIR]
        for fn in filenames:
            if not fn.endswith(".py") or fn in _NOT_A_CONSUMER:
                continue
            fp = os.path.join(dirpath, fn)
            try:
                src = open(fp, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            if "anatomic_location_status" not in src:
                continue
            for i, ln in enumerate(src.splitlines(), 1):
                if _READS_STATUS.search(ln):
                    readers.append({"file": os.path.relpath(fp, BASE)
                                    .replace("\\", "/"), "line": i,
                                    "code": ln.strip()[:120]})
    measured = {
        "field": "organs.<organ>.anatomic_location_status, as the pipeline team ships it",
        "resolver": "ipkg_scorer_fixed.resolve_site",
        "pool_size": len(pool),
        "lesion_rows_whose_sub_site_was_never_measured": n_unmeasured,
        "resolved_status_for_an_ABSENT_axis": sorted(absent_states),
        "resolved_status_for_an_UNMEASURED_axis": sorted(unmeas_states),
        "statuses_that_mean_both_at_once": collide,
        "executable_lines_in_this_tree_that_READ_the_status_field":
            len(readers),
        "readers": readers,
        "excluded_from_the_reader_scan":
            list(_NOT_A_CONSUMER) + ["%s/" % d for d in
                                     _NOT_A_CONSUMER_DIR],
        "what_counts_as_a_reader":
            "the field name QUOTED, next to a subscript, a .get(), an "
            "equality or a membership test. Prose does not quote it - the "
            "first version of this scan accepted an unquoted match after "
            "the English word 'in', and the pack documenting this finding "
            "then registered as a reader and flipped the row to PASS.",
        "why_the_reader_count_is_the_proof":
            "the pipeline team ships the reason for every void in a field of its own. If "
            "no line outside this suite reads it, then no surface can "
            "honour it, whatever the resolver does - the distinction exists "
            "in the delivery and nowhere downstream of it.",
        "breakdown": {"%s | %s | resolved=%s" % k: v["n"]
                      for k, v in sorted(by_state.items())},
        "examples": {"%s | %s | resolved=%s" % k: v["example"]
                     for k, v in sorted(by_state.items())},
        "why_it_matters":
            "a kidney row and a never-derived pancreas row arrive at the "
            "display surfaces in the same resolved status, and the surface "
            "words that status the way the kidney needs: 'the sub-site "
            "vocabulary does not apply to this organ'. On a pancreas that "
            "sentence is false - the pancreas has a head, a body and a "
            "tail; nobody measured which."}
    # The distinction survives to a surface if EITHER the resolver keeps the
    # two apart, OR something downstream reads the pipeline team's reason field directly.
    # Neither is true today, and a row that required only one of them would
    # be the weaker claim.
    ok = (n_unmeasured == 0) or (not collide) or bool(readers)
    rows.append(_r(
        "surfaces", "V11.3/subsite",
        "a sub-site that was never measured is not presented as a sub-site "
        "the organ does not have",
        PASS if ok else FAIL, measured,
        "no resolved status shared between an absent axis and an unmeasured "
        "one, and the reason field read by something"))
    return rows


def v11_surfaces(verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    say("== V11 the display contract")
    rows = []
    rows += v11_1_organ_dice_convention()
    rows += v11_2_pool_on_every_surface()
    rows += v11_3_withheld_is_not_negative()
    return rows


# ---------------------------------------------------------------------------
# V10  the runner
# ---------------------------------------------------------------------------
CHECK_ORDER = ("V1", "V2", "V3", "V4", "V9", "V11")

CHECK_TITLES = {
    "V1": "identity and geometry",
    "V2": "predicted-versus-reference agreement",
    "V3": "claim by claim",
    "V4": "anatomical sanity in world coordinates",
    "V9": "scan-quality assertions",
    "V11": "the display contract: convention, denominator, epistemic status",
}


def run_case(entry, checks, lock, verbose=True):
    ctx = load_case(entry, verbose=verbose)
    rows = []
    if "V1" in checks:
        rows += v1_identity(ctx, lock)
    if "V2" in checks:
        rows += v2_agreement(ctx)
    if "V3" in checks:
        rows += v3_claims(ctx, verbose=verbose)
    if "V4" in checks:
        rows += v4_anatomy(ctx, verbose=verbose)
    if "V9" in checks:
        rows += v9_scan_quality(ctx, verbose=verbose)
    head = {"case": ctx["cid"], "collection": ctx["collection"],
            "dataset": ctx["dataset"], "organ_of_record": ctx["organ"],
            "grid": ctx["grid"], "axcodes": ctx["pred_axcodes"],
            "prediction": _rel(ctx["pred_path"]),
            "ct": _rel(ctx["ct_path"]),
            "reference": ctx["ref_variant"]["label"],
            "reference_provenance": ctx["ref_variant"]["provenance"],
            "structures_measured": [n for n, _, _ in ctx["structures"]],
            "notes_inherited_from_the_audit": ctx["inherited_notes"],
            "load_seconds": ctx["load_seconds"]}
    del ctx
    return head, rows


def _tally(rows):
    t = {PASS: 0, FAIL: 0, SKIP: 0}
    for r in rows:
        t[r["status"]] = t.get(r["status"], 0) + 1
    return t


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------
_POLICY = [
    "Decision 1 of the audit: corrected values are SHOWN, never silently "
    "substituted. Nothing in this suite writes to a delivered file, a mask, "
    "a record or a sidecar; every FAIL below is a statement about a file "
    "that is still on disk exactly as the pipeline team issued it.",
    "Decision 2: the rater page carries no accuracy score. Every agreement "
    "figure in this report is marked audience=internal-evaluation. The "
    "semi-oracle disclosure is method rather than score and does belong on "
    "the rater page; it is quoted in the header for that use.",
    "A FAIL is a finding. A FAIL that carries a `known` field was already "
    "written down by the 2026-09-29 audit and is quoted there; a FAIL with "
    "no `known` field is new.",
]


def _scalar(x):
    return not isinstance(x, (dict, list, tuple))


def _md_value(v, width=110):
    """A measured value on one line of a table.

    The scalars first, then any short list of scalars - because on several
    checks the informative part IS a list (which fields missed, which slices,
    which words were outside the vocabulary), and a cell that says "4 fields"
    tells a reader nothing they can act on.
    """
    if isinstance(v, dict):
        bits = ["%s=%s" % (k, v[k]) for k in v if _scalar(v[k])]
        for k, val in v.items():
            if isinstance(val, (list, tuple)) and val and                     all(_scalar(x) for x in val):
                bits.append("%s=[%s]" % (k, ", ".join(str(x)
                                                      for x in val[:8])))
        s = "; ".join(bits) or "%d fields" % len(v)
    elif isinstance(v, (list, tuple)):
        s = ", ".join(str(x) for x in v[:8])
    else:
        s = str(v)
    s = s.replace("|", "/").replace(chr(10), " ")
    return s if len(s) <= width else s[:width - 3] + "..."


def case_markdown(head, rows):
    L = []
    L.append("# IPKG validation - %s (%s)" % (head["case"],
                                              head["collection"]))
    L.append("")
    L.append("Generated %s by `ipkg_validate.py`. %s." % (_utc(), AUDIT))
    L.append("")
    L.append("**Semi-oracle disclosure (this sentence, and only this "
             "sentence, belongs on the rater page).** %s"
             % SEMI_ORACLE_DISCLOSURE)
    L.append("")
    t = _tally(rows)
    L.append("| | |")
    L.append("|---|---|")
    L.append("| result | **%d PASS, %d FAIL, %d SKIP** |"
             % (t[PASS], t[FAIL], t[SKIP]))
    L.append("| grid | %s at %s mm, axis codes %s |"
             % ("x".join(str(v) for v in head["grid"]["shape"]),
                " x ".join("%.4g" % v for v in head["grid"]["spacing_mm"]),
                head["axcodes"]))
    L.append("| mask drawn | `%s` |" % head["prediction"])
    L.append("| CT | `%s` |" % head["ct"])
    L.append("| reference | %s |" % head["reference"])
    L.append("| structures | %s |" % ", ".join(head["structures_measured"]))
    L.append("")
    for grp in CHECK_ORDER:
        sub = [r for r in rows if r["id"].split(".")[0] == grp]
        if not sub:
            continue
        L.append("## %s %s" % (grp, CHECK_TITLES[grp]))
        L.append("")
        L.append("| check | status | measured | expected |")
        L.append("|---|---|---|---|")
        for r in sub:
            L.append("| %s %s | **%s** | %s | %s |"
                     % (r["id"], r["title"], r["status"],
                        _md_value(r["measured"]), _md_value(r["expected"], 70)))
        L.append("")
    fails = [r for r in rows if r["status"] == FAIL]
    if fails:
        L.append("## What failed, with the numbers")
        L.append("")
        for r in fails:
            L.append("### %s  %s" % (r["id"], r["title"]))
            L.append("")
            L.append("Expected: %s" % r["expected"])
            L.append("")
            if r.get("note"):
                L.append(r["note"])
                L.append("")
            if r.get("known"):
                L.append("Already documented: %s" % r["known"])
            else:
                L.append("**Not previously documented.**")
            L.append("")
            L.append("```json")
            L.append(json.dumps(r["measured"], indent=1, default=str))
            L.append("```")
            L.append("")
    if head.get("notes_inherited_from_the_audit"):
        L.append("## Notes carried over from the audit's own measurement")
        L.append("")
        for n in head["notes_inherited_from_the_audit"]:
            L.append("- %s" % n)
        L.append("")
    L.append("## Policy")
    L.append("")
    for p in _POLICY:
        L.append("- %s" % p)
    L.append("")
    return "\n".join(L)


def summary_markdown(doc):
    L = []
    L.append("# IPKG validation suite - summary")
    L.append("")
    L.append("Generated %s by `ipkg_validate.py`. %s."
             % (doc["generated_utc"], AUDIT))
    L.append("")
    L.append("**Semi-oracle disclosure.** %s" % SEMI_ORACLE_DISCLOSURE)
    L.append("")
    # A filtered run overwrites this file, so it has to say what it covered.
    # A summary that looks complete and is not is worse than no summary.
    L.append("This run covered cases %s and check groups %s."
             % (", ".join(doc["cases"]) or "(none)",
                ", ".join(doc.get("checks_run") or [])))
    L.append("")
    L.append("| case | collection | PASS | FAIL | SKIP | report |")
    L.append("|---|---|---|---|---|---|")
    for cid, c in doc["cases"].items():
        t = c["tally"]
        L.append("| %s | %s | %d | %d | %d | `validation/%s` |"
                 % (cid, c["head"]["collection"], t.get(PASS, 0),
                    t.get(FAIL, 0), t.get(SKIP, 0), c["report_markdown"]))
    if doc.get("corpus"):
        t = _tally(doc["corpus"])
        L.append("| (corpus) | population test | %d | %d | %d | this file |"
                 % (t.get(PASS, 0), t.get(FAIL, 0), t.get(SKIP, 0)))
    if doc.get("surfaces"):
        t = _tally(doc["surfaces"])
        L.append("| (surfaces) | the display contract | %d | %d | %d | "
                 "this file |"
                 % (t.get(PASS, 0), t.get(FAIL, 0), t.get(SKIP, 0)))
    L.append("")
    allrows = [r for c in doc["cases"].values() for r in c["rows"]]
    allrows += doc.get("corpus") or []
    allrows += doc.get("surfaces") or []
    fails = [r for r in allrows if r["status"] == FAIL]
    new = [r for r in fails if not r.get("known")]
    L.append("## %d checks, %d FAIL, %d of them not previously documented"
             % (len(allrows), len(fails), len(new)))
    L.append("")
    if new:
        L.append("### The findings the 2026-09-29 audit had not already "
                 "written down")
        L.append("")
        for r in new:
            L.append("**%s - %s %s**" % (r.get("case"), r["id"], r["title"]))
            L.append("")
            if r.get("note"):
                L.append(r["note"])
                L.append("")
            L.append("- expected: %s" % r["expected"])
            L.append("- measured: %s" % _md_value(r["measured"], 400))
            L.append("")
    if fails:
        L.append("### Every FAIL")
        L.append("")
        L.append("| case | check | new? | what did not hold |")
        L.append("|---|---|---|---|")
        for r in fails:
            L.append("| %s | %s %s | %s | %s |"
                     % (r.get("case"), r["id"], r["title"],
                        "**NEW**" if not r.get("known") else "known",
                        _md_value(r.get("note") or r["measured"], 150)))
        L.append("")
    L.append("## Policy")
    L.append("")
    for p in _POLICY:
        L.append("- %s" % p)
    L.append("")
    L.append("Per-case reports are beside this file. The JSON form of "
             "everything here is `validation/validation_report.json`, which "
             "is what an evaluation panel should read rather than this "
             "summary.")
    L.append("")
    return "\n".join(L)


def build(which=None, checks=None, corpus=True, verbose=True):
    _ensure_out()
    checks = set(checks or CHECK_ORDER)
    lock = _load_lock()
    entries = [FC.entry_for(w) for w in which] if which else list(FC.CASES)
    doc = {"schema": SCHEMA, "generated_utc": _utc(),
           "generator": "ipkg_validate.py", "audit": AUDIT,
           "checks_run": sorted(checks),
           "check_titles": CHECK_TITLES,
           "policy": _POLICY,
           "semi_oracle_disclosure": SEMI_ORACLE_DISCLOSURE,
           "semi_oracle_disclosure_audience": AUDIENCE_RATER,
           "tolerances": {
               "volume_cm3": VOL_TOL_CM3, "volume_fraction": VOL_TOL_FRAC,
               "voxels": VOXEL_TOL, "dice": DICE_TOL,
               "affine_mm": AFFINE_TOL_MM, "zoom_mm": ZOOM_TOL_MM,
               "slice_gate_px_at_256": SLICE_GATE_PX_AT_256},
           "cases": {}}
    for e in entries:
        if verbose:
            print("== %s" % e["display_id"])
        head, rows = run_case(e, checks, lock, verbose=verbose)
        md = case_markdown(head, rows)
        name = "report_%s.md" % head["case"]
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(md)
        with open(os.path.join(OUT_DIR, "report_%s.json" % head["case"]),
                  "w", encoding="utf-8") as fh:
            json.dump({"schema": SCHEMA, "generated_utc": _utc(),
                       "head": head, "rows": rows,
                       "tally": _tally(rows)}, fh, indent=1, default=str)
        doc["cases"][head["case"]] = {"head": head, "rows": rows,
                                      "tally": _tally(rows),
                                      "report_markdown": name,
                                      "report_json":
                                          "report_%s.json" % head["case"]}
        if verbose:
            t = _tally(rows)
            print("   %s: %d PASS, %d FAIL, %d SKIP"
                  % (head["case"], t[PASS], t[FAIL], t[SKIP]))
    if corpus and "V4" in checks:
        if verbose:
            print("== corpus population test")
        doc["corpus"] = v4_corpus(verbose=verbose)
    if corpus and "V11" in checks:
        doc["surfaces"] = v11_surfaces(verbose=verbose)
    # A flat findings list beside the per-case rows, so a panel that wants to
    # show "what this suite found" does not have to walk four reports and
    # decide for itself which FAIL was already known.
    allrows = [r for c in doc["cases"].values() for r in c["rows"]]
    allrows += doc.get("corpus") or []
    allrows += doc.get("surfaces") or []
    doc["tally"] = _tally(allrows)
    doc["n_checks"] = len(allrows)
    doc["findings"] = [
        {"case": r.get("case"), "id": r["id"], "title": r["title"],
         "new": not bool(r.get("known")), "note": r.get("note"),
         "known": r.get("known"), "expected": r["expected"],
         "audience": r.get("audience")}
        for r in allrows if r["status"] == FAIL]
    _save_lock(lock)
    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=str)
    with open(SUMMARY_MD, "w", encoding="utf-8") as fh:
        fh.write(summary_markdown(doc))
    return doc


def selftest(verbose=True):
    """The suite's own arithmetic, without touching a case.

    A validator with a broken comparator is worse than no validator, because
    it reports PASS. These are the four pieces that decide every verdict: the
    field comparator, the world-coordinate profile, the Dice, and the
    fingerprint lock's FAIL path - which is the one branch a normal run never
    reaches, because a normal run has an unchanged file.
    """
    say = print if verbose else (lambda *a, **k: None)
    fails = []

    def ck(name, cond, got=None):
        if cond:
            say("  ok    %s" % name)
        else:
            say("  FAIL  %s  (%s)" % (name, got))
            fails.append(name)

    # the comparator
    ck("cm3 within the rounding tolerance", _cmp("cm3", 176.24, 176.27)[0])
    ck("cm3 outside it", not _cmp("cm3", 176.24, 167.60)[0])
    ck("a relative match on a large volume",
       _cmp("cm3", 2723.62, 2726.00)[0])
    ck("voxels are exact", not _cmp("voxels", 3496, 3497)[0])
    ck("exact on strings", _cmp("exact", "head", "head")[0]
       and not _cmp("exact", "head", "tail")[0])

    # world coordinates on a synthetic RAS grid: a 1 mm isotropic volume whose
    # world x runs -10 mm upward, with one voxel set at index 3.
    aff = np.eye(4)
    aff[0, 3] = -10.0
    m = np.zeros((8, 4, 4), bool)
    m[3, 1, 1] = True
    ck("a world centroid follows the affine",
       abs(_world_centroid(m, aff)[0] - (-7.0)) < 1e-9,
       _world_centroid(m, aff))
    m[5, 1, 1] = True
    ck("a world median sits between two voxels",
       abs(_wpct(_world_profile(m, aff, 0), 50) - (-6.0)) < 1e-9,
       _wpct(_world_profile(m, aff, 0), 50))
    # a rotated grid has to be refused rather than measured along the wrong
    # line, which is the whole reason the alignment is checked
    rot = np.eye(4)
    rot[:3, :3] = np.array([[0.7, -0.7, 0.0], [0.7, 0.7, 0.0],
                            [0.0, 0.0, 1.0]])
    ck("a rotated grid is refused", _world_profile(m, rot, 0) is None)

    # Dice
    a = np.zeros((4, 4, 4), bool)
    b = np.zeros((4, 4, 4), bool)
    a[0:2] = True
    b[1:3] = True
    d, inter, na, nb = _dice(a, b)
    ck("Dice of two half-overlapping blocks is 0.5",
       abs(d - 0.5) < 1e-12 and inter == 16 and na == 32 and nb == 32, d)
    ck("Dice of two empty masks is 1.0 by convention",
       _dice(np.zeros((2, 2, 2), bool), np.zeros((2, 2, 2), bool))[0] == 1.0)

    # the lock's FAIL branch, on a scratch file of our own
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ipkg_validate_selftest_")
    p = os.path.join(tmp, "probe.bin")
    try:
        with open(p, "wb") as fh:
            fh.write(b"one")
        lock = {"files": {}}
        s1, _ = _lock_check(lock, "SELFTEST", "probe", p)
        s2, _ = _lock_check(lock, "SELFTEST", "probe", p)
        with open(p, "wb") as fh:
            fh.write(b"two")
        s3, m3 = _lock_check(lock, "SELFTEST", "probe", p)
        ck("the first sighting records rather than asserts", s1 == SKIP, s1)
        ck("an unchanged file passes", s2 == PASS, s2)
        ck("a changed file fails and prints both hashes",
           s3 == FAIL and m3.get("sha256") != m3.get("sha256_recorded"), s3)
    finally:
        try:
            os.remove(p)
            os.rmdir(tmp)
        except OSError:
            pass

    # the vocabulary table, which is the one check that is pure knowledge
    ck("a kidney has no head",
       "head" not in ORGAN_SUBSITE_VOCABULARY["kidney"]["allowed"])
    ck("a pancreas does", "head" in
       ORGAN_SUBSITE_VOCABULARY["pancreas"]["allowed"])

    # THE VOID CLASSIFIER, which is the comparator the 2026-10-04 revision
    # turns on. Every one of these reads "na" on the wire; the whole point
    # is that four of them are four different findings, and a classifier
    # that collapsed any two would put the population test back to sleep.
    ck("a kidney's 'na' is an ABSENT axis",
       _subsite_state({"anatomic_location": "na",
                       "anatomic_location_status": "not_applicable"})[0]
       == "absent")
    ck("a pancreas's never-derived 'na' is UNMEASURED, not absent",
       _subsite_state({"anatomic_location": "na",
                       "anatomic_location_status": "not_computed"})[0]
       == "unmeasured")
    ck("a refused orientation guard is UNMEASURED, not absent",
       _subsite_state({"anatomic_location": "na",
                       "anatomic_location_status": "guard_failed"})[0]
       == "unmeasured")
    ck("a computed word is MEASURED",
       _subsite_state({"anatomic_location": "head",
                       "anatomic_location_status": "computed"})
       == ("measured", "computed", "head"))
    ck("a status this suite has never seen is NOT waved through as a void",
       _subsite_state({"anatomic_location": "na",
                       "anatomic_location_status": "something_new"})[0]
       == "unstated")
    ck("a record older than the status field falls back on its value",
       _subsite_state({"anatomic_location": "tail"})[0] == "measured"
       and _subsite_state({"anatomic_location": "unknown"})[0] == "unstated")

    # the organ-Dice display detector, on four lines written to look like
    # the surfaces it sweeps. The false-positive direction matters as much
    # as the false-negative one: a check that cries about a captioned table
    # is a check nobody reads twice.
    named = ["## T4", "", "| Collection | median organ Dice (CONV A) |",
             "|---|---|", "| LiTS | **0.9589** |"]
    bare = ["## Cases", "", "| collection | organ Dice, measured |",
            "|---|---|", "| LiTS | liver **0.9449** |"]
    ck("a table whose header names convention A is not reported",
       not any(not _CONV_RE.search(sec)
               for _, _, sec in _organ_dice_displays(named)))
    ck("a table that names no convention is reported",
       any(not _CONV_RE.search(sec)
           for _, _, sec in _organ_dice_displays(bare)))
    ck("a table with no Dice column is not swept at all",
       not list(_organ_dice_displays(
           ["| collection | liver volume cm3 |", "|---|---|",
            "| LiTS | 1711.34 |"])))
    # THE ROW FORM. The manuscript tables put the structure and the figure
    # in different cells, and the same-cell rule walked past the whole of
    # tables/ until 2026-10-04 - 19 displays found where there are 127.
    ck("the structure and the figure in DIFFERENT cells are still a display",
       len(list(_organ_dice_displays(
           ["## T4 convention A", "",
            "| Collection | structure | median organ Dice |",
            "|---|---|---|", "| FLARE23 | liver | **0.9649** |"]))) == 1)
    # and the two false positives that rule produced on its first run, kept
    # here so they cannot come back: a COLLECTION name and a CASE ID, both
    # inside tumour-Dice tables.
    ck("a collection called 'MSD Pancreas' in a tumour table is not an "
       "organ figure",
       not list(_organ_dice_displays(
           ["| Collection | round-1 tumour Dice |", "|---|---|",
            "| MSD Pancreas | 0.8179 |"])))
    ck("a case id 'pancreas_125' in a lesion table is not an organ figure",
       not list(_organ_dice_displays(
           ["| case | delivered lesion Dice |", "|---|---|",
            "| pancreas_125 | 0.6175 |"])))
    ck("a memo dated before the convention is exempt, one dated after is not",
       _predates_the_convention("paper/X_2026-09-29.md")
       and not _predates_the_convention("paper/X_2026-10-02.md")
       and not _predates_the_convention("README.md"))

    # V11.3/subsite's READER scan, which switched itself off on its first
    # run: its pattern accepted an unquoted match after the English word
    # "in", so the sentence that DOCUMENTS the finding registered as a
    # reader and the row flipped FAIL -> PASS. These three cases are the
    # incident, kept so it cannot happen twice.
    _prose = ('    "the pipeline team puts the reason in anatomic_location_status: '
              'not_applicable, not_computed..."')
    _code_get = '    st = b.get("anatomic_location_status")'
    _code_sub = '    if rec["anatomic_location_status"] == "not_computed":'
    ck("a prose sentence naming the status field is NOT a reader",
       not _READS_STATUS.search(_prose))
    ck("a .get() of the status field IS a reader",
       bool(_READS_STATUS.search(_code_get)))
    ck("a subscript of the status field IS a reader",
       bool(_READS_STATUS.search(_code_sub)))

    # V11.3/claims' question matcher, which under-counted by three on its
    # first run: the query suite writes an organ as a noun, as an adjective
    # and not at all, and the first version knew only the noun. The count
    # it reported was 9 of 16; the true figure is 12 of 16, and a signed
    # pack had already quoted the 9.
    _dropped = ["liver", "spleen", "pancreas"]
    ck("the organ as a noun is matched",
       _question_rests_on("Q2 lesion in spleen", _dropped) == "spleen")
    ck("the organ as an ADJECTIVE is matched too ('splenic' is the spleen)",
       _question_rests_on("Q4b hepatic tumor > 65.4 cm3 AND splenic lesion",
                          _dropped) is not None)
    ck("a question that COUNTS organs rests on every withheld claim",
       _question_rests_on("Q5 lesions in >= 2 organs", _dropped)
       is not None)
    ck("a question about an organ the rule KEPT is not matched",
       _question_rests_on("Q2 lesion in kidney", _dropped) is None
       and _question_rests_on("Q7a high tumor burden in observed anatomy",
                              _dropped) is None)

    say("selftest: %d failed" % len(fails))
    return not fails


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="the IPKG clinical-correctness validation suite")
    ap.add_argument("--case", action="append", default=None,
                    help="a case id; repeatable. All four by default.")
    ap.add_argument("--check", action="append", default=None,
                    choices=list(CHECK_ORDER),
                    help="a check group; repeatable. All by default.")
    ap.add_argument("--corpus-only", action="store_true",
                    help="run only the population-level sub-site test")
    ap.add_argument("--selftest", action="store_true",
                    help="check the suite's own comparators and geometry, "
                         "without opening a case")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    verbose = not a.quiet

    if a.selftest:
        return 0 if selftest(verbose=verbose) else 1

    if a.corpus_only:
        _ensure_out()
        rows = v4_corpus(verbose=verbose)
        doc = {"schema": SCHEMA, "generated_utc": _utc(),
               "generator": "ipkg_validate.py", "audit": AUDIT,
               "checks_run": ["V4.6"], "check_titles": CHECK_TITLES,
               "policy": _POLICY,
               "semi_oracle_disclosure": SEMI_ORACLE_DISCLOSURE,
               "semi_oracle_disclosure_audience": AUDIENCE_RATER,
               "cases": {}, "corpus": rows}
        with open(os.path.join(OUT_DIR, "report_corpus.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, default=str)
        for r in rows:
            print("%-6s %-58s %s" % (r["status"], r["id"],
                                     _md_value(r["measured"], 80)))
        t = _tally(rows)
        print("corpus: %d PASS, %d FAIL, %d SKIP"
              % (t[PASS], t[FAIL], t[SKIP]))
        return 0 if t[FAIL] == 0 else 1

    doc = build(which=a.case, checks=a.check, verbose=verbose)
    allrows = [r for c in doc["cases"].values() for r in c["rows"]]
    allrows += doc.get("corpus") or []
    allrows += doc.get("surfaces") or []
    t = _tally(allrows)
    print("")
    print("%d checks: %d PASS, %d FAIL, %d SKIP"
          % (len(allrows), t[PASS], t[FAIL], t[SKIP]))
    for r in allrows:
        if r["status"] == FAIL:
            print("  FAIL %-14s %-10s %s%s"
                  % (r.get("case"), r["id"], r["title"],
                     "" if r.get("known") else "   [NEW]"))
    print("")
    print("reports: %s" % os.path.relpath(OUT_DIR, BASE))
    # A non-zero exit is deliberate: this suite is meant to be wired into a
    # pre-demo check, and a suite whose FAILs do not change an exit code is a
    # suite nobody notices.
    return 0 if t[FAIL] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
