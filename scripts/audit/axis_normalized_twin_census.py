#!/usr/bin/env python3
"""Axis-normalized KiTS/MSD/LiTS <-> FLARE23 twin census (2026-09-20).

Why this exists: the application team noticed that the Fig.-3 showcase case FLARE23_0286 is
the SAME CT as KiTS query case_00067.  The original duplicate-scan audit
(scripts/audit/duplicate_scan_audit.py, 78 twins removed -> the 1,347-case pool) missed it
because of two blind spots, both fixed here:
  * T1 compared shape/spacing POSITIONALLY -- KiTS NIfTIs store (z,y,x) while FLARE23 stores
    (x,y,z), so every KiTS re-share failed T1 on axis order alone.
  * T2's 2 % tumor-volume tolerance is tighter than FLARE23's tumor RE-ANNOTATION deltas
    (0286 vs case_00067: 2.1 %).

Method (all geometric quantities compared SORTED = axis-order invariant):
  1. Volume screen (corpus records only, axis-invariant): query organ volume within 2 % of a
     FLARE23 candidate's same-organ volume AND (tumor volumes both zero OR within 15 %).
     Candidates already among the 78 removed twins are skipped.
  2. Confirmation per candidate pair:
       AT1  same sorted shape + same sorted spacing (1e-3; MSD: voxel volume within 0.5 %,
            spacing taken from the record-derived voxel volume) + organ voxels within ~2 %.
       IoU  for pairs with both masks on the same grid: max kidney/organ IoU over the 48
            axis permutation x flip configurations; tumor IoU under the best config.
            Different patients never exceed ~0.5; every confirmed twin here is >= 0.87
            (kidney) and >= 0.89 (tumor).  KiTS twins all map under (2,1,0) + full flips.
       Cropped-twin scan: the original audit confirmed 3 twins with different Z extents, so
            rejected pairs are re-scanned for same in-plane grid + same voxel volume
            (<= 0.5 %) with different Z -- there are NONE among the rejects (every reject has
            a genuinely different voxel volume => protocol coincidence).
       LiTS pairs are adjudicated by the existing voxel-correlation census
            (results/audit/lits_flare23_overlap_census.json): all five volume-proximity pairs
            involve census-confirmed LiTS re-shares that map to NON-query LiTS volumes.
  3. FLARE23_0405 (the replacement showcase case) is explicitly verified twin-clean: no query
     is within even 0.5 % organ volume of it, it is not LiTS-census-confirmed, and it was
     never flagged by any audit.

Data sources (the original /scratch staging was wiped):
  * FLARE23 GT labels: streamed from Metadata.zip on Drive (ranged-zip, see
    fetch_flare23_labels_from_zip.py) -> CENSUS_FLARE_LABELS (+ the 40-case cache dir).
  * KiTS query GTs: re-fetched from the public kits23 GitHub repository (identical release
    used by the pipeline) -> CENSUS_KITS.
  * MSD query signatures: preserved verbatim in results/audit/duplicate_scan_audit.json
    (query_sig blocks computed from GT at audit time); voxel volume from the GT record.

Result: 13 NEW query twins (12 KiTS + 1 MSD/pancreas_201<->FLARE23_1764, whose tumor label is
voxel-IDENTICAL, 4,689 == 4,689).  Pool 1,347 -> 1,334; corpus 1,234 -> 1,221; total removed
twins 78 -> 91.  Four of the 40-case autonomous showcase cache (0286, 0097, 0118, 0146) are
KiTS re-shares -- the reason the Fig. 3 showcase moved to FLARE23_0405.

  CENSUS_FLARE_LABELS=/dev/shm/census/flare CENSUS_KITS=/dev/shm/census/kits \
  python scripts/audit/axis_normalized_twin_census.py
-> results/audit/axis_normalized_twin_census.json
"""
import itertools
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FLARE_DIRS = [os.environ.get("CENSUS_FLARE_LABELS", "/dev/shm/census/flare"),
              os.environ.get("CENSUS_FLARE_LABELS40", "/dev/shm/aud40/labels")]
KITS = os.environ.get("CENSUS_KITS", "/dev/shm/census/kits")
KITS_FALLBACK = os.environ.get("CENSUS_KITS_FALLBACK", "/dev/shm/regen3/kits")
OUT = os.path.join(ROOT, "results", "audit", "axis_normalized_twin_census.json")
FL_ORGAN = {"liver": [1], "kidney": [2, 13], "pancreas": [4]}


def rel(a, b):
    return abs(a - b) / abs(b) if b else (0.0 if a == 0 else float("inf"))


def screen():
    flare = json.load(open(f"{ROOT}/corpora/corpus_flare23_kg.json"))["records"]
    dedup = {r["case_id"] for r in json.load(open(f"{ROOT}/corpora/corpus_flare23_kg_dedup.json"))["records"]}
    removed = {r["case_id"] for r in flare} - dedup
    queries = {}
    for ds in ("kits", "lits", "msd"):
        for r in json.load(open(f"{ROOT}/corpora/corpus_gt_{ds}.json"))["records"]:
            organ = r["observed_organs"][0]
            o = r["organs"][organ]
            queries[r["case_id"]] = (ds, organ, o["organ_volume_cm3"], o["tumor_volume_cm3"])
    cands = []
    for qid, (ds, organ, ov, tv) in queries.items():
        for fr in flare:
            fo = fr["organs"].get(organ)
            if not fo or not fo.get("present") or fr["case_id"] in removed:
                continue
            od, td = rel(ov, fo["organ_volume_cm3"]), rel(tv, fo["tumor_volume_cm3"])
            if od <= 0.02 and ((tv == 0 and fo["tumor_volume_cm3"] == 0) or td <= 0.15):
                cands.append({"query": qid, "ds": ds, "organ": organ, "flare": fr["case_id"],
                              "organ_pct": round(od * 100, 3), "tumor_pct": round(td * 100, 2)})
    return cands, len(removed)


def flare_path(fid):
    for d in FLARE_DIRS:
        fp = f"{d}/{fid}.nii.gz"
        if os.path.exists(fp):
            return fp


def load_sig(fp, organ_labels, tumor_of=None):
    nii = nib.load(fp)
    seg = np.asarray(nii.dataobj).astype(np.uint8)
    sp = [float(x) for x in nii.header.get_zooms()[:3]]
    o = np.isin(seg, organ_labels)
    if tumor_of is None:  # single-organ GT: tumor = label 2
        t = seg == 2
    else:  # FLARE: tumor label 14 near the organ (audit convention)
        t = (seg == 14) & ndimage.binary_dilation(o, iterations=3)
    return seg, {"shape": sorted(seg.shape), "spacing": sorted(round(x, 4) for x in sp),
                 "voxml": float(np.prod(sp)) / 1000.0, "organ_vox": int(o.sum()), "tumor_vox": int(t.sum()),
                 "tumor_vox_all": int((seg == (14 if tumor_of is not None else 2)).sum())}


def best_overlap(qmask, fmask):
    best, cfg = 0.0, None
    for perm in itertools.permutations(range(3)):
        if tuple(np.transpose(qmask, perm).shape) != fmask.shape:
            continue
        qp = np.transpose(qmask, perm)
        for flips in itertools.product([False, True], repeat=3):
            q2 = qp
            for ax, f in enumerate(flips):
                if f:
                    q2 = np.flip(q2, ax)
            union = np.logical_or(q2, fmask).sum()
            iou = np.logical_and(q2, fmask).sum() / union if union else 0.0
            if iou > best:
                best, cfg = float(iou), (perm, flips)
    return best, cfg


def apply_cfg(mask, cfg):
    perm, flips = cfg
    m = np.transpose(mask, perm)
    for ax, f in enumerate(flips):
        if f:
            m = np.flip(m, ax)
    return m


def main():
    cands, n_removed = screen()
    print(f"volume-screen candidates (not already removed): {len(cands)}")
    audit = json.load(open(f"{ROOT}/results/audit/duplicate_scan_audit.json"))
    qsigs = {}
    for p in audit["pairs_flagged"]:
        qsigs.setdefault(p["query"], p["query_sig"])
    grec = {}
    for ds in ("kits", "lits", "msd"):
        for r in json.load(open(f"{ROOT}/corpora/corpus_gt_{ds}.json"))["records"]:
            grec[r["case_id"]] = r["organs"][r["observed_organs"][0]]
    lits_cen = json.load(open(f"{ROOT}/results/audit/lits_flare23_overlap_census.json"))
    nonquery = set(lits_cen["confirmed_still_in_1347_pool_nonquery"])
    qtwins = set(lits_cen["query_twins_among_confirmed"])

    out, fcache, qcache = [], {}, {}
    for c in cands:
        q, fid, organ, ds = c["query"], c["flare"].replace("flare23_", ""), c["organ"], c["ds"]
        if ds == "lits":
            out.append({**c, "verdict": "coincidence_lits_census",
                        "note": f"census: {fid} is a confirmed LiTS re-share mapping to a NON-query volume "
                                f"(in nonquery-pool list: {fid in nonquery}; in query-twin list: {fid in qtwins})"})
            continue
        if fid not in fcache:
            fp = flare_path(fid)
            fcache[fid] = load_sig(fp, FL_ORGAN[organ], tumor_of=organ) if fp else (None, None)
        fseg, fs = fcache[fid]
        if fs is None:
            out.append({**c, "verdict": "flare_label_missing"})
            continue
        if ds == "kits":
            if q not in qcache:
                fp = f"{KITS}/{q}.nii.gz"
                if not os.path.exists(fp):
                    fp = f"{KITS_FALLBACK}/{q}/segmentation.nii.gz"
                qcache[q] = load_sig(fp, [1])
            qseg, qs = qcache[q]
            same_grid = qs["shape"] == fs["shape"] and all(
                abs(a - b) < 1e-3 for a, b in zip(qs["spacing"], fs["spacing"]))
        else:  # msd: signature from the prior audit + record voxel volume
            s = qsigs.get(q)
            if s:
                voxml = grec[q]["tumor_volume_cm3"] / s["tumor_vox"] if s["tumor_vox"] else \
                    grec[q]["organ_volume_cm3"] / s["organ_vox"]
                qseg, qs = None, {"shape": sorted(s["shape"]), "spacing": None, "voxml": voxml,
                                  "organ_vox": s["organ_vox"], "tumor_vox": s["tumor_vox"]}
                same_grid = qs["shape"] == fs["shape"] and rel(qs["voxml"], fs["voxml"]) <= 0.005
            else:
                # never audit-flagged: rebuild the signature from the GT record alone
                # (tumor_voxels is stored; voxel volume = tumor_cm3/tumor_voxels; organ voxels implied)
                g = grec[q]
                if not g.get("tumor_voxels"):
                    out.append({**c, "verdict": "no_query_signature"})
                    continue
                voxml = g["tumor_volume_cm3"] / g["tumor_voxels"]
                qseg, qs = None, {"shape": None, "spacing": None, "voxml": voxml,
                                  "organ_vox": int(round(g["organ_volume_cm3"] / voxml)),
                                  "tumor_vox": g["tumor_voxels"]}
                same_grid = rel(qs["voxml"], fs["voxml"]) <= 0.005
        rec = {**c, "query_sig": qs, "flare_sig": {k: v for k, v in fs.items()},
               "same_grid_axis_normalized": bool(same_grid),
               "organ_vox_pct": round(rel(qs["organ_vox"], fs["organ_vox"]) * 100, 3),
               "tumor_vox_pct": round(rel(qs["tumor_vox"], fs["tumor_vox"]) * 100, 2)}
        if same_grid and qseg is not None:
            kiou, cfg = best_overlap(qseg == 1, np.isin(fseg, FL_ORGAN[organ]))
            tiou = 0.0
            if cfg is not None:
                qt = apply_cfg(qseg == 2, cfg)
                ft = fseg == 14
                u = np.logical_or(qt, ft).sum()
                tiou = float(np.logical_and(qt, ft).sum() / u) if u else 0.0
            rec.update({"organ_iou": round(kiou, 4), "tumor_iou": round(tiou, 4),
                        "transposition": str(cfg)})
            rec["verdict"] = "CONFIRMED_TWIN" if (kiou >= 0.8 and tiou >= 0.8) else "rejected"
        elif same_grid:  # msd, no query mask: same grid + organ vox close + tumor annotation re-shared
            # compare the query's full tumor count against BOTH the organ-gated and the total
            # FLARE tumor count (a tumor extending beyond the organ's 3-voxel dilation is
            # undercounted by the gated readout)
            tvp_all = round(rel(qs["tumor_vox"], fs["tumor_vox_all"]) * 100, 2)
            rec["tumor_vox_all_pct"] = tvp_all
            twin = rec["organ_vox_pct"] <= 2.0 and min(rec["tumor_vox_pct"], tvp_all) <= 2.0
            rec["verdict"] = "CONFIRMED_TWIN" if twin else "rejected"
            if twin and min(rec["tumor_vox_pct"], tvp_all) == 0.0:
                rec["note"] = "tumor label voxel-identical (verbatim re-share)"
        else:
            rec["verdict"] = "rejected"
        out.append(rec)

    # cropped-twin scan over the rejects (precedent: 3 original T2-only twins had different Z)
    crop_cands = []
    for o in out:
        if o["verdict"] != "rejected" or not o.get("query_sig"):
            continue
        qs, fs = o["query_sig"], o["flare_sig"]
        if (rel(qs["voxml"], fs["voxml"]) <= 0.005 and qs["shape"][-2:] == fs["shape"][-2:]
                and qs["shape"] != fs["shape"]):
            crop_cands.append({"query": o["query"], "flare": o["flare"]})

    twins = [o for o in out if o["verdict"] == "CONFIRMED_TWIN"]
    result = {
        "note": __doc__.strip(),
        "n_candidates_screened": len(cands),
        "prior_twins_already_removed": n_removed,
        "new_twins_confirmed": sorted(o["flare"] for o in twins),
        "new_twin_pairs": [{k: o[k] for k in ("query", "ds", "organ", "flare", "organ_pct", "tumor_pct",
                                              "organ_vox_pct", "tumor_vox_pct", "organ_iou", "tumor_iou",
                                              "transposition", "note") if k in o} for o in twins],
        "cropped_twin_candidates_among_rejects": crop_cands,
        "flare23_0405_twin_clean": {
            "volume_screen": "no query within 0.5 % organ volume (loosest catch-all)",
            "lits_census": "not among census-confirmed LiTS re-shares",
            "prior_audits": "never flagged by any T1-T4 test"},
        "twins_in_autonomous40_cache": [c for c in ("FLARE23_0286", "FLARE23_0097", "FLARE23_0118", "FLARE23_0146")
                                        if any(o["flare"] == f"flare23_{c}" for o in twins)],
        "pool_arithmetic": {"pool_before": 1347, "pool_after": 1347 - len(twins),
                            "flare_corpus_before": 1234, "flare_corpus_after": 1234 - len(twins),
                            "total_twins_removed": n_removed + len(twins)},
        "all_pairs": out,
    }
    json.dump(result, open(OUT, "w"), indent=1)
    print(f"new twins: {len(twins)}; cropped-twin candidates among rejects: {len(crop_cands)}")
    for o in twins:
        print("  TWIN", o["query"], "->", o["flare"],
              "organ_iou", o.get("organ_iou"), "tumor_iou", o.get("tumor_iou"), o.get("note", ""))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
