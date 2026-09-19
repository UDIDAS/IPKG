#!/usr/bin/env python3
"""Duplicate-scan audit of the 1,425-case retrieval pool: are any of the 113 single-organ queries (KiTS23 36,
LiTS 21, MSD Pancreas 56) the same scan as one of the 1,312 FLARE23 candidates?  FLARE23 was assembled from
public sources that include those collections, so twins would inflate the large-pool results (Tables 9-11).

Signatures per case, from the reference masks:
  * header: shape, spacing (LiTS .npy derivatives carry no header — geometry-based tests are skipped for them)
  * organ voxels, tumor voxels (tumor within the organ's 3-voxel dilation for FLARE23), tumor/organ voxel RATIO
    (spacing-invariant), number of tumor components, organ bounding-box aspect ratios (spacing-invariant per axis
    only if spacing matches; we use the in-plane aspect, which survives isotropic in-plane resampling)
  * physical organ / tumor volume (cm3) where a header exists
Tests, per (query, FLARE23 candidate) pair with the query's organ:
  T1 exact geometry: same shape AND same spacing (1e-3) AND organ voxels within 0.5 %          -> certain twin
  T2 physical:       organ volume within 1 % AND tumor volume within 2 % (both physical)          -> strong
  T3 invariant:      tumor/organ ratio within 1 % AND same component count AND in-plane aspect within 2 % -> candidate
  T4 HU confirm:     for pairs with both CTs in the radiomics cache, organ HU mean/std within 1 %   -> confirmation
Output: results/audit/duplicate_scan_audit.json  (all T1/T2 pairs, T3 candidates with their T4 status, and the
list of FLARE23 case ids to drop from the pool).
"""
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
C113 = os.path.join(VKG_DATA, "cohort113")
LABEL_DIRS = [os.path.join(VKG_DATA, d) for d in ("flare23_pool/labels", "flare23_labels_scan", "flare3d_raw/labels", "flare23_labels_audit/labels")]
OUT = os.path.join(ROOT, "results", "audit", "duplicate_scan_audit.json")
FL_ORGAN = {"liver": [1], "kidney": [2, 13], "pancreas": [4]}


def sig(mask_organ, mask_tumor, spacing):
    ov, tv = int(mask_organ.sum()), int(mask_tumor.sum())
    idx = np.argwhere(mask_organ)
    ext = (idx.max(0) - idx.min(0) + 1) if len(idx) else np.array([1, 1, 1])
    s = {"organ_vox": ov, "tumor_vox": tv, "ratio": tv / ov if ov else None, "n_tumor_comp": int(ndimage.label(mask_tumor)[1]),
         "aspect_xy": float(ext[0] / ext[1]) if ext[1] else None, "extent": ext.tolist(), "shape": list(mask_organ.shape)}
    if spacing is not None:
        cm3 = float(np.prod(spacing)) / 1000.0
        s.update({"spacing": [round(float(x), 4) for x in spacing], "organ_cm3": ov * cm3, "tumor_cm3": tv * cm3})
    return s


def q_case(job):
    ds, cid = job
    if ds == "kits":
        nii = nib.load(f"{C113}/kits/{cid}/segmentation.nii.gz"); seg = np.asarray(nii.dataobj).astype(np.uint8); sp = nii.header.get_zooms()[:3]
        o, t = seg == 1, seg == 2; organ = "kidney"
    elif ds == "msd":
        nii = nib.load(f"{C113}/msd/labelsTr/{cid}.nii.gz"); seg = np.asarray(nii.dataobj).astype(np.uint8); sp = nii.header.get_zooms()[:3]
        o, t = seg == 1, seg == 2; organ = "pancreas"
    else:
        seg = np.load(f"{C113}/lits/seg/segmentation-{cid.replace('volume-', '')}.npy").astype(np.uint8); sp = None
        o, t = seg == 1, seg == 2; organ = "liver"
    return cid, {"dataset": ds, "organ": organ, **sig(o, t, sp)}


def f_case(path):
    cid = os.path.basename(path)[:-7]
    nii = nib.load(path); seg = np.asarray(nii.dataobj).astype(np.uint8); sp = nii.header.get_zooms()[:3]
    tum = seg == 14; out = {}
    for organ, labs in FL_ORGAN.items():
        o = np.isin(seg, labs)
        if not o.any():
            continue
        t = tum & ndimage.binary_dilation(o, iterations=3)
        out[organ] = sig(o, t, sp)
    return cid, out


def close(a, b, tol):
    return a is not None and b is not None and b != 0 and abs(a - b) / abs(b) <= tol


def main():
    gt = {}
    for ds in ("kits", "lits", "msd"):
        for r in json.load(open(f"{ROOT}/corpora/corpus_gt_{ds}.json"))["records"]:
            gt[r["case_id"]] = ds
    flare_ids = [r["case_id"].replace("flare23_", "") for r in json.load(open(f"{ROOT}/corpora/corpus_flare23_kg.json"))["records"]]
    paths = {}
    for d in LABEL_DIRS:
        for f in glob.glob(f"{d}/*.nii.gz"):
            paths.setdefault(os.path.basename(f)[:-7], f)
    fpaths = [paths[c] for c in flare_ids if c in paths]
    print(f"queries {len(gt)}; FLARE23 labels available {len(fpaths)}/{len(flare_ids)}", flush=True)
    with ProcessPoolExecutor(32) as ex:
        Q = dict(ex.map(q_case, [(ds, c) for c, ds in gt.items()]))
        F = dict(ex.map(f_case, fpaths, chunksize=8))
    # radiomics cache for HU confirmation (organ HU mean/std/p90 are feature dims 1..3 of the union vector)
    hu = {}
    fc = os.path.join(VKG_DATA, "baseline_features.npz")
    if os.path.exists(fc):
        R = np.load(fc, allow_pickle=True)["R"].item()
        hu = {c: v["__union__"][1:4].tolist() for c, v in R.items()}
    pairs = []
    for q, qs in Q.items():
        organ = qs["organ"]
        for c, fs in F.items():
            if organ not in fs:
                continue
            f = fs[organ]; tests = {}
            if "spacing" in qs:
                tests["T1_exact_geometry"] = (qs["shape"] == f["shape"] and all(abs(a - b) < 1e-3 for a, b in zip(qs["spacing"], f["spacing"]))
                                              and close(qs["organ_vox"], f["organ_vox"], 0.005))
                tests["T2_physical"] = close(qs["organ_cm3"], f["organ_cm3"], 0.01) and (close(qs["tumor_cm3"], f["tumor_cm3"], 0.02) or (qs["tumor_vox"] == 0 and f["tumor_vox"] == 0))
            tests["T3_invariant"] = (close(qs["ratio"], f["ratio"], 0.01) if (qs["ratio"] and f["ratio"]) else (qs["tumor_vox"] == 0 and f["tumor_vox"] == 0)) \
                and qs["n_tumor_comp"] == f["n_tumor_comp"] and close(qs["aspect_xy"], f["aspect_xy"], 0.02)
            fq, ff = hu.get(q), hu.get(f"flare23_{c}")
            tests["T4_hu_confirm"] = (close(fq[0], ff[0], 0.01) and close(fq[1], ff[1], 0.01)) if (fq and ff) else None
            if any(v for k, v in tests.items() if k != "T4_hu_confirm"):
                pairs.append({"query": q, "dataset": qs["dataset"], "flare23": f"flare23_{c}", "organ": organ, "tests": tests,
                              "query_sig": {k: qs[k] for k in ("shape", "organ_vox", "tumor_vox", "ratio", "n_tumor_comp") if k in qs},
                              "flare_sig": {k: f[k] for k in ("shape", "organ_vox", "tumor_vox", "ratio", "n_tumor_comp")}})
    twins = sorted({p["flare23"] for p in pairs if p["tests"].get("T1_exact_geometry") or p["tests"].get("T2_physical") or (p["tests"]["T3_invariant"] and p["tests"]["T4_hu_confirm"])})
    cands = sorted({p["flare23"] for p in pairs if p["tests"]["T3_invariant"] and p["tests"]["T4_hu_confirm"] is None and not (p["tests"].get("T1_exact_geometry") or p["tests"].get("T2_physical"))})
    out = {"n_queries": len(Q), "n_flare23_signatures": len(F), "n_flare23_corpus": len(flare_ids),
           "pairs_flagged": pairs, "twins_confirmed": twins, "invariant_candidates_unconfirmed": cands,
           "definitions": __doc__.strip()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
    print(f"flagged pairs {len(pairs)}; confirmed twins {len(twins)}; unconfirmed invariant candidates {len(cands)}")
    for p in pairs[:20]:
        print("  ", p["query"], "->", p["flare23"], p["organ"], {k: v for k, v in p["tests"].items() if v})
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
