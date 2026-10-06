#!/usr/bin/env python3
"""Pancreatic sub-site (head / body / tail) in PATIENT coordinates: the replacement for the slice-axis rule in
build_predicted_corpus.py (the co-author 09-29 §3 / §6 item 11).

The delivered rule measured the tumor's offset along the ACQUISITION SLICE axis (axis 2 = inferior->superior on MSD)
with a +-0.12 dead band. That axis is craniocaudal, not the gland's long axis, and because the tail normally sits
cranial to the head the labels come out close to INVERTED (MSD GT, n=281: kappa vs the corrected rule -0.335).
It also ran for liver and kidney, which have no head/body/tail.

Corrected rule (pancreas only; every other organ -> "na"):
  1. voxel -> world mm through the NIfTI affine (nibabel RAS+: +x = patient RIGHT, so patient-left L = -x).
  2. gland = pancreas + tumor labels (tumor voxels are carved out of the pancreas label).
  3. PRIMARY  f_lr  = (mean L of tumor voxels - P1(L_gland)) / (P99(L_gland) - P1(L_gland)), clipped to [0, 1];
     head f < 1/3, body 1/3 <= f < 2/3, tail f >= 2/3  (operational thirds; the SEER/CAP anatomic boundaries -
     head right of the left border of the SMV, body up to the left border of the aorta, tail to the splenic
     hilum - need vessel masks the four datasets do not provide).
  4. CROSS-CHECK f_pca along the gland's first principal axis (mm, oriented toward patient-left).
  5. GUARDS (fail closed -> "unknown"): non-degenerate affine; the gland's patient-left third lies cranial to its
     patient-right third (tail rises toward the splenic hilum); LR and PCA thirds agree unless |f_lr - f_pca| is
     small (boundary cases are flagged, not voided).

  python pancreas_subsite.py --labels-dir /path/to/Task07_Pancreas/labelsTr      (GT audit, MSD label 1|2)
-> results/kg/pancreas_subsite_msd_gt.json
"""
import argparse
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import nibabel as nib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "kg", "pancreas_subsite_msd_gt.json")
SNOMED = {"head": "64163001", "body": "40133006", "tail": "73239005", "pancreas": "15776009"}   # tx.fhir.org, Intl 20250201
LABS = ("head", "body", "tail")


def thirds(f):
    return "head" if f < 1 / 3 else ("body" if f < 2 / 3 else "tail")


def old_rule(seg, olab=1, tlab=2, ax=2):
    """The delivered rule (build_predicted_corpus.py): slice-axis offset of the tumor slice set, +-0.12."""
    other = tuple(i for i in range(3) if i != ax)
    zc = np.where((seg == tlab).any(axis=other))[0]
    oc = np.where((seg == olab).any(axis=other))[0]
    if not len(zc) or not len(oc):
        return "na"
    off = (zc.mean() - oc.mean()) / (oc.max() - oc.min() + 1e-6)
    return "head" if off > 0.12 else ("tail" if off < -0.12 else "body")


def pancreas_subsite(gland, tumor, affine):
    """gland, tumor: boolean 3-D arrays (gland must include the tumor); affine: 4x4 voxel->RAS mm.
    Returns {"site", "f_lr", "f_pca", "site_pca", "guards": {...}, "snomed"}; site "unknown" when a guard fails."""
    A = np.asarray(affine, float)
    if not tumor.any() or not gland.any():
        return {"site": "na", "reason": "no tumor or no gland"}
    if abs(np.linalg.det(A[:3, :3])) < 1e-6:
        return {"site": "unknown", "reason": "degenerate affine"}
    W = lambda ijk: ijk.astype(float) @ A[:3, :3].T + A[:3, 3]
    gw, tw = W(np.argwhere(gland)), W(np.argwhere(tumor))
    Lg, Lt = -gw[:, 0], -tw[:, 0]
    lo, hi = np.percentile(Lg, 1), np.percentile(Lg, 99)
    f_lr = float(np.clip((Lt.mean() - lo) / max(hi - lo, 1e-6), 0, 1))
    mu = gw.mean(0)
    w, V = np.linalg.eigh(np.cov((gw - mu).T))
    pc = V[:, -1] if V[0, -1] <= 0 else -V[:, -1]            # orient toward patient-left (-x RAS)
    pg, pt = (gw - mu) @ pc, (tw - mu) @ pc
    plo, phi = np.percentile(pg, 1), np.percentile(pg, 99)
    f_pca = float(np.clip((pt.mean() - plo) / max(phi - plo, 1e-6), 0, 1))
    third = (Lg - Lg.min()) / (np.ptp(Lg) + 1e-6)
    tail_minus_head_S = float(gw[third > 2 / 3, 2].mean() - gw[third < 1 / 3, 2].mean())
    guards = {"tail_cranial_to_head": tail_minus_head_S > 0, "tail_minus_head_S_mm": round(tail_minus_head_S, 1),
              "lr_pca_agree": thirds(f_lr) == thirds(f_pca), "abs_f_diff": round(abs(f_lr - f_pca), 3),
              "pc1_var_frac": round(float(w[-1] / w.sum()), 3)}
    site = thirds(f_lr)
    if not guards["tail_cranial_to_head"]:
        site, reason = "unknown", "orientation guard failed (patient-left third not cranial) - review"
    elif not guards["lr_pca_agree"] and guards["abs_f_diff"] > 0.1:
        site, reason = "unknown", "LR and PCA thirds disagree by > 0.1 - review"
    else:
        reason = "ok" if guards["lr_pca_agree"] else "boundary case (LR/PCA thirds differ, |df| <= 0.1): LR used"
    return {"site": site, "reason": reason, "f_lr": round(f_lr, 3), "f_pca": round(f_pca, 3),
            "site_pca": thirds(f_pca), "guards": guards, "snomed": SNOMED.get(site)}


def _one(fp):
    n = nib.load(fp)
    seg = np.asarray(n.dataobj).astype(np.uint8)
    r = {"case": os.path.basename(fp)[:-7], "axcodes": "".join(nib.aff2axcodes(n.affine)), "old_rule": old_rule(seg)}
    r.update(pancreas_subsite(seg >= 1, seg == 2, n.affine))
    if "f_lr" in r:
        r["lr_thirds"] = thirds(r["f_lr"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default="/path/to/staging/acm_data/Task07_Pancreas/labelsTr")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    fs = sorted(glob.glob(f"{a.labels_dir}/pancreas_*.nii.gz"))
    with Pool(min(16, os.cpu_count() or 4)) as p:
        R = p.map(_one, fs)
    T = [r for r in R if r["site"] != "na"]
    n = len(T)
    agree = lambda x, y: sum(r[x] == r[y] for r in T) / n

    def kappa(x, y):
        pa, pb = Counter(r[x] for r in T), Counter(r[y] for r in T)
        pe = sum(pa[l] * pb[l] for l in LABS) / n ** 2
        return (agree(x, y) - pe) / (1 - pe)
    s = {"n_files": len(R), "n_with_tumor": n, "axcodes": dict(Counter(r["axcodes"] for r in R)),
         "dist_corrected": dict(Counter(r["site"] for r in T)), "dist_lr_thirds": dict(Counter(thirds(r["f_lr"]) for r in T)),
         "dist_pca_thirds": dict(Counter(r["site_pca"] for r in T)), "dist_old_rule": dict(Counter(r["old_rule"] for r in T)),
         "kappa_old_vs_lr_thirds": round(kappa("old_rule", "lr_thirds"), 3), "kappa_lr_vs_pca_thirds": round(kappa("lr_thirds", "site_pca"), 3),
         "lr_pca_agreement": round(agree("lr_thirds", "site_pca"), 4),
         "guard_tail_cranial_frac": round(sum(r["guards"]["tail_cranial_to_head"] for r in T) / n, 4),
         "confusion_corrected_vs_old": {x: {y: sum(r["site"] == x and r["old_rule"] == y for r in T) for y in LABS} for x in LABS + ("unknown",)},
         "snomed": SNOMED}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"note": __doc__.strip(), "summary": s, "records": R}, open(a.out, "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
