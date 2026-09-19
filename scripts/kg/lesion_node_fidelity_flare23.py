#!/usr/bin/env python3
"""Draft Table 4 — genuine LESION-node fidelity between predicted-mask and reference-mask graph constructions.

Replaces the FLARE22 Task-2 source (exp_kg_fidelity.py), which measured pancreas ORGAN geometry on 20 organ-only
cases (its 0.95 is a volume-tercile agreement). Here: the 576 FLARE23 cases with a CT, semi-oracle AUSAM tumor
masks (GT-box prompted, the protocol of the single-organ predicted corpora) vs the reference masks.

Lesion node = tumor voxels attributed to one organ (per-component winner-take-all over the 3-voxel organ
dilation, as the graph builder does). Predicted lesion nodes are matched to reference lesion nodes patient-wise
by Hungarian assignment on 3-D IoU (accept IoU >= 0.3, the closed-loop convention); attributes are compared on
matched pairs: volume (cm3), maximal Feret diameter (mm, convex-hull pairwise distance as exp_kg_fidelity.py),
centroid (mm), and tercile agreement for volume AND diameter (terciles defined on the reference distribution).
Also reported: node-level precision / recall of lesion nodes, and the same attributes on an IoU-free
"same-organ" matching (predicted and reference both carry a lesion node in the organ) as the lenient variant.
Output: results/kg/lesion_node_fidelity_flare23.json
"""
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
PRED = os.path.join(VKG_DATA, "flare23_predicted"); REF = os.path.join(VKG_DATA, "flare23_pool", "labels")
OUT = os.path.join(ROOT, "results", "kg", "lesion_node_fidelity_flare23.json")
POOLED = {"liver": [1], "kidney": [2, 13], "spleen": [3], "pancreas": [4]}; TUMOR = 14; IOU_MIN = 0.3


def feret_mm(mask, sp):
    idx = np.argwhere(mask)
    if len(idx) < 2:
        return 0.0
    pts = idx * np.asarray(sp)
    if len(pts) > 3000:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pts = pts[np.random.RandomState(0).choice(len(pts), 3000, replace=False)]
    return float(pdist(pts).max())


def lesions(mask, sp):
    """{organ: {'mask': bool array, 'volume_cm3', 'diameter_mm', 'centroid_mm', 'n_comp'}} — attributed lesion nodes."""
    cm3 = float(np.prod(sp)) / 1000.0; tum = mask == TUMOR
    dil = {o: ndimage.binary_dilation(np.isin(mask, labs), iterations=3) for o, labs in POOLED.items() if np.isin(mask, labs).any()}
    lbl, n = ndimage.label(tum); assigned = {o: np.zeros(mask.shape, bool) for o in dil}
    for c in range(1, n + 1):
        comp = lbl == c; ov = {o: int((comp & d).sum()) for o, d in dil.items()}
        best = max(ov, key=ov.get) if ov else None
        if best is not None and ov[best] > 0:
            assigned[best] |= comp
    out = {}
    for o, m in assigned.items():
        if m.any():
            out[o] = {"mask": m, "volume_cm3": float(m.sum() * cm3), "diameter_mm": feret_mm(m, sp),
                      "centroid_mm": (np.argwhere(m).mean(0) * np.asarray(sp)).tolist(), "n_comp": int(ndimage.label(m)[1])}
    return out


def one(cid):
    pn = nib.load(f"{PRED}/{cid}.nii.gz"); pm = np.asarray(pn.dataobj).astype(np.uint8); sp = [float(z) for z in pn.header.get_zooms()[:3]]
    rm = np.asarray(nib.load(f"{REF}/{cid}.nii.gz").dataobj).astype(np.uint8)
    P, R = lesions(pm, sp), lesions(rm, sp)
    po, ro = sorted(P), sorted(R)
    iou = np.zeros((len(po), len(ro)))
    for i, a in enumerate(po):
        for j, b in enumerate(ro):
            inter = np.logical_and(P[a]["mask"], R[b]["mask"]).sum(); uni = np.logical_or(P[a]["mask"], R[b]["mask"]).sum()
            iou[i, j] = inter / uni if uni else 0.0
    pairs = []
    if iou.size:
        r, c = linear_sum_assignment(-iou)
        pairs = [(po[i], ro[j], float(iou[i, j])) for i, j in zip(r, c) if iou[i, j] >= IOU_MIN]
    strip = lambda d: {k: v for k, v in d.items() if k != "mask"}
    return {"cid": cid, "pred": {o: strip(v) for o, v in P.items()}, "ref": {o: strip(v) for o, v in R.items()}, "matched_iou": pairs,
            "same_organ": [o for o in po if o in R]}


def stats(pairs_attr):
    """pairs_attr: list of (pred_dict, ref_dict)."""
    if not pairs_attr:
        return None
    gv = np.array([r["volume_cm3"] for _, r in pairs_attr]); pv = np.array([p["volume_cm3"] for p, _ in pairs_attr])
    gd = np.array([r["diameter_mm"] for _, r in pairs_attr]); pd_ = np.array([p["diameter_mm"] for p, _ in pairs_attr])
    gc = np.array([r["centroid_mm"] for _, r in pairs_attr]); pc = np.array([p["centroid_mm"] for p, _ in pairs_attr])
    mape = lambda a, b: float(np.mean(np.abs(a - b) / np.maximum(b, 1e-6))) * 100
    r_ = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None
    def terc(g, p):
        q = np.quantile(g, [1 / 3, 2 / 3]); return float(np.mean(np.digitize(g, q) == np.digitize(p, q))), [round(float(x), 2) for x in q]
    va, vq = terc(gv, pv); da, dq = terc(gd, pd_)
    return {"n_pairs": len(pairs_attr), "volume_MAPE_pct": round(mape(pv, gv), 1), "volume_pearson_r": round(r_(pv, gv), 3),
            "diameter_MAPE_pct": round(mape(pd_, gd), 1), "diameter_pearson_r": round(r_(pd_, gd), 3),
            "centroid_error_mm_mean": round(float(np.mean(np.linalg.norm(pc - gc, axis=1))), 2),
            "centroid_error_mm_median": round(float(np.median(np.linalg.norm(pc - gc, axis=1))), 2),
            "volume_tercile_agreement": round(va, 3), "volume_tercile_edges_cm3": vq,
            "diameter_tercile_agreement": round(da, 3), "diameter_tercile_edges_mm": dq,
            "multiplicity_agreement": round(float(np.mean([(p["n_comp"] >= 2) == (r["n_comp"] >= 2) for p, r in pairs_attr])), 3)}


def main():
    cids = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{PRED}/*.nii.gz") if os.path.exists(f"{REF}/{os.path.basename(f)}"))
    with ProcessPoolExecutor(32) as ex:
        res = list(ex.map(one, cids, chunksize=4))
    n_ref = sum(len(r["ref"]) for r in res); n_pred = sum(len(r["pred"]) for r in res); n_m = sum(len(r["matched_iou"]) for r in res)
    iou_pairs = [(r["pred"][a], r["ref"][b]) for r in res for a, b, _ in r["matched_iou"]]
    so_pairs = [(r["pred"][o], r["ref"][o]) for r in res for o in r["same_organ"]]
    per_organ = {o: stats([(r["pred"][a], r["ref"][b]) for r in res for a, b, _ in r["matched_iou"] if b == o]) for o in POOLED}
    out = {"n_cases": len(res), "n_cases_with_reference_lesion": sum(1 for r in res if r["ref"]),
           "protocol": "semi-oracle AUSAM (GT-box prompted) FLARE23 masks vs reference masks; lesion node = tumor voxels attributed to one organ; "
                       "Hungarian matching on 3-D IoU >= 0.3 (strict) and same-organ pairing (lenient)",
           "lesion_nodes": {"reference": n_ref, "predicted": n_pred, "matched_iou": n_m,
                            "node_precision": round(n_m / n_pred, 3) if n_pred else None, "node_recall": round(n_m / n_ref, 3) if n_ref else None,
                            "mean_iou_of_matched": round(float(np.mean([i for r in res for _, _, i in r["matched_iou"]])), 3) if n_m else None},
           "table4_strict_iou_matched": stats(iou_pairs), "table4_lenient_same_organ": stats(so_pairs), "per_organ_strict": per_organ,
           "per_case": [{k: v for k, v in r.items()} for r in res]}
    os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_cases", "n_cases_with_reference_lesion", "lesion_nodes", "table4_strict_iou_matched", "table4_lenient_same_organ")}, indent=1))
    print({o: (v["n_pairs"], v["volume_MAPE_pct"], v["diameter_MAPE_pct"]) if v else None for o, v in per_organ.items()})
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
