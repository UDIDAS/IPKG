#!/usr/bin/env python3
"""Draft Tables 12 & 13 (+ Table 3 autonomous columns): repair vs generic post-processing, and the
closed loop (graph rebuilt from each mask source vs the reference-mask graph).  CPU-only; consumes the raw
autonomous masks cached by scripts/segmentation/autonomous_infer_flare.py.

Mask sources per patient (FLARE23 full-label cases):
  (a) raw      : autonomous SAM3 (concept organs + generic tumor, tumor kept inside an organ)
  (b) postproc : GENERIC control (draft §4.6) — per-class largest connected component, then morphological
                 opening with a 3 mm spherical structuring element applied at 1 mm isotropic resampling
                 (spacing-independent); no anatomical knowledge.  Variant (b') = same but the tumor class is
                 exempt from LCC (opening only), reported as a sensitivity row.
  (c) repair   : ontology-guided repair (kg_guided_segment.repair): LCC per organ + tumor components not
                 adjacent to any organ removed (+ volume-band flags, which never edit the mask)
  ref          : reference masks (ceiling)

Per source and patient the graph record is built exactly as closing_the_loop.py did (raw_organ_info /
record): organ nodes = organs with >= MINVOX voxels; the lesion node of organ o = tumor voxels within a
3-voxel dilation of o (tvox, components, containment); burden terciles from the reference cohort.

Metrics
  Tab. 12  Dice per structure for (a),(b),(b'),(c); Δ vs raw with patient-level bootstrap CI + permutation p.
  Tab. 13  organ-volume MAPE (pooled over organ observations, as before; strict variant counts a missing
           organ node as 100 % error); organ/site fidelity; triple F1 (macro over patients); retrieval with
           the patients as queries (mAP, mismatch@10, spurious@10; reference-graph relevance).
           Contrasts (c)-(b) [primary], (c)-(a), (b)-(a) on triple F1 with Holm within family F3 + exact
           sign test for (c)-(b); secondary metrics with CIs and raw p.
  Tab. 3   autonomous per-organ volume r / MAPE (liver, kidney pooled L+R, pancreas, + spleen and per-side).

Node correspondence (draft §4.6): organ nodes by canonical identity; lesion nodes by patient-wise Hungarian
assignment on 3-D IoU of the per-organ tumor masks with acceptance IoU >= 0.3; unmatched predicted lesions
contribute all their triples as FP, unmatched reference lesions all theirs as FN.
Triple set per patient mirrors kg_build_graph.py: (case, depicts_organ, organ), (organ, mapped_to_concept,
organ-concept), (organ, has_lesion, lesion), (lesion, located_in, organ), (lesion, mapped_to_concept,
tumor-concept), (lesion, has_observation, burden|multiplicity|containment = value).  Case-level boilerplate
(from_dataset, of_patient) is omitted as identical across sources.

Usage: python closed_loop_tab12_13.py [--cases LIST.txt] [--workers 32]
Output: results/repair/closed_loop_tab12_13.json
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "retrieval"))
from kg_guided_segment import repair, ORGAN_LABELS, TUMOR          # noqa: E402
from retrieval_core_local import similarity, relevant, _dcg        # noqa: E402

VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
CASES = os.path.join(VKG_DATA, "flare_full_cases")
RAW = os.path.join(VKG_DATA, "autonomous_raw")
OUT = os.environ.get("VKG_OUT", os.path.join(_HERE, "..", "..", "results", "repair", "closed_loop_tab12_13.json"))

NAME = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}
STRUCTS = [("liver", 1), ("right_kidney", 2), ("spleen", 3), ("pancreas", 4), ("left_kidney", 13), ("tumor", TUMOR)]
SOURCES = ["raw", "postproc", "postproc_tumor_exempt", "repair", "ref"]
MINVOX = 50
IOU_MIN = 0.3
B_BOOT, B_PERM = 5000, 20000
rng = np.random.default_rng(12345)
CATS = ["burden_cat", "multiplicity", "containment", "anatomic_location"]


# ------------------------------------------------------------------ mask operations
def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else None


def _ball(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def lcc(m):
    lbl, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lbl, range(1, n + 1))
    return lbl == (int(np.argmax(sizes)) + 1)


def opening_3mm_iso(m, spacing, radius_mm=3.0):
    """Morphological opening with a spherical SE of radius_mm applied at 1 mm isotropic resampling."""
    if not m.any():
        return m
    zoom = np.asarray(spacing, float)                    # voxel -> mm : resample to 1 mm iso
    iso = ndimage.zoom(m.astype(np.uint8), zoom, order=0) > 0
    iso = ndimage.binary_opening(iso, structure=_ball(int(round(radius_mm))))
    back = ndimage.zoom(iso.astype(np.uint8), np.array(m.shape) / np.array(iso.shape), order=0) > 0
    out = np.zeros(m.shape, bool)
    sl = tuple(slice(0, min(a, b)) for a, b in zip(m.shape, back.shape))
    out[sl] = back[sl]
    return out


def postproc(mask, spacing, tumor_lcc=True):
    """Generic control: per-class LCC then 3 mm opening at 1 mm iso. Only removes voxels -> no overlaps."""
    out = np.zeros(mask.shape, np.uint8)
    for lab in list(ORGAN_LABELS.values()) + [TUMOR]:
        m = mask == lab
        if not m.any():
            continue
        if lab != TUMOR or tumor_lcc:
            m = lcc(m)
        m = opening_3mm_iso(m, spacing)
        out[m] = lab
    return out


def organ_info(mask, sp_cm3):
    """Per-organ voxel volume + associated tumor (within a 3-vox organ dilation) + the lesion mask itself."""
    tum = mask == TUMOR
    info, lesions = {}, {}
    for lab, name in NAME.items():
        om = mask == lab
        if om.sum() < MINVOX:
            continue
        od = ndimage.binary_dilation(om, iterations=3)
        t_in = tum & od
        info[name] = {"vox": int(om.sum()), "vol_cm3": round(int(om.sum()) * sp_cm3, 2),
                      "tvox": int(t_in.sum()), "tcomp": int(ndimage.label(t_in)[1]),
                      "t_contained": (float((tum & om).sum()) / t_in.sum() >= 0.9) if t_in.sum() else None}
        if t_in.any():
            lesions[name] = t_in
    return info, lesions


def process_case(cid):
    """Worker: all mask sources for one patient -> small dict (no arrays except the IoU matrices)."""
    nii = nib.load(f"{CASES}/{cid}_label.nii.gz")
    gt = np.asarray(nii.dataobj).astype(np.uint8)
    sp = [float(z) for z in nii.header.get_zooms()[:3]]
    sp_cm3 = float(np.prod(sp)) / 1000.0
    raw = np.asarray(nib.load(f"{RAW}/{cid}.nii.gz").dataobj).astype(np.uint8)
    masks = {"raw": raw,
             "postproc": postproc(raw, sp, tumor_lcc=True),
             "postproc_tumor_exempt": postproc(raw, sp, tumor_lcc=False),
             "repair": repair(raw, sp)[0],
             "ref": gt}
    out = {"cid": cid, "spacing": sp, "dice": {}, "info": {}, "iou": {}, "lesion_order": {}}
    ref_info, ref_les = organ_info(gt, sp_cm3)
    ref_order = sorted(ref_les)
    for src, m in masks.items():
        out["dice"][src] = {s: dice(m == lab, gt == lab) for s, lab in STRUCTS if (gt == lab).any()}
        info, les = organ_info(m, sp_cm3)
        out["info"][src] = info
        order = sorted(les)
        out["lesion_order"][src] = order
        iou = np.zeros((len(order), len(ref_order)))
        for i, a in enumerate(order):
            for j, b in enumerate(ref_order):
                inter = np.logical_and(les[a], ref_les[b]).sum()
                union = np.logical_or(les[a], ref_les[b]).sum()
                iou[i, j] = inter / union if union else 0.0
        out["iou"][src] = iou.tolist()
    out["lesion_order"]["ref"] = ref_order
    return out


# ------------------------------------------------------------------ graph records / triples
def record(cid, info, t1, t2):
    organs = {}
    for name, d in info.items():
        ht = d["tvox"] > 0
        organs[name] = {"present": True, "organ_volume_cm3": d["vol_cm3"], "has_tumor": ht,
                        "tumor_volume_cm3": d["tvox"], "tumor_voxels": d["tvox"],
                        "burden_cat": ("low" if d["tvox"] < t1 else ("high" if d["tvox"] >= t2 else "medium")) if ht else "none",
                        "multiplicity": ("multifocal" if d["tcomp"] >= 2 else "solitary") if ht else "none",
                        "containment": ("contained" if d["t_contained"] else "boundary") if ht else "none",
                        "anatomic_location": "na"}
    return {"case_id": cid, "dataset": "flare", "observed_organs": list(organs), "organs": organs}


def triples(rec, lesion_alias):
    """Typed triples of one patient graph. lesion_alias maps this source's lesion organ -> canonical lesion
    id (the matched reference lesion's organ, or 'UNMATCHED:<organ>')."""
    T = set()
    for o, d in rec["organs"].items():
        T.add(("case", "depicts_organ", f"organ:{o}"))
        T.add((f"organ:{o}", "mapped_to_concept", f"concept:{o}"))
        if d["has_tumor"]:
            L = f"lesion:{lesion_alias.get(o, 'UNMATCHED:' + o)}"
            T.add((f"organ:{o}", "has_lesion", L))
            T.add((L, "located_in", f"organ:{o}"))
            T.add((L, "mapped_to_concept", f"concept:tumor_{o}"))
            for f in ("burden_cat", "multiplicity", "containment"):
                if d[f] not in ("none", "na", None):
                    T.add((L, "has_observation", f"{f}={d[f]}"))
    return T


def match_lesions(iou, order, ref_order):
    """Hungarian on IoU, accept >= IOU_MIN. Returns {pred organ: ref organ}."""
    iou = np.asarray(iou)
    if iou.size == 0:
        return {}
    r, c = linear_sum_assignment(-iou)
    return {order[i]: ref_order[j] for i, j in zip(r, c) if iou[i, j] >= IOU_MIN}


def prf(pred, ref):
    tp = len(pred & ref); fp = len(pred - ref); fn = len(ref - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def organ_site_fidelity(rec, ref_rec, alias):
    """Proportion of reference nodes reproduced with correct facts, spurious predicted nodes in the
    denominator: organ nodes by identity; lesion nodes matched (IoU) with located_in + containment equal."""
    ro, po = set(ref_rec["organs"]), set(rec["organs"])
    organ_ok = len(ro & po)
    rl = {o for o, d in ref_rec["organs"].items() if d["has_tumor"]}
    pl = {o for o, d in rec["organs"].items() if d["has_tumor"]}
    les_ok = 0
    for o in pl:
        m = alias.get(o)
        if m is None or m not in rl:
            continue
        if m == o and rec["organs"][o]["containment"] == ref_rec["organs"][m]["containment"]:
            les_ok += 1                       # located_in target (organ identity) and containment agree
    n_ref = len(ro) + len(rl)
    n_spur = len(po - ro) + sum(1 for o in pl if alias.get(o) not in rl)
    denom = n_ref + n_spur
    return {"organ_identity": organ_ok / len(ro) if ro else None,
            "lesion_site": les_ok / len(rl) if rl else None,
            "fidelity": (organ_ok + les_ok) / denom if denom else None}


# ------------------------------------------------------------------ retrieval
def organ_agree(qp, cp):
    n = 0
    for k in CATS:
        va = qp.get(k)
        if va == cp.get(k) and va not in ("none", "na", "unknown", None):
            n += 1
    return n >= 2


def mismatch10_multi(q_rel, cand_rels):
    """Cross-organ mismatch generalised to multi-organ queries: over the query's tumor-bearing organs
    (reference record), a top-10 candidate is a mismatch if it agrees with none of them at the same organ
    but agrees with one of the query's organ phenotypes at a different organ. None if the query has no
    informative organ."""
    qorgs = [o for o, d in q_rel["organs"].items() if d["has_tumor"]]
    if not qorgs:
        return None
    flags = []
    for c in cand_rels:
        same = any(o in c["organs"] and organ_agree(q_rel["organs"][o], c["organs"][o]) for o in qorgs)
        elsewhere = any(organ_agree(q_rel["organs"][o], p) for o in qorgs for o2, p in c["organs"].items() if o2 != o)
        flags.append(1.0 if (not same and elsewhere) else 0.0)
    return float(np.mean(flags))


def retrieval(recs, relrecs):
    out = {}
    ids = sorted(recs)
    for q in ids:
        cands = [(similarity(recs[q], recs[b], "proposed"), b) for b in ids if b != q]
        cands = [(s, b) for s, b in cands if s is not None]
        if not cands:
            continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if relevant(relrecs[q], relrecs[b]) else 0 for _, b in cands]
        if sum(rels) == 0:
            continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        top10 = [b for _, b in cands[:10]]
        qobs = set(recs[q]["observed_organs"])
        out[q] = {"ap": ap / sum(rels), "p10": float(np.mean(rels[:10])),
                  "ndcg": _dcg(rels[:10]) / (_dcg(sorted(rels, reverse=True)[:10]) or 1),
                  "spur10": float(np.mean([0.0 if (qobs & set(recs[b]["observed_organs"])) else 1.0 for b in top10])),
                  "mism10": mismatch10_multi(relrecs[q], [relrecs[b] for b in top10])}
    return out


# ------------------------------------------------------------------ statistics
def paired(a, b):
    """a, b: {patient: value}; returns mean delta a-b with bootstrap CI and sign-flip permutation p."""
    keys = sorted(k for k in set(a) & set(b) if a[k] is not None and b[k] is not None)
    dif = np.array([a[k] - b[k] for k in keys], float)
    if not len(dif):
        return None
    obs = float(dif.mean())
    boots = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(B_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(B_PERM, len(dif)))
    p = float((np.sum(np.abs((signs * dif).mean(axis=1)) >= abs(obs)) + 1) / (B_PERM + 1))
    return {"n": len(keys), "delta": round(obs, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "p_raw": p,
            "improved": int((dif > 0).sum()), "worsened": int((dif < 0).sum()), "tied": int((dif == 0).sum())}


def sign_test(d):
    from math import comb
    n = d["improved"] + d["worsened"]
    if n == 0:
        return 1.0
    k = min(d["improved"], d["worsened"])
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def holm(pdict):
    items = sorted([(k, v) for k, v in pdict.items() if v is not None], key=lambda kv: kv[1]["p_raw"])
    m = len(items)
    for rank, (k, v) in enumerate(items):
        v["p_holm"] = round(min(1.0, (m - rank) * v["p_raw"]), 5)
    return pdict


def ci_mean(vals):
    v = np.array([x for x in vals if x is not None], float)
    if not len(v):
        return None
    boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B_BOOT)]
    return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None, help="text file with one case id per line (default: all cached)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cases:
        cids = [l.strip() for l in open(a.cases) if l.strip()]
    else:
        cids = sorted(os.path.basename(f)[:-len("_ct.nii.gz")] for f in glob.glob(f"{CASES}/*_ct.nii.gz"))
    cids = [c for c in cids if os.path.exists(f"{RAW}/{c}.nii.gz")]
    print(f"closed loop over {len(cids)} cases, {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        per = {r["cid"]: r for r in ex.map(process_case, cids)}
    print("mask processing done", flush=True)

    # ---- Tab. 12: Dice
    tab12 = {}
    for s, _ in STRUCTS:
        row = {}
        for src in SOURCES:
            vals = {c: per[c]["dice"][src].get(s) for c in cids if s in per[c]["dice"][src]}
            row[src] = {"mean": round(float(np.mean([v for v in vals.values() if v is not None])), 3),
                        "n": sum(v is not None for v in vals.values()), "ci95": ci_mean(vals.values())}
        for src in ("postproc", "postproc_tumor_exempt", "repair"):
            A = {c: per[c]["dice"][src].get(s) for c in cids}; Bv = {c: per[c]["dice"]["raw"].get(s) for c in cids}
            row[f"delta_{src}_minus_raw"] = paired(A, Bv)
        A = {c: per[c]["dice"]["repair"].get(s) for c in cids}; Bv = {c: per[c]["dice"]["postproc"].get(s) for c in cids}
        row["delta_repair_minus_postproc"] = paired(A, Bv)
        tab12[s] = row

    # ---- graph records (burden terciles from the reference cohort, as in closing_the_loop.py)
    gtv = sorted(d["tvox"] for c in cids for d in per[c]["info"]["ref"].values() if d["tvox"] > 0)
    t1, t2 = (np.percentile(gtv, [33.3, 66.6]) if gtv else (0, 0))
    recs = {src: {c: record(c, per[c]["info"][src], t1, t2) for c in cids} for src in SOURCES}

    # lesion-level overlap behind the lesion-node recall: best IoU of each reference lesion node over the
    # source's lesion nodes (0 when the source has none), pooled over patients
    lesion_iou = {}
    for src in SOURCES:
        best = []
        for c in cids:
            iou = np.asarray(per[c]["iou"][src]); nref = len(per[c]["lesion_order"]["ref"])
            if nref == 0:
                continue
            best += list(iou.max(axis=0)) if iou.size else [0.0] * nref
        b = np.array(best)
        lesion_iou[src] = {"n_reference_lesion_nodes": int(len(b)), "mean_best_iou": round(float(b.mean()), 3), "median_best_iou": round(float(np.median(b)), 3),
                           "frac_iou_ge_0.3": round(float((b >= 0.3).mean()), 3), "frac_iou_ge_0.1": round(float((b >= 0.1).mean()), 3),
                           "frac_iou_eq_0": round(float((b == 0).mean()), 3)}

    tab13, per_patient = {}, {}
    for src in SOURCES:
        mape_pool, mape_strict_pool, pat_mape, fid, f1s, prs = [], [], {}, {}, {}, {}
        for c in cids:
            info, ref = per[c]["info"][src], per[c]["info"]["ref"]
            errs = [abs(info[o]["vox"] - g["vox"]) / g["vox"] for o, g in ref.items() if o in info and g["vox"] > 0]
            strict = [abs(info[o]["vox"] - g["vox"]) / g["vox"] if o in info else 1.0 for o, g in ref.items() if g["vox"] > 0]
            mape_pool += errs; mape_strict_pool += strict
            pat_mape[c] = float(np.mean(strict)) if strict else None
            alias = match_lesions(per[c]["iou"][src], per[c]["lesion_order"][src], per[c]["lesion_order"]["ref"]) \
                if src != "ref" else {o: o for o in per[c]["lesion_order"]["ref"]}
            fid[c] = organ_site_fidelity(recs[src][c], recs["ref"][c], alias)
            P = triples(recs[src][c], alias); R = triples(recs["ref"][c], {o: o for o in per[c]["lesion_order"]["ref"]})
            p, r, f = prf(P, R); prs[c] = (p, r); f1s[c] = f
        ret = retrieval(recs[src], recs["ref"])
        # lesion-node bookkeeping: how many lesion nodes each graph carries and how many are real
        n_les = [sum(d["has_tumor"] for d in recs[src][c]["organs"].values()) for c in cids]
        n_ref_les = [sum(d["has_tumor"] for d in recs["ref"][c]["organs"].values()) for c in cids]
        matched = [len(match_lesions(per[c]["iou"][src], per[c]["lesion_order"][src], per[c]["lesion_order"]["ref"])) if src != "ref" else n_ref_les[i]
                   for i, c in enumerate(cids)]
        lesion_stats = {"mean_lesion_nodes_per_patient": round(float(np.mean(n_les)), 2),
                        "mean_reference_lesion_nodes": round(float(np.mean(n_ref_les)), 2),
                        "lesion_node_precision": round(sum(matched) / sum(n_les), 3) if sum(n_les) else None,
                        "lesion_node_recall": round(sum(matched) / sum(n_ref_les), 3) if sum(n_ref_les) else None}
        per_patient[src] = {"triple_f1": f1s, "fidelity": {c: fid[c]["fidelity"] for c in cids},
                            "mape_strict": pat_mape, "ap": {q: v["ap"] for q, v in ret.items()},
                            "mism10": {q: v["mism10"] for q, v in ret.items()}, "spur10": {q: v["spur10"] for q, v in ret.items()}}
        mean_ = lambda d: round(float(np.mean([v for v in d.values() if v is not None])), 3) if any(v is not None for v in d.values()) else None
        tab13[src] = {
            "volume_MAPE_pct": round(float(np.mean(mape_pool)) * 100, 1), "n_organ_obs": len(mape_pool),
            "volume_MAPE_pct_strict": round(float(np.mean(mape_strict_pool)) * 100, 1),
            "organ_site_fidelity": mean_({c: fid[c]["fidelity"] for c in cids}),
            "organ_identity": mean_({c: fid[c]["organ_identity"] for c in cids}),
            "lesion_site": mean_({c: fid[c]["lesion_site"] for c in cids}),
            "triple_P": round(float(np.mean([prs[c][0] for c in cids])), 3),
            "triple_R": round(float(np.mean([prs[c][1] for c in cids])), 3),
            "triple_F1": round(float(np.mean([f1s[c] for c in cids])), 3), "triple_F1_ci95": ci_mean(f1s.values()),
            **lesion_stats,
            "n_queries": len(ret), "mAP": mean_({q: v["ap"] for q, v in ret.items()}),
            "P@10": mean_({q: v["p10"] for q, v in ret.items()}), "nDCG": mean_({q: v["ndcg"] for q, v in ret.items()}),
            "mismatch@10": mean_({q: v["mism10"] for q, v in ret.items()}), "spurious@10": mean_({q: v["spur10"] for q, v in ret.items()}),
        }

    # ---- contrasts. Family F3 = triple F1: (c)-(b) primary, (c)-(a), (b)-(a).
    pp = per_patient
    F3 = {"(c)-(b) repair - postproc": paired(pp["repair"]["triple_f1"], pp["postproc"]["triple_f1"]),
          "(c)-(a) repair - raw": paired(pp["repair"]["triple_f1"], pp["raw"]["triple_f1"]),
          "(b)-(a) postproc - raw": paired(pp["postproc"]["triple_f1"], pp["raw"]["triple_f1"])}
    holm(F3)
    F3["(c)-(b) repair - postproc"]["sign_test_p"] = round(sign_test(F3["(c)-(b) repair - postproc"]), 5)
    secondary = {}
    for metric in ("fidelity", "mape_strict", "ap", "mism10", "spur10"):
        secondary[metric] = {"(c)-(b)": paired(pp["repair"][metric], pp["postproc"][metric]),
                             "(c)-(a)": paired(pp["repair"][metric], pp["raw"][metric]),
                             "(b)-(a)": paired(pp["postproc"][metric], pp["raw"][metric]),
                             "(b')-(a)": paired(pp["postproc_tumor_exempt"][metric], pp["raw"][metric])}

    # ---- Tab. 3 autonomous columns: per-organ volume r / MAPE vs reference
    tab3 = {}
    for src in ("raw", "postproc", "repair"):
        tab3[src] = {}
        groups = {"liver": ["liver"], "kidney (L+R pooled)": ["left_kidney", "right_kidney"], "pancreas": ["pancreas"],
                  "spleen": ["spleen"], "left_kidney": ["left_kidney"], "right_kidney": ["right_kidney"]}
        for g, organs in groups.items():
            pv, gv = [], []
            for c in cids:
                info, ref = per[c]["info"][src], per[c]["info"]["ref"]
                if all(o in ref for o in organs):
                    gv.append(sum(ref[o]["vol_cm3"] for o in organs))
                    pv.append(sum(info[o]["vol_cm3"] for o in organs if o in info))
            pv, gv = np.array(pv), np.array(gv)
            ok = gv > 0
            r = float(np.corrcoef(pv[ok], gv[ok])[0, 1]) if ok.sum() > 2 else None
            tab3[src][g] = {"n": int(ok.sum()), "r": round(r, 3) if r is not None else None,
                            "MAPE_pct": round(float(np.mean(np.abs(pv[ok] - gv[ok]) / gv[ok])) * 100, 1)}

    out = {"n_cases": len(cids), "cases": cids, "burden_terciles_vox": [float(t1), float(t2)],
           "definitions": {"postproc": "per-class LCC then opening, spherical SE r=3 mm at 1 mm isotropic (draft §4.6); "
                                       "postproc_tumor_exempt: tumor class opening only",
                           "repair": "kg_guided_segment.repair: LCC per organ + drop tumor components not within 3 vox of an organ",
                           "volume_MAPE_pct": "pooled over reference organ observations with a predicted organ node (closing_the_loop.py definition)",
                           "volume_MAPE_pct_strict": "as above but a missing predicted organ node counts as 100 % error",
                           "organ_site_fidelity": "(organ nodes with correct identity + lesion nodes matched by IoU>=0.3 with correct located_in and containment) / (reference nodes + spurious predicted nodes)",
                           "triple_F1": "macro over patients; triples mirror kg_build_graph.py edges; lesion nodes matched patient-wise by Hungarian IoU>=0.3",
                           "retrieval": "queries = the patients, candidates = the other patients of the same mask source, relevance from the reference-graph records; mismatch@10 generalised to multi-organ queries",
                           "stats": "paired patient bootstrap B=5000, sign-flip permutation B=20000, Holm within F3 (triple F1)"},
           "tab12_dice": tab12, "tab13": tab13, "lesion_node_best_iou": lesion_iou, "F3_triple_F1_contrasts": F3, "secondary_contrasts": secondary,
           "tab3_autonomous_volume_fidelity": tab3,
           "per_patient": {src: {k: v for k, v in pp[src].items()} for src in SOURCES}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    print("\n=== Tab. 12 Dice: raw | postproc | postproc(tumor exempt) | repair ===")
    for s, row in tab12.items():
        print(f"  {s:13s} {row['raw']['mean']:.3f} | {row['postproc']['mean']:.3f} | {row['postproc_tumor_exempt']['mean']:.3f} | {row['repair']['mean']:.3f}"
              f"   Δ(c-a) {row['delta_repair_minus_raw']['delta']:+.3f}  Δ(c-b) {row['delta_repair_minus_postproc']['delta']:+.3f} p={row['delta_repair_minus_postproc']['p_raw']:.4f}")
    print("\n=== Tab. 13 ===")
    for src in SOURCES:
        t = tab13[src]
        print(f"  {src:22s} MAPE {t['volume_MAPE_pct']:6.1f}% (strict {t['volume_MAPE_pct_strict']:6.1f}%)  fidelity {t['organ_site_fidelity']}  "
              f"tripleF1 {t['triple_F1']} (P {t['triple_P']} R {t['triple_R']})  lesion nodes/pt {t['mean_lesion_nodes_per_patient']} "
              f"(ref {t['mean_reference_lesion_nodes']}; prec {t['lesion_node_precision']} rec {t['lesion_node_recall']})  "
              f"mAP {t['mAP']}  mism@10 {t['mismatch@10']}  spur@10 {t['spurious@10']}")
    print("  lesion-node best IoU per reference lesion:", json.dumps(lesion_iou, indent=None))
    print("  F3:", json.dumps(F3, indent=None))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
