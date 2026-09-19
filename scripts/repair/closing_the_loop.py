#!/usr/bin/env python3
"""Closing-the-loop (draft p20): does KG-guided repair of AUTONOMOUS masks improve the REBUILT KG's
node fidelity (organ-volume MAPE) and downstream retrieval (mAP)?

Per FLARE23 case: fully-autonomous SAM3 (concept organs + generic tumor model, NO boxes) -> RAW mask;
KG-guided repair -> REPAIRED mask; GT from seg.nii.gz. Extract organ volumes + phenotypes from all three,
then compare:
  - node fidelity: organ-volume MAPE (raw vs GT) vs (repaired vs GT)   [the loop closes if repaired < raw]
  - retrieval: OAKG mAP over graphs built from raw / repaired / GT phenotypes, relevance scored vs GT
  - Dice raw vs repaired (sanity, matches kg_guided_eval)
Data: /scratch/user/acm_data/flare3d/{PID}/{CT,seg}.nii.gz  (multiclass seg; tumor=14).
-> results/closing_the_loop.json"""
import os, glob, json, sys
import numpy as np, nibabel as nib, torch
from scipy import ndimage
sys.path.insert(0, "/home/user/SWOG/src/scripts")
import infer_ensemble as IE
from run_pancreas_sam3 import _load_sam3_ckpt
from kg_guided_segment import repair, TUMOR
from kg_retrieval_v2 import similarity, relevant, _dcg
CASES = "/scratch/user/acm_data/flare3d"
TUMOR_CKPT = "/scratch/user/acm_data/ckpts/sam3_tumor_generic.pth"
OUT = "/home/user/SWOG/results/closing_the_loop.json"
NAME = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}
MINVOX = 50

def dice(a, b):
    s = a.sum() + b.sum(); return float(2 * np.logical_and(a, b).sum() / s) if s else None

def raw_organ_info(mask, sp_cm3):
    """Per-organ voxel volume + associated tumor voxels (tumor within a 3-vox organ dilation)."""
    tum = (mask == TUMOR); info = {}
    for lab, name in NAME.items():
        om = (mask == lab)
        if om.sum() < MINVOX: continue
        od = ndimage.binary_dilation(om, iterations=3); t_in = tum & od
        info[name] = {"vox": int(om.sum()), "vol_cm3": round(int(om.sum()) * sp_cm3, 2),
                      "tvox": int(t_in.sum()), "tcomp": int(ndimage.label(t_in)[1]),
                      "t_contained": (float((tum & om).sum()) / t_in.sum() >= 0.9) if t_in.sum() else None}
    return info

def record(cid, info, t1, t2):
    """corpus-style phenotype record from per-organ info + cohort burden tertiles (GT-derived)."""
    organs = {}
    for name, d in info.items():
        ht = d["tvox"] > 0
        organs[name] = {"present": True, "organ_volume_cm3": d["vol_cm3"], "has_tumor": ht,
                        "tumor_volume_cm3": round(d["tvox"] * 0 + d["tvox"], 2), "tumor_voxels": d["tvox"],
                        "burden_cat": ("low" if d["tvox"] < t1 else ("high" if d["tvox"] >= t2 else "medium")) if ht else "none",
                        "multiplicity": ("multifocal" if d["tcomp"] >= 2 else "solitary") if ht else "none",
                        "containment": ("contained" if d["t_contained"] else "boundary") if ht else "none",
                        "anatomic_location": "na"}
    return {"case_id": cid, "dataset": "flare", "observed_organs": list(organs), "organs": organs}

def mape(pred, gt):  # organ-volume MAPE over organs present in GT
    errs = []
    for name, g in gt.items():
        if name in pred and g["vox"] > 0:
            errs.append(abs(pred[name]["vox"] - g["vox"]) / g["vox"])
    return errs

def retrieval(recs):
    ids = list(recs); AP = []
    for q in ids:
        cands = [(similarity(recs[q], recs[b], "proposed"), b) for b in ids if b != q]
        cands = [(s, b) for s, b in cands if s is not None]
        if not cands: continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if relevant(gtrec[q], gtrec[b]) else 0 for _, b in cands]
        if sum(rels) == 0: continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r: hit += 1; ap += hit / (i + 1)
        AP.append(ap / sum(rels))
    return round(float(np.mean(AP)), 3) if AP else None, len(AP)

cids = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{CASES}/*/CT.nii.gz"))
n = int(sys.argv[1]) if len(sys.argv) > 1 else len(cids)
cids = cids[:n]
print(f"closing-the-loop over {len(cids)} FLARE23 cases", flush=True)
model, proc = IE.load_base(); tmodel = _load_sam3_ckpt(TUMOR_CKPT, "cuda")

raw_info, rep_info, gt_info, sp3 = {}, {}, {}, {}
dice_raw, dice_rep = {s: [] for s in NAME.values()}, {s: [] for s in NAME.values()}
for k, cid in enumerate(cids):
    ct = nib.load(f"{CASES}/{cid}/CT.nii.gz").get_fdata()
    nii = nib.load(f"{CASES}/{cid}/seg.nii.gz"); gt = np.asarray(nii.dataobj).astype(np.uint8)
    sp = nii.header.get_zooms()[:3]; sp_cm3 = float(np.prod(sp)) / 1000.0; sp3[cid] = sp_cm3
    pred = np.zeros(ct.shape, np.uint8)
    for concept, lab in IE.ORGANS:
        m = IE.seg_concept(ct, model, proc, concept); pred[m > 0] = lab
    tm = IE.seg_concept(ct, tmodel, proc, "tumor"); pred[(tm > 0) & (pred > 0)] = TUMOR
    rep, _ = repair(pred, sp)
    raw_info[cid] = raw_organ_info(pred, sp_cm3); rep_info[cid] = raw_organ_info(rep, sp_cm3); gt_info[cid] = raw_organ_info(gt, sp_cm3)
    for lab, name in NAME.items():
        g = (gt == lab)
        if g.sum() == 0: continue
        dice_raw[name].append(dice(pred == lab, g)); dice_rep[name].append(dice(rep == lab, g))
    print(f"[{k+1}/{len(cids)}] {cid}: organs raw {len(raw_info[cid])} rep {len(rep_info[cid])} gt {len(gt_info[cid])}", flush=True)
    torch.cuda.empty_cache()

# burden tertiles from GT per-organ tumor voxels
gtv = sorted(d["tvox"] for info in gt_info.values() for d in info.values() if d["tvox"] > 0)
t1, t2 = (np.percentile(gtv, [33.3, 66.6]) if gtv else (0, 0))
rawrec = {c: record(c, raw_info[c], t1, t2) for c in cids}
reprec = {c: record(c, rep_info[c], t1, t2) for c in cids}
gtrec = {c: record(c, gt_info[c], t1, t2) for c in cids}

mape_raw = np.concatenate([mape(raw_info[c], gt_info[c]) for c in cids]) if cids else np.array([])
mape_rep = np.concatenate([mape(rep_info[c], gt_info[c]) for c in cids]) if cids else np.array([])
map_raw, nr = retrieval(rawrec); map_rep, np_ = retrieval(reprec); map_gt, ng = retrieval(gtrec)

out = {
    "n_cases": len(cids),
    "node_fidelity_MAPE_pct": {"raw": round(float(mape_raw.mean()) * 100, 2), "repaired": round(float(mape_rep.mean()) * 100, 2),
                               "delta": round((float(mape_rep.mean()) - float(mape_raw.mean())) * 100, 2),
                               "n_organ_obs": int(len(mape_raw))},
    "retrieval_mAP": {"raw_graph": map_raw, "repaired_graph": map_rep, "gt_graph_upper_bound": map_gt},
    "dice": {s: {"raw": round(float(np.mean(dice_raw[s])), 3), "repaired": round(float(np.mean(dice_rep[s])), 3), "n": len(dice_raw[s])}
             for s in NAME.values() if dice_raw[s]},
    "interpretation": "loop closes if repaired MAPE < raw MAPE and repaired mAP > raw mAP (toward GT upper bound).",
}
json.dump(out, open(OUT, "w"), indent=2)
print("\n=== closing-the-loop ===")
print(json.dumps({k: out[k] for k in ("n_cases", "node_fidelity_MAPE_pct", "retrieval_mAP")}, indent=2))
