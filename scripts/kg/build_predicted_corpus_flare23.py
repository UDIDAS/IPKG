#!/usr/bin/env python3
"""PREDICTED-phenotype corpus for FLARE23 — the input the query suite (draft Tab. 6, rows Q4a/Q4b/Q5) and
the closed-loop comparisons need, built with the SAME protocol as corpus_predicted_{kits,lits,msd}.json
(build_predicted_corpus.py): per-dataset AUSAM organ model + tumor model, semi-oracle GT box per labeled
slice, 3-D assembly, then the shared phenotype schema.

Stage 1 (GPU, sharded):  python build_predicted_corpus_flare23.py infer --shard k/N
    organs : sam3_organ_generic_ausam_flare23.pth, text = organ name, GT box per slice (liver, right kidney,
             left kidney, spleen, pancreas — kidneys predicted per side, pooled into `kidney` afterwards)
    tumor  : sam3_tumor_ausam_flare.pth, text 'tumor', GT tumor box on tumor-present slices
    -> $VKG_DATA/flare23_predicted/{cid}.nii.gz (multi-label: 1/2/3/4/13 organs, 14 tumor) + {cid}.json
Stage 2 (CPU):           python build_predicted_corpus_flare23.py assemble
    per organ (kidneys pooled): organ_volume_cm3, has_tumor, tumor_volume_cm3, burden_cat (the GT corpus'
    volume terciles: <5.44 low, >=19.2 high), multiplicity (components of tumor within a 3-voxel organ
    dilation), containment (>=90 % inside), anatomic_location 'unknown', size_cat 'unknown'
    -> corpora/corpus_predicted_flare23.json
"""
import argparse
import glob
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.transform import resize

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "segmentation"))
import sam3_autonomous_local as IE                                   # noqa: E402  (hu_to_rgb, _load_sam3_ckpt, WIN)

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
VKG_DATA = IE.VKG_DATA
POOL = os.path.join(VKG_DATA, "flare23_pool")
OUTD = os.path.join(VKG_DATA, "flare23_predicted")
ORGAN_CKPT = os.path.join(VKG_DATA, "ckpts", "organ", "sam3_organ_generic_ausam_flare23.pth")
TUMOR_CKPT = os.path.join(VKG_DATA, "ckpts", "tumor", "sam3_tumor_ausam_flare.pth")
CORPUS_OUT = os.path.join(ROOT, "corpora", "corpus_predicted_flare23.json")
ORGANS = [("liver", [1], 1), ("right kidney", [2], 2), ("spleen", [3], 3), ("pancreas", [4], 4), ("left kidney", [13], 13)]
TUMOR = 14
MINPX = 50                      # as eval_ausam_3d.py: min organ px on the 256 mask to count a slice
T1, T2 = 5.44, 19.2             # cm3 terciles of the GT FLARE23 corpus (low < T1 <= medium < T2 <= high)


def bbox_from_mask(mask_2d, pad=3):
    ys, xs = np.where(mask_2d > 0)
    if len(xs) == 0:
        return None
    H, W = mask_2d.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(W - 1, int(xs.max()) + pad), min(H - 1, int(ys.max()) + pad)]


def predict_batch(model, proc, rgbs, boxes, text):
    import torch
    from torch.amp import autocast
    inp = proc(images=rgbs, text=[text] * len(rgbs), input_boxes=[[b] for b in boxes],
               input_boxes_labels=[[1]] * len(rgbs), return_tensors="pt")
    kw = {"pixel_values": inp["pixel_values"].to("cuda")}
    for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
        if inp.get(k) is not None:
            kw[k] = inp[k].to("cuda")
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        idx = out.pred_logits.sigmoid().argmax(dim=1)                       # extract_best_mask_soft, batched
        pm = out.pred_masks[torch.arange(len(rgbs)), idx].float().unsqueeze(1)
        pm = torch.nn.functional.interpolate(pm, size=(256, 256), mode="bilinear", align_corners=False)
    return (pm.sigmoid().squeeze(1).cpu().numpy() > 0.5)


def predict_volume(model, proc, ct, seg, labs, text, rgb_cache, batch=8):
    """Semi-oracle: run over slices where the GT structure is present (>= MINPX on 256), GT box prompt."""
    pred = np.zeros(ct.shape, bool)
    todo = []
    for z in range(ct.shape[2]):
        gm256 = resize(np.isin(seg[:, :, z], labs).astype(float), (256, 256), order=0, preserve_range=True) > 0.5
        if gm256.sum() < MINPX:
            continue
        todo.append((z, bbox_from_mask(gm256.astype(np.uint8), pad=3) or [0, 0, 255, 255]))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        prs = predict_batch(model, proc, [rgb_cache[z] for z, _ in chunk], [b for _, b in chunk], text)
        for (z, _), pr in zip(chunk, prs):
            pred[:, :, z] = resize(pr.astype(float), ct.shape[:2], order=0, preserve_range=True) > 0.5
    return pred


def infer(a):
    import torch
    from transformers import Sam3Processor
    k, n = (int(x) for x in a.shard.split("/"))
    os.makedirs(OUTD, exist_ok=True)
    cids = sorted(os.path.basename(f)[:-len("_0000.nii.gz")] for f in glob.glob(f"{POOL}/images/*_0000.nii.gz")
                  if os.path.exists(f"{POOL}/labels/{os.path.basename(f)[:-len('_0000.nii.gz')]}.nii.gz"))
    cids = [c for c in cids[k::n] if not os.path.exists(f"{OUTD}/{c}.json")][:a.limit]
    print(f"shard {k}/{n}: {len(cids)} cases to do", flush=True)
    if not cids:
        return
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    om = IE._load_sam3_ckpt(ORGAN_CKPT, "cuda"); tm = IE._load_sam3_ckpt(TUMOR_CKPT, "cuda")
    for i, cid in enumerate(cids):
        t0 = time.time()
        nii = nib.load(f"{POOL}/images/{cid}_0000.nii.gz"); ct = nii.get_fdata()
        seg = np.asarray(nib.load(f"{POOL}/labels/{cid}.nii.gz").dataobj).astype(np.uint8)
        sp = [float(z) for z in nii.header.get_zooms()[:3]]
        rgb = [IE.hu_to_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True), *IE.WIN).astype(np.uint8)
               for z in range(ct.shape[2])]
        ml = np.zeros(ct.shape, np.uint8); rec = {"cid": cid, "spacing": sp, "organs": {}}
        for name, labs, lab in ORGANS:
            if not np.isin(seg, labs).any():
                continue
            pm = predict_volume(om, proc, ct, seg, labs, name, rgb, a.batch)
            ml[pm] = lab
            g = np.isin(seg, labs); s = pm.sum() + g.sum()
            rec["organs"][name] = {"vox": int(pm.sum()), "gt_vox": int(g.sum()), "dice": round(float(2 * (pm & g).sum() / s), 4) if s else None}
        if (seg == TUMOR).any():
            tmask = predict_volume(tm, proc, ct, seg, [TUMOR], "tumor", rgb, a.batch)
            ml[tmask] = TUMOR
            g = seg == TUMOR; s = tmask.sum() + g.sum()
            rec["tumor"] = {"vox": int(tmask.sum()), "gt_vox": int(g.sum()), "dice": round(float(2 * (tmask & g).sum() / s), 4) if s else None}
        else:
            rec["tumor"] = {"vox": 0, "gt_vox": 0, "dice": None}
        nib.save(nib.Nifti1Image(ml, nii.affine), f"{OUTD}/{cid}.nii.gz")
        rec["seconds"] = round(time.time() - t0, 1)
        json.dump(rec, open(f"{OUTD}/{cid}.json", "w"), indent=1)
        print(f"[{i+1}/{len(cids)}] {cid} {rec['seconds']}s  " + " ".join(f"{o.split()[0][:3]}{o.split()[-1][0] if ' ' in o else ''} {v['dice']}" for o, v in rec["organs"].items())
              + f"  tumor {rec['tumor']['dice']}", flush=True)
        torch.cuda.empty_cache()


def assemble(a):
    POOLED = {"liver": [1], "kidney": [2, 13], "spleen": [3], "pancreas": [4]}
    recs = []
    for f in sorted(glob.glob(f"{OUTD}/*.json")):
        cid = os.path.basename(f)[:-5]
        nii = nib.load(f"{OUTD}/{cid}.nii.gz"); ml = np.asarray(nii.dataobj).astype(np.uint8)
        cm3 = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
        tum = ml == TUMOR
        # lesion -> organ attribution: each tumor COMPONENT goes to the single organ whose 3-voxel dilation it
        # overlaps most (winner-take-all, as flare23_predict.py attributed by overlap). A component touching no
        # organ is unattributed. (The naive rule "tumor within any organ's dilation" credits a lesion to every
        # neighbouring organ and fabricates cross-organ involvement — 68/576 cases in the first assembly.)
        dil = {o: ndimage.binary_dilation(np.isin(ml, labs), iterations=3) for o, labs in POOLED.items() if np.isin(ml, labs).any()}
        lbl, ncomp = ndimage.label(tum)
        assigned = {o: np.zeros(ml.shape, bool) for o in dil}
        for c in range(1, ncomp + 1):
            comp = lbl == c
            ov = {o: int((comp & d).sum()) for o, d in dil.items()}
            best = max(ov, key=ov.get) if ov else None
            if best is not None and ov[best] > 0:
                assigned[best] |= comp
        organs = {}
        for o, labs in POOLED.items():
            om = np.isin(ml, labs)
            if om.sum() == 0:
                continue
            t_in = assigned[o]
            tvox = int(t_in.sum()); ht = tvox > 0; tv = tvox * cm3
            organs[o] = {"present": True, "organ_volume_cm3": round(om.sum() * cm3, 2), "has_tumor": ht,
                         "tumor_volume_cm3": round(tv, 2), "tumor_voxels": tvox,
                         "burden_cat": ("low" if tv < T1 else ("high" if tv >= T2 else "medium")) if ht else "none",
                         "multiplicity": ("multifocal" if ndimage.label(t_in)[1] >= 2 else "solitary") if ht else "none",
                         "containment": ("contained" if (tum & om).sum() / tvox >= 0.9 else "boundary") if ht else "none",
                         "anatomic_location": "unknown", "size_cat": "unknown"}
        recs.append({"case_id": f"flare23_{cid}", "dataset": "flare23", "granularity": "volume",
                     "observed_organs": sorted(organs), "organs": organs})
    json.dump({"dataset": "flare23", "n": len(recs), "note": "Semi-oracle AUSAM predicted phenotypes (GT-box prompts, "
               "sam3_organ_generic_ausam_flare23 + sam3_tumor_ausam_flare); kidneys pooled; burden terciles of the GT corpus; "
               "each tumor component attributed to the organ of maximal overlap (3-voxel dilation).",
               "records": recs}, open(CORPUS_OUT, "w"), indent=1)
    n_t = sum(any(o["has_tumor"] for o in r["organs"].values()) for r in recs)
    d = [json.load(open(f)) for f in sorted(glob.glob(f"{OUTD}/*.json"))]
    md = {o: round(float(np.mean([x["organs"][o]["dice"] for x in d if o in x["organs"] and x["organs"][o]["dice"] is not None])), 3)
          for o in ("liver", "right kidney", "spleen", "pancreas", "left kidney")}
    md["tumor"] = round(float(np.mean([x["tumor"]["dice"] for x in d if x["tumor"]["dice"] is not None])), 3)
    print(f"FLARE23 predicted corpus: {len(recs)} cases ({n_t} with tumor); mean semi-oracle Dice {md} -> {CORPUS_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["infer", "assemble"])
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    infer(a) if a.stage == "infer" else assemble(a)
