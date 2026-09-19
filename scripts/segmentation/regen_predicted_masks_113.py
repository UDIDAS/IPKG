#!/usr/bin/env python3
"""Regenerate the 113-cohort single-organ PREDICTED masks locally (v11 item 2, predicted half).

The frozen masks behind corpus_predicted_{kits,lits,msd}.json are unavailable (KiTS/MSD on the
Delta store, LiTS gone entirely), so the containment-v2 re-extraction regenerates them with the
same SEMI-ORACLE protocol as build_predicted_corpus.py — per labeled slice (GT structure
>= 50 px at 256^2): GT box + 3 px pad, text prompt, predict at 256^2, resize back to the native
slice — on the locally available weights:
    organ pass: base facebook/sam3 (text = organ name)
    tumor pass: base + the FLARE23-only tumor expert (~/hmmkg_ckpts/sam3_tumor_flare_only.pth)
The 2026-09-16 Drive round measured the FLARE expert within ~0.007 DSC of the per-dataset
experts under this protocol (LiTS 0.778 vs 0.785, MSD 0.815 vs 0.818, KiTS 0.903 vs 0.910), and
per-case fidelity vs the FROZEN corpora is recorded here (has_tumor concordance, organ/tumor
voxel ratios) so the downstream containment estimate carries its own validation.

Datasets/grids (matching the frozen corpora): kits native (slice axis 0), msd native (axis 2),
lits = the 256^2 liver-slab npy grid (axis 0) — the grid the frozen LiTS corpus was built on.
Masks are saved EXCLUSIVE (organ 1, tumor 2, tumor wins) for the delivered extractor; the raw
independent organ/tumor voxel counts (and their overlap) go to the per-case sidecar JSON.

  HF_TOKEN=... python regen_predicted_masks_113.py [--datasets kits lits msd] [--batch 8]
-> /path/to/staging/acm_data/masks_predicted_113_regen/{ds}/{cid}.nii.gz + {cid}.json
-> results/segmentation/regen_masks_113_fidelity.json (summary)
"""
import argparse
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from skimage.transform import resize

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
import sam3_autonomous_local as IE                      # noqa: E402  hu_to_rgb, WIN, _load_sam3_ckpt

KITS = "/path/to/staging/acm_data/kits_vol"
LITS = "/path/to/staging/acm_data/lits_npy"
MSD_I = "/path/to/staging/acm_data/msd_pancreas_images"
MSD_L = "/path/to/staging/acm_data/msd_pancreas_labels"
OUTD = "/path/to/staging/acm_data/masks_predicted_113_regen"
TUMOR_OVERLAY = os.path.expanduser("~/hmmkg_ckpts/sam3_tumor_flare_only.pth")
SUMMARY = os.path.join(ROOT, "results", "segmentation", "regen_masks_113_fidelity.json")
MINPX = 50
CFG = {"kits": {"organ": "kidney", "ax": 0}, "lits": {"organ": "liver", "ax": 0}, "msd": {"organ": "pancreas", "ax": 2}}


def load(ds, cid):
    if ds == "kits":
        nii = nib.load(f"{KITS}/{cid}/imaging.nii.gz")
        return nii.get_fdata(), np.asarray(nib.load(f"{KITS}/{cid}/segmentation.nii.gz").dataobj).astype(np.uint8), nii.affine
    if ds == "lits":
        vid = cid.replace("volume-", "")
        return np.load(f"{LITS}/ct/volume-{vid}.npy"), np.load(f"{LITS}/seg/segmentation-{vid}.npy").astype(np.uint8), np.eye(4)
    nii = nib.load(f"{MSD_I}/{cid}.nii.gz")
    return nii.get_fdata(), np.asarray(nib.load(f"{MSD_L}/{cid}.nii.gz").dataobj).astype(np.uint8), nii.affine


def bbox(m, pad=3):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    H, W = m.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(W - 1, int(xs.max()) + pad), min(H - 1, int(ys.max()) + pad)]


def sl(vol, ax, z):
    return vol[z] if ax == 0 else vol[:, :, z]


def put(pred, ax, z, s):
    if ax == 0:
        pred[z] = s
    else:
        pred[:, :, z] = s


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
        idx = out.pred_logits.sigmoid().argmax(dim=1)
        pm = out.pred_masks[torch.arange(len(rgbs)), idx].float().unsqueeze(1)
        pm = torch.nn.functional.interpolate(pm, size=(256, 256), mode="bilinear", align_corners=False)
    return (pm.sigmoid().squeeze(1).cpu().numpy() > 0.5)


def predict_volume(model, proc, shape, seg, lab, text, rgb_cache, ax, batch):
    pred = np.zeros(shape, bool)
    todo = []
    for z in range(shape[ax]):
        gm = sl(seg, ax, z) == lab
        gm256 = gm if gm.shape == (256, 256) else (resize(gm.astype(float), (256, 256), order=0, preserve_range=True) > 0.5)
        if gm256.sum() < MINPX:
            continue
        todo.append((z, bbox(gm256.astype(np.uint8), pad=3) or [0, 0, 255, 255]))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        prs = predict_batch(model, proc, [rgb_cache[z] for z, _ in chunk], [b for _, b in chunk], text)
        for (z, _), pr in zip(chunk, prs):
            tgt = sl(seg, ax, z).shape
            put(pred, ax, z, pr if tgt == (256, 256) else (resize(pr.astype(float), tgt, order=0, preserve_range=True) > 0.5))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["kits", "lits", "msd"])
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    import torch
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    om = IE._load_sam3_ckpt("/nonexistent", "cuda")
    tm = IE._load_sam3_ckpt(TUMOR_OVERLAY, "cuda")
    print("models loaded", flush=True)

    summary = {}
    for ds in a.datasets:
        frozen = {r["case_id"]: r for r in json.load(open(f"{ROOT}/corpora/corpus_predicted_{ds}.json"))["records"]}
        organ, ax = CFG[ds]["organ"], CFG[ds]["ax"]
        os.makedirs(f"{OUTD}/{ds}", exist_ok=True)
        rows = []
        for i, cid in enumerate(sorted(frozen)):
            if os.path.exists(f"{OUTD}/{ds}/{cid}.json"):
                rows.append(json.load(open(f"{OUTD}/{ds}/{cid}.json")))
                continue
            t0 = time.time()
            ct, seg, aff = load(ds, cid)
            rgb = [IE.hu_to_rgb(sl(ct, ax, z) if sl(ct, ax, z).shape == (256, 256)
                                else resize(sl(ct, ax, z), (256, 256), preserve_range=True, anti_aliasing=True),
                                *IE.WIN).astype(np.uint8) for z in range(ct.shape[ax])]
            omask = predict_volume(om, proc, ct.shape, seg, 1, organ, rgb, ax, a.batch)
            tmask = predict_volume(tm, proc, ct.shape, seg, 2, "tumor", rgb, ax, a.batch)
            m = np.zeros(ct.shape, np.uint8)
            m[omask] = 1
            m[tmask] = 2                                       # exclusive save: tumor wins
            nib.save(nib.Nifti1Image(m, aff), f"{OUTD}/{ds}/{cid}.nii.gz")
            fro = frozen[cid]["organs"][organ]
            fro_ovox = round(fro["organ_volume_cm3"] * 1000) if ds == "lits" else None   # lits pseudo = vox*0.001
            g_o, g_t = seg == 1, seg == 2
            rec = {"cid": cid, "ds": ds,
                   "organ_vox": int(omask.sum()), "tumor_vox": int(tmask.sum()),
                   "overlap_vox": int((omask & tmask).sum()),
                   "dice_organ_gt": round(float(2 * (omask & g_o).sum() / (omask.sum() + g_o.sum())), 4) if (omask.sum() + g_o.sum()) else None,
                   "dice_tumor_gt": round(float(2 * (tmask & g_t).sum() / (tmask.sum() + g_t.sum())), 4) if (tmask.sum() + g_t.sum()) else None,
                   "frozen_tumor_vox": fro["tumor_voxels"], "frozen_has_tumor": fro["has_tumor"],
                   "frozen_organ_vox_pseudo": fro_ovox,
                   "has_tumor": bool(tmask.sum() > 0), "seconds": round(time.time() - t0, 1)}
            json.dump(rec, open(f"{OUTD}/{ds}/{cid}.json", "w"), indent=1)
            rows.append(rec)
            print(f"[{ds} {i+1}/{len(frozen)}] {cid} o={rec['organ_vox']} t={rec['tumor_vox']} "
                  f"(frozen t={rec['frozen_tumor_vox']}) dice {rec['dice_organ_gt']}/{rec['dice_tumor_gt']} {rec['seconds']}s", flush=True)
            torch.cuda.empty_cache()
        ht = sum(1 for r in rows if r["has_tumor"] == r["frozen_has_tumor"])
        tv = [(r["tumor_vox"], r["frozen_tumor_vox"]) for r in rows if r["frozen_tumor_vox"] > 0 and r["tumor_vox"] > 0]
        ratio = float(np.median([a_ / b for a_, b in tv])) if tv else None
        corr = float(np.corrcoef([a_ for a_, _ in tv], [b for _, b in tv])[0, 1]) if len(tv) > 2 else None
        summary[ds] = {"n": len(rows), "has_tumor_concordance": f"{ht}/{len(rows)}",
                       "tumor_vox_median_ratio_regen_over_frozen": round(ratio, 3) if ratio else None,
                       "tumor_vox_corr": round(corr, 4) if corr else None,
                       "mean_dice_organ_gt": round(float(np.mean([r["dice_organ_gt"] for r in rows if r["dice_organ_gt"]])), 3),
                       "mean_dice_tumor_gt": round(float(np.mean([r["dice_tumor_gt"] for r in rows if r["dice_tumor_gt"]])), 3)}
        print(ds, summary[ds], flush=True)
    os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
    json.dump({"note": __doc__.strip(), "summary": summary}, open(SUMMARY, "w"), indent=1)
    print(f"-> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
