#!/usr/bin/env python3
"""One-pass re-run of the four final cases: every shipped mask comes with the metrics of THAT mask (the co-author 09-29 §6 items 7-9).

Arms (all share one preprocessing: HU window (-125, 225), 256x256 slices, argmax query, sigmoid > 0.5, resize back):
  so_pair       SEMI-ORACLE (the protocol behind the printed numbers): per labeled slice with GT structure >= 50 px at
                256^2, GT box + 3 px pad + text prompt. Weights = the dataset's own organ + tumor checkpoint pair;
                where the per-dataset tumor checkpoint is not recoverable (LiTS, MSD) the substitute is named in the
                manifest.
  displayed     NO inference: the masks shipped in final_cases_2026-09-19/masks/ (what raters saw), scored with the
                same metric code. The three single-organ ones were drawn by base facebook/sam3 (organ) + the
                FLARE23-only tumor expert; FLARE23_0405 by its own pair.
  so_displayed  (optional; needs the base facebook/sam3 weights) re-draw with the displayed weights.
  auto_pair     AUTONOMOUS: text prompt only, NO box, NO slice gate — every slice is prompted; a slice is kept when
                the top query's confidence >= 0.15 and the mask has >= 15 px (sam3_autonomous_local thresholds).
                Same weights as so_pair.

Masks are saved EXCLUSIVE (organ label(s), tumor wins) exactly as the app draws them, and every metric is computed on
that saved mask: Dice, NSD@1/2 mm, HD95 (mm), predicted/reference volume, out-of-reference volume, component count;
plus the same metrics after connected-component post-processing (organ: largest component per organ, 2 for a
bilateral 'kidney' label; tumor: drop components < 0.1 cm^3) so the method change in plan item 12 can be judged.
LiTS is the 256^2 npy slab with no header spacing: Dice is exact, mm metrics use the NOMINAL display spacing
(1.4844, 1.4844, 2.5) and are flagged.

  CKPT_DIR=/dev/shm/ud_ckpt CASE_DIR=/dev/shm/fc HF_HOME=... python rerun_final_cases.py [--arms ...] [--cases ...]
-> $OUT_DIR/{arm}/{case}.nii.gz + results/segmentation/final_cases_rerun_2026-09-29.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.transform import resize

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
import sam3_autonomous_local as IE                                       # noqa: E402

CKPT = os.environ.get("CKPT_DIR", "/dev/shm/ud_ckpt")
CASES = os.environ.get("CASE_DIR", "/dev/shm/fc")
OUT = os.environ.get("OUT_DIR", "/dev/shm/ud_rerun")
FLARE_ONLY = os.environ.get("FLARE_ONLY_CKPT", os.path.expanduser("~/hmmkg_ckpts/sam3_tumor_flare_only.pth"))  # Drive: response_2026-10-01/checkpoints/tumor/ (md5 9a7ac81c...); local copy removed 10-03
SUMMARY = os.path.join(ROOT, "results", "segmentation", "final_cases_rerun_2026-09-29.json")
MINPX, PAD = 50, 3
LITS_NOMINAL_SP = (2.5, 1.4844, 1.4844)          # slab axis order (z, y, x); display spacing, NOT header spacing
WEIGHTS_DTYPE = os.environ.get("WEIGHTS_DTYPE", "fp32")
BASE = "BASE"                                    # sentinel: facebook/sam3 with no fine-tuned state_dict

# structure name -> (GT labels, output label, text prompt); tumor handled separately
CASE_CFG = {
    "volume-76":    {"ds": "lits", "ax": 0, "organs": [("liver", [1], 1, "liver")], "tlab": [2], "tout": 2,
                     "organ_ckpt": "organ/sam3_organ_generic_ausam_lits.pth", "tumor_ckpt": None},
    "case_00067":   {"ds": "kits", "ax": 0, "organs": [("kidney", [1], 1, "kidney")], "tlab": [2], "tout": 2,
                     "organ_ckpt": "organ/sam3_organ_generic_ausam_kits.pth", "tumor_ckpt": "tumor/sam3_tumor_ausam_kits.pth"},
    "pancreas_125": {"ds": "msd", "ax": 2, "organs": [("pancreas", [1], 1, "pancreas")], "tlab": [2], "tout": 2,
                     "organ_ckpt": "organ/sam3_organ_generic_ausam_msd.pth", "tumor_ckpt": None},
    "FLARE23_0405": {"ds": "flare23", "ax": 2,
                     "organs": [("liver", [1], 1, "liver"), ("right kidney", [2], 2, "right kidney"),
                                ("spleen", [3], 3, "spleen"), ("pancreas", [4], 4, "pancreas"),
                                ("left kidney", [13], 13, "left kidney")],
                     "tlab": [14], "tout": 14,
                     "organ_ckpt": "organ/sam3_organ_generic_ausam_flare23.pth", "tumor_ckpt": "tumor/sam3_tumor_ausam_flare.pth"},
}
# per-dataset tumor checkpoints sam3_tumor_ausam_{lits,pancreas}.pth are not on this machine or the Drive archive;
# substitute = the FLARE per-dataset tumor expert (cross-dataset; within 0.007 DSC of the dedicated experts, 09-16 round)
TUMOR_SUBSTITUTE = "tumor/sam3_tumor_ausam_flare.pth"


def load_case(cid):
    if cid == "volume-76":
        return np.load(f"{CASES}/ct/volume-76.npy"), np.load(f"{CASES}/ct/segmentation-76.npy").astype(np.uint8), np.eye(4), None
    p = {"case_00067": ("ct/case_00067_imaging.nii.gz", "ct/case_00067_segmentation.nii.gz"),
         "pancreas_125": ("ct/pancreas_125_image.nii.gz", "ct/pancreas_125_label.nii.gz"),
         "FLARE23_0405": ("ct/ct_FLARE23_0405_0000.nii.gz", "gt/FLARE23_0405/seg.nii.gz")}[cid]
    img = nib.load(f"{CASES}/{p[0]}")
    seg = np.asarray(nib.load(f"{CASES}/{p[1]}").dataobj).astype(np.uint8)
    return img.get_fdata(), seg, img.affine, tuple(float(z) for z in img.header.get_zooms()[:3])


def sha256(path, _cache={}):
    if path == BASE:
        return None
    if path not in _cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        _cache[path] = h.hexdigest()
    return _cache[path]


def load_model(path):
    """Fine-tuned checkpoints are COMPLETE state_dicts (1,468/1,468 keys), so build SAM 3 from its config and load
    strict=True: the checkpoint alone defines the model (IE._load_sam3_ckpt loads strict=False over the base
    weights and silently returns the base model when the path is missing)."""
    import torch
    from transformers import Sam3Config, Sam3Model
    if path == BASE:
        return IE._load_sam3_ckpt("/nonexistent", "cuda")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = Sam3Model(Sam3Config.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN))
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)
    if WEIGHTS_DTYPE == "fp16":                  # shared GPUs: fp16 storage; matmuls already run fp16 under autocast
        model = model.half()
    return model.to("cuda").eval()


def sl(v, ax, z):
    return v[z] if ax == 0 else v[:, :, z]


def put(v, ax, z, s):
    if ax == 0:
        v[z] = s
    else:
        v[:, :, z] = s


def to256(a, order):
    if a.shape == (256, 256):
        return a
    return resize(a.astype(float), (256, 256), order=order, preserve_range=True, anti_aliasing=(order > 0))


def bbox(m):
    ys, xs = np.where(m)
    H, W = m.shape
    return [max(0, int(xs.min()) - PAD), max(0, int(ys.min()) - PAD), min(W - 1, int(xs.max()) + PAD), min(H - 1, int(ys.max()) + PAD)]


def run_model(model, proc, rgbs, text, boxes=None):
    import torch
    from torch.amp import autocast
    kw_in = dict(images=rgbs, text=[text] * len(rgbs), return_tensors="pt")
    if boxes is not None:
        kw_in.update(input_boxes=[[b] for b in boxes], input_boxes_labels=[[1]] * len(rgbs))
    inp = proc(**kw_in)
    kw = {k: (inp[k].to("cuda").half() if (k == "pixel_values" and WEIGHTS_DTYPE == "fp16") else inp[k].to("cuda")) for k in ("pixel_values", "input_ids", "attention_mask", "input_boxes", "input_boxes_labels")
          if inp.get(k) is not None}
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        pl = out.pred_logits.sigmoid()
        idx = pl.argmax(dim=1)
        conf = pl[torch.arange(len(rgbs)), idx]
        pm = out.pred_masks[torch.arange(len(rgbs)), idx].float().unsqueeze(1)
        pm = torch.nn.functional.interpolate(pm, size=(256, 256), mode="bilinear", align_corners=False)
    return pm.sigmoid().squeeze(1).cpu().numpy(), conf.tolist()


def predict(model, proc, rgb, seg, labs, text, ax, mode, batch=8):
    """mode 'so': GT-box on gated slices -> bool volume. mode 'auto': every slice, no box, confidence gate."""
    shape = seg.shape
    pred = np.zeros(shape, bool)
    Z = shape[ax]
    if mode == "so":
        todo = []
        for z in range(Z):
            g = to256(np.isin(sl(seg, ax, z), labs), 0) > 0.5
            if g.sum() >= MINPX:
                todo.append((z, bbox(g)))
    else:
        todo = [(z, None) for z in range(Z)]
    for i in range(0, len(todo), batch):
        ch = todo[i:i + batch]
        probs, confs = run_model(model, proc, [rgb[z] for z, _ in ch], text, [b for _, b in ch] if mode == "so" else None)
        for (z, _), pr, c in zip(ch, probs, confs):
            tgt = sl(seg, ax, z).shape
            if mode == "auto" and c < IE.CONF_MIN:
                continue
            m = pr if tgt == (256, 256) else resize(pr, tgt, order=1, preserve_range=True)
            m = m > 0.5
            if mode == "auto" and m.sum() < IE.VOX_MIN:
                continue
            put(pred, ax, z, m)
    return pred


def surface(pred, gt, sp):
    def sdt(m):
        s = m & ~ndimage.binary_erosion(m)
        return s, ndimage.distance_transform_edt(~s, sampling=sp)
    if not pred.any() or not gt.any():
        return {"nsd_1mm": 0.0 if (pred.any() or gt.any()) else 1.0, "nsd_2mm": 0.0 if (pred.any() or gt.any()) else 1.0, "hd95_mm": None}
    ps, pdt = sdt(pred)
    gs, gdt = sdt(gt)
    d_g2p, d_p2g = pdt[gs], gdt[ps]
    tot = len(d_g2p) + len(d_p2g)
    return {"nsd_1mm": round(float(((d_g2p <= 1).sum() + (d_p2g <= 1).sum()) / tot), 4),
            "nsd_2mm": round(float(((d_g2p <= 2).sum() + (d_p2g <= 2).sum()) / tot), 4),
            "hd95_mm": round(float(max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))), 2)}


def metrics(pred, gt, sp, vox_cm3):
    inter = int((pred & gt).sum())
    s = int(pred.sum()) + int(gt.sum())
    _, ncomp = ndimage.label(pred)
    r = {"dice": round(2 * inter / s, 4) if s else None,
         "pred_cm3": round(float(pred.sum()) * vox_cm3, 2), "ref_cm3": round(float(gt.sum()) * vox_cm3, 2),
         "outside_ref_cm3": round(float((pred & ~gt).sum()) * vox_cm3, 2),
         "outside_ref_pct_of_pred": round(100.0 * float((pred & ~gt).sum()) / max(1, int(pred.sum())), 1),
         "missed_ref_cm3": round(float((gt & ~pred).sum()) * vox_cm3, 2), "components": int(ncomp)}
    r.update(surface(pred, gt, sp))
    return r


def keep_largest(m, k):
    lab, n = ndimage.label(m)
    if n <= k:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:k] + 1
    return np.isin(lab, keep)


def drop_small(m, min_vox):
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, np.where(sizes >= min_vox)[0] + 1)


def arm_weights(arm, cfg):
    if arm == "so_displayed":
        return BASE, FLARE_ONLY, "displayed-09-19 weights (base organ + FLARE23-only tumor expert)"
    t = cfg["tumor_ckpt"]
    note = "per-dataset pair"
    if t is None:
        t, note = TUMOR_SUBSTITUTE, "per-dataset organ; tumor = FLARE expert SUBSTITUTE (per-dataset tumor ckpt not recoverable)"
    return os.path.join(CKPT, cfg["organ_ckpt"]), os.path.join(CKPT, t), note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["displayed", "so_pair", "auto_pair"])
    ap.add_argument("--cases", nargs="+", default=list(CASE_CFG))
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    import torch
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    res = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {"note": __doc__.strip(), "runs": {}}
    loaded = {}

    def get(path):
        if path not in loaded:
            for k in list(loaded):
                del loaded[k]
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            loaded[path] = load_model(path)
        return loaded[path]

    for cid in a.cases:
        cfg = CASE_CFG[cid]
        ct, seg, aff, sp_hdr = load_case(cid)
        sp = sp_hdr if sp_hdr else LITS_NOMINAL_SP
        vox_cm3 = float(np.prod(sp)) / 1000.0
        ax = cfg["ax"]
        rgb = [IE.hu_to_rgb(to256(sl(ct, ax, z), 1), *IE.WIN).astype(np.uint8) for z in range(ct.shape[ax])]
        for arm in a.arms:
            if arm == "so_displayed" and cid == "FLARE23_0405":
                continue                          # 0405 was drawn by its own pair; so_pair IS its displayed protocol
            if arm == "displayed":
                shipped = f"{CASES}/masks/{cid}.nii.gz"
                out = np.asarray(nib.load(shipped).dataobj).astype(np.uint8)
                assert out.shape == seg.shape, (cid, out.shape, seg.shape)
                disp = cid == "FLARE23_0405"
                opath = os.path.join(CKPT, cfg["organ_ckpt"]) if disp else BASE
                tpath = os.path.join(CKPT, cfg["tumor_ckpt"]) if disp else FLARE_ONLY
                note, mode, secs = "shipped mask (final_cases_2026-09-19/masks/), scored as drawn", "so", None
            else:
                opath, tpath, note = arm_weights(arm, cfg)
                mode = "auto" if arm.startswith("auto") else "so"
                t0 = time.time()
                om = get(opath)
                organ_masks = {name: predict(om, proc, rgb, seg, labs, text, ax, mode, a.batch) for name, labs, _, text in cfg["organs"]}
                del om                                                 # one model resident at a time (shared GPUs)
                tm = get(tpath)
                tmask = predict(tm, proc, rgb, seg, cfg["tlab"], "tumor", ax, mode, a.batch)
                del tm
                secs = round(time.time() - t0, 1)
                out = np.zeros(seg.shape, np.uint8)
                for (name, _, lab, _) in cfg["organs"]:
                    out[organ_masks[name]] = lab
                out[tmask] = cfg["tout"]                               # exclusive, tumor wins (as drawn)
                os.makedirs(f"{OUT}/{arm}", exist_ok=True)
                nib.save(nib.Nifti1Image(out, aff), f"{OUT}/{arm}/{cid}.nii.gz")
            rec = {"dataset": cfg["ds"], "arm": arm, "mode": mode, "weights_note": note,
                   "organ_ckpt": os.path.basename(opath) if opath != BASE else "facebook/sam3 (base)",
                   "organ_ckpt_sha256": sha256(opath), "tumor_ckpt": os.path.basename(tpath), "tumor_ckpt_sha256": sha256(tpath),
                   "weights_dtype": None if arm == "displayed" else WEIGHTS_DTYPE, "spacing_mm": list(sp), "spacing_source": "header" if sp_hdr else "NOMINAL display spacing (npy slab has none)",
                   "seconds_gpu_pass": secs, "mask_file": f"masks/{cid}.nii.gz" if arm == "displayed" else f"{arm}/{cid}.nii.gz", "structures": {}, "structures_ccpp": {}}
            min_vox = max(1, int(round(0.1 / vox_cm3)))
            for (name, labs, lab, _) in cfg["organs"]:
                p, g = out == lab, np.isin(seg, labs)
                rec["structures"][name] = metrics(p, g, sp, vox_cm3)
                k = 2 if (name == "kidney") else 1                     # KiTS 'kidney' label is bilateral
                rec["structures_ccpp"][name] = metrics(keep_largest(p, k), g, sp, vox_cm3)
            p, g = out == cfg["tout"], np.isin(seg, cfg["tlab"])
            rec["structures"]["tumor"] = metrics(p, g, sp, vox_cm3)
            rec["structures_ccpp"]["tumor"] = metrics(drop_small(p, min_vox), g, sp, vox_cm3)
            rec["ccpp_rule"] = f"organ: largest component (2 for bilateral kidney label); tumor: drop components < 0.1 cm3 ({min_vox} vox)"
            res["runs"].setdefault(cid, {})[arm] = rec
            print(f"[{cid} {arm}] {secs}s  " + "  ".join(f"{k} {v['dice']}" for k, v in rec["structures"].items()), flush=True)
            json.dump(res, open(SUMMARY, "w"), indent=1)
    print(f"-> {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
