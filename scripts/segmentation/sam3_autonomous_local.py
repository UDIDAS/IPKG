#!/usr/bin/env python3
"""Bundle-local autonomous SAM3 inference — vendored from the original workspace modules
`infer_ensemble.py` (load_base / concept_slice / seg_concept / ORGANS), `infer_sam3.py` (hu_to_rgb)
and `run_pancreas_sam3.py` (_load_sam3_ckpt / SAM3_MODEL_ID / HF token), with the module-level
/home/user and /scratch/user side effects removed so the repair / closed-loop experiments
(`scripts/repair/*.py`) run from any checkout.

The functions are verbatim copies (same window, thresholds, resize order, label ids). Originals kept at
gdrive:data/ssl_handoff_extras/flare23_model/code/.

Data root: $VKG_DATA (default /path/to/VKG_data), expected layout
  $VKG_DATA/flare_full_cases/{cid}_ct.nii.gz, {cid}_label.nii.gz    (40 FLARE23 cases, multiclass GT, tumor=14)
  $VKG_DATA/ckpts/tumor/sam3_tumor_generic.pth
"""
import os

import numpy as np
import torch
from skimage.transform import resize
from torch.amp import autocast

VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
CASES = os.path.join(VKG_DATA, "flare_full_cases")
TUMOR_CKPT = os.path.join(VKG_DATA, "ckpts", "tumor", "sam3_tumor_generic.pth")

SAM3_MODEL_ID = "facebook/sam3"
WIN = (-125, 225)
# organ concept -> output label id (matches FLARE/our KG scheme)
ORGANS = [("liver", 1), ("right kidney", 2), ("spleen", 3), ("pancreas", 4), ("left kidney", 13)]
TUMOR_LABEL = 14
VOX_MIN = 15          # min mask pixels per slice to keep
CONF_MIN = 0.15       # min top-detection confidence (sigmoid of pred_logit) to accept a slice


def _load_hf_token():
    """Token from env, else the standard HF cache file. Keeps secrets out of source."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (os.path.expanduser("~/.cache/huggingface/token"),
              os.path.expanduser("~/.huggingface/token")):
        if os.path.exists(p):
            with open(p) as f:
                t = f.read().strip()
            if t:
                return t
    return None


HF_TOKEN = _load_hf_token()


def hu_to_rgb(slice_hu, lo, hi):
    clip = np.clip(slice_hu, lo, hi)
    norm = ((clip - lo) / (hi - lo) * 255).astype(np.uint8)
    return np.stack([norm] * 3, axis=-1)


def _load_sam3_ckpt(ckpt_path, device):
    from transformers import Sam3Model
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
    model.to(device).eval()
    return model


def load_base():
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID, token=HF_TOKEN)
    return model.to("cuda").eval(), proc


def concept_slice(model, proc, rgb, concept):
    """Return (prob_map 256x256, confidence) for a text concept, no box."""
    inputs = proc(images=[rgb], text=[concept], return_tensors="pt")
    kw = {"pixel_values": inputs["pixel_values"].to("cuda")}
    for k in ("input_ids", "attention_mask"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k].to("cuda")
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        pl = out.pred_logits[0].sigmoid()
        i = int(pl.argmax())
        conf = float(pl[i])
        m = torch.nn.functional.interpolate(out.pred_masks[0:1, i:i + 1].float(),
                                            size=(256, 256), mode="bilinear", align_corners=False)
    return m.sigmoid().squeeze().cpu().numpy(), conf


def seg_concept(ct, model, proc, concept):
    X, Y, Z = ct.shape
    out = np.zeros(ct.shape, np.uint8)
    for z in range(Z):
        rgb = hu_to_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True),
                        *WIN).astype(np.uint8)
        prob, conf = concept_slice(model, proc, rgb, concept)
        if conf < CONF_MIN:
            continue
        m = (resize(prob, (X, Y), order=1, preserve_range=True) > 0.5).astype(np.uint8)
        if m.sum() >= VOX_MIN:
            out[:, :, z] = m
    return out


# ----------------------------------------------------------------------------------------------
# Batched variant (added on Delta, 2026-09-13): same per-slice computation as seg_concept(), but B
# slices go through the processor/model per forward. Every slice is resized to 256x256 first, so
# there is no padding and the per-slice logits are those of the single-slice path.
# ----------------------------------------------------------------------------------------------
def concept_batch(model, proc, rgbs, concept):
    """rgbs: list of HxWx3 uint8 -> (list of prob maps 256x256, list of confidences)."""
    inputs = proc(images=rgbs, text=[concept] * len(rgbs), return_tensors="pt")
    kw = {"pixel_values": inputs["pixel_values"].to("cuda")}
    for k in ("input_ids", "attention_mask"):
        if inputs.get(k) is not None:
            kw[k] = inputs[k].to("cuda")
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        pl = out.pred_logits.sigmoid()                      # (B, Q)
        idx = pl.argmax(dim=1)                              # (B,)
        conf = pl[torch.arange(len(rgbs)), idx]
        masks = out.pred_masks[torch.arange(len(rgbs)), idx].float().unsqueeze(1)   # (B,1,h,w)
        m = torch.nn.functional.interpolate(masks, size=(256, 256), mode="bilinear", align_corners=False)
    return list(m.sigmoid().squeeze(1).cpu().numpy()), conf.tolist()


def seg_concept_batched(ct, model, proc, concept, batch=8, rgb_cache=None):
    """Batched equivalent of seg_concept(). rgb_cache: optional precomputed list of 256x256 RGB slices
    (the HU windowing/resize is concept-independent, so callers share it across concepts)."""
    X, Y, Z = ct.shape
    if rgb_cache is None:
        rgb_cache = [hu_to_rgb(resize(ct[:, :, z], (256, 256), preserve_range=True, anti_aliasing=True),
                               *WIN).astype(np.uint8) for z in range(Z)]
    out = np.zeros(ct.shape, np.uint8)
    for z0 in range(0, Z, batch):
        probs, confs = concept_batch(model, proc, rgb_cache[z0:z0 + batch], concept)
        for j, (prob, conf) in enumerate(zip(probs, confs)):
            if conf < CONF_MIN:
                continue
            m = (resize(prob, (X, Y), order=1, preserve_range=True) > 0.5).astype(np.uint8)
            if m.sum() >= VOX_MIN:
                out[:, :, z0 + j] = m
    return out
