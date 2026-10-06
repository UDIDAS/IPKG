#!/usr/bin/env python3
"""Which weights drew the three single-organ masks shipped 09-19 (final_cases_2026-09-19/masks/)? (the study lead 10-03, the pipeline team item 1)

The package README said "same checkpoints"; the regeneration record (commit 23a5e04, regen_predicted_masks_113.py)
says base facebook/sam3 for the organ + sam3_tumor_flare_only for the tumor. This settles it by re-drawing:
  tumor half : sam3_tumor_flare_only with the regeneration script's OWN functions (predict_volume, bbox, batch
               inference: GT box + 3 px on slices with >= 50 GT px at 256^2) -> compared voxel-for-voxel with the
               shipped tumor label (masks are exclusive with tumor winning, so the shipped label-2 set IS the tumor
               pass's output)
  organ half : the per-dataset organ checkpoint (sam3_organ_generic_ausam_<ds>) under the same protocol -> if the
               shipped organ were drawn by it, it would match the same way; base facebook/sam3 itself cannot be
               re-run here (gated download), so the organ half is established by exclusion + the regeneration log.

  CKPT_DIR=... DATA=... python verify_displayed_weights.py -> results/audit/displayed_weights_check_2026-10-03.json
"""
import json
import os
import sys

import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "segmentation"))
import sam3_autonomous_local as IE                                       # noqa: E402
import regen_predicted_masks_113 as RG                                   # noqa: E402
from rerun_final_cases import load_model, sha256                        # noqa: E402

CK = os.environ.get("CKPT_DIR", "/path/to/staging/ckpt")
D = os.environ.get("DATA", "/path/to/staging")
FLARE_ONLY = os.environ.get("FLARE_ONLY_CKPT", os.path.expanduser("~/hmmkg_ckpts/sam3_tumor_flare_only.pth"))  # Drive: response_2026-10-01/checkpoints/tumor/ (md5 9a7ac81c...); local copy removed 10-03
OUT = os.path.join(ROOT, "results", "audit", "displayed_weights_check_2026-10-03.json")
OUT_BASE = os.path.join(ROOT, "results", "audit", "displayed_weights_base_organ_2026-10-05.json")
BASE_ORGAN = "--base-organ" in sys.argv   # 10-05: re-draw the organ half with base facebook/sam3, loaded exactly as
                                          # regen_predicted_masks_113.py did on 09-19 (IE._load_sam3_ckpt("/nonexistent"))
CASES = {"volume-76": ("lits", "liver", 0), "case_00067": ("kits", "kidney", 0), "pancreas_125": ("msd", "pancreas", 2)}


def load(c):
    if c == "volume-76":
        return np.load(f"{D}/fc/volume-76.npy"), np.load(f"{D}/fc/segmentation-76.npy").astype(np.uint8)
    p = {"case_00067": ("case_00067_imaging", "case_00067_segmentation"), "pancreas_125": ("pancreas_125_image", "pancreas_125_label")}[c]
    return nib.load(f"{D}/fc/{p[0]}.nii.gz").get_fdata(), np.asarray(nib.load(f"{D}/fc/{p[1]}.nii.gz").dataobj).astype(np.uint8)


def cmp(a, b):
    return {"voxels_redrawn": int(a.sum()), "voxels_shipped": int(b.sum()), "voxels_differing": int((a ^ b).sum()),
            "identical": bool(np.array_equal(a, b)), "dice": round(2 * int((a & b).sum()) / max(1, int(a.sum()) + int(b.sum())), 4)}


def main():
    from transformers import Sam3Processor
    import torch
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    data = {}
    for c, (ds, organ, ax) in CASES.items():
        ct, seg = load(c)
        rgb = [IE.hu_to_rgb(RG.sl(ct, ax, z) if RG.sl(ct, ax, z).shape == (256, 256)
                            else RG.resize(RG.sl(ct, ax, z), (256, 256), preserve_range=True, anti_aliasing=True), *IE.WIN).astype(np.uint8)
               for z in range(ct.shape[ax])]
        shipped = np.asarray(nib.load(f"{D}/fc_masks/{c}.nii.gz").dataobj)
        data[c] = (ct, seg, rgb, shipped)
    if BASE_ORGAN:
        return base_organ(proc, data)
    res = {"note": __doc__.strip(), "tumor_model": "sam3_tumor_flare_only", "tumor_sha256": sha256(FLARE_ONLY), "cases": {}}
    tm = load_model(FLARE_ONLY)
    for c, (ds, organ, ax) in CASES.items():
        ct, seg, rgb, shipped = data[c]
        t = RG.predict_volume(tm, proc, ct.shape, seg, 2, "tumor", rgb, ax, 8)
        res["cases"].setdefault(c, {})["tumor_half_vs_flare_only"] = cmp(t, shipped == 2)
        print(c, "tumor", res["cases"][c]["tumor_half_vs_flare_only"], flush=True)
    del tm
    torch.cuda.empty_cache()
    for c, (ds, organ, ax) in CASES.items():
        ct, seg, rgb, shipped = data[c]
        path = f"{CK}/sam3_organ_generic_ausam_{ds}.pth"
        om = load_model(path)
        o = RG.predict_volume(om, proc, ct.shape, seg, 1, organ, rgb, ax, 8)
        t_ship = shipped == 2
        res["cases"][c]["organ_half_vs_per_dataset_ckpt"] = {"ckpt": os.path.basename(path), **cmp(o & ~t_ship, shipped == 1)}
        print(c, "organ", res["cases"][c]["organ_half_vs_per_dataset_ckpt"], flush=True)
        del om
        torch.cuda.empty_cache()
    json.dump(res, open(OUT, "w"), indent=1)
    print("->", OUT)


def base_organ(proc, data):
    """Organ half re-drawn with base SAM 3 (no fine-tune), compared voxel-for-voxel with the shipped organ label."""
    import torch
    om = IE._load_sam3_ckpt("/nonexistent", "cuda")
    res = {"note": ("Organ half of the three shipped single-organ showcase masks re-drawn with base facebook/sam3, loaded "
                    "as the 09-19 regeneration loaded it (regen_predicted_masks_113.py at commit 23a5e04: "
                    "IE._load_sam3_ckpt('/nonexistent') -> base weights), same protocol and functions as the tumor check "
                    "of displayed_weights_check_2026-10-03.json. Masks are exclusive with tumor winning, so the redrawn "
                    "organ is compared after removing the shipped tumor voxels."),
           "organ_model": IE.SAM3_MODEL_ID + " (base, no fine-tune)", "cases": {}}
    for c, (ds, organ, ax) in CASES.items():
        ct, seg, rgb, shipped = data[c]
        o = RG.predict_volume(om, proc, ct.shape, seg, 1, organ, rgb, ax, 8)
        res["cases"][c] = {"organ_half_vs_base_sam3": cmp(o & ~(shipped == 2), shipped == 1)}
        print(c, "organ (base)", res["cases"][c]["organ_half_vs_base_sam3"], flush=True)
        torch.cuda.empty_cache()
    json.dump(res, open(OUT_BASE, "w"), indent=1)
    print("->", OUT_BASE)


if __name__ == "__main__":
    main()
