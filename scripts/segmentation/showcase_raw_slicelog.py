#!/usr/bin/env python3
"""Raw (ungated) prompt-free tumor masks + the per-slice keep/drop log for the four showcase cases (the study lead 10-03, the pipeline team item 2).

Same pass as round 2 (eval_tumor_promptfree.py): sam3_tumor_generic, text 'tumor' on EVERY slice, no box, no slice
gate, identical preprocessing and orientation (LiTS = MSD Task03 NIfTI rotated -90 deg in-plane to the training
orientation, saved back on the NIfTI grid). For each slice the log records the top query's confidence, the mask
area at 256^2 and at native resolution, and whether the slice was KEPT or DROPPED and why:
    drop_conf   top-query confidence < 0.15
    drop_area   confidence ok but the native-resolution mask has < 15 px
    kept        both thresholds met (the slice's mask enters the 3-D tumor)
No reference is read except to score, and to report (never to decide) whether the slice contains reference tumor.

The organ gate is then re-applied to the raw mask with round 2's rule (keep 3-D tumor components that touch the
TotalSegmentator host organ dilated 5 mm). The host organ is taken from the shipped round-2 showcase mask (label 1;
the gated tumor overwrote host voxels inside itself, so host = label 1 | shipped tumor). For the three single-organ
cases this is round 2's host exactly; FLARE23_0405 was gated on more TotalSegmentator classes than the shipped mask
carries, so its re-gate is reported as a check, and the shipped gated mask stays canonical.

  CKPT=/path/sam3_tumor_generic.pth DATA=/path/to/staging python showcase_raw_slicelog.py
-> $OUT_DIR/{case}_raw.nii.gz, {case}_slicelog.csv ; results/segmentation/showcase_raw_slicelog_2026-10-03.json
"""
import csv
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
import sam3_autonomous_local as IE                                        # noqa: E402
from rerun_final_cases import load_model, run_model, metrics, to256, sl, put, sha256   # noqa: E402

CKPT = os.environ.get("CKPT", "/path/to/staging/ckpt/sam3_tumor_generic.pth")
DATA = os.environ.get("DATA", "/path/to/staging")
OUT = os.environ.get("OUT_DIR", "/path/to/staging/out")
SUMMARY = os.path.join(ROOT, "results", "segmentation", "showcase_raw_slicelog_2026-10-03.json")
ROUND2 = os.path.join(ROOT, "results", "segmentation", "promptfree_round2_2026-10-01.json")
GATE_MM = 5.0
# case -> (dataset, image, label, tumor labels, slice axis)
CASES = {"volume-76": ("lits", "lits/Task03_Liver/imagesTr/liver_76.nii.gz", "lits/Task03_Liver/labelsTr/liver_76.nii.gz", [2], 2),
         "case_00067": ("kits", "fc/case_00067_imaging.nii.gz", "fc/case_00067_segmentation.nii.gz", [2], 0),
         "pancreas_125": ("msd", "fc/pancreas_125_image.nii.gz", "fc/pancreas_125_label.nii.gz", [2], 2),
         "FLARE23_0405": ("flare23", "fc/ct_FLARE23_0405_0000.nii.gz", "gt/seg.nii.gz", [14], 2)}


def main():
    from transformers import Sam3Processor
    os.makedirs(OUT, exist_ok=True)
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    model = load_model(CKPT)
    h = sha256(CKPT)
    r2 = json.load(open(ROUND2))["showcase"]
    res = {"note": __doc__.strip(), "model": os.path.basename(CKPT), "model_sha256": h,
           "thresholds": {"conf_min": IE.CONF_MIN, "px_min_native": IE.VOX_MIN, "prompt": "tumor", "box": None}, "cases": {}}
    for c, (ds, ip, lp, tlab, ax) in CASES.items():
        t0 = time.time()
        img = nib.load(f"{DATA}/{ip}")
        ct, seg = img.get_fdata(), np.asarray(nib.load(f"{DATA}/{lp}").dataobj)
        sp = tuple(float(z) for z in img.header.get_zooms()[:3])
        back = lambda m: m                                                  # noqa: E731
        if ds == "lits":                                                    # round-2 orientation (training .npy)
            ct, seg = np.rot90(ct, -1, axes=(0, 1)).copy(), np.rot90(seg, -1, axes=(0, 1)).copy()
            sp = (sp[1], sp[0], sp[2])
            back = lambda m: np.rot90(m, 1, axes=(0, 1))                    # noqa: E731
        gt = np.isin(seg, tlab)
        Z = ct.shape[ax]
        rgb = [IE.hu_to_rgb(to256(sl(ct, ax, z), 1), *IE.WIN).astype(np.uint8) for z in range(Z)]
        pred = np.zeros(seg.shape, bool)
        rows = []
        for z0 in range(0, Z, 8):
            zs = list(range(z0, min(Z, z0 + 8)))
            probs, confs = run_model(model, proc, [rgb[z] for z in zs], "tumor", None)
            for z, pr, cf in zip(zs, probs, confs):
                tgt = sl(seg, ax, z).shape
                m = (pr if tgt == (256, 256) else resize(pr, tgt, order=1, preserve_range=True)) > 0.5
                a256 = int((pr > 0.5).sum())
                if cf < IE.CONF_MIN:
                    dec = "drop_conf"
                elif m.sum() < IE.VOX_MIN:
                    dec = "drop_area"
                else:
                    dec = "kept"
                    put(pred, ax, z, m)
                rows.append({"slice": z, "top_conf": round(float(cf), 4), "mask_px_256": a256, "mask_px_native": int(m.sum()),
                             "decision": dec, "ref_tumor_px_on_slice": int(sl(gt, ax, z).sum())})
        raw = back(pred).astype(np.uint8)
        nib.save(nib.Nifti1Image(raw, img.affine), f"{OUT}/{c}_raw.nii.gz")
        with open(f"{OUT}/{c}_slicelog.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        vox_cm3 = float(np.prod(sp)) / 1000.0
        rec = {"dataset": ds, "slice_axis": ax, "n_slices": Z, "seconds_gpu": round(time.time() - t0, 1),
               "slices": {k: sum(r["decision"] == k for r in rows) for k in ("kept", "drop_conf", "drop_area")},
               "kept_slices_without_ref_tumor": sum(r["decision"] == "kept" and r["ref_tumor_px_on_slice"] == 0 for r in rows),
               "ref_tumor_slices_dropped": sum(r["decision"] != "kept" and r["ref_tumor_px_on_slice"] > 0 for r in rows),
               "tumor_raw": metrics(pred, gt, sp, vox_cm3), "raw_mask": f"{c}_raw.nii.gz", "slicelog": f"{c}_slicelog.csv"}
        rec["round2_raw_dice"] = r2[c]["tumor_raw"]["dice"]
        rec["raw_reproduces_round2"] = abs(rec["tumor_raw"]["dice"] - rec["round2_raw_dice"]) < 1e-3
        # re-gate on the shipped round-2 host organ
        shp = f"{DATA}/sc/{c}.nii.gz"
        if os.path.exists(shp):
            sm = np.asarray(nib.load(shp).dataobj)
            if ds == "lits":
                sm = np.rot90(sm, -1, axes=(0, 1))
            shipped_t = sm == tlab[0]
            host = (sm > 0) if ds == "flare23" else ((sm == 1) | shipped_t)
            near = ndimage.distance_transform_edt(~host, sampling=sp) <= GATE_MM
            lab, n = ndimage.label(pred)
            keep = np.unique(lab[near & pred]); keep = keep[keep > 0]
            gated = np.isin(lab, keep)
            rec["tumor_gated"] = metrics(gated, gt, sp, vox_cm3)
            rec["gate"] = {"components_in": int(n), "components_kept": int(len(keep)),
                           "host": "shipped round-2 TotalSegmentator host organ" + (" (subset of round-2 host classes)" if ds == "flare23" else "")}
            rec["gated_equals_shipped_voxelwise"] = bool(np.array_equal(gated, shipped_t))
            rec["gated_vs_shipped_voxel_diff"] = int((gated ^ shipped_t).sum())
        res["cases"][c] = rec
        print(c, rec["slices"], "raw dice", rec["tumor_raw"]["dice"], "r2", rec["round2_raw_dice"],
              "gated==shipped", rec.get("gated_equals_shipped_voxelwise"), rec.get("gated_vs_shipped_voxel_diff"), flush=True)
        json.dump(res, open(SUMMARY, "w"), indent=1)
    print("->", SUMMARY)


if __name__ == "__main__":
    main()
