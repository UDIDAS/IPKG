#!/usr/bin/env python3
"""Sub-site for the 56 PREDICTED MSD records from regenerated semi-oracle masks (the study lead 10-03, the pipeline team item 5).

The masks behind corpus_predicted_msd.json were not kept and its tumor checkpoint (sam3_tumor_ausam_pancreas) was
deleted 08-22, so the round-2 re-issue set every predicted MSD sub-site to 'unknown' -- except pancreas_125, which
was re-scored on its displayed mask ('head'). This script removes that inconsistency by regenerating all 56 under
the frozen corpus' protocol (build_predicted_corpus.py: GT box + 3 px on slices whose GT structure has >= 50 px at
256^2, text prompt, 3-D assembly) with the closest surviving weights:
    organ : sam3_organ_generic_ausam_msd.pth   (the SAME checkpoint that built the frozen records)
    tumor : sam3_tumor_flare_only.pth          (substitute; the 09-17 regeneration's tumor model, validated against
                                                the frozen MSD records: has_tumor 56/56, tumor-voxel r = 0.982)
and applying the corrected rule (scripts/kg/pancreas_subsite.py) to gland = organ | tumor. Fidelity to the frozen
records is reported per case (tumor voxels, has_tumor), together with the reference-label site for comparison.

  CKPT_DIR=... MSD_DIR=... python regen_msd_predicted_subsite.py
-> results/kg/msd_predicted_subsite_regen_2026-10-03.json
"""
import json
import os
import sys
import time

import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "segmentation"))
from pancreas_subsite import pancreas_subsite                            # noqa: E402
import sam3_autonomous_local as IE                                       # noqa: E402
from rerun_final_cases import load_model, predict, to256, sl, sha256     # noqa: E402

CK = os.environ.get("CKPT_DIR", "/path/to/staging/ckpt")
MSD = os.environ.get("MSD_DIR", "/path/to/staging/msd")
ORGAN = os.path.join(CK, "sam3_organ_generic_ausam_msd.pth")
TUMOR = os.environ.get("TUMOR_CKPT", os.environ.get("FLARE_ONLY_CKPT", os.path.expanduser("~/hmmkg_ckpts/sam3_tumor_flare_only.pth")))
OUT = os.path.join(ROOT, "results", "kg", "msd_predicted_subsite_regen_2026-10-03.json")
MASKS = os.environ.get("MASK_OUT", "/path/to/staging/msd_regen")


def main():
    from transformers import Sam3Processor
    import torch
    frozen = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_predicted_msd.json")))["records"]}
    gt_site = {r["case"]: r for r in json.load(open(os.path.join(ROOT, "results", "kg", "pancreas_subsite_msd_gt.json")))["records"]}
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    os.makedirs(MASKS, exist_ok=True)
    cases = sorted(frozen)
    vols = {}
    for c in cases:                                   # shared preprocessing: axis 2, HU window, 256^2
        img = nib.load(f"{MSD}/imagesTr/{c}.nii.gz")
        seg = np.asarray(nib.load(f"{MSD}/labelsTr/{c}.nii.gz").dataobj).astype(np.uint8)
        vols[c] = (img, seg)
    res = {"note": __doc__.strip(), "organ_ckpt": os.path.basename(ORGAN), "organ_sha256": sha256(ORGAN),
           "tumor_ckpt": os.path.basename(TUMOR), "tumor_sha256": sha256(TUMOR), "cases": {}}
    masks = {}
    for stage, path, labs, text in (("organ", ORGAN, [1], "pancreas"), ("tumor", TUMOR, [2], "tumor")):
        model = load_model(path)
        for c in cases:
            img, seg = vols[c]
            ct = img.get_fdata()
            rgb = [IE.hu_to_rgb(to256(sl(ct, 2, z), 1), *IE.WIN).astype(np.uint8) for z in range(ct.shape[2])]
            t0 = time.time()
            masks.setdefault(c, {})[stage] = predict(model, proc, rgb, seg, labs, text, 2, "so", 16)
            print(f"{stage} {c} {time.time() - t0:.1f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    for c in cases:
        img, seg = vols[c]
        o, t = masks[c]["organ"], masks[c]["tumor"]
        out = np.zeros(seg.shape, np.uint8)
        out[o] = 1
        out[t] = 2
        nib.save(nib.Nifti1Image(out, img.affine), f"{MASKS}/{c}.nii.gz")
        site = pancreas_subsite(o | t, t, img.affine) if t.any() else {"site": "na", "reason": "no predicted tumor"}
        fr = frozen[c]["organs"]["pancreas"]
        res["cases"][c] = {"site": site["site"], "reason": site.get("reason"), "f_lr": site.get("f_lr"), "f_pca": site.get("f_pca"),
                           "site_reference_label": gt_site.get(c, {}).get("site"),
                           "tumor_vox": int(t.sum()), "frozen_tumor_vox": fr.get("tumor_voxels"),
                           "has_tumor": bool(t.any()), "frozen_has_tumor": fr.get("has_tumor"),
                           "anatomic_location_delivered": fr.get("anatomic_location")}
    R = list(res["cases"].values())
    tv = [(r["tumor_vox"], r["frozen_tumor_vox"]) for r in R if r["tumor_vox"] and r["frozen_tumor_vox"]]
    from collections import Counter
    res["summary"] = {"n": len(R), "dist": dict(Counter(r["site"] for r in R)),
                      "agree_with_reference_label_site": sum(r["site"] == r["site_reference_label"] for r in R),
                      "has_tumor_concordance": sum(r["has_tumor"] == r["frozen_has_tumor"] for r in R),
                      "tumor_vox_corr_vs_frozen": round(float(np.corrcoef(*zip(*tv))[0, 1]), 4) if len(tv) > 2 else None,
                      "tumor_vox_median_ratio_vs_frozen": round(float(np.median([a / b for a, b in tv])), 3) if tv else None,
                      "pancreas_125": res["cases"].get("pancreas_125")}
    json.dump(res, open(OUT, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    print("->", OUT)


if __name__ == "__main__":
    main()
