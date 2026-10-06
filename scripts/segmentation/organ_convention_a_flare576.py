#!/usr/bin/env python3
"""The 576 FLARE23 semi-oracle masks re-scored under convention A from the existing files (the study lead 10-03 list, the pipeline team item 2).

Masks: masks_predicted_flare23/ (Drive; Zenodo deposit), drawn 09-14 by sam3_organ_generic_ausam_flare23 (organs, GT box)
+ sam3_tumor_ausam_flare (tumor, GT box); EXCLUSIVE labels 1 liver / 2 right kidney / 3 spleen / 4 pancreas /
13 left kidney / 14 tumor (tumor wins). Reference: the FLARE23 labels (same scheme), from the Drive Metadata.zip.
Kidney = right | left, as in the published organ table.

Mass attribution (both sides, same rule): each tumor component (label 14) goes to the organ holding the majority of
organ voxels in a 3-voxel shell around it (organ_convention_a.attribute_flare). Per organ o:
  Po = predicted organ label, Pm = predicted tumor attributed to o, R = reference organ label, M = reference mass of o
  dice_tumor_excluded  Dice(Po, R)             -- the published convention
  organ_dice           Dice(Po | Pm, R | M)    -- convention A: organ with mass, both sides
  parenchyma_dice      Dice(Po & ~M, R)
  parenchyma_recall    |Po & R| / |R|
  mass_coverage        |(Po | Pm) & M| / |M|   (null when the organ carries no reference mass)
Reported for all 576 and for the organ model's held-out subset (filtered organ split TEST), because 316 of the 576 are
that model's training cases. Semi-oracle: the GT box was given; these are interactive upper bounds.

  MASKS=... LABELS=... python organ_convention_a_flare576.py -> results/segmentation/organ_convention_a_flare576_2026-10-03.json
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
from organ_convention_a import FL_ORG, dice                               # noqa: E402
from scipy import ndimage                                                 # noqa: E402

MASKS = os.environ.get("MASKS", "/path/to/staging/masks/masks_predicted_flare23")
LABELS = os.environ.get("LABELS", "/path/to/staging/labels")
OUT = os.environ.get("OUT_JSON", os.path.join(ROOT, "results", "segmentation", "organ_convention_a_flare576_2026-10-03.json"))
SCORED = ("liver", "kidney", "spleen", "pancreas")


def attribute(seg):
    out = {o: np.zeros(seg.shape, bool) for o in FL_ORG}
    lab, n = ndimage.label(seg == 14)
    for i, sli in enumerate(ndimage.find_objects(lab), 1):
        pad = tuple(slice(max(0, s.start - 4), s.stop + 4) for s in sli)
        comp = lab[pad] == i
        shell = ndimage.binary_dilation(comp, iterations=3) & ~comp
        votes = {o: int(np.isin(seg[pad][shell], ls).sum()) for o, ls in FL_ORG.items()}
        best = max(votes, key=votes.get)
        if votes[best] > 0:
            out[best][pad] |= comp
    return out


def one(c):
    try:
        p = np.asarray(nib.load(f"{MASKS}/{c}.nii.gz").dataobj).astype(np.uint8)
        g = np.asarray(nib.load(f"{LABELS}/{c}.nii.gz").dataobj).astype(np.uint8)
    except Exception as e:                                                # noqa: BLE001
        return {"case": c, "error": str(e)[:100]}
    if p.shape != g.shape:
        return {"case": c, "error": f"shape {p.shape} vs {g.shape}"}
    pm, gm = attribute(p), attribute(g)
    rec = {"case": c, "organs": {}}
    for o in SCORED:
        Po, R = np.isin(p, FL_ORG[o]), np.isin(g, FL_ORG[o])
        if not R.any():
            continue
        Pm, M = pm[o] & ~Po, gm[o] & ~R
        rec["organs"][o] = {"dice_tumor_excluded": dice(Po, R), "organ_dice": dice(Po | Pm, R | M),
                            "parenchyma_dice": dice(Po & ~M, R), "parenchyma_recall": round(int((Po & R).sum()) / int(R.sum()), 4),
                            "mass_coverage": round(int(((Po | Pm) & M).sum()) / int(M.sum()), 4) if M.any() else None,
                            "mass_vox": int(M.sum())}
    return rec


def summarize(rows):
    s = {}
    for o in SCORED:
        R_ = [r["organs"][o] for r in rows if o in r.get("organs", {})]
        s[o] = {"n": len(R_), "n_with_mass": sum(x["mass_coverage"] is not None for x in R_)}
        for k in ("dice_tumor_excluded", "organ_dice", "parenchyma_dice", "parenchyma_recall", "mass_coverage"):
            v = [x[k] for x in R_ if x[k] is not None]
            s[o][k] = round(float(np.mean(v)), 4) if v else None
            if k in ("dice_tumor_excluded", "organ_dice") and v:
                s[o][k + "_median"] = round(float(np.median(v)), 4)
    return s


def main():
    ids = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{MASKS}/*.nii.gz"))
    with Pool(int(os.environ.get("NPROC", 14))) as pool:
        rows = pool.map(one, ids, chunksize=4)
    split = json.load(open(os.path.join(ROOT, "results", "audit", "split_manifests_2026-10-01.json")))["organ_pool_lkp"]["filtered"]["flare23"]
    role = {c: r for r in ("train", "val", "test") for c in split[r]}
    ok = [r for r in rows if "organs" in r]
    res = {"note": __doc__.strip(), "n_masks": len(ids), "n_scored": len(ok), "errors": [r for r in rows if "error" in r],
           "all_576": summarize(ok),
           "organ_model_test_subset": summarize([r for r in ok if role.get(r["case"]) == "test"]),
           "organ_model_train_or_val": summarize([r for r in ok if role.get(r["case"]) in ("train", "val")]),
           "not_in_organ_pool": summarize([r for r in ok if r["case"] not in role]),
           "cases": [{**r, "organ_split_role": role.get(r["case"], "not in organ pool")} for r in rows]}
    json.dump(res, open(OUT, "w"), indent=1)
    for k in ("all_576", "organ_model_test_subset", "organ_model_train_or_val", "not_in_organ_pool"):
        print(k, {o: (v["n"], v["dice_tumor_excluded"], v["organ_dice"], v["mass_coverage"]) for o, v in res[k].items()})
    print("->", OUT)


if __name__ == "__main__":
    main()
