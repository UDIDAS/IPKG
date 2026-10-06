#!/usr/bin/env python3
"""Semi-oracle ORGAN arm re-scored under convention A on the four organ test lists (the study lead 10-03 list, the pipeline team item 1).

Inference = the published protocol of eval_ausam_3d.py, unchanged: the dataset's own organ checkpoint
(sam3_organ_generic_ausam_<ds>), text = organ name, GT ORGAN box + 3 px on slices whose GT organ has >= 50 px at
256^2, argmax query, sigmoid > 0.5, nearest-neighbour resize back to the native slice. Patients = the filtered-split
organ TEST lists (LiTS 6, MSD 33, KiTS 20) and the 10 FLARE23 test cases of the published row. Only the SCORING changes:

  P   = the organ model's own mask (not carved by any tumor mask)
  R   = reference organ label (KiTS: kidney 1 UNION cyst 3; FLARE23 kidney = right 2 UNION left 13)
  M   = reference mass attributed to this organ: single-organ collections -> the whole tumor label (LiTS/MSD/KiTS 2);
        FLARE23 -> each reference tumor (14) component goes to the organ holding the majority of reference organ
        voxels in a 3-voxel shell around it (organs 1 liver / 2+13 kidney / 3 spleen / 4 pancreas)
  dice_tumor_excluded  Dice(P, R)            -- the published convention (reproduction check)
  organ_dice           Dice(P, R | M)        -- convention A: organ with mass
  parenchyma_dice      Dice(P & ~M, R)       -- the model's mask outside the mass vs the parenchyma
  parenchyma_recall    |P & R| / |R|
  mass_coverage        |P & M| / |M|         -- how much of the mass the organ mask swallows (null when no mass)
LiTS = the 256^2 training .npy (no header spacing; Dice-type metrics only, as published).

  CKPT_DIR=... DATA=... python organ_convention_a.py [--datasets lits msd kits flare23]
-> results/segmentation/organ_convention_a_2026-10-03.json
"""
import argparse
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
from rerun_final_cases import load_model, run_model, sha256              # noqa: E402

CK = os.environ.get("CKPT_DIR", "/path/to/staging/ckpt")
D = os.environ.get("DATA", "/path/to/staging")
OUT = os.environ.get("OUT_JSON", os.path.join(ROOT, "results", "segmentation", "organ_convention_a_2026-10-03.json"))
MINPX = 50
FL_ORG = {"liver": [1], "kidney": [2, 13], "spleen": [3], "pancreas": [4]}
CFG = {  # ds -> (axis, {organ: (box/gate labels as published, reference labels R)}, tumor labels)
    "lits": (0, {"liver": ([1], [1])}, [2]),
    "msd": (2, {"pancreas": ([1], [1])}, [2]),
    "kits": (0, {"kidney": ([1], [1, 3])}, [2]),
    "flare23": (2, {"liver": ([1], [1]), "kidney": ([2, 13], [2, 13]), "pancreas": ([4], [4])}, [14]),
}


def cases(ds):
    if ds == "flare23":
        return [c["case"] for c in json.load(open(os.path.join(ROOT, "results", "segmentation", "ausam_3d_flare23.json")))["cases"]]
    return json.load(open(os.path.join(ROOT, "results", "audit", "split_manifests_2026-10-01.json")))["organ_pool_lkp"]["filtered"][ds]["test"]


def load(ds, c):
    if ds == "lits":
        v = c.split("-")[1]
        return np.load(f"{D}/lits/ct/volume-{v}.npy"), np.load(f"{D}/lits/seg/segmentation-{v}.npy").astype(np.uint8), None
    p = {"msd": (f"{D}/msd/imagesTr/{c}.nii.gz", f"{D}/msd/labelsTr/{c}.nii.gz"),
         "kits": (f"{D}/kits/{c}/imaging.nii.gz", f"{D}/kits/{c}/segmentation.nii.gz"),
         "flare23": (f"{D}/flare/{c}/CT.nii.gz", f"{D}/flare/{c}/seg.nii.gz")}[ds]
    img = nib.load(p[0])
    return img.get_fdata(), np.asarray(nib.load(p[1]).dataobj).astype(np.uint8), tuple(float(z) for z in img.header.get_zooms()[:3])


def sl(v, ax, z):
    return v[z] if ax == 0 else v[:, :, z]


def predict(model, proc, ct, seg, labs, text, ax, batch=8):
    """eval_ausam_3d.py's organ pass, batched (slices are independent; no padding: every slice is 256^2)."""
    pred = np.zeros(seg.shape, bool)
    todo = []
    for z in range(seg.shape[ax]):
        gm256 = resize(np.isin(sl(seg, ax, z), labs).astype(float), (256, 256), order=0, preserve_range=True) > 0.5
        if gm256.sum() < MINPX:
            continue
        ys, xs = np.where(gm256)
        box = [max(0, int(xs.min()) - 3), max(0, int(ys.min()) - 3), min(255, int(xs.max()) + 3), min(255, int(ys.max()) + 3)]
        x = np.clip(resize(sl(ct, ax, z), (256, 256), preserve_range=True, anti_aliasing=True), *IE.WIN)
        rgb = np.stack([((x - IE.WIN[0]) / (IE.WIN[1] - IE.WIN[0]) * 255).astype(np.uint8)] * 3, -1)
        todo.append((z, rgb, box))
    for i in range(0, len(todo), batch):
        ch = todo[i:i + batch]
        probs, _ = run_model(model, proc, [r for _, r, _ in ch], text, [b for _, _, b in ch])
        for (z, _, _), pr in zip(ch, probs):
            m = resize((pr > 0.5).astype(float), sl(seg, ax, z).shape, order=0, preserve_range=True) > 0.5
            if ax == 0:
                pred[z] = m
            else:
                pred[:, :, z] = m
    return pred


def attribute_flare(seg):
    """Reference tumor components -> host organ by majority vote of reference organ voxels in a 3-voxel shell."""
    t = seg == 14
    out = {o: np.zeros(seg.shape, bool) for o in FL_ORG}
    lab, n = ndimage.label(t)
    for i, sli in enumerate(ndimage.find_objects(lab), 1):
        pad = tuple(slice(max(0, s.start - 4), s.stop + 4) for s in sli)
        comp = lab[pad] == i
        shell = ndimage.binary_dilation(comp, iterations=3) & ~comp
        votes = {o: int(np.isin(seg[pad][shell], ls).sum()) for o, ls in FL_ORG.items()}
        best = max(votes, key=votes.get)
        if votes[best] > 0:
            out[best][pad] |= comp
    return out


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return round(2 * int((a & b).sum()) / s, 4) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(CFG))
    a = ap.parse_args()
    from transformers import Sam3Processor
    import torch
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    pub = {ds: {c["case"]: c for c in json.load(open(os.path.join(ROOT, "results", "segmentation", f"ausam_3d_{ds}.json")))["cases"]} for ds in CFG}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"note": __doc__.strip(), "datasets": {}}
    for ds in a.datasets:
        ax, organs, tlab = CFG[ds]
        ck = f"{CK}/sam3_organ_generic_ausam_{ds}.pth"
        model = load_model(ck)
        rows = []
        for c in cases(ds):
            t0 = time.time()
            try:
                ct, seg, sp = load(ds, c)
            except Exception as e:                                  # noqa: BLE001
                rows.append({"case": c, "error": f"{type(e).__name__}: {str(e)[:100]}"}); print(c, "ERROR", e, flush=True); continue
            mass = attribute_flare(seg) if ds == "flare23" else {o: np.isin(seg, tlab) for o in organs}
            rec = {"case": c, "organs": {}}
            for o, (gl, rl) in organs.items():
                P = predict(model, proc, ct, seg, gl, o, ax)
                R = np.isin(seg, rl)
                M = mass[o] & ~R
                pp = pub[ds].get(c, {}).get("organs", {}).get(o, {})
                rec["organs"][o] = {"dice_tumor_excluded": dice(P, np.isin(seg, gl)), "published_dice": pp.get("dice3d"),
                                    "organ_dice": dice(P, R | M), "parenchyma_dice": dice(P & ~M, R),
                                    "parenchyma_recall": round(int((P & R).sum()) / max(1, int(R.sum())), 4),
                                    "mass_coverage": round(int((P & M).sum()) / int(M.sum()), 4) if M.any() else None,
                                    "mass_vox": int(M.sum()), "pred_vox": int(P.sum()), "published_pred_vox": pp.get("vox_pred")}
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)
            print(ds, c, {o: (v["dice_tumor_excluded"], v["published_dice"], v["organ_dice"]) for o, v in rec["organs"].items()}, flush=True)
            torch.cuda.empty_cache()
        summ = {}
        for o in organs:
            R_ = [r["organs"][o] for r in rows if "organs" in r]
            summ[o] = {"n": len(R_), **{k: round(float(np.mean([x[k] for x in R_ if x[k] is not None])), 4) if any(x[k] is not None for x in R_) else None
                                        for k in ("dice_tumor_excluded", "published_dice", "organ_dice", "parenchyma_dice", "parenchyma_recall", "mass_coverage")},
                       "n_with_mass": sum(x["mass_coverage"] is not None for x in R_),
                       "reproduces_published_mean": None}
            if summ[o]["published_dice"] is not None:
                summ[o]["reproduces_published_mean"] = abs(summ[o]["dice_tumor_excluded"] - summ[o]["published_dice"]) <= 0.005
        res["datasets"][ds] = {"checkpoint": os.path.basename(ck), "checkpoint_sha256": sha256(ck), "n_cases": len(rows),
                               "summary": summ, "cases": rows}
        json.dump(res, open(OUT, "w"), indent=1)
        print(ds, json.dumps(summ), flush=True)
        del model
        torch.cuda.empty_cache()
    print("->", OUT)


if __name__ == "__main__":
    main()
