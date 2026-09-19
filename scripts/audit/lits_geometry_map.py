#!/usr/bin/env python3
"""LiTS real volumes (gating item 5 / queue item 1): recover per-case geometry for the LiTS
cases from MSD Task03_Liver (the public LiTS release WITH NIfTI headers), then rebuild the
LiTS corpora with physical volumes.

The shipped LiTS data are 256x256 .npy volumes with NO headers (pseudo-volumes, 1 mm iso
assumed) and are LIVER-SLAB CROPS: every npy slice contains liver, so npy z-extent < native
z-extent. The mapping volume-N -> liver_M is recovered from LABELS alone:

  the npy per-slice liver-area profile is slid (both z-directions) over the native Task03
  per-slice profile; a match requires correlation >= 0.999 at the best offset AND the
  in-window area scale ~ (X/256)*(Y/256) (= 4 for 512x512 natives, tolerance 3.4-4.6).
  The numbering turned out to be IDENTITY (volume-N = liver_N), so identity is tested first
  (accepted at correlation >= 0.9995) and a full search over all 131 candidates (accepted
  only with a >= 0.003 gap to the runner-up) is the fallback. Offset and z-flip per case are
  recorded.

Cross-check: the dedup-audit anchor FLARE23_0102 = liver_64 — the staged predicted FLARE23
mask header (shape/spacing) is compared against liver_64's header, and the volume-N that maps
to liver_64 is reported.

Stage 2 (--rebuild): corpus_{gt,predicted}_lits.json get per-case spacing + physical volumes.
  gt:        native-grid voxel counts from the Task03 label x native voxel volume (real cm3).
  predicted: the frozen 256^2-grid voxel counts x effective voxel volume
             (sx*(X/256) * sy*(Y/256) * sz)  — the predicted masks themselves are gone.
  The former pseudo values are kept as organ_volume_cm3_pseudo / tumor_volume_cm3_pseudo;
  categoricals (burden terciles etc.) are voxel-count-based and unchanged.

  python lits_geometry_map.py [--rebuild]
-> results/audit/lits_geometry_map.json (+ rebuilt corpora with --rebuild)
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LABELS = "/path/to/staging/msd_task03/Task03_Liver/labelsTr"
NPY_SEG = "/path/to/staging/acm_data/lits_npy/seg"
FLARE_MASKS = "/path/to/staging/acm_data/flare23_masks/masks_predicted_flare23"
OUT = os.path.join(ROOT, "results", "audit", "lits_geometry_map.json")


def npy_profile(fp):
    seg = np.load(fp)                      # (z, 256, 256), liver=1 (+tumor=2)
    liver = (seg >= 1)
    prof = liver.reshape(liver.shape[0], -1).sum(1).astype(float)
    return prof, seg


def nii_profile(fp):
    seg = np.asarray(nib.load(fp).dataobj)  # (x, y, z), liver=1, tumor=2
    liver = (seg >= 1)
    prof = liver.sum(axis=(0, 1)).astype(float)
    return prof, seg


def slide_corr(short, full):
    """Best Pearson correlation of `short` over all offsets of `full`, both z-directions.
    Returns (corr, offset, flipped, scale) — scale = median full/short area in the window."""
    L = len(short)
    if len(full) < L or short.std() == 0:
        return -1.0, 0, False, 0.0
    best = (-1.0, 0, False, 0.0)
    for flip in (False, True):
        q = full[::-1] if flip else full
        w = np.lib.stride_tricks.sliding_window_view(q, L).astype(float)   # (n_off, L)
        wm = w - w.mean(1, keepdims=True)
        sm = short - short.mean()
        denom = np.sqrt((wm ** 2).sum(1) * (sm ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            cs = np.where(denom > 0, wm @ sm / denom, -1.0)
        o = int(np.argmax(cs))
        if cs[o] > best[0]:
            scale = float(np.median(q[o:o + L] / np.maximum(short, 1)))
            best = (float(cs[o]), o, flip, scale)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--npy", default=NPY_SEG)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    print("indexing Task03 labels ...", flush=True)
    task03 = {}
    for fp in sorted(glob.glob(f"{a.labels}/liver_*.nii.gz")):
        if os.path.basename(fp).startswith("._"):
            continue
        m = os.path.basename(fp)[len("liver_"):-len(".nii.gz")]
        nii = nib.load(fp)
        seg = np.asarray(nii.dataobj)
        prof = (seg >= 1).sum(axis=(0, 1)).astype(float)
        task03[m] = {"prof": prof, "n": int(seg.shape[2]), "shape": [int(x) for x in seg.shape],
                     "spacing": [float(z) for z in nii.header.get_zooms()[:3]],
                     "liver_vox": int((seg == 1).sum()),          # exclusive, the frozen corpus convention
                     "liver_union_vox": int((seg >= 1).sum()),
                     "tumor_vox": int((seg == 2).sum())}
    print(f"  {len(task03)} Task03 labels", flush=True)

    def scale_ok(t, s):
        ex = (t["shape"][0] / 256.0) * (t["shape"][1] / 256.0)
        return 0.85 * ex <= s <= 1.15 * ex

    mapping, ambiguous = {}, []
    for fp in sorted(glob.glob(f"{a.npy}/segmentation-*.npy")):
        vid = "volume-" + os.path.basename(fp)[len("segmentation-"):-len(".npy")]
        prof, seg = npy_profile(fp)
        m = c = off = flip = scale = None
        ident = vid.replace("volume-", "")
        if ident in task03:                                   # identity first (volume-N = liver_N)
            c, off, flip, scale = slide_corr(prof, task03[ident]["prof"])
            if c >= 0.9995 and scale_ok(task03[ident], scale):
                m = ident
        if m is None:                                         # full search fallback
            scores = []
            for cand, t in task03.items():
                cc, oo, ff, ss = slide_corr(prof, t["prof"])
                scores.append((cc, cand, oo, ff, ss))
            scores.sort(reverse=True)
            (c, m, off, flip, scale) = scores[0]
            if c < 0.999 or not scale_ok(task03[m], scale) or \
               (len(scores) > 1 and c - scores[1][0] < 0.003):
                ambiguous.append({"query": vid, "n_slices": len(prof),
                                  "top": [(cand, round(s, 5)) for s, cand, *_ in scores[:3]]})
                continue
        t = task03[m]
        vx = (seg == 2).sum()
        mapping[vid] = {"liver": f"liver_{m}", "corr": round(c, 6), "z_flipped": bool(flip),
                        "slab_offset": int(off), "area_scale": round(scale, 3),
                        "shape_native": t["shape"], "spacing": t["spacing"],
                        "npy_slices": int(seg.shape[0]),
                        "gt_liver_vox_native": t["liver_vox"], "gt_liver_union_vox_native": t["liver_union_vox"],
                        "gt_tumor_vox_native": t["tumor_vox"],
                        "npy_liver_vox": int((seg == 1).sum()), "npy_tumor_vox": int(vx)}

    # anchor cross-check: FLARE23_0102 = liver_64
    anchor = {}
    l64 = task03.get("64")
    if l64:
        anchor["liver_64"] = {"shape": l64["shape"], "spacing": l64["spacing"]}
        f0102 = os.path.join(FLARE_MASKS, "FLARE23_0102.nii.gz")
        if os.path.exists(f0102):
            nii = nib.load(f0102)
            anchor["FLARE23_0102_mask"] = {"shape": [int(x) for x in nii.shape],
                                           "spacing": [float(z) for z in nii.header.get_zooms()[:3]]}
        anchor["volume_mapping_to_liver_64"] = next((v for v, m in mapping.items() if m["liver"] == "liver_64"), None)

    out = {"note": __doc__.split("Stage 2")[0].strip(), "n_mapped": len(mapping),
           "n_ambiguous": len(ambiguous), "mapping": mapping, "ambiguous": ambiguous,
           "anchor_crosscheck": anchor}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    qids = [r["case_id"] for r in json.load(open(f"{ROOT}/corpora/corpus_gt_lits.json"))["records"]]
    missing = [q for q in qids if q not in mapping]
    print(f"mapped {len(mapping)} / ambiguous {len(ambiguous)}; 21 query cases missing: {missing}")
    print(f"anchor: {anchor}")
    print(f"-> {OUT}", flush=True)

    if not a.rebuild:
        return
    if missing:
        raise SystemExit(f"cannot rebuild: unmapped query cases {missing}")
    for kind in ("gt", "predicted"):
        fp = f"{ROOT}/corpora/corpus_{kind}_lits.json"
        d = json.load(open(fp))
        for r in d["records"]:
            m = mapping[r["case_id"]]
            sx, sy, sz = m["spacing"]
            X, Y, _ = m["shape_native"]
            eff_cm3 = (sx * X / 256) * (sy * Y / 256) * sz / 1000.0     # 256^2-grid voxel -> cm3
            nat_cm3 = sx * sy * sz / 1000.0
            org = r["organs"]["liver"]
            org["organ_volume_cm3_pseudo"] = org["organ_volume_cm3"]
            org["tumor_volume_cm3_pseudo"] = org["tumor_volume_cm3"]
            if kind == "gt":                                             # native GT counts -> real cm3
                org["organ_volume_cm3"] = round(m["gt_liver_vox_native"] * nat_cm3, 2)
                org["tumor_volume_cm3"] = round(m["gt_tumor_vox_native"] * nat_cm3, 2)
            else:                                                        # frozen 256^2-grid predicted counts
                org_vox = round(org["organ_volume_cm3_pseudo"] * 1000)   # pseudo used voxel*0.001
                org["organ_volume_cm3"] = round(org_vox * eff_cm3, 2)
                org["tumor_volume_cm3"] = round(org["tumor_voxels"] * eff_cm3, 2)
            r["spacing_mm"] = m["spacing"]
            r["msd_task03_id"] = m["liver"]
        d["note"] = d.get("note", "") + (
            " REAL VOLUMES (2026-09-16): per-case geometry recovered from MSD Task03_Liver headers via "
            "label-profile matching (results/audit/lits_geometry_map.json); organ/tumor_volume_cm3 are physical "
            "(gt: native-grid counts from the full Task03 label; predicted: frozen 256^2 counts x effective voxel "
            "volume — the npy copies are liver-slab crops, so predicted volumes remain slab-limited); former "
            "pseudo values kept as *_pseudo; categoricals (voxel-tercile) unchanged.")
        json.dump(d, open(fp, "w"), indent=1)
        print(f"rebuilt {fp}")


if __name__ == "__main__":
    main()
