#!/usr/bin/env python3
"""Confirm/refute the 17 exact-grid collisions from the 687-record prescreen (2026-09-24).

Input: results/audit/unscanned687_grid_prescreen_2026-09-24.json (grid_collisions).
For each collision (an unmasked pool FLARE23 record sharing sorted shape + spacing with a
query volume):

  * KiTS / MSD queries - the definitive test: download both GT labels (query from the Drive
    collection copy, FLARE from Metadata.zip) and run the census's 48-configuration
    (permutation x flip) mask-overlap search.  Organ IoU >= 0.8 = same scan (different
    patients never exceed ~0.5; the eleven second-pass twins scored 0.87-0.96 despite full
    re-annotation), organ IoU well below = protocol coincidence (same scanner protocol, two
    patients).
  * LiTS queries - no native-grid Task03 GT is on this Drive (only downsampled npy slabs),
    so the strongest local test is native voxel-count comparison against
    lits_geometry_map.json's gt_{liver,tumor}_vox_native: every census-confirmed LiTS
    re-share matched liver GT voxels EXACTLY (liver_delta_pct 0.0 - FLARE inherited the
    Task03 liver labels).  Exact/near-exact = twin; a large delta = coincidence; anything
    between is flagged for KS (he has Task03 mounted) - stated per pair.
  * A collision whose FLARE record the LiTS overlap census ALREADY attributes to a
    different, non-query LiTS volume is closed as benign (grid coincidence between two LiTS
    scans), with the attribution shown.

  ZIP_INDEX=/dev/shm/zipidx/metadata_zip_index.json python confirm_grid_collisions.py
-> results/audit/unscanned687_collision_confirm_2026-09-24.json  (exit 1 if any pair is a
   confirmed or unresolved possible twin)
"""
import itertools
import json
import os
import struct
import subprocess
import sys
import zlib

import nibabel as nib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ZIP = "drive_UD:data/VKG datasets/Metadata.zip"
IDX = os.environ.get("ZIP_INDEX", "/dev/shm/zipidx/metadata_zip_index.json")
DRIVE = "drive_UD:data/VKG datasets"
CACHE = "/dev/shm/collide"
PRE = os.path.join(ROOT, "results", "audit", "unscanned687_grid_prescreen_2026-09-24.json")
OUT = os.path.join(ROOT, "results", "audit", "unscanned687_collision_confirm_2026-09-24.json")
FLARE_ORGAN = {"kits": [2, 13], "msd": [4], "lits": [1]}
QUERY_ORGAN, QUERY_TUMOR = 1, 2   # kits: kidney/tumor; Task07: pancreas/tumor; Task03: liver/tumor


def rcat(remote, off=None, cnt=None):
    cmd = ["rclone", "cat"]
    if off is not None:
        cmd += ["--offset", str(off), "--count", str(cnt)]
    r = subprocess.run(cmd + [remote], stdout=subprocess.PIPE)
    assert r.returncode == 0, remote
    return r.stdout


def flare_label(fid):
    dst = f"{CACHE}/{fid}.nii.gz"
    if not os.path.exists(dst):
        ent = json.load(open(IDX))[f"Metadata/labels/{fid}.nii.gz"]
        method, csz, usz, off = ent
        lh = rcat(ZIP, off, 30)
        n, m = struct.unpack("<HH", lh[26:30])
        data = rcat(ZIP, off + 30 + n + m, csz)
        raw = data if method == 0 else zlib.decompress(data, -15)
        assert len(raw) == usz
        open(dst, "wb").write(raw)
    return np.asarray(nib.load(dst).dataobj).astype(np.uint8)


def query_label(q):
    if q.startswith("case_"):
        rem, name = f"{DRIVE}/KiTS/{q}/segmentation.nii.gz", f"{q}.nii.gz"
    else:
        rem, name = f"{DRIVE}/Pancreas/labelsTr/{q}.nii.gz", f"{q}.nii.gz"
    dst = f"{CACHE}/{name}"
    if not os.path.exists(dst):
        subprocess.run(["rclone", "copyto", rem, dst], check=True)
    return np.asarray(nib.load(dst).dataobj).astype(np.uint8)


def best_overlap(qm, fm):
    best, cfg = 0.0, None
    for perm in itertools.permutations(range(3)):
        if tuple(np.transpose(qm, perm).shape) != fm.shape:
            continue
        qp = np.transpose(qm, perm)
        for flips in itertools.product([False, True], repeat=3):
            m = qp
            for ax, fl in enumerate(flips):
                if fl:
                    m = np.flip(m, ax)
            u = np.logical_or(m, fm).sum()
            iou = float(np.logical_and(m, fm).sum() / u) if u else 0.0
            if iou > best:
                best, cfg = iou, (perm, flips)
    return best, cfg


def main():
    os.makedirs(CACHE, exist_ok=True)
    cols = json.load(open(PRE))["grid_collisions"]
    lmap = json.load(open(os.path.join(ROOT, "results/audit/lits_geometry_map.json")))["mapping"]
    lcen = json.load(open(os.path.join(ROOT, "results/audit/lits_flare23_overlap_census.json")))
    rows, twins, flags = [], [], []
    for c in cols:
        fid = c["flare"].replace("flare23_", "")
        q = c["query"]
        ds = "kits" if q.startswith("case_") else ("lits" if q.startswith("volume-") else "msd")
        row = {"flare": c["flare"], "query": q, "dataset": ds, "grid": c["shape"],
               "spacing": c["flare_spacing"]}
        if ds == "lits":
            prior = lcen["confirmed"].get(fid)
            fs = flare_label(fid)
            fl, ft = int((fs == 1).sum()), int((fs == 14).sum())
            gl = lmap[q]["gt_liver_vox_native"]; gt = lmap[q]["gt_tumor_vox_native"]
            row.update({"flare_liver_vox": fl, "query_liver_vox_native": gl,
                        "flare_tumor_vox": ft, "query_tumor_vox_native": gt,
                        "liver_delta_pct": round(abs(fl - gl) / gl * 100, 3)})
            if prior:
                row["lits_census_attribution"] = prior
                row["verdict"] = (f"BENIGN - census already attributes {fid} to {prior['volume']} "
                                  "(non-query); grid coincidence between two LiTS scans")
                if prior["volume"] == q:
                    row["verdict"] = "CONFIRMED_TWIN (census attribution matches the query)"
                    twins.append(row)
            elif ft > 0 and ft == gt:
                row["verdict"] = "CONFIRMED_TWIN (tumor GT voxel counts EXACTLY equal - the LiTS " \
                                 "census's T2 criterion: FLARE inherited the Task03 tumor label " \
                                 "verbatim; the liver delta is FLARE's own liver re-annotation)"
                twins.append(row)
            elif row["liver_delta_pct"] <= 0.5:
                row["verdict"] = "CONFIRMED_TWIN (liver GT voxels match the query's native count; " \
                                 "every census-confirmed LiTS re-share matched at delta 0.0)"
                twins.append(row)
            elif row["liver_delta_pct"] >= 5:
                row["verdict"] = "coincidence (liver GT differs by " + str(row["liver_delta_pct"]) + " %)"
            else:
                row["verdict"] = "FLAG for KS Task03 mask-overlap (borderline liver delta)"
                flags.append(row)
        else:
            qs = query_label(q)
            fs = flare_label(fid)
            qo, qt = qs == QUERY_ORGAN, qs == QUERY_TUMOR
            fo, ft = np.isin(fs, FLARE_ORGAN[ds]), fs == 14
            iou, cfg = best_overlap(qo, fo)
            tiou = None
            if cfg is not None and qt.any():
                m = np.transpose(qt, cfg[0])
                for ax, fl in enumerate(cfg[1]):
                    if fl:
                        m = np.flip(m, ax)
                u = np.logical_or(m, ft).sum()
                tiou = round(float(np.logical_and(m, ft).sum() / u), 4) if u else 0.0
            row.update({"organ_iou": round(iou, 4), "tumor_iou": tiou,
                        "transposition": str(cfg),
                        "organ_vox": [int(qo.sum()), int(fo.sum())],
                        "tumor_vox": [int(qt.sum()), int(ft.sum())]})
            if iou >= 0.8:
                row["verdict"] = "CONFIRMED_TWIN"
                twins.append(row)
            elif tiou is not None and tiou >= 0.8 and min(row["tumor_vox"]) >= 1000:
                row["verdict"] = ("CONFIRMED_TWIN (by tumor: IoU "
                                  f"{tiou} on a {min(row['tumor_vox'])}-voxel tumor; the organ "
                                  "disagreement is re-annotation, two patients cannot share a tumor)")
                twins.append(row)
            elif iou <= 0.5:
                row["verdict"] = "coincidence (different patients; organ IoU far below twin range)"
            else:
                row["verdict"] = "FLAG (ambiguous IoU)"
                flags.append(row)
        rows.append(row)
        print(row["flare"], "vs", q, "->", row["verdict"])
    report = {"confirm": "unscanned687_collision_confirm_2026-09-24",
              "n_collisions": len(cols), "rows": rows,
              "n_confirmed_twins": len(twins), "confirmed_twins": twins,
              "n_flags": len(flags), "flags": flags}
    json.dump(report, open(OUT, "w"), indent=1)
    print(f"\n{len(twins)} confirmed twin(s), {len(flags)} flag(s), "
          f"{len(cols) - len(twins) - len(flags)} coincidence(s)/benign")
    sys.exit(1 if (twins or flags) else 0)


if __name__ == "__main__":
    main()
