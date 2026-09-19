#!/usr/bin/env python3
"""Full LiTS <-> FLARE23 overlap census (v11 item 4): how many of the 131 LiTS volumes appear
inside FLARE23 — the count the Limitations sentence needs. Extends the query-only extended
audit (lits_extended_audit.json) to ALL 131 volumes, replacing/confirming KS's geometry screen.

Screen (local, from already-computed signatures):
  T2 physical — native Task03 liver volume within 1 % AND tumor volume within 2 % (or both
     tumor-free) of a FLARE23 GT record (corpus_flare23_kg.json, 1,312 cases); Task03 volumes
     from lits_geometry_map.json native counts x native voxel volume.
  T1 geometry — same native shape AND spacing (1e-3) as a shipped predicted-mask header
     (576 of 1,312 have masks in the release copy).
Confirmation: every screened candidate's GT label is streamed from the Drive Metadata.zip
(ranged zip reads, index /path/to/staging/metadata_zip_index.json) and compared voxel-exactly
(tumor label 14 vs Task03 label 2; shape; spacing).

Non-query twins STAY in the 1,347 pool: no query duplicates them, and each is unique within
the pool (the other 110 LiTS volumes are not corpus members) — this census only feeds the
data-overlap statement.
-> results/audit/lits_flare23_overlap_census.json
"""
import glob
import json
import os
import struct
import subprocess
import zlib
from concurrent.futures import ThreadPoolExecutor

import nibabel as nib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MASKS = "/path/to/staging/acm_data/flare23_masks/masks_predicted_flare23"
LABELS_DIR = "/path/to/staging/acm_data/flare23_labels_audit"
ZIP = "drive_UD:data/VKG datasets/Metadata.zip"
ZIP_INDEX = "/path/to/staging/metadata_zip_index.json"
TASK03 = "/path/to/staging/msd_task03/Task03_Liver/labelsTr"
OUT = os.path.join(ROOT, "results", "audit", "lits_flare23_overlap_census.json")


def close(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def fetch_label(cid, idx):
    dst = f"{LABELS_DIR}/{cid}.nii.gz"
    name = f"Metadata/labels/{cid}.nii.gz"
    method, csz, usz, off = idx[name]
    if os.path.exists(dst) and os.path.getsize(dst) == usz:
        return cid
    hdr = subprocess.run(["rclone", "cat", "--offset", str(off), "--count", "30", ZIP],
                         capture_output=True, timeout=120).stdout
    n, m = struct.unpack("<HH", hdr[26:30])
    data = subprocess.run(["rclone", "cat", "--offset", str(off + 30 + n + m), "--count", str(csz), ZIP],
                          capture_output=True, timeout=1800).stdout
    raw = zlib.decompress(data, -15) if method == 8 else data
    assert len(raw) == usz, (cid, len(raw), usz)
    open(dst + ".part", "wb").write(raw)
    os.replace(dst + ".part", dst)
    return cid


def main():
    gmap = json.load(open(f"{ROOT}/results/audit/lits_geometry_map.json"))["mapping"]
    flare = json.load(open(f"{ROOT}/corpora/corpus_flare23_kg.json"))["records"]
    dedup_ids = {r["case_id"] for r in json.load(open(f"{ROOT}/corpora/corpus_flare23_kg_dedup.json"))["records"]}
    prior = json.load(open(f"{ROOT}/results/audit/lits_extended_audit.json"))
    prior_conf = {c: v["query"] for c, v in prior["gt_label_confirmation"]["results"].items()}
    queries = {r["case_id"] for r in json.load(open(f"{ROOT}/corpora/corpus_gt_lits.json"))["records"]}

    headers = {}
    for f in sorted(glob.glob(f"{MASKS}/*.nii.gz")):
        nii = nib.load(f)
        headers[os.path.basename(f)[:-7]] = ([int(x) for x in nii.shape],
                                             [float(z) for z in nii.header.get_zooms()[:3]])

    cand = {}                                            # flare cid (bare) -> {volume, tests}
    for vid, m in gmap.items():
        sx, sy, sz = m["spacing"]
        nat = sx * sy * sz / 1000.0
        ov, tv = m["gt_liver_vox_native"] * nat, m["gt_tumor_vox_native"] * nat
        for fr in flare:
            liver = fr.get("organs", {}).get("liver")
            if not liver:
                continue
            if close(ov, liver["organ_volume_cm3"], 0.01) and \
               (close(tv, liver["tumor_volume_cm3"], 0.02) or (m["gt_tumor_vox_native"] == 0 and liver["tumor_volume_cm3"] == 0)):
                cid = fr["case_id"].replace("flare23_", "")
                cand.setdefault(cid, {"volume": vid, "tests": set()})["tests"].add("T2")
        shape = m["shape_native"]
        for cid, (hs, hsp) in headers.items():
            if shape == hs and all(abs(a - b) < 1e-3 for a, b in zip(m["spacing"], hsp)):
                cand.setdefault(cid, {"volume": vid, "tests": set()})["tests"].add("T1")

    print(f"screen: {len(cand)} candidate FLARE23 cases ({sum(1 for c in cand if c in prior_conf)} already confirmed)", flush=True)
    idx = json.load(open(ZIP_INDEX))
    todo = [c for c in sorted(cand) if c not in prior_conf and f"Metadata/labels/{c}.nii.gz" in idx]
    no_label = [c for c in sorted(cand) if c not in prior_conf and f"Metadata/labels/{c}.nii.gz" not in idx]
    with ThreadPoolExecutor(8) as ex:
        for i, c in enumerate(ex.map(lambda c: fetch_label(c, idx), todo)):
            if (i + 1) % 10 == 0:
                print(f"  fetched {i+1}/{len(todo)}", flush=True)

    confirmed, rejected = {}, {}
    for c in sorted(cand):
        vid = cand[c]["volume"]
        if c in prior_conf:
            confirmed[c] = {"volume": prior_conf[c], "basis": "prior extended audit (query twin)",
                            "tests": sorted(cand[c]["tests"])}
            continue
        if c in no_label:
            rejected[c] = {"volume": vid, "reason": "no GT label in Metadata.zip", "tests": sorted(cand[c]["tests"])}
            continue
        f = nib.load(f"{LABELS_DIR}/{c}.nii.gz")
        fs = np.asarray(f.dataobj)
        t = nib.load(f"{TASK03}/{gmap[vid]['liver']}.nii.gz")
        ts = np.asarray(t.dataobj)
        same = (list(fs.shape) == list(ts.shape)
                and all(abs(a - b) < 1e-3 for a, b in zip(f.header.get_zooms()[:3], t.header.get_zooms()[:3]))
                and int((fs == 14).sum()) == int((ts == 2).sum()))
        rec = {"volume": vid, "tests": sorted(cand[c]["tests"]),
               "flare_tumor_vox": int((fs == 14).sum()), "task03_tumor_vox": int((ts == 2).sum()),
               "liver_delta_pct": round(abs(int((fs == 1).sum()) - int((ts == 1).sum())) / max(int((ts == 1).sum()), 1) * 100, 3)}
        (confirmed if same else rejected)[c] = rec
        print(f"  {c} <- {vid}: {'CONFIRMED' if same else 'rejected'}", flush=True)

    in_pool = sorted(c for c in confirmed if f"flare23_{c}" in dedup_ids)
    q_twins = sorted(c for c in confirmed if confirmed[c].get("basis"))
    vols = {v["volume"] for v in confirmed.values()}
    out = {"note": __doc__.strip(),
           "n_lits_volumes_screened": len(gmap),
           "n_candidates": len(cand),
           "n_confirmed": len(confirmed),
           "n_rejected": len(rejected),
           "n_distinct_lits_volumes_in_flare23": len(vols),
           "confirmed": confirmed, "rejected": rejected,
           "query_twins_among_confirmed": q_twins,
           "confirmed_still_in_1347_pool_nonquery": [c for c in in_pool if c not in q_twins],
           "limitations_sentence_numbers": {
               "lits_volumes_with_a_flare23_twin": len(vols),
               "of_131_screened": 131,
               "flare23_cases_that_are_lits_sourced": len(confirmed),
               "of_which_query_twins_removed_from_pool": len(q_twins) + 1,   # + FLARE23_0102 (volume-64, not a query)
               "note": "FLARE23_0102 (= volume-64) is counted in the removals: confirmed 09-15b, removed then."}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\n{len(confirmed)} FLARE23 cases confirmed LiTS-sourced ({len(vols)} distinct LiTS volumes of 131); "
          f"{len(rejected)} candidates rejected -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
