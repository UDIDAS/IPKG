#!/usr/bin/env python3
"""Fetch the CT + reference label of every FLARE23 CT sub-pool candidate out of the 87 GB Drive Metadata.zip by
ranged reads (no full download), into the layout baselines_tab7_tab8.py reads:
  $VKG_DATA/flare23_pool/images/<FLARE23_xxxx>_0000.nii.gz   and   .../labels/<FLARE23_xxxx>.nii.gz
The zip's central directory (ZIP64) is parsed here and cached as $VKG_DATA/metadata_zip_index.json.

  VKG_DATA=... python fetch_flare23_ct_subpool.py results/retrieval/dedup/ct_subpool_1311_2026-10-06.json
"""
import json
import os
import struct
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

ZIP = os.environ.get("METADATA_ZIP", "drive_UD:data/VKG datasets/Metadata.zip")
OUT = os.path.join(os.environ.get("VKG_DATA", "/path/to/VKG_data"), "flare23_pool")
INDEX = os.path.join(os.environ.get("VKG_DATA", "/path/to/VKG_data"), "metadata_zip_index.json")


def cat(off, n, tries=5):
    for _ in range(tries):
        r = subprocess.run(["rclone", "cat", "--offset", str(off), "--count", str(n), ZIP], capture_output=True, timeout=3600)
        if len(r.stdout) == n:
            return r.stdout
    raise IOError(f"short read at {off}+{n}")


def size():
    r = subprocess.run(["rclone", "lsjson", ZIP], capture_output=True, text=True, timeout=300)
    return json.loads(r.stdout)[0]["Size"]


def central_directory():
    if os.path.exists(INDEX):
        return json.load(open(INDEX))
    total = size()
    tail = cat(total - 70000, 70000)
    cd_size, cd_off = struct.unpack("<II", tail[tail.rfind(b"PK\x05\x06") + 12:][:8])
    if 0xFFFFFFFF in (cd_size, cd_off):
        j = tail.rfind(b"PK\x06\x06")
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd, k, ent = cat(cd_off, cd_size), 0, {}
    while cd[k:k + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[k + 10:k + 12])
        csz, usz = struct.unpack("<II", cd[k + 20:k + 28])
        n, m, c = struct.unpack("<HHH", cd[k + 28:k + 34])
        off, = struct.unpack("<I", cd[k + 42:k + 46])
        name = cd[k + 46:k + 46 + n].decode()
        ex, e = cd[k + 46 + n:k + 46 + n + m], 0
        while e < len(ex):                                   # ZIP64 extra field: only the saturated values follow
            hid, hl = struct.unpack("<HH", ex[e:e + 4])
            if hid == 1:
                q = e + 4
                if usz == 0xFFFFFFFF:
                    usz, = struct.unpack("<Q", ex[q:q + 8]); q += 8
                if csz == 0xFFFFFFFF:
                    csz, = struct.unpack("<Q", ex[q:q + 8]); q += 8
                if off == 0xFFFFFFFF:
                    off, = struct.unpack("<Q", ex[q:q + 8]); q += 8
            e += 4 + hl
        ent[name] = [method, csz, usz, off]
        k += 46 + n + m + c
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    json.dump(ent, open(INDEX, "w"))
    return ent


def fetch(job):
    name, (method, csz, usz, off), dst = job
    if os.path.exists(dst) and os.path.getsize(dst) == usz:
        return dst, "cached"
    try:
        n, m = struct.unpack("<HH", cat(off, 30)[26:30])
        data = cat(off + 30 + n + m, csz)
        raw = zlib.decompress(data, -15) if method == 8 else data
        assert len(raw) == usz
        open(dst + ".part", "wb").write(raw); os.replace(dst + ".part", dst)
        return dst, "ok"
    except Exception as e:                                   # noqa: BLE001
        return dst, f"ERR {e}"


def main():
    cands = [c.replace("flare23_", "") for c in json.load(open(sys.argv[1]))["flare23_candidates_with_ct"]]
    ent = central_directory()
    byname = {os.path.basename(k): k for k in ent}
    os.makedirs(f"{OUT}/images", exist_ok=True); os.makedirs(f"{OUT}/labels", exist_ok=True)
    jobs, missing = [], []
    for c in cands:
        img = byname.get(f"{c}_0000.nii.gz"); lab = f"Metadata/labels/{c}.nii.gz"
        if img is None or lab not in ent:
            missing.append(c); continue
        jobs += [(img, ent[img], f"{OUT}/images/{c}_0000.nii.gz"), (lab, ent[lab], f"{OUT}/labels/{c}.nii.gz")]
    print(f"{len(cands)} candidates, {len(jobs)} files, missing in zip: {missing}", flush=True)
    bad = []
    with ThreadPoolExecutor(12) as ex:
        for i, (dst, st) in enumerate(ex.map(fetch, jobs), 1):
            if st.startswith("ERR"):
                bad.append((dst, st))
            if i % 50 == 0:
                print(i, "/", len(jobs), flush=True)
    print("FETCH_DONE", len(jobs), "errors", len(bad), bad[:5], flush=True)


if __name__ == "__main__":
    main()
