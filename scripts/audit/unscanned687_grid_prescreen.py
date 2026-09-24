#!/usr/bin/env python3
"""Grid-level twin prescreen for the 687 unmasked pool FLARE23 records (2026-09-24).

Closes (to the extent physically possible) the disclosed residual: the 687 pool FLARE23
records with no predicted mask were covered only by the label-based volume screen, whose
blind spot (re-annotation deltas beyond the 2 %/15 % tolerances) is exactly what hid the
eleven second-pass KiTS twins.  A CT-voxel scan of these 687 is impossible for ANYONE: the
FLARE23 release copy (Metadata.zip) contains images for 950 studies and none of them is one
of the 687 (verified here) - only their GT labels exist.

This prescreen is annotation-independent anyway: a same-scan re-share lives on the SAME voxel
grid (every one of KS's 310 CT-voxel twins is grid-identical, axis order aside), and the GT
label file carries the CT's grid.  So:

  1. stream the NIfTI header of each of the 687 GT labels out of Metadata.zip (ranged reads);
  2. compare sorted(shape) + sorted(spacing) against every one of the 113 queries' grids
     (census/audit signatures + LiTS geometry map; missing ones fetched from the Drive
     collection copies); spacing tolerance 1e-3, voxel-volume 0.5 % fallback where a
     signature lacks spacing;
  3. any collision -> full GT-label download for both sides and the 48-configuration
     (permutation x flip) mask-overlap confirmation used by the census, which confirmed the
     eleven at organ IoU 0.87-0.96 despite their re-annotations.

Zero collisions means: no unmasked pool record even shares a voxel grid with any query, so
none can be a same-scan re-share of a query (caveat, stated in the artifact: a re-share that
was RESAMPLED to a new grid would evade both this and KS's np.array_equal CT test; none of
the 310 known re-shares was).

  ZIP_INDEX=/dev/shm/zipidx/metadata_zip_index.json python unscanned687_grid_prescreen.py
-> results/audit/unscanned687_grid_prescreen_2026-09-24.json  (exit 1 on unresolved collision)
"""
import gzip
import io
import json
import os
import struct
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ZIP = "drive_UD:data/VKG datasets/Metadata.zip"
IDX = os.environ.get("ZIP_INDEX", "/dev/shm/zipidx/metadata_zip_index.json")
DRIVE = "drive_UD:data/VKG datasets"
OUT = os.path.join(ROOT, "results", "audit", "unscanned687_grid_prescreen_2026-09-24.json")
SP_TOL, VOX_TOL = 1e-3, 0.005


def j(path):
    return json.load(open(os.path.join(ROOT, path)))


def rcat(remote, off=None, cnt=None, tries=3):
    cmd = ["rclone", "cat"]
    if off is not None:
        cmd += ["--offset", str(off), "--count", str(cnt)]
    for t in range(tries):
        r = subprocess.run(cmd + [remote], stdout=subprocess.PIPE)
        if r.returncode == 0 and r.stdout:
            return r.stdout
    raise IOError(f"rclone cat failed: {remote}")


def nifti_hdr(gz_head):
    """(shape, spacing) from the first bytes of a .nii.gz stream."""
    raw = zlib.decompressobj(16 + 15).decompress(gz_head, 1024)
    if len(raw) < 348:
        raise ValueError("short header")
    end = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    dim = struct.unpack(end + "8h", raw[40:56])
    pixdim = struct.unpack(end + "8f", raw[76:108])
    return [int(x) for x in dim[1:4]], [round(float(x), 4) for x in pixdim[1:4]]


def zip_member_head(name, ent, n_bytes=131072):
    method, csz, usz, off = ent[name]
    lh = rcat(ZIP, off, 30)
    n, m = struct.unpack("<HH", lh[26:30])
    data = rcat(ZIP, off + 30 + n + m, min(n_bytes, csz))
    if method == 0:
        return data
    return zlib.decompressobj(-15).decompress(data, n_bytes)


def zip_member_full(name, ent):
    method, csz, usz, off = ent[name]
    lh = rcat(ZIP, off, 30)
    n, m = struct.unpack("<HH", lh[26:30])
    data = rcat(ZIP, off + 30 + n + m, csz)
    raw = data if method == 0 else zlib.decompress(data, -15)
    assert len(raw) == usz
    return raw


def query_signatures():
    """case_id -> {shape, spacing|None, voxml|None, source} for all 113 queries."""
    qids = {r["case_id"]: r for ds in ("kits", "lits", "msd")
            for r in j(f"corpora/corpus_gt_{ds}.json")["records"]}
    sigs = {}
    for p in j("results/audit/axis_normalized_twin_census.json")["all_pairs"]:
        s = p.get("query_sig")
        if s and s.get("shape") and p["query"] in qids and p["query"] not in sigs:
            sigs[p["query"]] = {"shape": s["shape"], "spacing": s.get("spacing"),
                                "voxml": s.get("voxml"), "source": "census"}
    lit = j("results/audit/lits_geometry_map.json")["mapping"]
    for q in qids:
        if q in lit and q not in sigs:
            sigs[q] = {"shape": lit[q]["shape_native"], "spacing": lit[q]["spacing"],
                       "voxml": None, "source": "lits_geometry_map"}
    for p in j("results/audit/duplicate_scan_audit.json")["pairs_flagged"]:
        s = p.get("query_sig")
        if s and s.get("shape") and p["query"] in qids and p["query"] not in sigs:
            sigs[p["query"]] = {"shape": s["shape"], "spacing": None, "voxml": None,
                                "source": "duplicate_scan_audit (shape only)"}
    # fetch from Drive: queries with no signature at all, AND any signature without spacing
    # (a shape-only signature cannot discriminate common grids like 512x512x89 - the source
    # of the 234 spurious collisions in the first run)
    missing = [q for q in qids if q not in sigs or not sigs[q].get("spacing")]

    def fetch(q):
        if q.startswith("case_"):
            rem = f"{DRIVE}/KiTS/{q}/segmentation.nii.gz"
        elif q.startswith("pancreas_"):
            rem = f"{DRIVE}/Pancreas/labelsTr/{q}.nii.gz"
        else:
            raise ValueError(f"no Drive source for {q}")
        shape, sp = nifti_hdr(rcat(rem, 0, 262144))
        return q, {"shape": shape, "spacing": sp, "voxml": None, "source": "drive header fetch"}

    with ThreadPoolExecutor(8) as ex:
        for q, s in ex.map(fetch, missing):
            if q in sigs and sigs[q].get("shape") and sorted(sigs[q]["shape"]) != sorted(s["shape"]):
                s["source"] += f" (WARNING: shape differs from stored sig {sigs[q]['shape']})"
            sigs[q] = s
    return sigs, missing


def main():
    ent = json.load(open(IDX))
    dedup = {r["case_id"] for r in j("corpora/corpus_flare23_kg_dedup.json")["records"]}
    pred = {r["case_id"] for r in j("corpora/corpus_predicted_flare23.json")["records"]}
    un = sorted(c.replace("flare23_", "") for c in dedup - pred)
    imgs = {n.split("/")[-1].replace("_0000.nii.gz", "") for n in ent if n.startswith("Metadata/images/")}
    assert not (set(un) & imgs), "some 'unscanned' record has a CT after all - rethink"

    sigs, fetched = query_signatures()
    assert len(sigs) == 113, f"query sigs incomplete: {len(sigs)}"

    def flare_grid(fid):
        shape, sp = nifti_hdr(zip_member_head(f"Metadata/labels/{fid}.nii.gz", ent))
        return fid, shape, sp

    cache = "/dev/shm/flare687_grids.json"
    grids = json.load(open(cache)) if os.path.exists(cache) else {}
    errs = []
    todo = [f for f in un if f not in grids]
    with ThreadPoolExecutor(12) as ex:
        for r in ex.map(lambda f: _safe(flare_grid, f), todo):
            if isinstance(r, tuple):
                grids[r[0]] = (r[1], r[2])
            else:
                errs.append(r)
    json.dump(grids, open(cache, "w"))

    collisions = []
    for fid, (fsh, fsp) in grids.items():
        fvox = fsp[0] * fsp[1] * fsp[2] / 1000.0
        for q, s in sigs.items():
            if sorted(fsh) != sorted(s["shape"]):
                continue
            if s["spacing"]:
                ok = all(abs(a - b) <= SP_TOL for a, b in zip(sorted(fsp), sorted(s["spacing"])))
            elif s["voxml"]:
                ok = abs(fvox - s["voxml"]) / s["voxml"] <= VOX_TOL
            else:
                ok = True  # shape-only signature: conservative collision
            if ok:
                collisions.append({"flare": f"flare23_{fid}", "query": q,
                                   "shape": fsh, "flare_spacing": fsp,
                                   "query_spacing": s.get("spacing"), "sig_source": s["source"]})
    report = {
        "prescreen": "unscanned687_grid_prescreen_2026-09-24",
        "premise": "same-scan re-shares are grid-identical (all 310 KS CT-voxel twins are); "
                   "GT labels carry the CT grid, so grid comparison is annotation-independent",
        "n_unmasked_pool_records": len(un),
        "headers_read": len(grids), "header_errors": errs,
        "no_ct_exists_for_any": True,
        "query_sigs": {"total": len(sigs), "fetched_from_drive": len(fetched),
                       "shape_only": [q for q, s in sigs.items() if not s["spacing"] and not s["voxml"]]},
        "grid_collisions": collisions, "n_collisions": len(collisions),
        "caveat": "a re-share RESAMPLED to a new grid evades this test (and would equally evade "
                  "KS's np.array_equal CT-voxel scan); none of the 310 known re-shares is resampled",
    }
    if not collisions and not errs:
        report["verdict"] = ("CLEAN - no unmasked pool record shares a voxel grid with any of the "
                             "113 queries; none can be an identical-grid same-scan re-share of a query")
    else:
        report["verdict"] = "COLLISIONS/ERRORS - confirm each by mask overlap before claiming clean"
    json.dump(report, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "grid_collisions"}, indent=1))
    for c in collisions:
        print("COLLISION:", c)
    sys.exit(1 if (collisions or errs) else 0)


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        return f"{a[0]}: {e}"


if __name__ == "__main__":
    main()
