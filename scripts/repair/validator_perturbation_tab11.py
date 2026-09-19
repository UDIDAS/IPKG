#!/usr/bin/env python3
"""Draft Table 11 — validator response to controlled perturbation of observed organ nodes (§4.7).

Validator = the KG plausibility check applied by kg_guided_segment.repair(): an organ node is FLAGGED when
its volume falls outside the cohort band [p1, p99] of kg_atlas.json.  Because repair() also enforces
"organ = one connected structure", the component-count check (n_components > 1) is reported as a second
channel, and the union as the combined validator.

Nodes (reference-verified, so unperturbed = the false-positive baseline):
  * FLARE23 closed-loop test patients (n=40): liver, right_kidney, spleen, pancreas, left_kidney from the
    reference masks — 200 nodes (physical cm3 from the NIfTI spacing); these patients are excluded from the
    atlas cohort.
  * KiTS23 test cohort (n=36): pooled-kidney GT volume from corpora/corpus_gt_kits.json (out-of-cohort check
    of the band; single-organ). LiTS/MSD GT corpora carry voxel-count-derived pseudo-volumes and are excluded.

Family 1 — volume scaling: volume x f, f in {0.5,0.7,0.85,1.0,1.15,1.3,1.5,2.0}; flagged % per f, per organ.
Family 2 — mask-level (FLARE23 masks only): erosion 3 mm, dilation 3 mm (spherical SE at 1 mm isotropic),
  +one disconnected spherical component of 10 % organ volume at a random location inside the body and
  outside every organ (seed 0); volume recomputed and the validator re-run.

Output: results/repair/validator_perturbation_tab11.json
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
CASES = os.path.join(VKG_DATA, "flare_full_cases")
ATLAS = os.path.join(VKG_DATA, "kg_atlas.json")
OUT = os.environ.get("VKG_OUT", os.path.join(ROOT, "results", "repair", "validator_perturbation_tab11.json"))
NAME = {1: "liver", 2: "right_kidney", 3: "spleen", 4: "pancreas", 13: "left_kidney"}
FACTORS = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0]


def _ball(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def morph_iso(m, spacing, op, radius_mm=3.0):
    zoom = np.asarray(spacing, float)
    iso = ndimage.zoom(m.astype(np.uint8), zoom, order=0) > 0
    f = ndimage.binary_erosion if op == "erode" else ndimage.binary_dilation
    iso = f(iso, structure=_ball(int(round(radius_mm))))
    back = ndimage.zoom(iso.astype(np.uint8), np.array(m.shape) / np.array(iso.shape), order=0) > 0
    out = np.zeros(m.shape, bool); sl = tuple(slice(0, min(a, b)) for a, b in zip(m.shape, back.shape))
    out[sl] = back[sl]
    return out


def add_spurious(m, body_free, frac, rng):
    """Add one sphere (in voxel units, isotropic in index space) of frac*volume(m) inside body_free."""
    vox = int(round(frac * m.sum()))
    r = max(1, int(round((3 * vox / (4 * np.pi)) ** (1 / 3))))
    cand = np.argwhere(body_free)
    ball = _ball(r)
    for _ in range(500):
        c = cand[rng.integers(len(cand))]
        if any(ci - r < 0 or ci + r + 1 > n for ci, n in zip(c, m.shape)):
            continue                                            # keep the sphere fully inside the volume
        sl = tuple(slice(ci - r, ci + r + 1) for ci in c)
        if body_free[sl][ball].all():
            out = m.copy(); sub = out[sl]; sub[ball] = True; out[sl] = sub
            return out, vox
    return None, vox


def ncomp_sig(m, frac=0.01, minvox=50):
    """Number of connected components with size >= max(minvox, frac * organ volume) — ignores label specks."""
    lbl, n = ndimage.label(m)
    if n <= 1:
        return n
    sizes = ndimage.sum(m, lbl, range(1, n + 1))
    return int((sizes >= max(minvox, frac * m.sum())).sum())


def process_case(cid):
    nii = nib.load(f"{CASES}/{cid}_label.nii.gz"); gt = np.asarray(nii.dataobj).astype(np.uint8)
    sp = [float(z) for z in nii.header.get_zooms()[:3]]; cm3 = float(np.prod(sp)) / 1000.0
    ct = np.asarray(nib.load(f"{CASES}/{cid}_ct.nii.gz").dataobj)
    body = ndimage.binary_erosion(ct > -500, iterations=5)
    body_free = body & (gt == 0)
    rng = np.random.default_rng(0)
    rows = []
    for lab, name in NAME.items():
        m = gt == lab
        if not m.any():
            continue
        row = {"cid": cid, "organ": name, "vol_cm3": m.sum() * cm3, "ncomp": int(ndimage.label(m)[1]), "ncomp_sig": ncomp_sig(m)}
        e = morph_iso(m, sp, "erode"); d = morph_iso(m, sp, "dilate")
        row["erode_cm3"] = e.sum() * cm3; row["erode_ncomp"] = int(ndimage.label(e)[1]); row["erode_ncomp_sig"] = ncomp_sig(e)
        row["dilate_cm3"] = d.sum() * cm3; row["dilate_ncomp"] = int(ndimage.label(d)[1]); row["dilate_ncomp_sig"] = ncomp_sig(d)
        s, vox = add_spurious(m, body_free, 0.10, rng)
        if s is not None:
            row["spur_cm3"] = s.sum() * cm3; row["spur_ncomp"] = int(ndimage.label(s)[1]); row["spur_ncomp_sig"] = ncomp_sig(s)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", default=None); ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    atlas = json.load(open(ATLAS))["organs"]
    band = {o: (atlas[o]["volume_cm3"]["p1"], atlas[o]["volume_cm3"]["p99"]) for o in atlas}
    flag = lambda o, v: not (band[o][0] <= v <= band[o][1])

    cids = [l.strip() for l in open(a.cases)] if a.cases else \
        sorted(os.path.basename(f)[:-len("_ct.nii.gz")] for f in glob.glob(f"{CASES}/*_ct.nii.gz"))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for rs in ex.map(process_case, cids) for r in rs]
    kits = json.load(open(os.path.join(ROOT, "corpora", "corpus_gt_kits.json")))["records"]
    rows += [{"cid": r["case_id"], "organ": "kidney", "vol_cm3": r["organs"]["kidney"]["organ_volume_cm3"], "ncomp": None} for r in kits]

    organs = sorted(set(r["organ"] for r in rows))
    scaling = {}
    for o in ["ALL"] + organs:
        sub = [r for r in rows if o == "ALL" or r["organ"] == o]
        scaling[o] = {"n_nodes": len(sub), **{f"f={f:g}": round(100 * np.mean([flag(r["organ"], r["vol_cm3"] * f) for r in sub]), 1) for f in FACTORS}}
    # smallest f>1 / largest f<1 with >=90 % detection
    thr = {}
    for o, d in scaling.items():
        up = [f for f in FACTORS if f > 1 and d[f"f={f:g}"] >= 90]; dn = [f for f in FACTORS if f < 1 and d[f"f={f:g}"] >= 90]
        thr[o] = {"smallest_f_above_1_with_90pct": min(up) if up else None, "largest_f_below_1_with_90pct": max(dn) if dn else None}

    masklevel = {}
    fl = [r for r in rows if r["ncomp"] is not None]
    for o in ["ALL"] + sorted(set(r["organ"] for r in fl)):
        sub = [r for r in fl if o == "ALL" or r["organ"] == o]
        d = {"n_nodes": len(sub)}
        for pert, key in (("unperturbed", "vol_"), ("erosion_3mm", "erode_"), ("dilation_3mm", "dilate_"), ("spurious_component_10pct", "spur_")):
            nk = "ncomp" if key == "vol_" else f"{key}ncomp"
            ss = [r for r in sub if f"{key}cm3" in r]
            vb = [flag(r["organ"], r[f"{key}cm3"]) for r in ss]
            cc_raw = [r[nk] > 1 for r in ss]
            cc = [r[nk + "_sig"] > 1 for r in ss]
            d[pert] = {"n": len(ss), "volume_band_flagged_pct": round(100 * np.mean(vb), 1),
                       "component_any_size_flagged_pct": round(100 * np.mean(cc_raw), 1),
                       "component_flagged_pct": round(100 * np.mean(cc), 1),
                       "either_flagged_pct": round(100 * np.mean([x or y for x, y in zip(vb, cc)]), 1),
                       "mean_volume_change_pct": round(100 * float(np.mean([(r[f"{key}cm3"] - r["vol_cm3"]) / r["vol_cm3"] for r in ss])), 1)}
        masklevel[o] = d

    out = {"atlas": {"path": ATLAS, "band": "p1..p99", "cohort": json.load(open(ATLAS)).get("source"), "n_cohort": json.load(open(ATLAS)).get("n_cases"),
                     "bands_cm3": band},
           "nodes": {"flare23_reference_organ_nodes": len(fl), "kits_gt_kidney_nodes": len(kits), "total_scaling_nodes": len(rows)},
           "volume_scaling_flagged_pct": scaling, "scaling_thresholds": thr, "mask_level": masklevel,
           "note": "unperturbed (f=1.0) row = false-positive rate on reference-verified nodes. component channel = >=2 components each "
                   ">= max(50 vox, 1 % of organ volume) (label specks ignored); component_any_size = raw n_components>1, which the "
                   "FLARE23 reference labels themselves trip 35 % of the time"}
    json.dump(out, open(OUT, "w"), indent=1)
    print("volume scaling — flagged % (rows: organ; cols: f)"); print("  organ         n  " + "  ".join(f"{f:>5g}" for f in FACTORS))
    for o, d in scaling.items():
        print(f"  {o:13s} {d['n_nodes']:3d}  " + "  ".join(f"{d[f'f={f:g}']:5.1f}" for f in FACTORS))
    print("mask-level (FLARE23) — either-channel flagged % [volume-band / component(>=1%)]")
    for o, d in masklevel.items():
        print(f"  {o:13s} n={d['n_nodes']:3d}  " + "  ".join(f"{p}: {d[p]['either_flagged_pct']:5.1f} [{d[p]['volume_band_flagged_pct']:.1f}/{d[p]['component_flagged_pct']:.1f}]"
                                                for p in ("unperturbed", "erosion_3mm", "dilation_3mm", "spurious_component_10pct")))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
