#!/usr/bin/env python3
"""Host-assignment agreement check: app rule vs pipeline Eq. 3, on the frozen predicted
FLARE23 masks — coauthor review item, prepared 2026-09-15. RUNS WHERE THE MASKS LIVE (cluster or
a machine with masks_predicted_flare23.zip extracted); CPU only.

The two rules are NOT identical in form, so agreement is an empirical question:
  * Pipeline Eq. 3 (the evaluated extractor, app_handoff/extract_phenotypes.py): each tumor
    connected component is attributed to the ONE organ whose 3-VOXEL binary dilation overlaps
    it most (argmax, competitive across organs; zero-overlap components stay unattributed).
    On anisotropic grids the reach is anisotropic: 3 voxels of 2 mm slices = 6 mm in z.
  * App rule (segx_measure.host_attach_tolerance_mm / host_split): a component belongs to the
    host organ when its minimum surface distance is <= max(2.0 mm, 1.05 * max(spacing)) —
    metric, isotropic, per-host threshold (2.1 mm on 2 mm-slice scans).
  Disagreements are expected exactly for components sitting 2.1–6 mm from an organ in z, and
  for components near TWO organs (argmax vs per-host threshold can differ).

  When two organs sit at exactly the same minimum distance (within tolerance), the app-rule
  host is broken alphabetically (deterministic; fixed 2026-09-16 — the earlier set-iteration
  tie-break made the result PYTHONHASHSEED-dependent by a few components per run;
  n_app_distance_ties in the output counts affected components).

Usage:  python check_host_rule_vs_eq3.py --masks <dir with FLARE23_*.nii.gz predicted masks>
Output: host_rule_agreement.json — per-component agreement rate, disagreement census
        (distance bands, organ pairs), and per-case has_tumor flips per organ.
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np
from scipy import ndimage

POOLED = {"liver": [1], "kidney": [2, 13], "spleen": [3], "pancreas": [4]}
TUMOR = 14
HOST_ATTACH_FLOOR_MM, HOST_ATTACH_VOXEL_FRAC = 2.0, 1.05


def app_tolerance(spacing):
    return max(HOST_ATTACH_FLOOR_MM, HOST_ATTACH_VOXEL_FRAC * float(max(spacing)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", default="host_rule_agreement.json")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.masks, "*.nii.gz")))
    agree = disagree = ties = 0
    census = {}
    flips = []
    for i, f in enumerate(files):
        nii = nib.load(f)
        seg = np.asarray(nii.dataobj).astype(np.uint8)
        sp = [float(z) for z in nii.header.get_zooms()[:3]]
        tol = app_tolerance(sp)
        organs = {o: np.isin(seg, labs) for o, labs in POOLED.items() if np.isin(seg, labs).any()}
        dil = {o: ndimage.binary_dilation(m, iterations=3) for o, m in organs.items()}
        edt = {o: ndimage.distance_transform_edt(~m, sampling=sp) for o, m in organs.items()}
        lbl, n = ndimage.label(seg == TUMOR)
        case_flips = {}
        for c in range(1, n + 1):
            comp = lbl == c
            ov = {o: int((comp & d).sum()) for o, d in dil.items()}
            eq3 = max(ov, key=ov.get) if ov and max(ov.values()) > 0 else None
            dist = {o: float(edt[o][comp].min()) for o in organs}
            app = {o for o, d in dist.items() if d <= tol}
            # tie-break deterministically (alphabetical) on exactly equal min distances: iterating the raw
            # set made the winner PYTHONHASHSEED-dependent, so agreement varied by a few components per run
            app_host = min(sorted(app), key=lambda o: dist[o]) if app else None
            if len(app) > 1 and sorted(dist[o] for o in app)[0] == sorted(dist[o] for o in app)[1]:
                ties += 1
            if eq3 == app_host:
                agree += 1
            else:
                disagree += 1
                band = "2.1-6mm" if app_host is None and eq3 and dist.get(eq3, 99) <= 6 else "other"
                key = f"{eq3 or 'none'}->{app_host or 'none'}:{band}"
                census[key] = census.get(key, 0) + 1
                for o in {eq3, app_host} - {None}:
                    case_flips[o] = case_flips.get(o, 0) + 1
        if case_flips:
            flips.append({"case": os.path.basename(f)[:-7], "organs": case_flips})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(files)}] agree {agree} / disagree {disagree}", flush=True)
    total = agree + disagree
    out = {"n_cases": len(files), "n_components": total,
           "component_agreement": round(agree / total, 4) if total else None,
           "n_app_distance_ties": ties,
           "disagreement_census": census, "cases_with_hosttumor_flips": flips,
           "rules": __doc__.split("Usage:")[0].strip()}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"agreement {out['component_agreement']} over {total} components -> {a.out}")


if __name__ == "__main__":
    main()
