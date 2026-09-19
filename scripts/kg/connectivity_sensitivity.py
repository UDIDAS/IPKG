#!/usr/bin/env python3
"""D4 (v12): the application counts tumor components at one connectivity while the pipeline's
`ndimage.label` default is 6-connectivity (faces only) — the draft cites 1,037 vs 908 components
on LiTS and asks: align the application to 26 or state the difference in §3.11.

This quantifies the decision on every mask set we hold: total tumor components at 6- vs
26-connectivity, per-case multiplicity flips (solitary <-> multifocal, the ONLY pipeline
statement that depends on the choice), and the Table 11 multiplicity-precision row recomputed
with BOTH sides at 26 (vs the frozen both-at-6).

Mask sets: GT — LiTS (131 native Task03 labels), KiTS (36), MSD (56), FLARE23 (cached GT labels);
predicted — FLARE23 (576 frozen semi-oracle), 113-cohort (regenerated).
-> results/kg/connectivity_sensitivity.json
"""
import glob
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
S26 = np.ones((3, 3, 3), np.uint8)
SETS = {
    "lits_gt_131": ("/path/to/staging/msd_task03/Task03_Liver/labelsTr/liver_*.nii.gz", 2),
    "kits_gt_36": ("/path/to/staging/acm_data/kits_vol/*/segmentation.nii.gz", 2),
    "msd_gt_56": ("/path/to/staging/acm_data/msd_pancreas_labels/pancreas_*.nii.gz", 2),
    "flare23_gt_cached": ("/path/to/staging/acm_data/flare23_labels_audit/*.nii.gz", 14),
    "flare23_predicted_576": ("/path/to/staging/acm_data/flare23_masks/masks_predicted_flare23/*.nii.gz", 14),
    "cohort113_predicted_regen": ("/path/to/staging/acm_data/masks_predicted_113_regen/*/*.nii.gz", 2),
}


def one(job):
    name, fp, tlab = job
    seg = np.asarray(nib.load(fp).dataobj)
    t = seg == tlab
    if not t.any():
        return name, fp, 0, 0
    return name, fp, int(ndimage.label(t)[1]), int(ndimage.label(t, structure=S26)[1])


def main():
    # restrict msd glob to the 56 evaluated cases
    msd56 = {r["case_id"] for r in json.load(open(f"{ROOT}/corpora/corpus_gt_msd.json"))["records"]}
    jobs = []
    for name, (pat, tlab) in SETS.items():
        for fp in sorted(glob.glob(pat)):
            base = os.path.basename(fp)
            if base.startswith("._"):
                continue
            if name == "msd_gt_56" and base[:-len(".nii.gz")] not in msd56:
                continue
            jobs.append((name, fp, tlab))
    print(f"{len(jobs)} masks", flush=True)
    with Pool(14) as p:
        res = list(p.imap_unordered(one, jobs, chunksize=4))

    out = {}
    flips6to26 = {}
    for name in SETS:
        rows = [(fp, c6, c26) for n, fp, c6, c26 in res if n == name]
        tumor = [(fp, c6, c26) for fp, c6, c26 in rows if c6 > 0]
        flips = [(os.path.basename(fp)[:-7], c6, c26) for fp, c6, c26 in tumor
                 if (c6 >= 2) != (c26 >= 2)]
        flips6to26[name] = flips
        out[name] = {"n_masks": len(rows), "n_with_tumor": len(tumor),
                     "components_6conn": sum(c for _, c, _ in tumor),
                     "components_26conn": sum(c for _, _, c in tumor),
                     "pct_more_at_6conn": round(100 * (sum(c for _, c, _ in tumor) / max(sum(c for _, _, c in tumor), 1) - 1), 1),
                     "multiplicity_flips_6_vs_26": len(flips),
                     "flip_cases": [f"{c} ({a}->{b})" for c, a, b in flips[:25]]}
        print(name, {k: v for k, v in out[name].items() if k != "flip_cases"}, flush=True)

    # Table 11 multiplicity row under 26-conn both sides (113 cohort: regen predicted vs GT)
    def mult_map(name, id_fn, conn_idx):
        return {id_fn(fp): ("multifocal" if (c[conn_idx] >= 2) else "solitary")
                for fp, *c in [(fp, c6, c26) for n, fp, c6, c26 in res if n == name and (c6 > 0 or c26 > 0)]}

    gmap = json.load(open(f"{ROOT}/results/audit/lits_geometry_map.json"))["mapping"]
    liver2vol = {m["liver"]: v for v, m in gmap.items()}
    def gt_id(fp):
        b = os.path.basename(fp)[:-len(".nii.gz")]
        return liver2vol.get(b, b) if b.startswith("liver_") else (fp.split("/")[-2] if "kits_vol" in fp else b)
    def pr_id(fp):
        return os.path.basename(fp)[:-len(".nii.gz")]

    t11 = {}
    for conn_idx, lab in ((0, "6conn_both"), (1, "26conn_both")):
        gt_m = {}
        for name in ("lits_gt_131", "kits_gt_36", "msd_gt_56"):
            gt_m.update(mult_map(name, gt_id, conn_idx))
        pr_m = mult_map("cohort113_predicted_regen", pr_id, conn_idx)
        pairs = [(pr_m[c], gt_m[c]) for c in pr_m if c in gt_m]
        t11[lab] = {"n": len(pairs), "multiplicity_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3)}

    out["table11_multiplicity_sensitivity_113"] = {
        **t11, "note": ("Predicted (regenerated masks) vs GT multiplicity agreement with both sides at the same "
                        "connectivity; the frozen Table 11 row (0.927) is both-at-6 on the frozen masks — this "
                        "measures the CHOICE's effect, not a replacement value.")}
    out["note"] = __doc__.strip()
    fp = f"{ROOT}/results/kg/connectivity_sensitivity.json"
    json.dump(out, open(fp, "w"), indent=1)
    print("table11 sensitivity:", t11)
    print(f"-> {fp}", flush=True)


if __name__ == "__main__":
    main()
