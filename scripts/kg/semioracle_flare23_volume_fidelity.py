#!/usr/bin/env python3
"""Draft Table 3, semi-oracle FLARE23 organ-node fidelity for ALL five KG organs incl. spleen (never exported:
the v6 semi-oracle FLARE23 run covered liver/kidney/pancreas on 10 cases).  Uses the per-case JSONs written
by build_predicted_corpus_flare23.py (GT-box prompted AUSAM organ model, 576 FLARE23 cases): predicted vs
reference organ volume r (Pearson), MAPE, and mean Dice.  Output: results/kg/semioracle_flare23_volume_fidelity.json
"""
import glob
import json
import os

import numpy as np

VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "kg", "semioracle_flare23_volume_fidelity.json")
GROUPS = {"liver": ["liver"], "kidney (L+R pooled)": ["right kidney", "left kidney"], "right_kidney": ["right kidney"],
          "left_kidney": ["left kidney"], "spleen": ["spleen"], "pancreas": ["pancreas"]}


def main():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{VKG_DATA}/flare23_predicted/*.json"))]
    out = {"n_cases": len(recs), "protocol": "semi-oracle (GT-box prompted) sam3_organ_generic_ausam_flare23; volumes in voxels x spacing", "organs": {}}
    for g, names in GROUPS.items():
        pv, gv, dsc = [], [], []
        for r in recs:
            sp = float(np.prod(r["spacing"])) / 1000.0
            if all(n in r["organs"] for n in names):
                pv.append(sum(r["organs"][n]["vox"] for n in names) * sp); gv.append(sum(r["organs"][n]["gt_vox"] for n in names) * sp)
                dsc += [r["organs"][n]["dice"] for n in names if r["organs"][n]["dice"] is not None]
        pv, gv = np.array(pv), np.array(gv); ok = gv > 0
        out["organs"][g] = {"n": int(ok.sum()), "volume_r": round(float(np.corrcoef(pv[ok], gv[ok])[0, 1]), 3),
                            "volume_MAPE_pct": round(float(np.mean(np.abs(pv[ok] - gv[ok]) / gv[ok])) * 100, 1),
                            "mean_dice": round(float(np.mean(dsc)), 3), "mean_gt_volume_cm3": round(float(gv[ok].mean()), 1)}
        print(f"  {g:22s} n={int(ok.sum()):3d}  r={out['organs'][g]['volume_r']:.3f}  MAPE={out['organs'][g]['volume_MAPE_pct']:5.1f}%  Dice={out['organs'][g]['mean_dice']:.3f}")
    json.dump(out, open(OUT, "w"), indent=1); print(f"-> {OUT}")


if __name__ == "__main__":
    main()
