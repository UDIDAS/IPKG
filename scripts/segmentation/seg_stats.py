#!/usr/bin/env python3
"""Bootstrap 95% CIs for every 3-D segmentation cell (DSC, NSD@2mm, HD95) — organ & tumor, per dataset.
Uses the per-patient `cases` arrays. -> results/seg_stats.json"""
import json, glob
import numpy as np
rng = np.random.default_rng(7)
RES = "/home/user/SWOG/results"
NAME = {"flare_task2":"FLARE-Task2","flare23":"FLARE23","kits":"KiTS23","msd":"MSD","lits":"LiTS"}

def ci(vals, B=5000):
    a = np.array([v for v in vals if isinstance(v,(int,float))], float)
    if len(a) < 2: return None
    m = float(a.mean()); boots = [a[rng.integers(0,len(a),len(a))].mean() for _ in range(B)]
    return {"mean": round(m,3), "ci95":[round(float(np.percentile(boots,2.5)),3), round(float(np.percentile(boots,97.5)),3)], "n": len(a)}

out = {"organ": {}, "tumor": {}}
# organ
for f in sorted(glob.glob(f"{RES}/ausam_3d_*.json")):
    d = json.load(open(f)); ds = d.get("dataset")
    if not ds or "cases" not in d: continue
    organs = {}
    for org in d["mean_3d_dice"]:
        cs = [c["organs"][org] for c in d["cases"] if org in c.get("organs",{})]
        dsc = [o.get("dice3d") for o in cs]
        nsd = [o.get("nsd3d",{}).get("2.0mm") for o in cs]
        hd  = [o.get("hd95_mm") for o in cs]
        organs[org] = {"DSC": ci(dsc), "NSD2mm": ci(nsd), "HD95": ci(hd)}
    out["organ"][ds] = organs
# tumor
for f in sorted(glob.glob(f"{RES}/tumor_3d_*.json")):
    d = json.load(open(f)); ds = d.get("dataset")
    if "cases" not in d: continue
    dsc = [c["dice3d"] for c in d["cases"]]
    nsd = [c["nsd3d"].get("2.0mm") for c in d["cases"] if isinstance(c.get("nsd3d"),dict)]
    hd  = [c.get("hd95_mm") for c in d["cases"]]
    out["tumor"][ds] = {"DSC": ci(dsc), "NSD2mm": ci(nsd), "HD95": ci(hd)}

json.dump(out, open(f"{RES}/seg_stats.json","w"), indent=2)
print("ORGAN — DSC (95% CI):")
for ds,orgs in out["organ"].items():
    for org,m in orgs.items():
        c=m["DSC"];
        if c: print(f"  {NAME.get(ds,ds):12s} {org:9s} DSC {c['mean']} {c['ci95']} (n={c['n']})")
print("\nTUMOR — DSC (95% CI):")
for ds,m in out["tumor"].items():
    c=m["DSC"]
    if c: print(f"  {NAME.get(ds,ds):12s} DSC {c['mean']} {c['ci95']} (n={c['n']})")
