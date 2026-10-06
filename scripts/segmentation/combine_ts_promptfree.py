#!/usr/bin/env python3
"""Round-2 combined autonomous pass (IPKG, the study lead 10-01 point 3): TotalSegmentator organs + prompt-free SAM 3 tumor
(sam3_tumor_generic on every collection; sam3_tumor_ausam_kits on its own KiTS test list).

Nothing here reads the reference except to SCORE. Per case:
  organ   = TotalSegmentator v2 --fast (3 mm network) host-organ label(s)
            (LiTS liver 5, MSD pancreas 7, KiTS kidneys 2|3; FLARE23 any of spleen/kidneys/liver/stomach/pancreas/colon).
  tumor   = eval_tumor_promptfree.py mask (text 'tumor', every slice, no box, no slice gate)
  gated   = tumor connected components that touch the TS host organ dilated by GATE_MM (our own organ mask, never
            the reference). Reported beside the raw tumor, not instead of it.
Scores: tumor Dice / NSD@1,2 mm / HD95 / components (raw, gated); organ Dice under BOTH label conventions
(organ-excludes-tumor = reference organ label only; organ-with-mass = reference organ | tumor), so the convention is
declared rather than chosen after the fact (the co-author 10-01 point 2).
MSD sub-site: scripts/kg/pancreas_subsite.py on (TS pancreas | gated tumor, gated tumor), compared with the same
rule on the reference (results/kg/pancreas_subsite_msd_gt.json).
Exposure: results/audit/promptfree_twin_exposure_2026-10-01.json flags test cases whose FLARE23 twin was in the
tumor model's TRAIN/VAL; summaries are given for all and for the twin-clean subset.

  python combine_ts_promptfree.py -> results/segmentation/promptfree_round2_2026-10-01.json
"""
import glob
import json
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "kg"))
from pancreas_subsite import pancreas_subsite                            # noqa: E402

DS = os.environ.get("VKG_DATASETS", "/path/to/cluster/projects/datasets")
TS = os.environ.get("VKG_TS", f"{DS}/derived/totalseg_fast")
FLARE = os.environ.get("VKG_FLARE23", "/path/to/cluster/projects/VKG_data/flare23_pool")
RUNBASE = os.environ.get("OUT_DIR", "/path/to/cluster/work/vkg_runs/promptfree")
# model -> (split whose TEST list it is scored on, datasets, twin-exposure applies)
MODELS = {"sam3_tumor_generic": ("global", ("lits", "msd", "kits", "flare23"), True),
          "sam3_tumor_ausam_kits": ("filtered", ("kits",), False)}
OUT = os.path.join(ROOT, "results", "segmentation", "promptfree_round2_2026-10-01.json")
EXPO = os.path.join(ROOT, "results", "audit", "promptfree_twin_exposure_2026-10-01.json")
SUBGT = os.path.join(ROOT, "results", "kg", "pancreas_subsite_msd_gt.json")
GATE_MM = 5.0
# dataset -> (reference organ labels, reference tumor labels, TS host labels, TS dir, file stems)
CFG = {"lits": ([1], [2], [5], "msd_liver", lambda c: f"liver_{c.split('-')[1]}"),
       "msd": ([1], [2], [7], "msd_pancreas", lambda c: c),
       "kits": ([1], [2], [2, 3], "kits23", lambda c: c),
       "flare23": (None, [14], [1, 2, 3, 5, 6, 7, 20], "flare23", lambda c: c)}   # FLARE lesions: any TS abdominal organ
TS_FLARE = os.environ.get("VKG_TS_FLARE", "/path/to/cluster/work/vkg_data/totalseg_fast")   # same TS call, run 10-01


def ref_path(ds, c):
    s = CFG[ds][4](c)
    return {"lits": f"{DS}/MSD/Task03_Liver/labelsTr/{s}.nii.gz", "msd": f"{DS}/MSD/Task07_Pancreas/labelsTr/{s}.nii.gz",
            "kits": f"{DS}/KiTS23/{s}/segmentation.nii.gz", "flare23": f"{FLARE}/labels/{s}.nii.gz"}[ds]


def surf(p, g, sp):
    if not p.any() or not g.any():
        return {"nsd_1mm": 0.0 if (p.any() or g.any()) else 1.0, "nsd_2mm": 0.0 if (p.any() or g.any()) else 1.0, "hd95_mm": None}
    def sdt(m):                                   # crop to the joint bounding box for speed (exact: EDT is local)
        s = m & ~ndimage.binary_erosion(m)
        return s
    bb = ndimage.find_objects((p | g).astype(np.uint8))[0]
    bb = tuple(slice(max(0, b.start - 2), b.stop + 2) for b in bb)
    P, G = p[bb], g[bb]
    ps, gs = sdt(P), sdt(G)
    pdt, gdt = ndimage.distance_transform_edt(~ps, sampling=sp), ndimage.distance_transform_edt(~gs, sampling=sp)
    d1, d2 = pdt[gs], gdt[ps]
    tot = len(d1) + len(d2)
    return {"nsd_1mm": round(float(((d1 <= 1).sum() + (d2 <= 1).sum()) / tot), 4),
            "nsd_2mm": round(float(((d1 <= 2).sum() + (d2 <= 2).sum()) / tot), 4),
            "hd95_mm": round(float(max(np.percentile(d1, 95), np.percentile(d2, 95))), 2)}


def score(p, g, sp):
    s = int(p.sum()) + int(g.sum())
    v = float(np.prod(sp)) / 1000.0
    r = {"dice": round(2 * int((p & g).sum()) / s, 4) if s else None, "pred_cm3": round(p.sum() * v, 2),
         "ref_cm3": round(g.sum() * v, 2), "outside_ref_cm3": round((p & ~g).sum() * v, 2), "components": int(ndimage.label(p)[1])}
    r.update(surf(p, g, sp))
    return r


def one(args):
    """Cached per case: recomputed only when the tumor mask or the TS mask is newer than the cached record."""
    model, ds, c, showcase = args
    cp = f"{RUNBASE}/combined_cache/{model}/{ds}/{c}{'_showcase' if showcase else ''}.json"
    pm = f"{RUNBASE}/{model}/{ds}/{c}.nii.gz"
    tsp = f"{TS_FLARE if ds == 'flare23' else TS}/{CFG[ds][3]}/{CFG[ds][4](c)}.nii.gz"
    if os.path.exists(cp) and os.path.exists(pm) and os.path.getmtime(cp) > max(os.path.getmtime(pm), os.path.getmtime(tsp) if os.path.exists(tsp) else 0) \
            and (os.path.exists(tsp) or "tumor_gated" not in json.load(open(cp))):
        r = json.load(open(cp))
        if os.path.exists(tsp) == ("tumor_gated" in r):
            return r
    r = _one(args)
    if "error" not in r:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        json.dump(r, open(cp, "w"))
    return r


def _one(args):
    model, ds, c, showcase = args
    olab, tlab, host, tsd, stem = CFG[ds]
    pm = f"{RUNBASE}/{model}/{ds}/{c}.nii.gz"
    if not os.path.exists(pm):
        return {"case": c, "dataset": ds, "model": model, "showcase": showcase, "error": "no prompt-free mask"}
    rn = nib.load(ref_path(ds, c))
    ref = np.asarray(rn.dataobj)
    sp = tuple(float(z) for z in rn.header.get_zooms()[:3])
    tum = np.asarray(nib.load(pm).dataobj) > 0
    gt_t = np.isin(ref, tlab)
    rec = {"case": c, "dataset": ds, "model": model, "showcase": showcase, "spacing_mm": list(sp), "tumor_raw": score(tum, gt_t, sp)}
    tsp = f"{TS_FLARE if ds == 'flare23' else TS}/{tsd}/{stem(c)}.nii.gz"
    if not os.path.exists(tsp):
        rec["note"] = "no TotalSegmentator mask for this case: raw tumor only"
        return rec
    tsn = nib.load(tsp)
    ts = np.asarray(tsn.dataobj)
    # same voxel grid required (shape + direction/spacing); FLARE23 label headers carry a zeroed origin, so the
    # translation is not compared (voxel-wise scoring is unaffected)
    assert ts.shape == ref.shape and np.allclose(tsn.affine[:3, :3], rn.affine[:3, :3], atol=1e-3), (c, ts.shape, ref.shape)
    hostm = np.isin(ts, host)
    near = np.zeros_like(hostm)
    if hostm.any() and tum.any():                 # EDT only inside the tumor bbox + gate margin (exact for <= GATE_MM)
        m = [int(np.ceil(GATE_MM / s)) + 1 for s in sp]
        bb = tuple(slice(max(0, b.start - k), b.stop + k) for b, k in zip(ndimage.find_objects(tum.astype(np.uint8))[0], m))
        near[bb] = ndimage.distance_transform_edt(~hostm[bb], sampling=sp) <= GATE_MM
    lab, n = ndimage.label(tum)
    keep = np.unique(lab[near & tum]); keep = keep[keep > 0]
    gated = np.isin(lab, keep)
    rec["tumor_gated"] = score(gated, gt_t, sp)
    rec["gate"] = {"rule": f"keep tumor components touching TS host organ dilated {GATE_MM} mm", "components_in": int(n), "components_kept": int(len(keep))}
    if showcase:                                  # app mask in the delivered label scheme, exclusive (tumor wins)
        out = np.zeros(ts.shape, np.uint8)
        lut = {"flare23": {5: 1, 2: 2, 1: 3, 7: 4, 3: 13}}.get(ds, {h: 1 for h in host})
        for a_, b_ in lut.items():
            out[ts == a_] = b_
        out[gated] = tlab[0]
        os.makedirs(f"{RUNBASE}/showcase_combined", exist_ok=True)
        nib.save(nib.Nifti1Image(out, tsn.affine), f"{RUNBASE}/showcase_combined/{c}.nii.gz")
        rec["mask_file"] = f"masks/showcase/{c}.nii.gz"
        rec["mask_labels"] = {str(v): k for k, v in {"organ (TotalSegmentator)": 1, "tumor (prompt-free SAM 3, organ-gated)": tlab[0]}.items()} \
            if ds != "flare23" else {"1": "liver", "2": "right kidney", "3": "spleen", "4": "pancreas", "13": "left kidney", "14": "tumor (prompt-free, gated)"}
    if olab is None:                              # FLARE23 multi-organ: organ scoring is the co-author's TotalSegmentator arm
        return rec
    gt_o = np.isin(ref, olab)
    rec["organ_ts"] = {"excludes_tumor": score(hostm & ~gated, gt_o, sp),       # TS organ minus our tumor vs ref organ
                       "with_mass": score(hostm | gated, gt_o | gt_t, sp)}    # whole organ (incl. mass) vs ref organ|tumor
    if ds == "msd":
        rec["subsite"] = pancreas_subsite(hostm | gated, gated, tsn.affine) if gated.any() else {"site": "unknown", "reason": "no gated tumor"}
    return rec


def boot_ci(v, n=2000, seed=0):
    """Percentile bootstrap 95% CI of the mean over patients."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return None
    m = np.random.default_rng(seed).choice(v, (n, len(v))).mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def summ(rows, key, sub):
    v = [r[key][sub] for r in rows if key in r and r[key].get(sub) is not None]
    return round(float(np.mean(v)), 4) if v else None


def main():
    man = json.load(open(os.path.join(ROOT, "results", "audit", "split_manifests_2026-10-01.json")))["tumor_pools"]
    expo = json.load(open(EXPO))["datasets"]
    key = {"lits": "lits", "msd": "pancreas", "kits": "kits", "flare23": "flare"}
    jobs = [(m, ds, c, False) for m, (split, dss, _) in MODELS.items() for ds in dss for c in man[split][key[ds]]["test"]]
    jobs += [("sam3_tumor_generic", ds, c, True) for ds, c in
             (("lits", "volume-76"), ("kits", "case_00067"), ("msd", "pancreas_125"), ("flare23", "FLARE23_0405"))]
    with Pool(int(os.environ.get("NPROC", 12))) as p:
        R = p.map(one, jobs)
    gts = {r["case"]: r for r in json.load(open(SUBGT))["records"]}
    meta = {}
    for f in glob.glob(f"{RUNBASE}/*/*/*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            if "error" not in d:
                meta.setdefault((d["model"], d["dataset"], d["case"]), d)
    out = {"note": __doc__.strip(), "models": {}, "showcase": {}}
    for m, (split, dss, twins) in MODELS.items():
        out["models"][m] = {"test_split": split, "datasets": {}}
        for ds in dss:
            ex = set(expo[ds]["exposed_cases"]) if twins else set()
            rows = [r for r in R if r["model"] == m and r["dataset"] == ds and not r["showcase"] and "error" not in r]
            for r in rows:
                r["exposed_via_twin"] = r["case"] in ex
                md = meta.get((m, ds, r["case"]), {}); r["model_sha256"] = md.get("model_sha256"); r["seconds_gpu"] = md.get("seconds")
                if ds == "msd" and "subsite" in r:
                    r["subsite_ref"] = gts.get(r["case"], {}).get("site")
            S = {}
            for name, rs in (("all", rows), ("twin_clean", [r for r in rows if not r["exposed_via_twin"]])):
                S[name] = {"n": len(rs)}
                for k in ("tumor_raw", "tumor_gated"):
                    S[name][k] = {x: summ(rs, k, x) for x in ("dice", "nsd_2mm", "hd95_mm", "components")}
                    S[name][k]["dice_ci95"] = boot_ci([r[k]["dice"] for r in rs if k in r and r[k].get("dice") is not None])
                if ds != "flare23" and rs:
                    S[name]["organ_excludes_tumor_dice"] = round(float(np.mean([r["organ_ts"]["excludes_tumor"]["dice"] for r in rs])), 4)
                    S[name]["organ_with_mass_dice"] = round(float(np.mean([r["organ_ts"]["with_mass"]["dice"] for r in rs])), 4)
            if ds == "msd":
                ok = [r for r in rows if r.get("subsite_ref") in ("head", "body", "tail")]
                S["subsite_vs_reference_rule"] = {"n_ref_called": len(ok), "agree": sum(r["subsite"]["site"] == r["subsite_ref"] for r in ok),
                                                  "pred_unknown": sum(r["subsite"]["site"] == "unknown" for r in ok)}
            n_exp = len(man[split][key[ds]]["test"])
            out["models"][m]["datasets"][ds] = {"n_test_list": n_exp, "n_scored": len(rows), "summary": S, "cases": rows}
            print(m, ds, f"{len(rows)}/{n_exp}", json.dumps(S))
    for r in R:
        if r["showcase"]:
            md = meta.get((r["model"], r["dataset"], r["case"]), {}); r["model_sha256"] = md.get("model_sha256")
            out["showcase"][r["case"]] = r
            print(r["case"], {k: r.get(k, {}).get("dice") for k in ("tumor_raw", "tumor_gated")}, r.get("subsite", {}).get("site"))
    json.dump(out, open(OUT, "w"), indent=1)
    print("->", OUT)

if __name__ == "__main__":
    main()
