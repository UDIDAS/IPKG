#!/usr/bin/env python3
"""Draft Tables 7 & 8, baseline rows (v)-(vii'): radiomics-vector and image-embedding retrieval with
per-query metrics (P@5/10, mAP, nDCG, spurious@10, Mismatch@10) and paired Δ/CI/Holm-p against the proposed
configuration (ii) — the per-query rerun the v7 round could not do without CT data.

Baselines are the ones of baselines_retrieval.py / baseline_image_embedding.py (same 11-dim first-order HU +
shape descriptor from GT regions, z-scored, cosine; same ResNet50-ImageNet mean-pooled 2048-dim slice
embedding), generalised to multi-organ candidates:
  * features are computed PER ORGAN (organ mask = that organ's labels, tumor = tumor voxels within a 3-voxel
    dilation of it) and for the UNION of labeled organs;
  * organ-agnostic (v)/(vii): cosine on the union features, all candidates;
  * organ-conditioned (vi)/(vii'): candidates must share an observed organ with the query; similarity = mean
    over shared organs of the per-organ cosine.
For single-organ cases (KiTS/LiTS/MSD) per-organ == union, so Tab. 7 numbers reduce to the original scripts.

Pools
  Tab. 7: 113 reference-evaluable queries x 112 candidates; relevance from the reference (GT) records.
  Tab. 8: 113 queries x (113 + FLARE23 candidates WITH a CT in the release copy — 576 of 1,312); relevance
          construction-defined (pool records: predicted for the 113, GT-KG for FLARE23), as in the v7 run.
          KG rows (i)-(iv) are re-run on the same sub-pool so the comparison is like-for-like.

Data ($VKG_DATA): cohort113/{kits/<case>/imaging|segmentation.nii.gz, lits/ct|seg/*.npy, msd/imagesTr|labelsTr},
flare23_pool/{images,labels}. Features are cached in $VKG_DATA/baseline_features.npz.
Output: results/retrieval/baselines_tab7_tab8.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import retrieval_core_local as core                                     # noqa: E402
from retrieval_core_local import load_corpus, relevant, _dcg             # noqa: E402
import retrieval_tables_789 as T                                         # noqa: E402  (per_query, paired, holm, agg)

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
VKG_DATA = os.environ.get("VKG_DATA", "/path/to/VKG_data")
C113 = os.path.join(VKG_DATA, "cohort113")
POOL = os.path.join(VKG_DATA, "flare23_pool")
FEAT = os.path.join(VKG_DATA, "baseline_features.npz")
OUT = os.path.join(os.environ.get("VKG_RES_DIR", os.path.join(ROOT, "results", "retrieval")), "baselines_tab7_tab8.json")
WIN = (-125, 225)
AXIAL = {"msd": 2, "lits": 0, "kits": 0, "flare23": 2}
# label scheme per dataset: organ key -> labels; tumor label
SCHEME = {"kits": ({"kidney": [1]}, 2), "lits": ({"liver": [1]}, 2), "msd": ({"pancreas": [1]}, 2),
          "flare23": ({"liver": [1], "kidney": [2, 13], "spleen": [3], "pancreas": [4]}, 14)}


# ------------------------------------------------------------------ loading
def load(ds, cid):
    if ds == "kits":
        ct = nib.load(f"{C113}/kits/{cid}/imaging.nii.gz").get_fdata()
        seg = np.asarray(nib.load(f"{C113}/kits/{cid}/segmentation.nii.gz").dataobj).astype(int)
    elif ds == "lits":
        vid = cid.replace("volume-", "")
        ct = np.load(f"{C113}/lits/ct/volume-{vid}.npy"); seg = np.load(f"{C113}/lits/seg/segmentation-{vid}.npy").astype(int)
    elif ds == "msd":
        ct = nib.load(f"{C113}/msd/imagesTr/{cid}.nii.gz").get_fdata()
        seg = np.asarray(nib.load(f"{C113}/msd/labelsTr/{cid}.nii.gz").dataobj).astype(int)
    else:
        c = cid.replace("flare23_", "")
        ct = nib.load(f"{POOL}/images/{c}_0000.nii.gz").get_fdata()
        seg = np.asarray(nib.load(f"{POOL}/labels/{c}.nii.gz").dataobj).astype(int)
    return ct, seg


# ------------------------------------------------------------------ radiomics
def feats(ct, O, Tm):
    """11-dim first-order HU + shape descriptor (verbatim baselines_retrieval.feats on given masks)."""
    def stats(mask):
        if mask.sum() == 0:
            return [0., 0., 0.]
        v = ct[mask]; return [float(v.mean()), float(v.std()), float(np.percentile(v, 90))]
    ovol = float(O.sum()); tvol = float(Tm.sum())
    ncomp = float(ndimage.label(Tm)[1]) if tvol > 0 else 0.
    om, tm = stats(O), stats(Tm)
    return np.array([np.log1p(ovol), om[0], om[1], om[2], 1.0 if tvol > 0 else 0.0, np.log1p(tvol), tm[0], tm[1],
                     ncomp, (tvol / ovol) if ovol > 0 else 0.0, tm[0] - om[0]], float)


def sl(vol, ax, z):
    return vol[z] if ax == 0 else (vol[:, z] if ax == 1 else vol[:, :, z])


def radiomics_case(job):
    ds, cid = job
    ct, seg = load(ds, cid)
    organs, tl = SCHEME[ds]
    Tall = seg == tl
    out = {}
    union_O = np.zeros(seg.shape, bool)
    for o, labs in organs.items():
        O = np.isin(seg, labs)
        if not O.any():
            continue
        union_O |= O
        Tm = Tall & ndimage.binary_dilation(O, iterations=3) if len(organs) > 1 else Tall
        out[o] = feats(ct, O, Tm)
    out["__union__"] = feats(ct, union_O, Tall)
    # slice picks for the embedding stage (16 per organ + union), returned so the GPU stage needs no seg
    ax = AXIAL[ds]; picks = {}
    for o, labs in list(organs.items()) + [("__union__", None)]:
        M = union_O if o == "__union__" else np.isin(seg, labs)
        zs = [z for z in range(M.shape[ax]) if sl(M, ax, z).sum() > 0]
        if zs:
            picks[o] = [int(zs[i]) for i in np.linspace(0, len(zs) - 1, min(16, len(zs))).astype(int)]
    lo, hi = WIN
    slices = {}
    for z in sorted(set(z for v in picks.values() for z in v)):
        s = sl(ct, ax, z).astype(np.float32); s = np.clip(s, lo, hi); slices[z] = (s - lo) / (hi - lo)
    return cid, out, picks, slices


# ------------------------------------------------------------------ embedding
def embed_all(cases, slices_by_case, picks_by_case, dev="cuda"):
    import torch, torch.nn.functional as Fn
    from torchvision.models import resnet50, ResNet50_Weights
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); net.fc = torch.nn.Identity(); net.eval().to(dev)
    E = {}
    for cid in cases:
        slices, picks = slices_by_case[cid], picks_by_case[cid]
        zs = sorted(slices)
        batch = []
        for z in zs:
            t = torch.from_numpy(slices[z])[None, None]
            t = Fn.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            batch.append(t.repeat(1, 3, 1, 1))
        with torch.no_grad():
            f = net(torch.cat(batch).to(dev)).cpu().numpy()
        fz = {z: f[i] for i, z in enumerate(zs)}
        E[cid] = {o: np.mean([fz[z] for z in v], axis=0) for o, v in picks.items()}
    return E


# ------------------------------------------------------------------ retrieval with feature vectors
def cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def per_query_feat(F, sim_recs, rel_recs, qids, cids, organ_conditioned):
    """Mirror of retrieval_tables_789.per_query with a feature-cosine ranking."""
    out = {}
    for qi in qids:
        qo = set(sim_recs[qi]["observed_organs"]) & set(F[qi])
        cands = []
        for bi in cids:
            if bi == qi:
                continue
            if organ_conditioned:
                shared = qo & set(sim_recs[bi]["observed_organs"]) & set(F[bi])
                if not shared:
                    continue
                s = float(np.mean([cos(F[qi][o], F[bi][o]) for o in shared]))
            else:
                s = cos(F[qi]["__union__"], F[bi]["__union__"])
            cands.append((s, bi))
        if not cands:
            continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if relevant(rel_recs[qi], rel_recs[bi]) else 0 for _, bi in cands]
        if sum(rels) == 0:
            continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        top10 = [bi for _, bi in cands[:10]]
        qobs = set(sim_recs[qi]["observed_organs"])
        m = {"p5": float(np.mean(rels[:5])), "p10": float(np.mean(rels[:10])), "ap": ap / sum(rels),
             "ndcg": _dcg(rels[:10]) / (_dcg(sorted(rels, reverse=True)[:10]) or 1),
             "spur10": float(np.mean([0.0 if (qobs & set(sim_recs[bi]["observed_organs"])) else 1.0 for bi in top10]))}
        m["mism10"] = T.mismatch10(rel_recs[qi], [rel_recs[bi] for bi in top10]) if len(rel_recs[qi]["observed_organs"]) == 1 else None
        out[qi] = m
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--skip-pool", action="store_true", help="Tab. 7 only (no FLARE23 candidates)")
    a = ap.parse_args()
    core.ORGAN_UNIVERSE = T.UNIVERSE
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"])
    gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    flare_all = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg.json")))["records"]}
    q113 = [i for i in pred if i in gt]
    pool_ids = [] if a.skip_pool else [c for c in flare_all if os.path.exists(f"{POOL}/images/{c.replace('flare23_', '')}_0000.nii.gz")
                                        and os.path.exists(f"{POOL}/labels/{c.replace('flare23_', '')}.nii.gz")]
    flare = {c: flare_all[c] for c in pool_ids}
    print(f"113 queries; FLARE23 candidates with CT: {len(flare)} / {len(flare_all)}", flush=True)

    # ---- features (cached)
    jobs = [(gt[c]["dataset"], c) for c in q113] + [("flare23", c) for c in pool_ids]
    if os.path.exists(FEAT) and not a.recompute:
        z = np.load(FEAT, allow_pickle=True); R, E = z["R"].item(), z["E"].item()
        jobs = [j for j in jobs if j[1] not in R or j[1] not in E]
    else:
        R, E = {}, {}
    if jobs:
        print(f"extracting features for {len(jobs)} cases ...", flush=True)
        slices_by, picks_by = {}, {}
        with ProcessPoolExecutor(a.workers) as ex:
            for k, (cid, r, picks, slices) in enumerate(ex.map(radiomics_case, jobs, chunksize=2)):
                R[cid] = r; picks_by[cid] = picks; slices_by[cid] = slices
                if (k + 1) % 25 == 0:
                    print(f"  radiomics {k+1}/{len(jobs)}", flush=True)
        print("embedding ...", flush=True)
        E.update(embed_all([j[1] for j in jobs], slices_by, picks_by))
        np.savez(FEAT, R=np.array(R, dtype=object), E=np.array(E, dtype=object))
    def zscore(cases):
        """z-score the radiomics vectors over the union features of the table's own population (as the original)."""
        allR = np.array([R[c]["__union__"] for c in cases]); mu, sd = allR.mean(0), allR.std(0) + 1e-8
        return {c: {o: (v - mu) / sd for o, v in R[c].items()} for c in cases}

    results = {"flare23_candidates_with_ct": len(flare), "flare23_candidates_total": len(flare_all), "n_queries": len(q113),
               "definitions": __doc__}
    tables = [("tab7", None, gt, q113)] + ([] if a.skip_pool else [("tab8_subpool", None, None, q113 + sorted(flare))])
    for tab, sim, rel, cids in tables:
        Rz = zscore(cids)
        if tab == "tab7":
            sim_kg_pred, sim_kg_gt, relr = pred, gt, gt
        else:
            pool_pred = {**pred, **flare}; pool_gt = {**gt, **flare}
            sim_kg_pred, sim_kg_gt, relr = pool_pred, pool_gt, pool_pred
        pq = {"(i) Reference phenotypes + gamma (upper bound)": T.per_query(sim_kg_gt, relr if tab == "tab7" else sim_kg_gt, q113, cids, "proposed"),
              "(ii) Predicted phenotypes + gamma (proposed)":   T.per_query(sim_kg_pred, relr, q113, cids, "proposed"),
              "(iii) Predicted, coverage-blind (no gamma)":     T.per_query(sim_kg_pred, relr, q113, cids, "coverage_blind"),
              "(iv) Predicted, organ-agnostic baseline":        T.per_query(sim_kg_pred, relr, q113, cids, "base"),
              "(v) Radiomics-vector (organ-agnostic)":          per_query_feat(Rz, sim_kg_pred, relr, q113, cids, False),
              "(vi) Radiomics-vector (organ-conditioned)":      per_query_feat(Rz, sim_kg_pred, relr, q113, cids, True),
              "(vii) Image-embedding (organ-agnostic)":         per_query_feat(E, sim_kg_pred, relr, q113, cids, False),
              "(vii') Image-embedding (organ-conditioned)":     per_query_feat(E, sim_kg_pred, relr, q113, cids, True)}
        rows = {k: T.agg(v) for k, v in pq.items()}
        ref = "(ii) Predicted phenotypes + gamma (proposed)"
        f1 = {k: T.paired(pq[k], pq[ref]) for k in pq if k != ref}
        T.holm(f1)
        for k, v in f1.items():
            rows[k]["dmAP_vs_ii"] = v
        results[tab] = {"n_candidates": len(cids), "rows": rows,
                        "holm_family": "F1: all seven comparator-vs-(ii) mAP contrasts of this table"}
        print(f"\n{tab}: {len(cids)} candidates")
        for k, v in rows.items():
            d = v.get("dmAP_vs_ii")
            print(f"  {k:46s} P@10={v['P@10']} mAP={v['mAP']} nDCG={v['nDCG']} spur={v['spurious@10']} mism={v['mismatch@10']}"
                  + (f"  dmAP {d['delta']:+.3f} {d['ci95']} p_holm={d['p_holm']}" if d else ""))
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
