#!/usr/bin/env python3
"""External retrieval baselines (paper configs v–vi): radiomics + organ-conditioned radiomics.
Rank the 113-patient cohort by radiomic-feature similarity — first-order HU + shape features extracted
from the CT within the GT organ (label 1) and tumor (label 2) regions — and score RELEVANCE against GT
tumor phenotypes with the SAME relevant() the KG retrieval uses. This is the non-KG comparator the paper
sets up. MSD + LiTS load locally; KiTS streams per-case from HF (temp in /dev/shm, deleted).
-> results/baselines_retrieval.json"""
import os, json, sys, urllib.request
import numpy as np, nibabel as nib
from scipy import ndimage
sys.path.insert(0, "/home/user/SWOG/src/scripts")
from kg_retrieval_v2 import relevant, _dcg
RES = "/home/user/SWOG/results"
PAN = "/scratch/user/acm_data/Pancreas"

def load_msd(c):
    seg = np.asarray(nib.load(f"{PAN}/labelsTr/{c}.nii.gz").dataobj).astype(int)
    ct = nib.load(f"{PAN}/imagesTr/{c}.nii.gz").get_fdata()
    return ct, seg
def load_lits(c):
    vid = c.replace("volume-", "")
    ct = np.load(f"/scratch/user/acm_data/Data/ct/volume-{vid}.npy")
    seg = np.load(f"/scratch/user/acm_data/Data/seg/segmentation-{vid}.npy").astype(int)
    return ct, seg
KITS_IMG = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/{c}.nii.gz"
KITS_SEG = "https://raw.githubusercontent.com/neheller/kits23/main/dataset/{c}/segmentation.nii.gz"
import time
def load_kits(c):
    sp_f, im_f = f"/dev/shm/_rk_{c}_seg.nii.gz", f"/dev/shm/_rk_{c}_img.nii.gz"
    try:
        for attempt in range(5):                      # retry/backoff for HF 429 rate-limits
            try:
                urllib.request.urlretrieve(KITS_SEG.format(c=c), sp_f)
                urllib.request.urlretrieve(KITS_IMG.format(c=c), im_f)
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    time.sleep(10 * (attempt + 1)); continue
                raise
        ct = nib.load(im_f).get_fdata(); seg = np.asarray(nib.load(sp_f).dataobj).astype(int)
        return ct, seg
    finally:
        for p in (sp_f, im_f):
            if os.path.exists(p): os.remove(p)
LOAD = {"msd": load_msd, "lits": load_lits, "kits": load_kits}

def feats(ct, seg):
    """11-dim first-order HU + shape descriptor from GT organ (==1) and tumor (==2)."""
    O = (seg == 1); T = (seg == 2)
    def stats(mask):
        if mask.sum() == 0: return [0., 0., 0.]
        v = ct[mask]; return [float(v.mean()), float(v.std()), float(np.percentile(v, 90))]
    ovol = float(O.sum()); tvol = float(T.sum())
    ncomp = float(ndimage.label(T)[1]) if tvol > 0 else 0.
    om, tm = stats(O), stats(T)
    return np.array([np.log1p(ovol), om[0], om[1], om[2],
                     1.0 if tvol > 0 else 0.0, np.log1p(tvol), tm[0], tm[1],
                     ncomp, (tvol / ovol) if ovol > 0 else 0.0, tm[0] - om[0]], float)

recs = {}
for ds in ["kits", "lits", "msd"]:
    for r in json.load(open(f"{RES}/corpus_gt_{ds}.json"))["records"]:
        recs[r["case_id"]] = r
ids = list(recs)
F = {}
for i, cid in enumerate(ids):
    ds = recs[cid]["dataset"]
    try:
        ct, seg = LOAD[ds](cid); F[cid] = feats(ct, seg)
        print(f"  [{i+1}/{len(ids)}] {ds}/{cid}", flush=True)
    except Exception as e:
        print(f"  SKIP {ds}/{cid}: {e}", flush=True)
ids = [i for i in ids if i in F]
X = np.array([F[i] for i in ids]); mu, sd = X.mean(0), X.std(0) + 1e-8; Z = (X - mu) / sd
zi = {cid: Z[k] for k, cid in enumerate(ids)}

def cos(a, b): return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))
def obs(cid): return set(recs[cid]["observed_organs"])

def evaluate(organ_conditioned):
    P5, P10, AP, ND, SPUR = [], [], [], [], []
    for q in ids:
        cands = []
        for b in ids:
            if b == q: continue
            if organ_conditioned and not (obs(q) & obs(b)): continue
            cands.append((cos(zi[q], zi[b]), b))
        if not cands: continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if relevant(recs[q], recs[b]) else 0 for _, b in cands]
        if sum(rels) == 0: continue
        P5.append(np.mean(rels[:5])); P10.append(np.mean(rels[:10]))
        hit = ap = 0.
        for i, r in enumerate(rels):
            if r: hit += 1; ap += hit / (i + 1)
        AP.append(ap / sum(rels))
        ND.append(_dcg(rels[:10]) / (_dcg(sorted(rels, reverse=True)[:10]) or 1))
        SPUR.append(np.mean([0. if (obs(q) & obs(b)) else 1. for _, b in cands[:10]]))
    r = lambda x: round(float(np.mean(x)), 3) if x else None
    return {"P@5": r(P5), "P@10": r(P10), "mAP": r(AP), "nDCG": r(ND), "spurious@10": r(SPUR), "n_queries": len(ND)}

out = {"n_patients": len(ids),
       "features": "11-dim first-order HU (mean/std/p90) + shape (log-volume, tumor n-components, tumor/organ ratio, tumor-organ HU contrast) from GT organ(=1)/tumor(=2) regions, z-scored",
       "radiomics (organ-agnostic)": evaluate(False),
       "radiomics_organ_conditioned": evaluate(True)}
json.dump(out, open(f"{RES}/baselines_retrieval.json", "w"), indent=2)
print("\n=== external baselines (retrieval) ===")
print(json.dumps(out, indent=2))
