#!/usr/bin/env python3
"""External retrieval baseline (paper config vii): image-embedding retrieval.
Embed each patient with a pretrained ResNet50 (ImageNet) over HU-windowed organ-region slices
(mean-pooled → 2048-dim), rank by cosine similarity, score RELEVANCE vs GT tumor phenotypes with the
same relevant() the KG retrieval uses. Organ-agnostic + organ-conditioned variants. Merges its keys
into results/baselines_retrieval.json. MSD+LiTS local; KiTS streams (retry/backoff)."""
import os, json, sys, urllib.request, urllib.error, time
import numpy as np, nibabel as nib, torch
sys.path.insert(0, "/home/user/SWOG/src/scripts")
from kg_retrieval_v2 import relevant, _dcg
RES = "/home/user/SWOG/results"
PAN = "/scratch/user/acm_data/Pancreas"
WIN = (-125, 225); AXIAL = {"msd": 2, "lits": 0, "kits": 0}; DEV = "cuda:0"

def load_msd(c):
    seg = np.asarray(nib.load(f"{PAN}/labelsTr/{c}.nii.gz").dataobj).astype(int)
    return nib.load(f"{PAN}/imagesTr/{c}.nii.gz").get_fdata(), seg
def load_lits(c):
    vid = c.replace("volume-", "")
    return (np.load(f"/scratch/user/acm_data/Data/ct/volume-{vid}.npy"),
            np.load(f"/scratch/user/acm_data/Data/seg/segmentation-{vid}.npy").astype(int))
KI = "https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/main/images/{c}.nii.gz"
KS = "https://raw.githubusercontent.com/neheller/kits23/main/dataset/{c}/segmentation.nii.gz"
def load_kits(c):
    sp, im = f"/dev/shm/_ie_{c}_s.nii.gz", f"/dev/shm/_ie_{c}_i.nii.gz"
    try:
        for a in range(5):
            try:
                urllib.request.urlretrieve(KS.format(c=c), sp); urllib.request.urlretrieve(KI.format(c=c), im); break
            except urllib.error.HTTPError as e:
                if e.code == 429 and a < 4: time.sleep(10 * (a + 1)); continue
                raise
        return nib.load(im).get_fdata(), np.asarray(nib.load(sp).dataobj).astype(int)
    finally:
        for p in (sp, im):
            if os.path.exists(p): os.remove(p)
LOAD = {"msd": load_msd, "lits": load_lits, "kits": load_kits}

from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as Fn
net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2); net.fc = torch.nn.Identity(); net.eval().to(DEV)

def sl(vol, ax, z): return vol[z] if ax == 0 else (vol[:, z] if ax == 1 else vol[:, :, z])
def embed(ct, seg, ax):
    O = (seg == 1)
    zs = [z for z in range(O.shape[ax]) if sl(O, ax, z).sum() > 0]
    if not zs: return None
    pick = [zs[i] for i in np.linspace(0, len(zs) - 1, min(16, len(zs))).astype(int)]
    lo, hi = WIN; batch = []
    for z in pick:
        s = sl(ct, ax, z).astype(np.float32); s = np.clip(s, lo, hi); s = (s - lo) / (hi - lo)
        t = torch.from_numpy(s)[None, None]; t = Fn.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        batch.append(t.repeat(1, 3, 1, 1))
    x = torch.cat(batch).to(DEV)
    with torch.no_grad():
        f = net(x).mean(0).cpu().numpy()          # mean-pool slice features -> 2048-dim
    return f

recs = {}
for ds in ["kits", "lits", "msd"]:
    for r in json.load(open(f"{RES}/corpus_gt_{ds}.json"))["records"]: recs[r["case_id"]] = r
ids = list(recs); E = {}
for i, cid in enumerate(ids):
    ds = recs[cid]["dataset"]
    try:
        ct, seg = LOAD[ds](cid); e = embed(ct, seg, AXIAL[ds])
        if e is not None: E[cid] = e
        print(f"  [{i+1}/{len(ids)}] {ds}/{cid}", flush=True)
    except Exception as ex:
        print(f"  SKIP {ds}/{cid}: {ex}", flush=True)
ids = [i for i in ids if i in E]

def cos(a, b): return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))
def obs(cid): return set(recs[cid]["observed_organs"])
def evaluate(organ_conditioned):
    P5, P10, AP, ND, SPUR = [], [], [], [], []
    for q in ids:
        cands = [(cos(E[q], E[b]), b) for b in ids if b != q and (not organ_conditioned or (obs(q) & obs(b)))]
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

out = json.load(open(f"{RES}/baselines_retrieval.json")) if os.path.exists(f"{RES}/baselines_retrieval.json") else {}
out["n_patients_image_embedding"] = len(ids)
out["image_embedding (organ-agnostic)"] = evaluate(False)
out["image_embedding_organ_conditioned"] = evaluate(True)
out["image_embedding_note"] = "ResNet50 (ImageNet) over HU-windowed organ-region slices, mean-pooled 2048-dim; cosine retrieval."
json.dump(out, open(f"{RES}/baselines_retrieval.json", "w"), indent=2)
print("\n=== image-embedding baseline ===")
print(json.dumps({k: out[k] for k in out if "image_embedding" in k}, indent=2))
