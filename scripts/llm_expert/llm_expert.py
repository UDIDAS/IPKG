#!/usr/bin/env python3
"""LLM-as-expert relevance study (fills the p1/p18/p19 κ + expert-relevance-mAP TBDs).
A local instruct model, BLINDED to the automatic label, judges whether a candidate patient is
clinically relevant to a query patient from phenotype text alone. Then:
  - Cohen's κ: LLM-expert relevance vs the pipeline's automatic relevance (validates the proxy)
  - expert-relevance mAP: re-score the predicted+γ ranking under LLM-expert relevance
Local model on GPU 1, sharing compute. -> results/llm_expert_study.json

NOTE: the locally-cached Llama-3.2-3B-Instruct is too weak to act as a radiologist-expert
(κ≈0.089, raw agreement 0.467 ≈ chance) — DO NOT report the 3B numbers. Re-run with a stronger
judge (Gemini/Claude) by swapping the generation backend; the pipeline/metrics are unchanged."""
import os, glob, json
os.environ["HF_HOME"] = "/scratch/user/hf_cache"; os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np, torch, sys
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, "/home/user/SWOG/src/scripts")
from kg_retrieval_v2 import similarity, relevant
RES = "/home/user/SWOG/results"
rng = np.random.default_rng(0)

def load(kind):
    recs = []
    for f in glob.glob(f"{RES}/corpus_{kind}_*.json"): recs += json.load(open(f))["records"]
    return {r["case_id"]: r for r in recs}
pred, gt = load("predicted"), load("gt")
ids = [i for i in pred if i in gt]

def describe(rec):
    parts = []
    for org in rec.get("observed_organs", []):
        o = rec["organs"].get(org, {})
        s = f"{org} (volume {o.get('organ_volume_cm3','?')} cm3"
        if o.get("has_tumor"):
            s += f"; tumor present, burden {o.get('burden_cat')}, {o.get('multiplicity')}, {o.get('containment')} containment, {o.get('anatomic_location')} location)"
        else: s += "; no tumor)"
        parts.append(s)
    return "; ".join(parts) or "no organs observed"

# stratified query sample (proportional across datasets), ~20 queries
by_ds = {}
for i in ids: by_ds.setdefault(pred[i]["dataset"], []).append(i)
queries = []
for ds, lst in by_ds.items():
    lst = sorted(lst); k = max(1, round(20 * len(lst) / len(ids)))
    queries += [lst[j] for j in np.linspace(0, len(lst) - 1, k).astype(int)]
queries = queries[:20]

# per query: (a) ranked top-10 by similarity -> expert-mAP; (b) CLASS-BALANCED auto-pos/auto-neg -> fair κ
meta = []
for q in queries:
    cands = []
    for b in ids:
        if b == q: continue
        s = similarity(pred[q], pred[b], "proposed")
        if s is not None: cands.append((s, b))
    cands.sort(key=lambda x: -x[0])
    for rank, (_, b) in enumerate(cands[:10]):
        meta.append({"q": q, "b": b, "rank": rank, "rank_role": True, "kappa_role": False})
    pos = [b for b in ids if b != q and relevant(gt[q], gt[b])]
    neg = [b for b in ids if b != q and not relevant(gt[q], gt[b])]
    kp = [pos[i] for i in rng.choice(len(pos), min(8, len(pos)), replace=False)] if pos else []
    kn = [neg[i] for i in rng.choice(len(neg), min(8, len(neg)), replace=False)] if neg else []
    for b in kp + kn:
        meta.append({"q": q, "b": b, "rank": None, "rank_role": False, "kappa_role": True})
upairs = sorted({(m["q"], m["b"]) for m in meta})   # unique (q,b) to label once

print(f"{len(queries)} queries, {len(upairs)} pairs — loading model...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
MID = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.2-3B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()
SYS = ("You are a radiologist doing case-based retrieval. Given a QUERY patient and a CANDIDATE patient described "
       "by organ and tumor phenotypes, decide if the candidate is clinically relevant to the query — i.e. similar "
       "tumor burden, multiplicity, containment, and anatomic location in a shared organ. Answer with only YES or NO.")
def judge_batch(batch):
    prompts = [tok.apply_chat_template([{"role":"system","content":SYS},
              {"role":"user","content":f"QUERY: {describe(pred[q])}\nCANDIDATE: {describe(pred[b])}\nRelevant? Answer YES or NO."}],
              tokenize=False, add_generation_prompt=True) for q,b in batch]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
    outs = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [1 if "yes" in o.lower() else 0 for o in outs]

tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
lab = {}
for i in range(0, len(upairs), 16):
    chunk = upairs[i:i + 16]
    for qb, v in zip(chunk, judge_batch(chunk)): lab[qb] = v
    print(f"  {min(i+16,len(upairs))}/{len(upairs)}", flush=True)

# κ on the CLASS-BALANCED pairs (removes the base-rate confound)
kp_pairs = [(m["q"], m["b"]) for m in meta if m["kappa_role"]]
auto_lab = [1 if relevant(gt[q], gt[b]) else 0 for q, b in kp_pairs]
llm_lab = [lab[(q, b)] for q, b in kp_pairs]
kappa = float(cohen_kappa_score(auto_lab, llm_lab))
agree = float(np.mean([a == l for a, l in zip(auto_lab, llm_lab)]))
al, ll = np.array(auto_lab), np.array(llm_lab)
confusion = {"auto1_llm1": int(((al == 1) & (ll == 1)).sum()), "auto1_llm0": int(((al == 1) & (ll == 0)).sum()),
             "auto0_llm1": int(((al == 0) & (ll == 1)).sum()), "auto0_llm0": int(((al == 0) & (ll == 0)).sum())}
diag = {"n_kappa_pairs": len(kp_pairs), "auto_positive_rate": round(float(al.mean()), 3),
        "llm_positive_rate": round(float(ll.mean()), 3), "confusion": confusion}

# expert-relevance mAP vs automatic mAP over the SAME queries (top-10 ranked)
def mAP(lab_fn):
    APs = []
    for q in queries:
        rows = [(m["rank"], lab_fn(m["q"], m["b"])) for m in meta if m["q"] == q and m["rank_role"]]
        rows.sort(); rels = [r for _, r in rows]
        if sum(rels) == 0: continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r: hit += 1; ap += hit / (i + 1)
        APs.append(ap / sum(rels))
    return round(float(np.mean(APs)), 3), len(APs)
expert_mAP, ne = mAP(lambda q, b: lab[(q, b)])
auto_mAP, na = mAP(lambda q, b: 1 if relevant(gt[q], gt[b]) else 0)

out = {"model": MID, "n_queries": len(queries), "n_pairs": len(upairs),
       "cohen_kappa_expert_vs_automatic": round(kappa, 3), "raw_agreement": round(agree, 3),
       "expert_relevance_mAP": expert_mAP, "automatic_relevance_mAP": auto_mAP, "n_scored_queries": ne,
       "diagnostics": diag,
       "caveat": f"LLM-as-expert judge = {MID}. Interpret with the diagnostics: low κ reflects (dis)agreement "
                 "between holistic LLM judgment and the STRICT automatic relevance rule (exact match on all 4 tumor "
                 "categoricals), not necessarily model weakness."}
json.dump(out, open(f"{RES}/llm_expert_study.json", "w"), indent=2)
print("\n=== LLM-as-expert study ===")
for k, v in out.items(): print(f"  {k}: {v}")
