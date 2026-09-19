#!/usr/bin/env python3
"""Draft Table 15, last row — comparators under the LLM-expert judge.  Same 20 queries, same judge and
prompt as llm_expert_local.py; for every configuration the top-10 lists are re-derived on the 113-case
reference-evaluable pool, all (query, candidate) pairs are labelled once (labels already stored by the
Qwen3-32B run are reused; only new pairs go to the model), and expert-relevance mAP is reported per
configuration next to automatic-relevance mAP.  Configurations: (ii) proposed, (iii) coverage-blind,
(iii') imputed-absent, (iv) organ-agnostic, (v)-(vii') radiomics / image-embedding (organ-agnostic and
organ-conditioned) from the feature cache of baselines_tab7_tab8.py.
Usage: python llm_expert_comparators.py <hf-model-id> <tag>     (device_map=auto)
Output: results/llm_expert/llm_expert_comparators_<tag>.json
"""
import json
import os
import re
import sys

os.environ.setdefault("HF_HOME", "/path/to/hf_cache")
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "retrieval"))
import retrieval_core_local as core                                     # noqa: E402
from retrieval_core_local import similarity, relevant, load_corpus     # noqa: E402
import retrieval_tables_789 as T                                        # noqa: E402
import baselines_tab7_tab8 as B                                         # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RES = os.path.join(ROOT, "results", "llm_expert")
core.ORGAN_UNIVERSE = T.UNIVERSE
rng = np.random.default_rng(0)

pred, gt = load_corpus("predicted", datasets=["kits", "lits", "msd"]), load_corpus("gt", datasets=["kits", "lits", "msd"])
ids = [i for i in pred if i in gt]

# ---- identical query sample to llm_expert.py / llm_expert_local.py (stratified, linspace, 20)
by_ds = {}
for i in ids:
    by_ds.setdefault(pred[i]["dataset"], []).append(i)
queries = []
for ds, lst in by_ds.items():
    lst = sorted(lst); k = max(1, round(20 * len(lst) / len(ids)))
    queries += [lst[j] for j in np.linspace(0, len(lst) - 1, k).astype(int)]
queries = queries[:20]

# ---- top-10 per configuration
z = np.load(B.FEAT, allow_pickle=True); R, E = z["R"].item(), z["E"].item()
allR = np.array([R[c]["__union__"] for c in ids]); mu, sd = allR.mean(0), allR.std(0) + 1e-8
Rz = {c: {o: (v - mu) / sd for o, v in R[c].items()} for c in ids}


def top10_kg(mode):
    out = {}
    for q in queries:
        cands = [(similarity(pred[q], pred[b], mode), b) for b in ids if b != q]
        cands = [(s, b) for s, b in cands if s is not None]; cands.sort(key=lambda x: -x[0])
        out[q] = [b for _, b in cands[:10]]
    return out


def top10_feat(F, organ_conditioned):
    out = {}
    for q in queries:
        qo = set(pred[q]["observed_organs"]) & set(F[q]); cands = []
        for b in ids:
            if b == q:
                continue
            if organ_conditioned:
                shared = qo & set(pred[b]["observed_organs"]) & set(F[b])
                if not shared:
                    continue
                s = float(np.mean([B.cos(F[q][o], F[b][o]) for o in shared]))
            else:
                s = B.cos(F[q]["__union__"], F[b]["__union__"])
            cands.append((s, b))
        cands.sort(key=lambda x: -x[0]); out[q] = [b for _, b in cands[:10]]
    return out


CONFIGS = {"(ii) proposed": top10_kg("proposed"), "(iii) coverage-blind": top10_kg("coverage_blind"),
           "(iii') imputed-absent": top10_kg("imputed_absent"), "(iv) organ-agnostic": top10_kg("base"),
           "(v) radiomics (organ-agnostic)": top10_feat(Rz, False), "(vi) radiomics (organ-conditioned)": top10_feat(Rz, True),
           "(vii) image-embedding (organ-agnostic)": top10_feat(E, False), "(vii') image-embedding (organ-conditioned)": top10_feat(E, True)}
pairs = sorted({(q, b) for t in CONFIGS.values() for q, bs in t.items() for b in bs})

# ---- labels: reuse the stored ones, judge only the new pairs
MID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-32B"
TAG = sys.argv[2] if len(sys.argv) > 2 else re.sub(r"[^A-Za-z0-9.]+", "_", MID.split("/")[-1])
lab = {}
prev = os.path.join(RES, f"llm_expert_study_{TAG}.json")
if os.path.exists(prev):
    for k, v in json.load(open(prev)).get("pair_labels", {}).items():
        q, b = k.split("|"); lab[(q, b)] = v
todo = [p for p in pairs if p not in lab]
print(f"{len(queries)} queries, {len(pairs)} pairs over {len(CONFIGS)} configurations; {len(lab)} labels reused, {len(todo)} to judge", flush=True)

if todo:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MID)
    model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="auto").eval()
    _CT_KW = {"enable_thinking": False} if "qwen3" in MID.lower() else {}
    SYS = ("You are a radiologist doing case-based retrieval. Given a QUERY patient and a CANDIDATE patient described "
           "by organ and tumor phenotypes, decide if the candidate is clinically relevant to the query — i.e. similar "
           "tumor burden, multiplicity, containment, and anatomic location in a shared organ. Answer with only YES or NO.")

    def describe(rec):
        parts = []
        for org in rec.get("observed_organs", []):
            o = rec["organs"].get(org, {}); s = f"{org} (volume {o.get('organ_volume_cm3','?')} cm3"
            if o.get("has_tumor"):
                s += f"; tumor present, burden {o.get('burden_cat')}, {o.get('multiplicity')}, {o.get('containment')} containment, {o.get('anatomic_location')} location)"
            else:
                s += "; no tumor)"
            parts.append(s)
        return "; ".join(parts) or "no organs observed"

    tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
    for i in range(0, len(todo), 16):
        chunk = todo[i:i + 16]
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYS},
                   {"role": "user", "content": f"QUERY: {describe(pred[q])}\nCANDIDATE: {describe(pred[b])}\nRelevant? Answer YES or NO."}],
                   tokenize=False, add_generation_prompt=True, **_CT_KW) for q, b in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        outs = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for qb, o in zip(chunk, outs):
            lab[qb] = 1 if "yes" in o.lower() else 0
        print(f"  {min(i+16, len(todo))}/{len(todo)}", flush=True)


def mAP(top, lab_fn):
    APs = []
    for q in queries:
        rels = [lab_fn(q, b) for b in top[q]]
        if sum(rels) == 0:
            APs.append(0.0); continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        APs.append(ap / sum(rels))
    return round(float(np.mean(APs)), 3)


def mAP_skipzero(top, lab_fn):        # llm_expert.py convention: queries with no relevant in top-10 are skipped
    APs = []
    for q in queries:
        rels = [lab_fn(q, b) for b in top[q]]
        if sum(rels) == 0:
            continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1; ap += hit / (i + 1)
        APs.append(ap / sum(rels))
    return (round(float(np.mean(APs)), 3) if APs else None), len(APs)


rows = {}
for name, top in CONFIGS.items():
    e, ne = mAP_skipzero(top, lambda q, b: lab[(q, b)]); a, na = mAP_skipzero(top, lambda q, b: 1 if relevant(gt[q], gt[b]) else 0)
    rows[name] = {"expert_relevance_mAP": e, "n_scored_queries_expert": ne, "automatic_relevance_mAP": a, "n_scored_queries_auto": na,
                  "expert_relevance_mAP_zero_for_empty": mAP(top, lambda q, b: lab[(q, b)]),
                  "expert_P@10": round(float(np.mean([np.mean([lab[(q, b)] for b in top[q]]) for q in queries])), 3)}
    print(f"  {name:44s} expert mAP {e} (n={ne})  auto mAP {a}  expert P@10 {rows[name]['expert_P@10']}")
# ---- F4: paired inference vs (ii) under judge relevance (Tab. 15), on the queries scorable under (ii)
def ap_of(top, q, lab_fn):
    rels = [lab_fn(q, b) for b in top[q]]
    if sum(rels) == 0:
        return 0.0
    hit = ap = 0.0
    for i, r in enumerate(rels):
        if r:
            hit += 1; ap += hit / (i + 1)
    return ap / sum(rels)


scorable = [q for q in queries if sum(lab[(q, b)] for b in CONFIGS["(ii) proposed"][q]) > 0]
ref_ap = {q: {"ap": ap_of(CONFIGS["(ii) proposed"], q, lambda q, b: lab[(q, b)])} for q in scorable}
F4 = {}
for name in ("(v) radiomics (organ-agnostic)", "(vi) radiomics (organ-conditioned)", "(vii) image-embedding (organ-agnostic)", "(vii') image-embedding (organ-conditioned)"):
    F4[name] = T.paired({q: {"ap": ap_of(CONFIGS[name], q, lambda q, b: lab[(q, b)])} for q in scorable}, ref_ap)
T.holm(F4)
excluded = {"(iv) organ-agnostic": f"excluded from F4: {sum(1 for q in queries if sum(lab[(q, b)] for b in CONFIGS['(iv) organ-agnostic'][q]) == 0)}/20 queries have no judge-relevant candidate in its top-10"}
print(f"F4 (n={len(scorable)} queries scorable under (ii)):")
for k, v in F4.items():
    print(f"  {k:44s} dAP {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  p={v['p_raw']:.2g}  Holm={v['p_holm']}")
print("  ", excluded)

out = {"model": MID, "n_queries": len(queries), "n_pairs_labelled": len(lab), "n_pairs_new": len(todo),
       "F4_paired_vs_ii_judge_relevance": {"n_queries": len(scorable), "queries": scorable, "contrasts": F4, **excluded,
                                           "stats": "paired query bootstrap B=5000, sign-flip permutation B=20000, Holm over the four baseline contrasts; AP=0 for a query with no judge-relevant candidate in a configuration's top-10"},
       "note": "same queries/judge/prompt as llm_expert_study; mAP over top-10, queries with no expert-relevant candidate skipped "
               "(llm_expert.py convention) — the zero-for-empty variant is also given; KG configurations on predicted phenotypes, "
               "baselines from the cached CT features (baselines_tab7_tab8.py)",
       "rows": rows, "pair_labels": {f"{q}|{b}": v for (q, b), v in lab.items()}}
os.makedirs(RES, exist_ok=True); json.dump(out, open(os.path.join(RES, f"llm_expert_comparators_{TAG}.json"), "w"), indent=2)
print("->", os.path.join(RES, f"llm_expert_comparators_{TAG}.json"))
