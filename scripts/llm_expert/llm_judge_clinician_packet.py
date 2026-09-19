#!/usr/bin/env python3
"""LLM-judge over the FROZEN 30-query clinician packet — the disclosed INTERIM for §5.8
(decision memo: docs/VKG_clinician_study_decision_2026-09-17.md; NOT a substitute for the
radiologist study: no human κ, no CT channel — the judge reads the same phenotype summaries
the scorer is built from).

Protocol identical to the §4.10 study (llm_expert_local.py, 2026-09-13): same judge
(Qwen/Qwen3-32B unless overridden), same system prompt, same describe() phenotype text, same
deterministic YES/NO decoding, blinded to configuration, rank, and the construction-rule flag.
Judged pairs: every (query, pooled candidate) of results/study/clinician_study_30q.json —
the pooled union covers all five top-10 lists, so nDCG@10 per configuration is exact.

Reported (the tab:clinician analysis hooks, judge-labeled):
  - nDCG@10 and P@10 per configuration (ii, iii, iii', vi, vii'); B-block queries carry only
    (ii)/(iii') by design and contribute only there.
  - F5-style contrasts vs (ii): paired query bootstrap B=5,000 CI + sign-flip permutation
    B=20,000, Holm over the four contrasts (retrieval_tables_789 semantics).
  - Cohen's κ judge-vs-construction-rule over all pooled pairs, overall and per block.
  HF_HOME=... python llm_judge_clinician_packet.py [<hf-model-id> [<tag>]]
-> results/llm_expert/llm_judge_clinician_packet_<tag>.json
"""
import json
import os
import re
import sys

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "retrieval"))
from retrieval_core_local import load_corpus                      # noqa: E402

RES = os.path.join(ROOT, "results", "llm_expert")
MID = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-32B"
TAG = sys.argv[2] if len(sys.argv) > 2 else re.sub(r"[^A-Za-z0-9.]+", "_", MID.split("/")[-1])
CFGS = ["ii", "iii", "iii_prime", "vi", "vii_prime"]

# ---- records: the pool's recorded representation (identical to the packet builder) ----
pred = load_corpus("predicted", datasets=["kits", "lits", "msd"])
flare = {r["case_id"]: r for r in json.load(open(f"{ROOT}/corpora/corpus_flare23_kg_dedup.json"))["records"]}
REC = {**pred, **flare}
packet = json.load(open(f"{ROOT}/results/study/clinician_study_30q.json"))


def describe(rec):                                     # verbatim from llm_expert_local.py
    parts = []
    for org in rec.get("observed_organs", []):
        o = rec["organs"].get(org, {})
        s = f"{org} (volume {o.get('organ_volume_cm3','?')} cm3"
        if o.get("has_tumor"):
            s += (f"; tumor present, burden {o.get('burden_cat')}, {o.get('multiplicity')}, "
                  f"{o.get('containment')} containment, {o.get('anatomic_location')} location)")
        else:
            s += "; no tumor)"
        parts.append(s)
    return "; ".join(parts) or "no organs observed"


cells = packet["queries"]
pairs = sorted({(c["query_id"], p["case_id"]) for c in cells for p in c["pooled_candidates"]})
print(f"{len(cells)} queries, {len(pairs)} pooled pairs — loading {MID} ...", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer      # noqa: E402
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16, device_map="auto")
model.eval()
tok.padding_side = "left"
tok.pad_token = tok.pad_token or tok.eos_token
_CT_KW = {"enable_thinking": False} if "qwen3" in MID.lower() else {}
SYS = ("You are a radiologist doing case-based retrieval. Given a QUERY patient and a CANDIDATE patient described "
       "by organ and tumor phenotypes, decide if the candidate is clinically relevant to the query — i.e. similar "
       "tumor burden, multiplicity, containment, and anatomic location in a shared organ. Answer with only YES or NO.")


def judge_batch(batch):
    prompts = [tok.apply_chat_template(
        [{"role": "system", "content": SYS},
         {"role": "user", "content": f"QUERY: {describe(REC[q])}\nCANDIDATE: {describe(REC[b])}\nRelevant? Answer YES or NO."}],
        tokenize=False, add_generation_prompt=True, **_CT_KW) for q, b in batch]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
    outs = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [1 if "yes" in o.lower() else 0 for o in outs]


lab = {}
for i in range(0, len(pairs), 16):
    chunk = pairs[i:i + 16]
    for qb, v in zip(chunk, judge_batch(chunk)):
        lab[qb] = v
    if (i // 16) % 5 == 0 or i + 16 >= len(pairs):
        print(f"  {min(i+16, len(pairs))}/{len(pairs)}", flush=True)

# ---- per-query nDCG@10 / P@10 per configuration under judge labels ----
def dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


pq = {c: {} for c in CFGS}
for cell in cells:
    q = cell["query_id"]
    for cfg in CFGS:
        lst = cell["top10"].get(cfg) or []
        if not lst:
            continue
        rels = [lab[(q, b)] for b in lst]
        ideal = sorted((lab[(q, b)] for b in cell["pooled_candidates_ids"]) if "pooled_candidates_ids" in cell
                       else (lab[(q, p["case_id"])] for p in cell["pooled_candidates"]), reverse=True)[:10]
        pq[cfg][cell["cell"]] = {"ndcg10": (dcg(rels) / dcg(ideal)) if dcg(ideal) > 0 else None,
                                 "p10": float(np.mean(rels))}

rows = {}
for cfg in CFGS:
    v = [m["ndcg10"] for m in pq[cfg].values() if m["ndcg10"] is not None]
    p = [m["p10"] for m in pq[cfg].values()]
    rows[cfg] = {"n_queries": len(pq[cfg]), "n_scored": len(v),
                 "ndcg10": round(float(np.mean(v)), 3), "p10": round(float(np.mean(p)), 3)}

# ---- F5-style contrasts vs (ii): paired bootstrap CI + sign-flip permutation, Holm ----
rng = np.random.default_rng(12345)
contrasts = {}
for cfg in ("iii", "iii_prime", "vi", "vii_prime"):
    common = [k for k in pq[cfg] if k in pq["ii"]
              and pq[cfg][k]["ndcg10"] is not None and pq["ii"][k]["ndcg10"] is not None]
    d = np.array([pq[cfg][k]["ndcg10"] - pq["ii"][k]["ndcg10"] for k in common])
    if not len(d):
        continue
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    signs = rng.choice([-1, 1], size=(20000, len(d)))
    perm = (signs * d).mean(axis=1)
    p = float((np.abs(perm) >= abs(d.mean())).mean())
    contrasts[cfg] = {"n": len(d), "delta_ndcg10": round(float(d.mean()), 3),
                      "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                      "p_raw": max(p, 1 / 20000)}
order = sorted(contrasts, key=lambda k: contrasts[k]["p_raw"])
for i, k in enumerate(order):
    contrasts[k]["p_holm"] = round(min(1.0, contrasts[k]["p_raw"] * (len(order) - i)), 5)

# ---- κ vs construction rule, overall + per block ----
auto = {(c["query_id"], p["case_id"]): int(bool(p["relevant_construction_rule"]))
        for c in cells for p in c["pooled_candidates"]}
block_of = {(c["query_id"], p["case_id"]): c["block"] for c in cells for p in c["pooled_candidates"]}
ks = {}
for scope in ("all", "A", "B", "C"):
    sel = [qb for qb in pairs if scope == "all" or block_of[qb] == scope]
    a, l = [auto[qb] for qb in sel], [lab[qb] for qb in sel]
    ks[scope] = {"n_pairs": len(sel), "kappa": round(float(cohen_kappa_score(a, l)), 3) if len(set(a)) > 1 and len(set(l)) > 1 else None,
                 "agreement": round(float(np.mean([x == y for x, y in zip(a, l)])), 3),
                 "judge_positive_rate": round(float(np.mean(l)), 3), "rule_positive_rate": round(float(np.mean(a)), 3)}

out = {"model": MID, "n_queries": len(cells), "n_pairs": len(pairs),
       "note": ("INTERIM for §5.8 — judge-labeled results on the identical frozen packet; NOT the radiologist "
                "study (no human inter-rater κ, no CT channel; the judge reads the same phenotype summaries the "
                "scorer is built from — see the decision memo). Protocol byte-identical to §4.10. nDCG@10 gain = "
                "judge YES/NO; ideal from the labeled pooled union. B-block queries carry only (ii)/(iii')."),
       "rows_by_configuration": rows,
       "f5_contrasts_vs_ii": contrasts,
       "kappa_judge_vs_construction_rule": ks,
       "pair_labels": {f"{q}|{b}": v for (q, b), v in lab.items()}}
os.makedirs(RES, exist_ok=True)
fp = f"{RES}/llm_judge_clinician_packet_{TAG}.json"
json.dump(out, open(fp, "w"), indent=1)
print("\nrows:", json.dumps(rows, indent=1))
print("contrasts:", json.dumps(contrasts, indent=1))
print("kappa:", json.dumps(ks, indent=1))
print(f"-> {fp}")
