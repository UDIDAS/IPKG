#!/usr/bin/env python3
"""F5 analysis harness for the §5.7 clinician relevance study (prespecified; option A, the study lead
2026-09-18). Runs the moment the two filled sheets come back — tab:clinician in one pass.

Inputs: the two filled rater sheets + KEY.csv (build_grading_sheets.py) + the frozen packet;
optionally the judge-on-packet labels for the three-way comparison.

Reports (the §4.11/§5.7 analysis hooks, verbatim):
  - quadratic-weighted κ between raters, overall and per block (A/B/C);
  - nDCG@10 per configuration (ii, iii, iii', vi, vii'), gain = MEAN of the two raters'
    0-3 grades (DCG gain 2^g - 1; ideal from the graded pooled union of the query);
  - F5 contrasts vs (ii): paired query bootstrap B=5,000 CI + sign-flip permutation B=20,000,
    Holm over the four contrasts;
  - binarized grades (mean >= 2): agreement + κ with the construction rule and with the
    Qwen3-32B judge (packet run), i.e. the direct three-way comparison.

  python analyze_clinician_grades.py [--sheets results/study/grading_sheets] [--out ...]
-> results/study/clinician_study_results.json
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CFGS = ["ii", "iii", "iii_prime", "vi", "vii_prime"]


def dcg(gains):
    return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", default=f"{ROOT}/results/study/grading_sheets")
    ap.add_argument("--judge", default=f"{ROOT}/results/llm_expert/llm_judge_clinician_packet_Qwen3_32B.json")
    ap.add_argument("--out", default=f"{ROOT}/results/study/clinician_study_results.json")
    a = ap.parse_args()

    key = list(csv.DictReader(open(f"{a.sheets}/KEY.csv")))
    packet = json.load(open(f"{ROOT}/results/study/clinician_study_30q.json"))
    cells = {c["cell"]: c for c in packet["queries"]}

    grades = {}                                            # (query_id, cand_id) -> {rater: grade}
    for rater in ("rater1", "rater2"):
        sheet = {r["item_id"]: r for r in csv.DictReader(open(f"{a.sheets}/{rater}_sheet.csv"))}
        for k in key:
            if k["rater"] != rater:
                continue
            g = sheet[k["item_id"]]["grade_0_to_3"].strip()
            if g == "":
                raise SystemExit(f"blank grade: {rater} item {k['item_id']}")
            grades.setdefault((k["query_id"], k["candidate_id"]), {})[rater] = int(g)
    assert all(len(v) == 2 for v in grades.values()), "every pair needs both raters"
    mean_g = {qb: (v["rater1"] + v["rater2"]) / 2.0 for qb, v in grades.items()}

    # κ between raters (quadratic), overall + per block
    block_of = {(k["query_id"], k["candidate_id"]): k["block"] for k in key}
    kappa = {}
    for scope in ("all", "A", "B", "C"):
        sel = [qb for qb in grades if scope == "all" or block_of[qb] == scope]
        r1, r2 = [grades[qb]["rater1"] for qb in sel], [grades[qb]["rater2"] for qb in sel]
        kappa[scope] = {"n_pairs": len(sel),
                        "quadratic_kappa": round(float(cohen_kappa_score(r1, r2, weights="quadratic")), 3),
                        "raw_agreement": round(float(np.mean([x == y for x, y in zip(r1, r2)])), 3)}

    # nDCG@10 per configuration, gain = mean grade
    pq = {c: {} for c in CFGS}
    for cell, c in cells.items():
        q = c["query_id"]
        pooled = [(q, p["case_id"]) for p in c["pooled_candidates"]]
        ideal = sorted((mean_g[qb] for qb in pooled), reverse=True)[:10]
        for cfg in CFGS:
            lst = c["top10"].get(cfg) or []
            if not lst:
                continue
            gains = [mean_g[(q, b)] for b in lst]
            pq[cfg][cell] = dcg(gains) / dcg(ideal) if dcg(ideal) > 0 else None
    rows = {cfg: {"n_queries": len(v),
                  "ndcg10": round(float(np.mean([x for x in v.values() if x is not None])), 3)}
            for cfg, v in pq.items()}

    # F5 contrasts vs (ii)
    rng = np.random.default_rng(12345)
    contrasts = {}
    for cfg in ("iii", "iii_prime", "vi", "vii_prime"):
        common = [k for k in pq[cfg] if k in pq["ii"] and pq[cfg][k] is not None and pq["ii"][k] is not None]
        d = np.array([pq[cfg][k] - pq["ii"][k] for k in common])
        if not len(d):
            continue
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        signs = rng.choice([-1, 1], size=(20000, len(d)))
        p = float((np.abs((signs * d).mean(axis=1)) >= abs(d.mean())).mean())
        contrasts[cfg] = {"n": len(d), "delta_ndcg10": round(float(d.mean()), 3),
                          "ci95": [round(float(np.percentile(boots, 2.5)), 3),
                                   round(float(np.percentile(boots, 97.5)), 3)],
                          "p_raw": max(p, 1 / 20000)}
    order = sorted(contrasts, key=lambda k: contrasts[k]["p_raw"])
    for i, k in enumerate(order):
        contrasts[k]["p_holm"] = round(min(1.0, contrasts[k]["p_raw"] * (len(order) - i)), 5)

    # binarized (mean >= 2) vs construction rule and vs the judge
    rule = {(c["query_id"], p["case_id"]): int(bool(p["relevant_construction_rule"]))
            for c in cells.values() for p in c["pooled_candidates"]}
    binz = {qb: int(g >= 2) for qb, g in mean_g.items()}
    third = {"vs_construction_rule": rule}
    if os.path.exists(a.judge):
        jl = json.load(open(a.judge))["pair_labels"]
        third["vs_llm_judge"] = {tuple(k.split("|")): v for k, v in jl.items()}
    comparisons = {}
    for name, other in third.items():
        sel = [qb for qb in binz if qb in other]
        b, o = [binz[qb] for qb in sel], [other[qb] for qb in sel]
        comparisons[name] = {"n": len(sel),
                             "agreement": round(float(np.mean([x == y for x, y in zip(b, o)])), 3),
                             "kappa": round(float(cohen_kappa_score(b, o)), 3)}

    out = {"design": "prespecified (§4.11), option A (team decision 2026-09-18): 2 blinded radiologists, 0-3 scale, "
                     "30 queries / de-duplicated 1,347 pool; IRB gate honored before grading",
           "kappa_between_raters": kappa,
           "ndcg10_by_configuration": rows,
           "f5_contrasts_vs_ii": contrasts,
           "binarized_ge2_comparisons": comparisons}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1)[:1500])
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
