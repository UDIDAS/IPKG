#!/usr/bin/env python3
"""F5 analysis of the clinician relevance study, as fixed by the exclusion protocol of 2026-10-08
(tag clinical-exclusion-protocol-2026-10-08; results/study/clinical_exclusion_protocol_2026-10-08.md).

GRADED PACKET. The radiologists hold the 20 September draw committed at fc15ed9 (30 queries, 755 pairs per rater).
The packet, KEY and sheet templates are read from that commit and checked against the sha256 values recorded in the
protocol; the script refuses to run on any other version (the grading_sheets/ files at HEAD are the 24 September
redraw, which was never sent). Returned sheets must carry exactly the fc15ed9 item ids and query/candidate texts.

EXCLUSION. The 24 pairs of results/study/twin_pairs_packet_fc15ed9_2026-10-08.json (sha256-pinned; 3 self-twin) are
excluded identically for every configuration; every outcome is reported WITH the exclusion (primary) and WITHOUT it.
Grades of the remaining pairs are used as collected. A blank grade is allowed only on an excluded pair.

OUTCOMES (protocol wording):
  - nDCG@10 per configuration (ii, iii, iii', vi, vii') over the graded candidates that remain in each configuration's
    original top-10 list, excluded ranks removed and NOT back-filled from lower ranks; gain = mean of the two raters'
    0-3 grades (2^g - 1); the ideal list is the best remaining graded pairs of the query's pooled union, to the same
    depth. --gap-mode close (default): remaining items keep their order and close up; --gap-mode keep: an excluded
    position keeps its rank with zero gain (study lead to confirm which reading is intended);
  - quadratic-weighted kappa between raters over the graded pairs, overall and per block;
  - agreement and kappa of binarized grades (mean >= 2) with the construction rule and with the LLM judge
    (judge labels cover exactly the fc15ed9 pairs);
  - F5 contrasts vs (ii) over the queries scorable for both members of a contrast, the number reported; paired query
    bootstrap 95 % CI (B = 5,000) and sign-flip permutation p = (k + 1)/(B + 1) (B = 20,000); step-down Holm within F5.

  python analyze_clinician_grades.py --returned <dir with rater1_sheet.csv, rater2_sheet.csv> [--gap-mode close|keep]
-> results/study/clinician_study_results.json
"""
import argparse
import csv
import hashlib
import io
import json
import os
import subprocess

import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PACKET_COMMIT = "fc15ed9"
PINNED = {  # sha256 at fc15ed9, as recorded in clinical_exclusion_protocol_2026-10-08.md
    "results/study/grading_sheets/KEY.csv": "56d2a273302f1b22ea51860cd52678dfddde09b13ed62fbfda8c71c64760bc61",
    "results/study/grading_sheets/rater1_sheet.csv": "3f06502b74aec2899ed752627929272b14e977d42381d00678719b2f283adac2",
    "results/study/grading_sheets/rater2_sheet.csv": "909ec5533b23167a5ce0ba824a5bbf5e0b10d1cf144c202b4590ac6a9c625446",
    "results/study/clinician_study_30q.json": "89f89f7706b806404b9f67f0fb6b25360cc39cdebd9f77a08634dc891b823ade"}
EXCL = os.path.join(ROOT, "results", "study", "twin_pairs_packet_fc15ed9_2026-10-08.json")
EXCL_SHA = "ea81dd6355ccf05d3f151f265758a698e9aa582394ac058a2e7c9f208f2c36c1"
CFGS = ["ii", "iii", "iii_prime", "vi", "vii_prime"]
B_BOOT, B_PERM, SEED = 5000, 20000, 12345


def at_packet(path):
    raw = subprocess.run(["git", "show", f"{PACKET_COMMIT}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
    got = hashlib.sha256(raw).hexdigest()
    if got != PINNED[path]:
        raise SystemExit(f"{path} at {PACKET_COMMIT} has sha256 {got}, protocol records {PINNED[path]}: refusing")
    return raw.decode("utf-8")


def dcg(gains):
    return sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))


def stepdown_holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, run, m = {}, 0.0, len(items)
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p)); out[k] = run
    return out


def analyse(cells, grades, block_of, rule, judge, excluded, gap_mode):
    keep = {qb for qb in grades if qb not in excluded}
    mean_g = {qb: sum(grades[qb].values()) / 2.0 for qb in keep}
    kappa = {}
    for scope in ("all", "A", "B", "C"):
        sel = sorted(qb for qb in keep if scope == "all" or block_of[qb] == scope)
        r1, r2 = [grades[qb]["rater1"] for qb in sel], [grades[qb]["rater2"] for qb in sel]
        kappa[scope] = {"n_pairs": len(sel),
                        "quadratic_kappa": round(float(cohen_kappa_score(r1, r2, weights="quadratic")), 3) if sel else None,
                        "raw_agreement": round(float(np.mean([x == y for x, y in zip(r1, r2)])), 3) if sel else None}
    pq = {c: {} for c in CFGS}
    for cell, c in cells.items():
        q = c["query_id"]
        pooled = sorted((mean_g[(q, p["case_id"])] for p in c["pooled_candidates"] if (q, p["case_id"]) in keep), reverse=True)
        for cfg in CFGS:
            lst = (c["top10"].get(cfg) or [])[:10]
            if not lst:
                continue
            if gap_mode == "close":
                gains = [mean_g[(q, b)] for b in lst if (q, b) in keep]
            else:
                gains = [mean_g[(q, b)] if (q, b) in keep else 0.0 for b in lst]
            depth = len(gains)
            ideal = pooled[:depth]
            if depth and dcg(ideal) > 0:
                pq[cfg][cell] = dcg(gains) / dcg(ideal)
    rows = {cfg: {"n_queries_scorable": len(v), "ndcg10": round(float(np.mean(list(v.values()))), 3) if v else None}
            for cfg, v in pq.items()}
    rng = np.random.default_rng(SEED); contrasts = {}
    for cfg in ("iii", "iii_prime", "vi", "vii_prime"):
        common = sorted(k for k in pq[cfg] if k in pq["ii"])
        d = np.array([pq[cfg][k] - pq["ii"][k] for k in common])
        if not len(d):
            contrasts[cfg] = {"n_queries_scorable_both": 0}; continue
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B_BOOT)])
        signs = rng.choice([-1.0, 1.0], size=(B_PERM, len(d)))
        k = int(np.sum(np.abs((signs * d).mean(axis=1)) >= abs(d.mean())))
        contrasts[cfg] = {"n_queries_scorable_both": len(d), "delta_ndcg10": round(float(d.mean()), 3),
                          "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
                          "p_raw": (k + 1) / (B_PERM + 1)}
    adj = stepdown_holm({c: v["p_raw"] for c, v in contrasts.items() if "p_raw" in v})
    for c, p in adj.items():
        contrasts[c]["p_holm_F5"] = round(p, 6)
    binz = {qb: int(g >= 2) for qb, g in mean_g.items()}
    comps = {}
    for name, other in (("vs_construction_rule", rule), ("vs_llm_judge", judge)):
        sel = sorted(qb for qb in binz if qb in other)
        b, o = [binz[qb] for qb in sel], [other[qb] for qb in sel]
        comps[name] = {"n": len(sel), "agreement": round(float(np.mean([x == y for x, y in zip(b, o)])), 3) if sel else None,
                       "kappa": round(float(cohen_kappa_score(b, o)), 3) if sel and len(set(b + o)) > 1 else None}
    return {"n_pairs_analysed": len(keep), "kappa_between_raters": kappa, "ndcg10_by_configuration": rows,
            "f5_contrasts_vs_ii": contrasts, "binarized_ge2_comparisons": comps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--returned", required=True, help="directory with the two filled sheets")
    ap.add_argument("--gap-mode", choices=("close", "keep"), default="close")
    ap.add_argument("--judge", default=os.path.join(ROOT, "results", "llm_expert", "llm_judge_clinician_packet_Qwen3_32B.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "study", "clinician_study_results.json"))
    a = ap.parse_args()
    key = list(csv.DictReader(io.StringIO(at_packet("results/study/grading_sheets/KEY.csv"))))
    packet = json.loads(at_packet("results/study/clinician_study_30q.json"))
    cells = {c["cell"]: c for c in packet["queries"]}
    ex_raw = open(EXCL, "rb").read()
    if hashlib.sha256(ex_raw).hexdigest() != EXCL_SHA:
        raise SystemExit("exclusion list does not match the sha256 recorded in the protocol: refusing")
    excluded = {(p["query_id"], p["candidate_id"]) for p in json.loads(ex_raw)["pairs"]}
    grades = {}
    for rater in ("rater1", "rater2"):
        template = {r["item_id"]: r for r in csv.DictReader(io.StringIO(at_packet(f"results/study/grading_sheets/{rater}_sheet.csv")))}
        filled = {r["item_id"]: r for r in csv.DictReader(open(os.path.join(a.returned, f"{rater}_sheet.csv"), encoding="utf-8"))}
        if set(filled) != set(template):
            raise SystemExit(f"{rater}: returned item ids differ from the fc15ed9 packet ({len(set(filled) ^ set(template))} differ)")
        for iid, t in template.items():
            if (filled[iid]["query_phenotype"], filled[iid]["candidate_phenotype"]) != (t["query_phenotype"], t["candidate_phenotype"]):
                raise SystemExit(f"{rater} item {iid}: texts differ from the fc15ed9 packet: wrong sheet version")
        for k in key:
            if k["rater"] != rater:
                continue
            qb = (k["query_id"], k["candidate_id"]); g = filled[k["item_id"]]["grade_0_to_3"].strip()
            if g == "":
                if qb in excluded:
                    continue
                raise SystemExit(f"blank grade on a non-excluded pair: {rater} item {k['item_id']}")
            if g not in ("0", "1", "2", "3"):
                raise SystemExit(f"grade outside 0-3: {rater} item {k['item_id']}: {g!r}")
            grades.setdefault(qb, {})[rater] = int(g)
    incomplete = [qb for qb, v in grades.items() if len(v) != 2 and qb not in excluded]
    if incomplete:
        raise SystemExit(f"{len(incomplete)} non-excluded pairs lack one rater's grade")
    grades = {qb: v for qb, v in grades.items() if len(v) == 2}
    block_of = {(k["query_id"], k["candidate_id"]): k["block"] for k in key}
    rule = {(c["query_id"], p["case_id"]): int(bool(p["relevant_construction_rule"])) for c in cells.values() for p in c["pooled_candidates"]}
    judge = {tuple(k.split("|")): v for k, v in json.load(open(a.judge))["pair_labels"].items()} if os.path.exists(a.judge) else {}
    out = {"design": ("prespecified; two blinded radiologists, 0-3 scale, 30 queries; packet fc15ed9 (755 pairs, drawn on the "
                      "1,347-record pool; all 30 queries remain in the 1,311 pool); protocol tag clinical-exclusion-protocol-2026-10-08"),
           "gap_mode": a.gap_mode,
           "with_exclusion_primary": analyse(cells, grades, block_of, rule, judge, excluded, a.gap_mode),
           "without_exclusion": analyse(cells, grades, block_of, rule, judge, set(), a.gap_mode),
           "n_excluded_pairs": len(excluded)}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("gap_mode", "n_excluded_pairs")}), "->", a.out)


if __name__ == "__main__":
    main()
