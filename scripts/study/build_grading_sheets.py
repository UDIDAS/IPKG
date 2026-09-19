#!/usr/bin/env python3
"""Blinded grading sheets for the §5.7 clinician relevance study (option A, team decision 2026-09-18).

From the frozen packet (results/study/clinician_study_30q.json): one CSV per rater with every
(query, pooled candidate) pair — query order and within-query candidate order independently
randomized per rater (seeded); items carry OPAQUE ids only. Raters see the two phenotype
summaries and a blank 0–3 grade column; never the configuration, rank, construction-rule flag,
or case id. KEY.csv (coordinator only — do NOT send to raters) maps item ids to case ids for
the CT viewing station.

⚠ IRB GATE (the study lead): grading must not begin before the IRB / not-human-subjects determination for
the public de-identified data is in hand.

-> results/study/grading_sheets/{rater1_sheet.csv, rater2_sheet.csv, KEY.csv,
   GRADING_INSTRUCTIONS.md}
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUTD = os.path.join(ROOT, "results", "study", "grading_sheets")
SEEDS = {"rater1": 20260918, "rater2": 20260919}

packet = json.load(open(f"{ROOT}/results/study/clinician_study_30q.json"))
cells = packet["queries"]

os.makedirs(OUTD, exist_ok=True)
key_rows = []
for rater, seed in SEEDS.items():
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cells))
    rows = []
    for qi, ci in enumerate(order):
        c = cells[ci]
        cand_order = rng.permutation(len(c["pooled_candidates"]))
        for cj, pi in enumerate(cand_order):
            p = c["pooled_candidates"][pi]
            item = f"{rater[-1]}-{qi+1:02d}-{cj+1:02d}"
            rows.append({"item_id": item, "query_no": qi + 1,
                         "query_phenotype": c["query_summary"],
                         "candidate_phenotype": p["summary"],
                         "grade_0_to_3": "", "notes": ""})
            key_rows.append({"rater": rater, "item_id": item, "cell": c["cell"], "block": c["block"],
                             "query_id": c["query_id"], "candidate_id": p["case_id"]})
    fp = f"{OUTD}/{rater}_sheet.csv"
    with open(fp, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0]))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"{rater}: {len(rows)} items ({len(cells)} queries) -> {fp}")

with open(f"{OUTD}/KEY.csv", "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(key_rows[0]))
    wcsv.writeheader()
    wcsv.writerows(key_rows)
print(f"KEY (coordinator only) -> {OUTD}/KEY.csv")

open(f"{OUTD}/GRADING_INSTRUCTIONS.md", "w").write("""# Grading instructions — clinician relevance study (§5.7, option A)

**Do not begin before the IRB / not-human-subjects determination is confirmed by the study lead.**

You will grade, for each item, whether the CANDIDATE patient is clinically relevant to the
QUERY patient as a retrieved similar case, using the two phenotype descriptions and the CT
images provided by the coordinator for that item. Items are in random order; do not discuss
grades with the other rater until both sheets are returned.

Scale (0–3; anchors PROPOSED — the study lead to confirm before distribution):
  3 — highly relevant: same organ involvement and closely matching tumor burden, multiplicity
      and containment; would cite as a matched prior case
  2 — relevant: shared organ involvement with most phenotype aspects matching
  1 — marginally relevant: shared organ or partial phenotype overlap only
  0 — not relevant: no meaningful phenotype correspondence

Grade every row (no blanks). Use `notes` for anything you want on record (image quality,
ambiguity, disagreement with the description).

Analysis (prespecified): quadratic-weighted κ between raters (overall + blocks); nDCG@10 per
configuration with the mean of the two grades as gain; F5 paired contrasts vs configuration
(ii), Holm-corrected; binarized (≥2) agreement with the construction rule and the LLM judge.
""")
print(f"instructions -> {OUTD}/GRADING_INSTRUCTIONS.md")
