#!/usr/bin/env python3
"""Word twin of make_v12_dedup_cells.py: the de-duplicated 1,347-pool table cells as a
shareable .docx (same sources `results/retrieval/dedup/`, same rows/columns as the md memo;
Table C.19 rendered as a compact table instead of raw JSON).
-> docs/VKG_v12_dedup_table_cells_2026-09-17.docx"""
import json
import os
import subprocess

from docx import Document
from docx.shared import Pt, RGBColor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = os.path.join(ROOT, "results", "retrieval", "dedup")
OUT = os.path.join(ROOT, "docs", "VKG_v12_dedup_table_cells_2026-09-17.docx")

t8 = json.load(open(f"{D}/retrieval_tab8_stress.json"))
t9 = json.load(open(f"{D}/retrieval_tab9_stratified.json"))
bl = json.load(open(f"{D}/baselines_tab7_tab8.json"))
ip = json.load(open(f"{D}/retrieval_iii_prime.json"))
c19 = json.load(open(f"{D}/tab10_example_ranks.json"))
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)


def H(t, l=1):
    doc.add_heading(t, level=l)


def P(t, b=False, i=False, sz=10.5, sp=6):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold, r.italic, r.font.size = b, i, Pt(sz)
    p.paragraph_format.space_after = Pt(sp)


def T(headers, rows, fs=9, bold_rows=()):
    tb = doc.add_table(rows=1, cols=len(headers))
    tb.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        cc = tb.rows[0].cells[i]
        cc.text = ""
        r = cc.paragraphs[0].add_run(h)
        r.bold, r.font.size = True, Pt(fs)
    for ri, row in enumerate(rows):
        cells = tb.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(fs)
            r.bold = ri in bold_rows
    doc.add_paragraph()


def dcell(v):
    d = v.get("dmAP_vs_ii")
    if not isinstance(d, dict):
        return "—", "—"
    return f"{d['delta']:+.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]", str(d.get("p_holm"))


MET = ["P@5", "P@10", "mAP", "nDCG", "Spur.@10", "Mism.@10", "ΔmAP vs (ii) [95% CI]", "p (Holm)"]


def metric_rows(rows_dict, iii_prime=None):
    out = []
    for k, v in rows_dict.items():
        d, p = dcell(v)
        spur = "n/a" if k.startswith(("(v", "(vi", "(vii")) else v["spurious@10"]
        out.append([k, v["P@5"], v["P@10"], v["mAP"], v["nDCG"], spur, v["mismatch@10"], d, p])
    if iii_prime is not None:
        v = iii_prime
        d, p = dcell(v)
        out.append(["(iii′) imputed-absent (suppl.)", v["P@5"], v["P@10"], v["mAP"], v["nDCG"],
                    v["spurious@10"], v["mismatch@10"], d, p])
    return out


H("v12 hand-over — de-duplicated 1,347-pool table cells", 0)
P(f"Generated from results/retrieval/dedup/ at {sha}. Word twin of "
  "docs/VKG_v12_dedup_table_cells_2026-09-17.md (same sources, same cells); regenerate both after any "
  "dedup rerun (scripts/reports/make_v12_dedup_cells{_docx}.py).", i=True, sz=9.5)
P("Pool: 1,347 = 113 single-organ queries + 1,234 FLARE23 (78 query twins removed; "
  "corpora/corpus_flare23_kg_dedup.json). Baseline (v)–(vii′) cells: 656-case CT sub-pool "
  "(113 + 543 dedup FLARE23 with a CT), per-query depth-30 re-rank validated against the frozen "
  "Delta 113-pool numbers (third-decimal reproduction, all eight configurations).")

H("Draft Table 8 — primary controlled benchmark (113-case pool, reference relevance)")
T(["Configuration"] + MET, metric_rows(bl["tab7"]["rows"], ip["tab7_pool"]["(iii') imputed-absent"]), bold_rows=(1,))

H("Draft Table 9 — large-pool stress test (1,347-case de-duplicated pool)")
H("KG rows, full 1,347 pool", 2)
T(["Configuration"] + MET, metric_rows(t8["rows"], ip["tab8_pool_1425"]["(iii') imputed-absent"]), bold_rows=(1,))
H("All rows, 656-case CT sub-pool (KG rows re-run for the like-for-like Δ)", 2)
T(["Configuration"] + MET, metric_rows(bl["tab8_subpool"]["rows"]), bold_rows=(1,))

H("Draft Table 10 — stratified retrieval (de-duplicated pool)")
rows = []
for key, v in t9["strata"].items():
    dd = v.get("delta_obs_mAP")
    if isinstance(dd, dict):
        dtxt, ptxt = f"{dd['delta']:+.3f} [{dd['ci95'][0]:.3f}, {dd['ci95'][1]:.3f}]", str(dd.get("p_holm"))
    else:
        dtxt, ptxt = "0.000 (exact tie)", "—"
    for m in ("proposed", "coverage_blind"):
        a = v[m]
        rows.append([key if m == "proposed" else "", m, a["n_queries"], a["P@5"], a["P@10"], a["mAP"],
                     a["nDCG"], a.get("mismatch@10", "n/a"), dtxt if m == "proposed" else "",
                     ptxt if m == "proposed" else ""])
T(["Stratum", "Method", "n_q", "P@5", "P@10", "mAP", "nDCG", "Mism.@10", "Δobs mAP [95% CI]", "p (Holm)"], rows)

H("(iii′) rows for Table 10 (returned to the main table team-confirmed 09-18; 1,347 pool)", 2)
K3 = "(iii') imputed-absent"
rows = []
for ds in ("kits", "lits", "msd"):
    v = ip["tab9_a_within_dataset"][ds][K3]
    d, pp = dcell(v)
    rows.append([f"(a) within-dataset — {ds}", v["n_queries"], v["P@5"], v["P@10"], v["mAP"], v["nDCG"],
                 v["mismatch@10"], d, pp])
for key, lab in (("tab9_b_single_to_flare23", "(b) single-organ → FLARE23"),
                 ("tab9_c_flare23_to_single", "(c) FLARE23 → single-organ")):
    v = ip[key][K3]
    d, pp = dcell(v)
    rows.append([lab, v["n_queries"], v["P@5"], v["P@10"], v["mAP"], v["nDCG"],
                 v["mismatch@10"] if v["mismatch@10"] is not None else "n/a", d, pp])
T(["Stratum", "n_q", "P@5", "P@10", "mAP", "nDCG", "Mism.@10", "ΔmAP vs (ii) [95% CI]", "p"], rows)

H("Table C.19 — example query ranks (de-duplicated pool, size " + str(c19["pool_size"]) + ")")
rows = []
for q in c19["queries"]:
    for cfg in ("proposed", "coverage_blind", "imputed_absent"):
        r = q[cfg]
        rows.append([q["title"] if cfg == "proposed" else "", q["query"] if cfg == "proposed" else "", cfg,
                     r["first_relevant_flare23_rank"], r["rank_within_flare23_only"],
                     r["single_organ_candidates_above"], r["n_relevant_flare23_candidates"], r["n_ranked"]])
T(["Query", "id", "Configuration", "first relevant FLARE23 rank", "rank within FLARE23",
   "single-organ above", "n relevant FLARE23", "n ranked"], rows)

P("CANONICAL (team item 1): Table 8 row (vii) Δ = −0.263 [−0.290, −0.237] (this file); the frozen Delta file's "
  "−0.262 [−0.290, −0.236] is the same quantity within embedding float nondeterminism (identical mAP and Holm p). "
  "Spur.@10 = n/a for baseline rows per the caption (path-based; the JSONs keep the no-shared-organ rate).", i=True, sz=9.5)
P("(iii′) full columns for both pools: results/retrieval/dedup/retrieval_iii_prime.json. Study set: "
  "results/study/clinician_study_30q.json (final, UNGRADED — grades/κ/nDCG@10/F5 pend recruitment).", i=True, sz=9.5)

doc.save(OUT)
print(f"-> {OUT}")
