#!/usr/bin/env python3
"""Word companion, regenerated to the v15 numbering (2026-09-18 PM, per YL's option-A email):
the stratified table RETURNS to Table 10 WITH the (iii') rows (YL item 2); summarization ->
Table 11, LLM-judge -> Table 12; closed loop stays supplementary; the clinician study is
Section 5.7 (option A decided: two radiologists, 0-3 scale, prespecified design; judge-on-packet
reported only as a labelled interim if grades miss submission). Also closes YL items 1 and 3:
canonical (vii) = -0.263 [-0.290, -0.237] (dedup file); baseline Spur.@10 = n/a per caption.
-> docs/VKG_v7_Tables6-9_Results_Reference.docx  (filename kept for continuity; the content
   header states the v14 provenance. Supersedes the 2026-09-17 v12 companion.)

Numbering is PROVISIONAL until the renumbered draft confirms it (derived from the 09-18 list,
not read off a v14 pdf). Companion to results/README.md §7 "2026-09-18 hand-over";
cells memo: docs/VKG_v12_dedup_table_cells_2026-09-17.{md,docx}.
"""
import json
import os

from docx import Document
from docx.shared import Pt, RGBColor

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "VKG_v7_Tables6-9_Results_Reference.docx")

t7 = json.load(open(f"{RES}/retrieval/dedup/retrieval_tab7_primary.json"))
t8 = json.load(open(f"{RES}/retrieval/dedup/retrieval_tab8_stress.json"))
t9 = json.load(open(f"{RES}/retrieval/dedup/retrieval_tab9_stratified.json"))
bl = json.load(open(f"{RES}/retrieval/dedup/baselines_tab7_tab8.json"))
ip = json.load(open(f"{RES}/retrieval/dedup/retrieval_iii_prime.json"))
q6 = json.load(open(f"{RES}/queries/query_suite_tab6.json"))
sm = json.load(open(f"{RES}/summarization/summarization_metrics.json"))
cv = json.load(open(f"{RES}/kg/containment_reextraction_113_predicted.json"))

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)
RED, GREY = (150, 0, 0), (90, 90, 90)


def H(t, l=1):
    doc.add_heading(t, level=l)


def P(t, b=False, i=False, c=None, sz=10.5, sp=6):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold, r.italic, r.font.size = b, i, Pt(sz)
    if c:
        r.font.color.rgb = RGBColor(*c)
    p.paragraph_format.space_after = Pt(sp)


def T(headers, rows, fs=9):
    tb = doc.add_table(rows=1, cols=len(headers))
    tb.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        cc = tb.rows[0].cells[i]
        cc.text = ""
        rr = cc.paragraphs[0].add_run(h)
        rr.bold, rr.font.size = True, Pt(fs)
    for row in rows:
        cs = tb.add_row().cells
        for i, v in enumerate(row):
            cs[i].text = ""
            rr = cs[i].paragraphs[0].add_run(str(v))
            rr.font.size = Pt(fs)
            if str(v).startswith("TBD"):
                rr.font.color.rgb = RGBColor(*RED)
                rr.bold = True
    doc.add_paragraph()


def dstr(row):
    d = row.get("dmAP_vs_ii")
    if not isinstance(d, dict):
        return "—", "—"
    return f"{d['delta']:+.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]", f"{d.get('p_holm')}"


doc.add_heading("VKG v15 — Tables Results Reference (2026-09-24; de-duplicated 1,309-case pool, third pass)", 0)
P("Drop-in values for the draft's retrieval/query tables on the FINAL de-duplicated pool "
  "(1,309 = 113 + 1,196 FLARE23 after removing all 116 query twins — 78 original + 13 census + 11 CT-voxel + 14 grid-census; results/retrieval/dedup/). "
  "Numbering follows the v15 layout (YL 09-18 PM email): the stratified table RETURNS to Table 10 with "
  "the (iii′) rows; summarization → Table 11; LLM-judge → Table 12; closed loop supplementary; clinician "
  "study = §5.7 (option A). CONFIRMED against the 09-18 23:51 draft (Tables 10/11/12 as mapped; Table 13 stays in the MAIN text). One draft catch: the Table 9 caption must read the 1,309-case de-duplicated corpus (any 1,425/1,347/1,334/1,323 is stale). Values unchanged from v14 except the "
  "restored Table 10 and its (iii′) rows. Stats: "
  "paired query bootstrap B=5,000; sign-flip permutation B=20,000; Holm per family. Supersedes the "
  "2026-09-17 v12 companion; hand-over: results/README.md §7 \"2026-09-18 hand-over\".",
  i=True, c=GREY, sp=10)

P("⚠ Before filling: do NOT quote the v6 γ-ablation contrast (mAP 0.714, spurious@10 0.32). Those rows "
  "reproduce only under a legacy organ-universe artifact in which every KiTS query ties at score 0. "
  "Corrected contrasts: γ ablation −0.039 (reference pool) / −0.061 (1,347-case de-duplicated pool); "
  "organ-agnostic −0.479 / −0.667 with Mismatch@10 0.55–0.56. And replace every \"689-case sub-pool\" "
  "with 656, every organ-agnostic-baseline \"0.24–0.45\" with (v) 0.409 · (vi) 0.460 · (vii) 0.245 · "
  "(vii′) 0.248.", b=True, c=RED, sp=10)

# ---------------- v14 numbering map ----------------
H("v15 numbering map (provisional — YL 09-18 PM email; confirm against the v15 pdf)", 1)
T(["v15 table", "Former", "Content", "Values live in"],
  [["Table 6", "Table 6", "Ten-family query suite, 113 cohort", "this doc + results/queries/query_suite_tab6.json"],
   ["Table 7", "Table 7", "FLARE23 reasoning suite (576 cases)", "results/README.md §4d + queries/query_suite_flare23.json"],
   ["Table 8", "Table 8", "Primary controlled retrieval benchmark (113 pool)", "this doc + retrieval/dedup/"],
   ["Table 9", "Table 9", "Large-pool stress test (1,309 + 636-case CT sub-pool)", "this doc + retrieval/dedup/"],
   ["Table 10", "Table 10 (returns)", "Stratified retrieval + (iii′) rows (YL item 2)", "this doc + retrieval/dedup/retrieval_{tab9_stratified,iii_prime}.json"],
   ["Table 11", "Table 11", "Summarization statement precision (containment v2)", "this doc + kg/containment_reextraction_113_predicted.json"],
   ["Table 12", "Table 12", "LLM relevance-judge study (Qwen3-32B)", "results/llm_expert/llm_expert_study_qwen3_32b.json"],
   ["§5.7", "§5.8", "Clinician relevance study (option A; interim judge-on-packet exists)", "study/clinician_study_30q.json + llm_expert/llm_judge_clinician_packet_Qwen3_32B.json"],
   ["Table 13", "Table 13", "Closed loop (CONFIRMED main text, p.20 of the 09-18 23:51 draft)", "results/repair/closed_loop_tab12_13.json + README §6b"]])

# ---------------- Table 6 ----------------
H("Draft Table 6 — ten-family query suite (§5.4)", 1)
P("Caption fills: n = 113 cases (reference-evaluable cohort; system verdicts from predicted records, "
  "reference labels from GT records). Q2/Q3/Q7b instantiated per organ (liver, kidney, pancreas) and "
  "pooled. Indet. = N_U/(N_T+N_F+N_U).")
name6 = {"Q2 organ-specific lookup (pooled liver/kidney/pancreas)": "Q2 Organ-specific lookup (pooled)",
         "Q3 phenotype retrieval: largest burden bin (pooled)": "Q3 Largest size-bin (pooled)",
         "Q4a compositional: hepatic>5cm AND renal lesion": "Q4a Hepatic>5cm ∧ renal",
         "Q4b compositional: hepatic>5cm AND splenic lesion": "Q4b Hepatic>5cm ∧ splenic",
         "Q5 cross-organ involvement (>=2 organs)": "Q5 Cross-organ involvement",
         "Q7a high tumor burden (observed anatomy)": "Q7a High tumor burden",
         "Q7b multifocal disease in one organ (pooled)": "Q7b Multifocal in one organ"}
rows = []
for r in q6["cohort_113"]["rows"]:
    f = lambda x: "—" if x is None else x
    ir = r["indet_rate"]
    rows.append([name6[r["family"]], r["N_T"], r["N_F"], r["N_U"], f(r["precision"]),
                 f(r["recall"]), f(r["F1"]), f"{100*ir:.1f}" if ir is not None else "—"])
m = q6["cohort_113"]["macro_over_evaluable"]
rows.append(["Macro (evaluable families)", "", "", "", m["macro_P"], m["macro_R"], m["macro_F1"], ""])
rows.append(["Q9 Similar-case retrieval", "→ Table 8/9 (mAP 0.972 / 0.999)", "", "", "", "", "", ""])
lat = q6["q10_latency_sparql_ms"]["flare23 (1,312 cases)"]["across_queries"]
rows.append(["Q10 Latency (SPARQL stage)",
             f"median {lat['median_ms']} ms, IQR {lat['IQR_ms'][0]}–{lat['IQR_ms'][1]} ms "
             "(1,312-case graph); 11–16 ms per-dataset graphs. Single-pass end-to-end: 21.3 s/case "
             "IQR [15.0, 35.8] (36 KiTS, A6000; q10_joint_timing_kits.json) / KS 12-study cold path "
             "86.2 s IQR 42.0–150.4; deployed-app upload-to-report remains app-side",
             "", "", "", "", "", ""])
T(["ID / Family", "N_T", "N_F", "N_U", "Precision", "Recall", "F1", "Indet. (%)"], rows)
P("Q4a/Q4b/Q5 are 100% indeterminate on single-organ cases BY OBSERVABILITY — the designed abstention, "
  "not a failure. On the FLARE23 reference graph they are determinate: Q4a 0 T / 1,312 F; Q4b 0 T / "
  "1,310 F (2 U); Q5 0 T / 1,311 F (1 U); Q7a 203 T / 1,109 F. No FLARE23 case has tumors in ≥2 core "
  "organs. System-vs-reference P/R/F1 for those rows: Table 7 (README §4d).", i=True, c=GREY)

# ---------------- Table 8 ----------------
H("Draft Table 8 — primary controlled retrieval benchmark (§5.5)", 1)
P("Caption fills: n_q = 113 queries, 113 reference-evaluable candidates (112 per query); ranking on "
  "predicted phenotypes, relevance from reference-mask (GT) phenotypes; family F1.")
rows = []
label7 = {"(i) Reference phenotypes + gamma (upper bound)": "(i) Reference + γ (upper bound)",
          "(ii) Predicted phenotypes + gamma (proposed)": "(ii) Predicted + γ (proposed)",
          "(iii) Predicted, coverage-blind (no gamma)": "(iii) Predicted, coverage-blind",
          "(iv) Predicted, organ-agnostic baseline": "(iv) Predicted, organ-agnostic"}
for k, v in t7["rows"].items():
    d, p = dstr(v)
    rows.append([label7[k], v["P@10"], v["mAP"], v["nDCG"], v["mismatch@10"], d, p])
bl_label = {"(v) Radiomics-vector (organ-agnostic)": "(v) Radiomics-vector (organ-agnostic)",
            "(vi) Radiomics-vector (organ-conditioned)": "(vi) Radiomics-vector (organ-conditioned)",
            "(vii) Image-embedding (organ-agnostic)": "(vii) Image-embedding (organ-agnostic)",
            "(vii') Image-embedding (organ-conditioned)": "(vii′) Image-embedding (organ-conditioned)"}
for k, v in bl["tab7"]["rows"].items():
    if k in label7:
        continue                                          # KG rows already listed from the primary file
    d, p = dstr(v)
    rows.append([bl_label[k], v["P@10"], v["mAP"], v["nDCG"], v["mismatch@10"], d, p])
d_ip, p_ip = dstr(ip["tab7_pool"]["(iii') imputed-absent"])
v_ip = ip["tab7_pool"]["(iii') imputed-absent"]
rows.append(["(iii′) Predicted, imputed-absent (suppl.)", v_ip["P@10"], v_ip["mAP"], v_ip["nDCG"],
             v_ip["mismatch@10"], d_ip, p_ip])
T(["Configuration", "P@10", "mAP", "nDCG", "Mism.@10", "mAP vs (ii) [95% CI]", "p (Holm)"], rows)
P("All (v)–(vii′) cells incl. paired Δ/CI/p and Mism.@10 are from the per-query local CT rerun "
  "(retrieval/dedup/baselines_tab7_tab8.json) — validated: these 113-pool rows reproduce the frozen "
  "Delta aggregates to the third decimal. CANONICAL (YL item 1): row (vii) Δ = −0.263 [−0.290, −0.237] "
  "(this source); the frozen Delta file's −0.262 [−0.290, −0.236] is the same quantity within embedding "
  "float nondeterminism (identical mAP and Holm p).", i=True, c=GREY)

# ---------------- Table 9 ----------------
H("Draft Table 9 — large-pool stress test on the DE-DUPLICATED pool (§5.5)", 1)
P("Caption fills: 113 queries × 1,309-case de-duplicated corpus (113 + 1,196 FLARE23 after removing "
  "all 116 query twins; kidneys pooled); construction-defined relevance. Baseline rows (v)–(vii′) are "
  "on the 636-case CT sub-pool (113 + 523 dedup FLARE23 candidates with a CT in the release copy), "
  "with KG rows re-run on the same sub-pool for the like-for-like Δ.")
rows = []
for k, v in t8["rows"].items():
    d, p = dstr(v)
    rows.append([label7[k], v["P@5"], v["P@10"], v["mAP"], v["nDCG"],
                 v["spurious@10"], v["mismatch@10"], d, p])
d_ip, p_ip = dstr(ip["tab8_pool_1425"]["(iii') imputed-absent"])
v_ip = ip["tab8_pool_1425"]["(iii') imputed-absent"]
rows.append(["(iii′) Predicted, imputed-absent (suppl.)", v_ip["P@5"], v_ip["P@10"], v_ip["mAP"],
             v_ip["nDCG"], v_ip["spurious@10"], v_ip["mismatch@10"], d_ip, p_ip])
for k, v in bl["tab8_subpool"]["rows"].items():
    d, p = dstr(v)
    lab = (bl_label.get(k) or label7.get(k, k)) + "  [656 sub-pool]"
    spur = "n/a" if k in bl_label else v["spurious@10"]
    rows.append([lab, v["P@5"], v["P@10"], v["mAP"], v["nDCG"],
                 spur, v["mismatch@10"], d, p])
T(["Configuration", "P@5", "P@10", "mAP", "nDCG", "Spur.@10", "Mism.@10",
   "mAP vs (ii) [95% CI]", "p (Holm)"], rows)
P("Full-pool rows (i)–(iv)/(iii′): 1,347 candidates. Sub-pool rows: 656 candidates (features need a "
  "CT; the sub-pool KG rows are listed for the like-for-like paired Δ). Spur.@10 = n/a for the baseline "
  "rows (YL item 3 CONFIRMED: path-based per the caption, undefined for non-graph methods; the JSONs "
  "keep the no-shared-organ rate under that key if a number is ever wanted).", i=True, c=GREY)

# ---------------- Table 10 (former 11) ----------------
H("Draft Table 11 — summarization statement precision, containment v2 (§5.6)", 1)
P("Statement-level factual consistency (predicted-phenotype summary vs GT, n=113). Containment row = "
  "v2 (re-extraction from the regenerated fidelity-gated masks; was 0.844 degenerate-boundary).")
pt = sm["per_type_statement_precision"]
cvp = cv["table11_containment_precision_v2"]
rows = [["present", pt["present"]["n"], pt["present"]["precision"]],
        ["has_tumor", pt["has_tumor"]["n"], pt["has_tumor"]["precision"]],
        ["burden_cat", pt["burden_cat"]["n"], pt["burden_cat"]["precision"]],
        ["multiplicity", pt["multiplicity"]["n"], pt["multiplicity"]["precision"]],
        ["containment (v2)", cvp["n_statements"], cvp["precision"]],
        ["anatomic_location", pt["anatomic_location"]["n"], pt["anatomic_location"]["precision"]],
        ["OVERALL (v2)", 662, "0.944  (= 625/662)"],
        ["tumor-presence", 113, sm["tumor_presence_statement_precision"]]]
T(["Statement type", "n", "Precision"], rows)
P("Overall v2 = (113 + 109 + 104 + 101 + 95 + 103)/662 = 625/662 = 0.944. Containment v2 per dataset: "
  f"kits {cvp['per_dataset']['kits']['precision']} · lits {cvp['per_dataset']['lits']['precision']} · "
  f"msd {cvp['per_dataset']['msd']['precision']}. The frozen summarization_metrics.json intentionally "
  "still reads 0.940 / 0.844 (side-file convention). Unobserved-organ discipline 1.000 by construction.",
  i=True, c=GREY)

# ---------------- Table 11 (former 12) ----------------
H("Draft Table 12 — LLM relevance-judge study (§4.10)", 1)
P("Qwen3-32B, blinded, structured summaries only (20 stratified queries / 469 pairs; κ on 320 "
  "class-balanced pairs): κ 0.669 (substantial), raw agreement 0.834, expert-relevance mAP 0.983, "
  "specificity 0.994. Comparators under the same judge: expert P@10 0.89 for (ii)/(iii)/(iii′) vs "
  "0.325–0.73 for (iv)–(vii′); F4 paired contrasts all Holm p ≤ 3e-4 ((iv) excluded, 11/20 queries "
  "unscorable). Files: llm_expert/llm_expert_study_qwen3_32b.json, "
  "llm_expert/llm_expert_comparators_qwen3_32b.json. Values unchanged from v12 — renumber only.")

# ---------------- Supplementary: stratified (former Table 10) ----------------
H("Draft Table 10 — stratified retrieval, RETURNED to the main sequence with the (iii′) rows (§5.5, family F2)", 1)
s = t9["strata"]
rows = []
for key, lab, mism in [("(a) within-dataset (identity check)", "(a) Within-dataset (control)", True),
                       ("(b) single-organ -> FLARE23", "(b) Single-organ → FLARE23", True),
                       ("(c) FLARE23 -> single-organ", "(c) FLARE23 → single-organ", False)]:
    v = s[key]
    d = v.get("delta_obs_mAP")
    if isinstance(d, dict):
        dtxt, ptxt = f"{d['delta']:+.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]", f"{d.get('p_holm')}"
    else:
        dtxt, ptxt = "0.000 (exact tie — check passes)", "—"
    for mm, ml in (("proposed", "proposed"), ("coverage_blind", "coverage-blind")):
        a = v[mm]
        rows.append([lab if mm == "proposed" else "", ml, a["n_queries"], a["P@10"], a["mAP"],
                     a["mismatch@10"] if mism else "n/a",
                     dtxt if mm == "proposed" else "", ptxt if mm == "proposed" else ""])
K3 = "(iii') imputed-absent"
for ds in ("kits", "lits", "msd"):
    v3 = ip["tab9_a_within_dataset"][ds][K3]
    d3, p3 = dstr(v3)
    rows.append([f"(a) — {ds}", "imputed-absent (iii′)", v3["n_queries"], v3["P@10"], v3["mAP"],
                 v3["mismatch@10"], d3, p3])
for key3, lab3 in (("tab9_b_single_to_flare23", "(b) Single-organ → FLARE23"),
                   ("tab9_c_flare23_to_single", "(c) FLARE23 → single-organ")):
    v3 = ip[key3][K3]
    d3, p3 = dstr(v3)
    rows.append([lab3, "imputed-absent (iii′)", v3["n_queries"], v3["P@10"], v3["mAP"],
                 v3["mismatch@10"] if v3["mismatch@10"] is not None else "n/a", d3, p3])
T(["Stratum", "Method", "n_q", "P@10", "mAP", "Mism.@10", "Δ vs (ii) [95% CI]", "p (Holm)"], rows)
P("The within-dataset identity check PASSES (rankings identical in all three datasets). Δobs is also "
  "exactly 0 in strata (b) and (c) — structural: the coverage-blind observed-absent penalty is a "
  "rank-preserving rescaling on any coverage-homogeneous pool. §5.5's sentence predicting positive Δobs "
  "in (b)/(c) must be reworded: the observability gap is a mixed-pool phenomenon (Table 9 row (iii), "
  "−0.061), and the strata localize where it vanishes. Mism.@10 is defined for single-organ queries, "
  "hence n/a in (c).", b=True, c=RED)

# ---------------- Supplementary: closed loop (former Table 13) ----------------
H("Draft Table 13 — closed loop (§6; CONFIRMED main text in the 09-18 23:51 draft)", 1)
P("Moves out of the main sequence unchanged: raw / generic post-proc / ontology repair triple F1 "
  "0.355 / 0.683 / 0.457 ((c)−(b) −0.226, p_Holm 1.5e-4); volume MAPE 120.2 % → 34.0 % → 29.9 %; "
  "retrieval mAP 0.606 → 0.424 → 0.399 (reference ceiling 0.918). The loop closes at the node level "
  "only. Values: results/repair/closed_loop_tab12_13.json; §6.1 rewrite: "
  "docs/VKG_v8_closed_loop_findings_2026-09-13.md. Renumber only — no cell changes.")

# ---------------- still red (v14 state) ----------------
H("Still red at v15 (2026-09-18 PM) — everything pipeline-side is filled", 1)
P("All v7-era red items above the line are CLOSED: closed-loop Tabs (09-13 GPU round), baseline "
  "cells + paired stats (09-17 local re-rank), Qwen3-32B judge (09-13), Q10 single-pass IQR (09-16), "
  "Table 10 containment v2 → overall 0.944 (09-17b). Remaining:", i=True, c=GREY, sp=4)
for item in ["Clinician study GRADES (§5.7, tab:clinician) — OPTION A DECIDED (YL 09-18): two board-certified "
             "abdominal radiologists, 0–3 scale, prespecified design unchanged; grading sheets + F5 harness are "
             "in scripts/study/; grading start gated on the IRB / not-human-subjects determination; if grades "
             "miss submission, report the judge-on-packet as a labelled interim "
             "(llm_expert/llm_judge_clinician_packet_Qwen3_32B.json), never as a substitute",
             "Data availability: release channel for the 576 Table 4 semi-oracle masks (complete on the "
             "Drive exchange masks_predicted_flare23/, 229.7 MiB) — public link vs archive DOI",
             "12 application screenshots after the §3.11 consistency fixes — app UI",
             "Deployed-app Q10 upload-to-report latency — app UI",
             "Case-B manuscript edit pass + bib — Overleaf, not this bundle",
             "App scorer weighting confirmation — app owner"]:
    P("⏳ " + item, c=RED, sp=2)

doc.save(OUT)
print(f"-> {OUT}")
