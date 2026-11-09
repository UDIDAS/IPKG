# Knowledge That Knows What It Cannot See

**Observability-Aware Ontology Grounding of Imaging Phenotypes for Cross-Dataset Reasoning in Abdominal CT**

Project: **IPKG** (imaging-phenotype knowledge graphs) — the framework's application carries the same name.

Public companion repository for the manuscript (Journal of Biomedical Informatics, under review, 2026).
It contains the **result files, the scripts that produced them, the shipped phenotype corpora
and knowledge graphs, and the duplicate-scan audits**, organized so the main results can be
reproduced from this repository alone on CPU, and the compute-heavy stages from public data.

**Repository state for the submitted paper:** tag **`paper-submission`** (created at submission;
release candidates are tagged `IPKG-v1-rc*`). Every table cell in the paper names its source
file below and in [`TABLE_MAP.md`](TABLE_MAP.md).

## What the paper shows (headline numbers, all reproducible here)

| Result | Value | Source |
|:--|:--|:--|
| Primary retrieval benchmark, predicted phenotypes + γ (Table 4, config ii), corrected relevance labels | mAP **0.967** (113-case pool); organ-conditioned baselines 0.880–0.930; random-order floor 0.880; Holm p = 6.5×10⁻⁴ for every F1 contrast | `results/retrieval/dedup/final_rows_2026-10-08.json` (inputs: `baselines_corrected_labels_2026-10-05.json`) |
| Large-pool stress test on the de-duplicated **1,311**-case pool (Table 5) | (ii) mAP **1.000** full pool and on the 638-case CT sub-pool (113 + 525 FLARE23 candidates with a CT); external baselines 0.237–0.476 on the sub-pool, Holm p = 6.5×10⁻⁴ | full pool `results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json`; sub-pool `final_rows_2026-10-08.json`, `ct_subpool_1311_2026-10-06.json` |
| Duplicate-scan censuses (Appendix B): **114 twins removed** from the cross-dataset pool (78 + 13 by the axis-normalized census + 11 by the CT-voxel second pass + 14 by the grid-census third pass, less two records removed in error and restored (FLARE23_0042, FLARE23_0102); complete census `flare23_twin_census_complete_2026-10-03.json` — all annotation-independent, mask-overlap- or exact-tumor-voxel-confirmed); FLARE23 re-shares **116/131** LiTS volumes; residual: **673** pool FLARE23 records (no CT in any release copy) are label- AND grid-screened — none shares a voxel grid with any query (census JSON `second_pass`/`third_pass` sections; elimination gate 4/4 PASS: `twin_elimination_gate_2026-09-24.json`; disclosed caveat: a resampled re-share evades the grid and CT-voxel tests equally) | pool 1,425 → **1,311** | `results/audit/` (eight audit files) |
| Statement-level report consistency (Table E.3) | **0.946** (573 of 606 statements; containment, closed rule, 0.872) | `results/summarization/statement_precision_table_e3.json` |
| LLM relevance-judge study (Table 6) | κ 0.669, judge-relevance mAP 0.983 | `results/llm_expert/llm_expert_study_qwen3_32b.json` |
| Q10 joint single-pass timing (Appendix F) | 21.3 s per case on 36 KiTS23 volumes, IQR 15.0–35.8 | `results/queries/q10_joint_timing_kits.json` |

## Layout

```
corpora/        per-patient phenotype records (GT + predicted, all datasets), the de-duplicated
                FLARE23 pool (1,196 records; the 1,311-pool version with 1,198 records is
                corpora/reissue_2026-10-03/), sub-site-corrected re-issues in corpora/reissue_2026-10-01/, and containment_v2/ side corpora (see its README)
kg_graphs/      built knowledge graphs (imaging_kg_*.ttl, unified_mmkg_*.json), schema,
                ontology mappings (SNOMED CT / NCIt), organ-volume atlas
queries/        the 10 SPARQL query families (.rq), the query-suite evaluators, runners
results/        every result JSON behind the paper's tables, by pipeline stage; the three
                duplicate-scan audit files are under results/audit/
scripts/        the producing scripts, mirrored by stage (segmentation / kg / retrieval /
                summarization / repair / llm_expert / audit / study / reports)
ipkg-v2/        application-side audit instruments: clinical-correctness validator (slice-gate and
                box test), rendered-page walk, convention-A scorer, 1,311-pool table builder, split
                pack, cut-over record (see ipkg-v2/README.md)
```

## Quick start (CPU, ~10 minutes)

```bash
pip install -r requirements.txt          # core: numpy scipy scikit-image scikit-learn nibabel rdflib
# 1,309-vintage cross-check of the retrieval graph rows (Tables 4, 5, G.1) on the shipped corpora:
export VKG_FLARE_CORPUS=$PWD/corpora/corpus_flare23_kg_dedup.json VKG_RES_DIR=/tmp/vkg_repro
python scripts/retrieval/retrieval_tables_789.py
python scripts/retrieval/retrieval_iii_prime.py
# Reproduce the query-suite verdicts (Table 3):
python queries/query_suite_eval.py
```
Expected (published labels, 1,309-vintage cross-check; the manuscript's values are the 1,311-case run in `results/retrieval/dedup/final_rows_2026-10-08.json`): Table 4 (ii) mAP 0.972 · Table 5 (ii) 0.999 full pool / 0.997 sub-pool, (iii) −0.061, (iv) −0.666 · (iii′) −0.264
[−0.279, −0.239] — byte-comparable to `results/retrieval/dedup/`. The corrected-label rows are re-scored by
`scripts/retrieval/baselines_corrected_labels.py` and `baselines_subpool_1311.py` (they need the baseline feature cache;
see `REPRODUCING.md`); both refuse to write unless their reproduction gates pass. Full instructions, including
the GPU stages and the audits: [`REPRODUCING.md`](REPRODUCING.md).

## Data availability

All imaging data are public releases: **FLARE23** (MICCAI FLARE 2023), **LiTS** (via MSD
Task03_Liver, which carries the original NIfTI headers), **KiTS23**, **MSD Pancreas**. This
repository ships derived per-patient phenotype records and graphs only — no image data, no PHI
(all sources are de-identified public challenge sets). The 576 semi-oracle predicted FLARE23
masks behind Table E.2 (the intermediate segmentation outputs the KG records are extracted from)
are available from the project data share
<https://drive.google.com/open?id=1uxsSQpWIWbxSDAiDrLJMHpJ4N5uOq4HQ>
(576 masks as `masks_predicted_flare23.zip`, `MANIFEST.md5` checksums, `burden_tercile_cutpoints.json`,
the FLARE23_0405 case-study mask, and the FLARE23 citation/licence notes, CC BY-NC-SA 4.0) — cite this
link plus the repository tag `paper-submission` (the submitted state) in the Data Availability statement.

## Clinician relevance study (§3.17 / §4.6)

The study design is prespecified in the manuscript (§3.17). The instantiated 30-query packet,
grading sheets, and interim judge labels are **withheld from this public snapshot until grading
completes**, to avoid contaminating the blinded raters; they will be added in a post-study
release together with the graded results and the design document.

## Determinism notes

Retrieval/statistics are seeded (query bootstrap B=5,000, sign-flip permutation B=20,000): seed
20261003 for the final Table 4 and Table 5 CT sub-pool contrasts (`baselines_corrected_labels.py`,
`baselines_subpool_1311.py`), seed 20260920 for the 1,309-vintage stratum (c) cross-check CIs (`tab9c_metric_cis.py`),
and seed 12345 elsewhere (other retrieval and closed-loop statistics, F5 analysis, study-set
instantiation). Two historical nondeterminism sources were
found and fixed and are documented in the scripts: a `PYTHONHASHSEED`-dependent tie-break in
the host-rule check, and a corpus-glob hazard (see `corpora/containment_v2/README.md`).

## License and citation

License: to be finalized by the authors before the repository is made public.
Cite as: *Knowledge That Knows What It Cannot See: Observability-Aware Ontology Grounding of Imaging Phenotypes for Cross-Dataset Reasoning in Abdominal CT.* Journal of Biomedical Informatics, under review, 2026. (Full citation added upon acceptance.)
