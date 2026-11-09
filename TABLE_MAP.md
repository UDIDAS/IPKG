# Paper table → script → result file

Numbering follows the submitted manuscript (main Tables 1–6, supplementary Tables A.1–L.1). "Reproduce"
points to the section of [`REPRODUCING.md`](REPRODUCING.md); Level A = CPU from this repository alone.
Files under `ipkg-v2/` are the application-side instruments; see [`ipkg-v2/README.md`](ipkg-v2/README.md).
`python scripts/audit/check_draft_cells.py <manuscript.pdf>` checks every table below against these files, row by row.

## Main tables

| Paper | Content | Producing script(s) | Result file(s) | Reproduce |
|:--|:--|:--|:--|:--|
| Table 1 | Statement of Significance | — | — | — |
| Table 2 | Organ-node fidelity (volume r / MAPE): semi-oracle single-organ rows (each checkpoint's held-out list: LiTS 6, KiTS23 20, MSD 33) and FLARE23 10-case rows; spleen from the 576-case run; autonomous raw/repaired columns | `scripts/kg/analyze_3d_kg.py`, `scripts/kg/semioracle_flare23_volume_fidelity.py`, `scripts/repair/closed_loop_tab12_13.py` | `results/kg/ausam_3d_summary.json` (`node_fidelity`), `results/kg/semioracle_flare23_volume_fidelity.json`, `results/repair/closed_loop_tab12_13.json` (`tab3_autonomous_volume_fidelity`) | C |
| Table 3 | Reasoning suite, 113 reference-evaluable cases (bracketed values: earlier two-dataset pilot, not regenerated here) | `queries/query_suite_eval.py` | `results/queries/query_suite_tab6.json` | A3 |
| Table 4 | Primary retrieval benchmark (113 queries, corrected arm, Holm over F1) | `scripts/retrieval/emit_final_rows_2026-10-07.py` (inputs from `baselines_corrected_labels.py`) | `results/retrieval/dedup/final_rows_2026-10-08.json` (`tab:retrieval-primary`), `tab_retrieval_primary_2026-10-08.tex`; labels `results/retrieval/inputs/relevance_labels_published_corrected_113.json` | C (baseline feature cache) |
| Table 5 | Large-pool stress test, 1,311-case pool: full-pool graph rows (raw p) and the 638-case CT sub-pool block (Holm over F1) | full pool: `ipkg-v2/build_retrieval_tables.py`; sub-pool: `scripts/retrieval/baselines_subpool_1311.py`, `emit_final_rows_2026-10-07.py` | full pool: `results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json` (`tab:retrieval` → `corrected`); sub-pool: `final_rows_2026-10-08.json` (`tab:retrieval ct-subpool`), `tab_retrieval_ct_subpool_rows_2026-10-08.tex`; membership `ct_subpool_1311_2026-10-06.json` | C |
| Table 6 | LLM relevance judge (Qwen3-32B; Qwen2.5-7B column), configurations under judge labels with Holm over F4; oncologist case-study block | `scripts/llm_expert/llm_expert_local.py`, `llm_expert_comparators.py`, `scripts/repair/holm_f3_f4_2026-10-08.py` | `results/llm_expert/llm_expert_study_qwen3_32b.json`, `llm_expert_study.json` (Qwen2.5-7B), `llm_expert_comparators_qwen3_32b.json`, `results/repair/holm_f3_f4_2026-10-08.json` (F4); oncologist block: see Table K.1 | C (2×48 GB GPUs) |

## Supplementary tables

| Paper | Content | Producing script(s) | Result file(s) | Reproduce |
|:--|:--|:--|:--|:--|
| Table A.1 | Semi-oracle held-out 3D segmentation (organs on each checkpoint's test list; tumors) | `scripts/segmentation/seg_stats.py` (from `eval_ausam_3d.py`, `eval_tumor_3d.py`) | `results/segmentation/seg_stats.json`; LiTS organ also `organ_convention_a_2026-10-03.json` | C |
| Table A.2 | Text-prompted, reference slice-gated Dice | `scripts/training/train_organ_generic.py` (organ), `scripts/segmentation/train_tumor_incremental.py` (tumor, all-four stage) | `results/segmentation/organ_generic_kg.json` (`per_organ_dataset`), `tumor_incremental.json` (`s4_flare`) | C |
| Table A.3 | Autonomous substrate, 157-scan shared cohort (tumor raw/gated, twin-clean; TotalSegmentator organ; convention A in the caption) | `scripts/segmentation/rescore_cohort157.py`, `combine_ts_promptfree.py`; convention A: `ipkg-v2/build_cohort_columns.py` (scorer `ipkg-v2/ipkg_organ_convention.py`) | `results/segmentation/cohort157_rescore_2026-10-03.json` (tumor), `promptfree_round2_2026-10-01.json` (organ), `organ_convention_a_autonomous_summary_2026-10-04.json` (convention A, FLARE23 organ cells); FLARE23 54-scan twin-clean intervals (n = 14) `ipkg-v2/substrate/flare23_twinclean_14.json` | C |
| Table A.4 | Organ-volume fidelity on FLARE-Task2 | `scripts/kg/analyze_3d_kg.py` | `results/kg/ausam_3d_summary.json` (`node_fidelity/*/flare_task2`) | C |
| Table A.5 | Tumor Dice for one model on a growing dataset set | `scripts/segmentation/train_tumor_incremental.py` | `results/segmentation/tumor_incremental.json` | C |
| Table B.1 | Datasets, coverage regimes, resulting KGs | `scripts/kg/kg_build_graph.py`, `scripts/kg/build_predicted_corpus*.py` | `kg_graphs/unified_mmkg*.json` (node/edge counts), `corpora/*.json` | A2 |
| Table E.1 | Ten query families (12 instantiations) | `queries/query_suite_eval.py` (semantics in docstring), `queries/*.rq` | — | A3 |
| Table E.2 | Lesion-node fidelity (256 of 274 matched nodes) | `scripts/kg/lesion_node_fidelity_flare23.py` | `results/kg/lesion_node_fidelity_flare23.json` | C (released masks) |
| Table E.3 | Statement-level report consistency (573 of 606 statements) | `scripts/summarization/statement_precision_table_e3.py` (rows from `summ_metrics.py` and `containment_reextraction_113_predicted.py`) | `results/summarization/statement_precision_table_e3.json` | A4 |
| Table E.4 | Reasoning suite, 576 FLARE23 predicted graphs | `queries/query_suite_flare23.py` | `results/queries/query_suite_flare23.json` | A3 |
| Table G.1 | Retrieval stratified by observability, 1,311 pool, corrected arm (Holm over F2) | `ipkg-v2/build_retrieval_tables.py` | `results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json` (`tab:retrieval-strata` → `corrected`; its `delta_obs` is (ii) minus the ablation, the table prints the ablation minus (ii)) | C |
| Table G.2 | Rank of the intended FLARE23 target, 1,311 pool, corrected arm | `ipkg-v2/build_retrieval_tables.py` | `retrieval_tables_1311_2026-10-07.json` (`tab:cross-examples` → `corrected`) | C |
| Table G.3 | Validator response to controlled perturbation | `scripts/repair/validator_perturbation_tab11.py` | `results/repair/validator_perturbation_tab11.json` | C |
| Table I.1 | Repair of autonomous organ masks (n = 40) | `scripts/repair/closed_loop_tab12_13.py` | `results/repair/closed_loop_tab12_13.json` (`tab12_dice`) | C |
| Table I.2 | Closed-loop evaluation (n = 40; Holm over F3) | `scripts/repair/closed_loop_tab12_13.py`, `holm_f3_f4_2026-10-08.py` | `results/repair/closed_loop_tab12_13.json`, `holm_f3_f4_2026-10-08.json` (F3) | C |
| Table K.1 | Oncologist case-study evaluation | application survey | not in this repository (see the manuscript's Data and code availability statement) | — |
| Table L.1 | Ontology mapping coverage (13/15) | `scripts/kg/export_ontology_mappings.py` | `results/kg/ontology_mappings_export_check.json`, `kg_graphs/ontology_mappings.json` | A2 |

The `p_holm` fields inside `retrieval_tables_1311_2026-10-07.json` predate the step-down correction and are
not cited: the full-pool rows of Table 5 report raw p, F1 Holm values are in `final_rows_2026-10-08.json`, and
both F2 contrasts have raw p = 1.0.

## Analyses added after submission (not in the submitted manuscript)

| Analysis | Content | Producing script | Result file(s) | Reproduce |
|:--|:--|:--|:--|:--|
| Closed-world false negatives on restricted views (provisional table name "Table E.5" in the checker) | The 576 semi-oracle FLARE23 predicted graphs of Table E.4 restricted to one organ each (liver-only, kidney-only, pancreas-only, as LiTS / KiTS23 / MSD records carry them); Q2, Q3, Q4a, Q4b, Q5, Q7b under the three-valued evaluator and a closed-world evaluator that scores every unsupported constraint F, both against the full four-organ reference verdict; per family and restricted organ: IPKG N_T / N_F / N_U, closed-world N_T / N_F and closed-world false negatives (reference T, closed-world F); Q2, Q3, Q7b also split into observed-organ and hidden-organ instantiations | `queries/query_suite_restricted_views.py` | `results/queries/restricted_views_closed_world.json`, `restricted_views_closed_world_table.tex` (Table 3 format) | A3 |

## Audit instruments (Supplementary Appendix C) and supporting artifacts

| Item | Source file(s) | Producer |
|:--|:--|:--|
| Slice-gate and box test (checks V2.2–V2.4) and the 158-check clinical-correctness validator (128 pass / 20 fail / 10 skip) | `ipkg-v2/validation/validation_report.json` | `ipkg-v2/ipkg_validate.py` |
| Rendered-page walk (rater page carries no accuracy figure) | `ipkg-v2/validation/ui/ui_walk.json`, `ui_walk.txt` | `ipkg-v2/ipkg_ui_walk.py` |
| Organ-convention scorer | `results/segmentation/organ_convention_a_2026-10-03.json`, `organ_convention_a_flare576_2026-10-03.json`, `organ_convention_a_autonomous_summary_2026-10-04.json` | `scripts/segmentation/organ_convention_a*.py`, `ipkg-v2/ipkg_organ_convention.py` |
| Draft-cell checker: rebuilds every row of every table above from its result file and requires it, cell by cell and in order, under that table's caption in the manuscript PDF (`python scripts/audit/check_draft_cells.py <manuscript.pdf>`; exit 0 = every row matches; Table K.1 and Table 3's bracketed pilot values outside the check) | the result files of the tables above | `scripts/audit/check_draft_cells.py` |
| Duplicate-scan censuses (Appendix B) and the complete cross-collection twin census | `results/audit/duplicate_scan_audit.json`, `lits_extended_audit.json`, `lits_flare23_overlap_census.json`, `axis_normalized_twin_census.json`, `flare23_twin_census_complete_2026-10-03.json`, `twin_elimination_gate_2026-09-24.json` | `scripts/audit/` |
| Shared-cohort split pack with per-patient exposure | `ipkg-v2/split_pack/ipkg_split_pack.json` (`.csv`); `results/audit/split_manifests_2026-10-01.json`, `promptfree_twin_exposure_2026-10-03.json` | `ipkg-v2/split_pack/build_split_pack.py`; `scripts/audit/rebuild_split_manifests.py`, `promptfree_twin_exposure.py` |
| Cut-over record of the final rating round | `ipkg-v2/cutover/README.md` | — |
| NLQ tumor-volume routing audit (§5.4) | `results/app_checks/nlq_tumor_volume_routing_2026-10-12.json` | `scripts/app_checks/nlq_tumor_volume_routing.py` |
| Corrected application verified at the rated build | `results/app_checks/verify_ipkg-corrected-2026-10-09_rendered.json`, `live_ranking_ipkg-corrected-2026-10-09.json` | `scripts/app_checks/` |
| Pool restoration 1,309 → 1,311 and its effect on rankings | `results/retrieval/pool_restore_impact_2026-10-03.json` | `scripts/retrieval/pool_restore_impact.py` |
| Pancreatic sub-site correction and record re-issue | `results/kg/pancreas_subsite_msd_gt.json`, `subsite_reissue_2026-10-01.json`; `corpora/reissue_2026-10-01/` | `scripts/kg/pancreas_subsite.py`, `reissue_subsite.py` |
| Showcase-mask provenance (voxel-exact re-draw) | `results/audit/displayed_weights_check_2026-10-03.json`, `displayed_weights_base_organ_2026-10-05.json` | `scripts/audit/verify_displayed_weights.py` |
| TotalSegmentator release (2.17/2.18; re-run masks and per-organ differences) | `results/audit/ts_version/` (`summary_2026-10-08.json`) | `scripts/audit/ts_version/ts_reproduce_showcase.py` |
| Checkpoint inventory with SHA-256 | `results/segmentation/checkpoint_manifest_2026-10-01.json` | `scripts/audit/complete_checkpoint_hashes.py` |
| Holm re-adjustment of result files whose generators cannot be re-run | `results/audit/holm_readjust_log_2026-10-08.json` | `scripts/audit/readjust_holm_stepdown_2026-10-08.py` |
| Q10 latency detail (Appendix F; joint pass on 36 KiTS23 volumes) | `results/queries/q10_joint_timing_kits.json` (+ `q10_stage_timing_proxy.json`) | `scripts/queries/q10_joint_timing_kits.py` |
| Clinician relevance study (§3.17; prespecified, grades pending) | packet, sheets and graded results withheld until grading completes | `scripts/study/build_clinician_query_set.py`, `build_grading_sheets.py`, `analyze_clinician_grades.py` |

Repository checks not reported in the manuscript: host-rule vs Eq. 3 agreement
(`results/host_rule_agreement.json`, `scripts/app_checks/check_host_rule_vs_eq3.py`) and the 6- vs 26-connectivity
sensitivity (`results/kg/connectivity_sensitivity.json`, `scripts/kg/connectivity_sensitivity.py`).

> **Vintage:** `results/retrieval/dedup/retrieval_tab7_primary.json`, `retrieval_tab8_stress.json`,
> `retrieval_tab9_stratified.json`, `baselines_tab7_tab8.json`, `tab9c_metric_cis.json` and `tab10_example_ranks.json`
> are cross-checks at the 1,309-case vintage under the published labels (Level A1 reproduces them). The manuscript's
> values are the 1,311-case corrected-arm files named in the tables above.
