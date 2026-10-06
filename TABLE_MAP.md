# Paper table/figure → script → result file

Numbering follows the submitted draft. "Reproduce" points to the section of
[`REPRODUCING.md`](REPRODUCING.md); Level A = CPU from this repository alone.

| Paper | Content | Producing script(s) | Result file(s) | Reproduce |
|:--|:--|:--|:--|:--|
| Table 1 | Datasets, coverage regimes, resulting KGs | `scripts/kg/kg_build_graph.py`, `scripts/kg/build_predicted_corpus*.py` | `kg_graphs/unified_mmkg*.json` (node/edge counts), `corpora/*.json` | A2 |
| Table 2 | Ten query families (12 instantiations) | `queries/query_suite_eval.py` (semantics in docstring) | `results/queries/query_suite_tab6.json` | A3 |
| Table 3 | Organ-node fidelity (volume r / MAPE) | `scripts/kg/semioracle_flare23_volume_fidelity.py`, `scripts/repair/closed_loop_tab12_13.py` (autonomous col.) | `results/kg/semioracle_flare23_volume_fidelity.json`, `results/kg/ausam_3d_summary.json`, `results/repair/closed_loop_tab12_13.json` | C |
| Table 4 | Lesion-node fidelity (256 IoU-matched nodes) | `scripts/kg/lesion_node_fidelity_flare23.py` | `results/kg/lesion_node_fidelity_flare23.json` | C (needs the released masks) |
| Table 5 | Ontology mapping coverage (13/15) | `scripts/kg/export_ontology_mappings.py` | `results/kg/ontology_mappings_export_check.json`, `kg_graphs/ontology_mappings.json` | A2 |
| Table 6 | Reasoning suite, 113 reference-evaluable cases | `queries/query_suite_eval.py` | `results/queries/query_suite_tab6.json` | A3 |
| Table 7 | Reasoning suite, 576 FLARE23 predicted graphs | `queries/query_suite_flare23.py` | `results/queries/query_suite_flare23.json` | A3 |
| Table 8 | Primary retrieval benchmark (113 pool, all 8 configs + (iii′)) | `scripts/retrieval/baselines_tab7_tab8.py`, `retrieval_tables_789.py`, `retrieval_iii_prime.py` | `results/retrieval/dedup/baselines_tab7_tab8.json` (canonical, incl. (v)–(vii′) paired stats), `retrieval_tab7_primary.json`, `retrieval_iii_prime.json` | A1 (KG rows), C (baseline features) |
| Table 9 | Large-pool stress test — **1,309** de-duplicated pool; baselines on the **636**-case CT sub-pool | same three scripts | `results/retrieval/dedup/retrieval_tab8_stress.json` + `baselines_tab7_tab8.json` (`tab8_subpool`); stratum-(c) per-metric CIs: `tab9c_metric_cis.json` | A1 / C |
| Table 10 | Stratified retrieval incl. the (iii′) rows | `retrieval_tables_789.py`, `retrieval_iii_prime.py` | `results/retrieval/dedup/retrieval_tab9_stratified.json`, `retrieval_iii_prime.json` (`tab9_*`) | A1 |
| Table 11 | Statement-level report precision (containment v2) | `scripts/summarization/summ_metrics.py` (frozen), `scripts/kg/containment_reextraction_113_predicted.py` (v2 row + overall) | `results/summarization/summarization_metrics.json`, `results/kg/containment_reextraction_113_{gt,predicted}.json` | A4 / C |
| Table 12 | LLM relevance-judge study (Qwen3-32B) | `scripts/llm_expert/llm_expert_local.py`, `llm_expert_comparators.py` | `results/llm_expert/llm_expert_study_qwen3_32b.json`, `llm_expert_comparators_qwen3_32b.json` | C (32B judge, 2 GPUs) |
| Table 13 | Closed-loop evaluation (40 FLARE23 patients) | `scripts/repair/closed_loop_tab12_13.py`, `validator_perturbation_tab11.py` | `results/repair/closed_loop_tab12_13.json`, `validator_perturbation_tab11.json` | C |
| §3.11 host-rule check | App rule vs Eq. 3 agreement (0.9853/749) | `scripts/app_checks/check_host_rule_vs_eq3.py` | `results/host_rule_agreement.json` | C (needs the released masks) |
| §5.4 Q10 timing | Joint single-pass workflow timing | `scripts/queries/q10_joint_timing_kits.py` | `results/queries/q10_joint_timing_kits.json` (+ `q10_stage_timing_proxy.json`) | C |
| §4.1 audits | Duplicate-scan audit (58+1 twins) · extended LiTS audit (19 query twins) · full overlap census (114/131) | `scripts/audit/duplicate_scan_audit.py`, `lits_extended_audit.py`, `lits_flare23_overlap_census.py`, `lits_geometry_map.py` | `results/audit/duplicate_scan_audit.json`, `lits_extended_audit.json`, `lits_flare23_overlap_census.json`, `lits_geometry_map.json` | B |
| §5.7 clinician study | Prespecified design (packet withheld until grading completes) | `scripts/study/build_clinician_query_set.py`, `build_grading_sheets.py`, `analyze_clinician_grades.py` | design prespecified in manuscript §4.11; packet + graded results post-study | — |
| Connectivity note (§3.11/D4) | 6- vs 26-connectivity sensitivity | `scripts/kg/connectivity_sensitivity.py` | `results/kg/connectivity_sensitivity.json` | B/C |
| Figs 3, B.4–B.6 | Application screenshots | application UI (not in this repository) | — | — |

Canonical retrieval results: `results/retrieval/dedup/` — the de-duplicated **1,309**-pool reruns
(09-20 census + 09-22 CT-voxel second pass + 09-24 grid-census third pass;
`results/audit/axis_normalized_twin_census.json` incl. §second_pass/§third_pass — the paper's numbers).
The pre-dedup "frozen original" files were removed 2026-09-24 to prevent stale-version transcription;
tags `IPKG-v1-rc1..rc4` retain them. Canonical example: Table 8 row (vii) Δ = −0.263 [−0.290, −0.237].

## Additions in IPKG-v1-rc6 (1,311-case pool, corrected relevance labels)

| Paper item | Source file | Producer |
|:--|:--|:--|
| Table 6 (primary benchmark), all rows under the corrected relevance labels; same-organ random floor | `results/retrieval/dedup/baselines_corrected_labels_2026-10-05.json` (labels: `results/retrieval/inputs/relevance_labels_published_corrected_113.json`) | `scripts/retrieval/baselines_corrected_labels.py` |
| Table 7 lower block (CT sub-pool, 638 cases), both label sets; sub-pool candidate list (525 of 1,198) | `results/retrieval/dedup/baselines_subpool_1311_2026-10-06.json`, `ct_subpool_1311_2026-10-06.json` | `scripts/retrieval/baselines_subpool_1311.py`, `fetch_flare23_ct_subpool.py` |
| Pool restoration 1,309 → 1,311 and its effect on rankings | `results/retrieval/pool_restore_impact_2026-10-03.json` | `scripts/retrieval/pool_restore_impact.py` |
| Complete cross-collection twin census | `results/audit/flare23_twin_census_complete_2026-10-03.json` | `scripts/audit/twin_census_complete.py` |
| Split manifests and per-patient exposure | `results/audit/split_manifests_2026-10-01.json`, `promptfree_twin_exposure_2026-10-03.json` | `scripts/audit/rebuild_split_manifests.py`, `promptfree_twin_exposure.py` |
| Autonomous arm (TotalSegmentator organs + prompt-free tumor) and the 157-scan cohort table | `results/segmentation/promptfree_round2_2026-10-01.json`, `cohort157_rescore_2026-10-03.json` | `scripts/segmentation/combine_ts_promptfree.py`, `rescore_cohort157.py` |
| Organ Dice under convention A (organ with mass) | `results/segmentation/organ_convention_a_2026-10-03.json`, `organ_convention_a_flare576_2026-10-03.json` | `scripts/segmentation/organ_convention_a*.py` |
| Pancreatic sub-site correction and record re-issue | `results/kg/pancreas_subsite_msd_gt.json`, `subsite_reissue_2026-10-01.json`; `corpora/reissue_2026-10-01/` | `scripts/kg/pancreas_subsite.py`, `reissue_subsite.py` |
| Showcase-mask provenance (voxel-exact re-draw of both halves) | `results/audit/displayed_weights_check_2026-10-03.json`, `displayed_weights_base_organ_2026-10-05.json` | `scripts/audit/verify_displayed_weights.py` |
| TotalSegmentator release (2.17/2.18) | `results/audit/totalsegmentator_version_check_2026-10-06.json` | `scripts/audit/ts_version/ts_reproduce_showcase.py` |
| Checkpoint inventory with SHA-256 | `results/segmentation/checkpoint_manifest_2026-10-01.json` | `scripts/audit/complete_checkpoint_hashes.py` |
