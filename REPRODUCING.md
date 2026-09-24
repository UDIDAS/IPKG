# Reproducing the results

Three levels. Level A needs only this repository and CPU (~10 min total). Level B adds public
label downloads (no GPU). Level C are the compute stages (GPU + public CT downloads); every
Level-C output JSON is shipped, so C is verification, not a prerequisite.

Environment: Python ≥ 3.10, `pip install -r requirements.txt`. GPU stages additionally need
`torch`/`torchvision`/`transformers` with CUDA and (for segmentation) access to the gated
`facebook/sam3` checkpoint on Hugging Face.

Determinism: all statistics are seeded (bootstrap B=5,000 / permutation B=20,000, seed 12345).
Feature-based baselines are architecture-deterministic but can wobble ±0.001 on paired-Δ third
decimals across GPU hardware (documented in TABLE_MAP; the shipped JSONs are canonical).

## Level A — CPU, from the shipped corpora

**A1. Retrieval tables (Tables 8–10 KG rows, (iii′), C.19):**
```bash
export VKG_FLARE_CORPUS=$PWD/corpora/corpus_flare23_kg_dedup.json VKG_RES_DIR=/tmp/vkg_repro
python scripts/retrieval/retrieval_tables_789.py      # Tables 8/9/10 KG rows
python scripts/retrieval/retrieval_iii_prime.py       # (iii') columns, both pools
python scripts/retrieval/tab10_example_ranks.py       # Table C.19 example ranks
```
Compare `/tmp/vkg_repro/*.json` against `results/retrieval/dedup/` — headline gates:
Table 9 (ii) mAP 0.999 (full 1,309 pool; 0.997 on the 636-case CT sub-pool) · (iii) Δ −0.061 [−0.074, −0.050] · (iv) Δ −0.665 [−0.698, −0.629] · (iii′) Δ −0.268 [−0.288, −0.249] ·
Table 10 (c) mAP 0.650 (n=516) · C.19 ranks identical.

**A2. Graphs and ontology (Tables 1, 5):**
```bash
python scripts/kg/kg_build_graph.py corpora/corpus_flare23_kg_dedup.json   # rebuild a KG
python scripts/kg/export_ontology_mappings.py                              # 13/15 coverage
```

**A3. Query suite (Tables 2, 6, 7):**
```bash
python queries/query_suite_eval.py        # 113 cohort; SPARQL latency section is hardware-bound
python queries/query_suite_flare23.py     # 576 predicted FLARE23 graphs
```

**A4. Summarization (Table 11 frozen part):**
```bash
python scripts/summarization/summ_metrics.py
```
The containment-v2 row and the 0.944 overall come from
`results/kg/containment_reextraction_113_predicted.json` (Level C to regenerate; shipped).

## Level B — audits (public label downloads, CPU)

The LiTS geometry recovery and the twin audits need MSD **Task03_Liver `labelsTr`**
(~250 MB from the MSD release) and the FLARE23 ground-truth labels:
```bash
python scripts/audit/lits_geometry_map.py             # identity mapping volume-N = liver_N, all 131 at r>=0.99998
python scripts/audit/lits_extended_audit.py           # 19/21 LiTS queries are FLARE23 twins
python scripts/audit/lits_flare23_overlap_census.py   # 114/131 LiTS volumes inside FLARE23
python scripts/kg/connectivity_sensitivity.py         # 6- vs 26-connectivity: 8/802 flips
```
Paths at the top of each script point to the staged label directories; edit to your download
locations. Every candidate twin is confirmed by voxel-exact ground-truth label comparison —
the shipped `results/audit/*.json` list every pair and every rejection.

## Level C — GPU / CT stages (verification of shipped results)

| Stage | Script | Data needed | Shipped output |
|:--|:--|:--|:--|
| Baseline features + rows (Tables 8/9 (v)–(vii′)) | `scripts/retrieval/baselines_tab7_tab8.py` (`$VKG_DATA` layout in docstring) | KiTS/LiTS/MSD + FLARE23 CTs (public releases) | `results/retrieval/dedup/baselines_tab7_tab8.json` |
| Depth-30 rankings | `scripts/retrieval/export_rankings_113.py` | feature cache from above | (application bundle) |
| Q10 joint timing | `scripts/queries/q10_joint_timing_kits.py` | KiTS CTs + SAM3 | `results/queries/q10_joint_timing_kits.json` |
| Containment v2 | `scripts/kg/containment_reextraction_flare23.py`, `..._113_gt.py`, `..._113_predicted.py` | released masks / GT labels | `results/kg/containment_reextraction_*.json` |
| Segmentation & closed loop | `scripts/segmentation/*`, `scripts/repair/*` | full CT datasets + SAM3 | `results/segmentation/`, `results/repair/` |
| LLM judge | `scripts/llm_expert/llm_expert_local.py Qwen/Qwen3-32B` | 2×48 GB GPUs | `results/llm_expert/llm_expert_study_qwen3_32b.json` |

**Validation gate used throughout:** any re-run of the 113-pool baseline table must reproduce
the shipped rows to the third decimal (mAP/P@10/nDCG) before its numbers are used — the same
gate that caught a corpus-loading hazard during development (`corpora/containment_v2/README.md`).
