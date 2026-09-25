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
| Primary retrieval benchmark, predicted + γ (Table 8, config ii) | mAP **0.972** (113-case pool) | `results/retrieval/dedup/baselines_tab7_tab8.json` |
| Large-pool stress test on the de-duplicated **1,309**-case pool (Table 9) | mAP **0.997** (636-case CT sub-pool; full pool 0.999); baselines ≤ 0.483, all Holm p ≤ 3.5×10⁻⁴ below (ii) | same file + `retrieval_tab8_stress.json` |
| Duplicate-scan audits: **116 twins removed** from the cross-dataset pool (78 + 13 by the axis-normalized census + 11 by the CT-voxel second pass + 14 by the grid-census third pass — all annotation-independent, mask-overlap- or exact-tumor-voxel-confirmed); FLARE23 re-shares **116/131** LiTS volumes; residual: **673** pool FLARE23 records (no CT in any release copy) are label- AND grid-screened — none shares a voxel grid with any query (census JSON `second_pass`/`third_pass` sections; elimination gate 4/4 PASS: `twin_elimination_gate_2026-09-24.json`; disclosed caveat: a resampled re-share evades the grid and CT-voxel tests equally) | pool 1,425 → **1,309** | `results/audit/` (eight audit files) |
| Statement-level report precision (Table 11, containment v2) | overall **0.944** (containment 0.872) | `results/kg/containment_reextraction_113_predicted.json` |
| LLM relevance-judge study (Table 12) | κ 0.669, judge-relevance mAP 0.983 | `results/llm_expert/llm_expert_study_qwen3_32b.json` |
| Host-rule vs Eq. 3 agreement (§3.11) | **0.9853** over 749 tumor components | `results/host_rule_agreement.json` |
| End-to-end Q10 workflow timing | 21.3 s/case median, IQR [15.0, 35.8] | `results/queries/q10_joint_timing_kits.json` |

## Layout

```
corpora/        per-patient phenotype records (GT + predicted, all datasets), the de-duplicated
                FLARE23 pool (1,196 records), and containment_v2/ side corpora (see its README)
kg_graphs/      built knowledge graphs (imaging_kg_*.ttl, unified_mmkg_*.json), schema,
                ontology mappings (SNOMED CT / NCIt), organ-volume atlas
queries/        the 10 SPARQL query families (.rq), the query-suite evaluators, runners
results/        every result JSON behind the paper's tables, by pipeline stage; the three
                duplicate-scan audit files are under results/audit/
scripts/        the producing scripts, mirrored by stage (segmentation / kg / retrieval /
                summarization / repair / llm_expert / audit / study / reports)
```

## Quick start (CPU, ~10 minutes)

```bash
pip install -r requirements.txt          # core: numpy scipy scikit-image scikit-learn nibabel rdflib
# Reproduce the retrieval tables on the shipped corpora (Tables 8-10 KG rows + (iii')):
export VKG_FLARE_CORPUS=$PWD/corpora/corpus_flare23_kg_dedup.json VKG_RES_DIR=/tmp/vkg_repro
python scripts/retrieval/retrieval_tables_789.py
python scripts/retrieval/retrieval_iii_prime.py
# Reproduce the query-suite verdicts (Tables 6):
python queries/query_suite_eval.py
```
Expected: Table 8 (ii) mAP 0.972 · Table 9 (ii) 0.999 full pool / 0.997 sub-pool, (iii) −0.061, (iv) −0.666 · (iii′) −0.264
[−0.279, −0.239] — byte-comparable to `results/retrieval/dedup/`. Full instructions, including
the GPU stages and the audits: [`REPRODUCING.md`](REPRODUCING.md).

## Data availability

All imaging data are public releases: **FLARE23** (MICCAI FLARE 2023), **LiTS** (via MSD
Task03_Liver, which carries the original NIfTI headers), **KiTS23**, **MSD Pancreas**. This
repository ships derived per-patient phenotype records and graphs only — no image data, no PHI
(all sources are de-identified public challenge sets). The 576 semi-oracle predicted FLARE23
masks behind Table 4 (the intermediate segmentation outputs the KG records are extracted from)
are available in the shared Drive folder
[`masks_predicted_flare23/`](https://drive.google.com/drive/folders/1INMkyghy0mouV0o61qWvBcOuAQ-Wt0Pl)
(576 x `.nii.gz`, 229.7 MiB, with the FLARE23 citation/license notes) — cite this link plus the
repository tag `IPKG-v1-rc5` in the Data Availability statement.

## Clinician relevance study (§4.11 / §5.7)

The study design is prespecified in the manuscript (§4.11). The instantiated 30-query packet,
grading sheets, and interim judge labels are **withheld from this public snapshot until grading
completes**, to avoid contaminating the blinded raters; they will be added in a post-study
release together with the graded results and the design document.

## Determinism notes

Retrieval/statistics are seeded (query bootstrap B=5,000, sign-flip permutation B=20,000,
seed 12345; study-set instantiation seed 12345). Two historical nondeterminism sources were
found and fixed and are documented in the scripts: a `PYTHONHASHSEED`-dependent tie-break in
the host-rule check, and a corpus-glob hazard (see `corpora/containment_v2/README.md`).

## License and citation

License: to be finalized by the authors before the repository is made public.
Cite as: *Knowledge That Knows What It Cannot See: Observability-Aware Ontology Grounding of Imaging Phenotypes for Cross-Dataset Reasoning in Abdominal CT.* Journal of Biomedical Informatics, under review, 2026. (Full citation added upon acceptance.)
