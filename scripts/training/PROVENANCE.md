# Provenance of the four training modules (verbatim copies, no edits)

These are imported by `scripts/segmentation/{eval_ausam_3d,eval_crossdataset_organ,train_tumor_incremental,run_pancreas_sam3}.py` and were missing from the delivery (the co-author 09-29 §6 item 5). Each file here is byte-identical to the training repository's branch `AUSAM-MMKG`, path `src/scripts/<file>`.

| File | Last commit touching it | Git blob | sha256 |
|:--|:--|:--|:--|
| `train_organ_generic.py` | 3b01029 (2026-08-05) | c5848b524fcd | 7d5a8abc82a749c5… |
| `train_tumor_generic_v3.py` | 3b0ed95 (2026-08-03) | 6d2ced4c271f | 87ef2599c3ba1d8d… |
| `run_pancreas_nifti.py` | fcaf78c (2026-06-27) | ed77ba5ebbc6 | 623481307cc58d56… |
| `run_flare.py` | fcaf78c (2026-06-27) | c90f1740551c | 2e42c2818c679711… |

Paths inside them point at the training cluster's `/home/user/SWOG` and `/scratch/user/acm_data`; set them to your data roots. `train_organ_generic.py` holds the shared `patient_split` (seed 42) used by every split in `results/audit/split_manifests_2026-10-01.json`.
