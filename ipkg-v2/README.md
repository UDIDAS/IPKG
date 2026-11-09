# ipkg-v2/ — application-side audit instruments

Instruments and records from the application tree that the manuscript cites (Supplementary Appendix C and the
Data and code availability statement). Each file is the committed version, anonymised: local paths, host names
and team initials are replaced by placeholders or roles, and identifiers carrying initials are renamed
consistently across script, CSV and JSON. Nothing else was changed.

| Path | What it is |
|:--|:--|
| `ipkg_validate.py` | The 158-check clinical-correctness validator over the delivered case records, including the slice-gate and box test (checks V2.2–V2.4) |
| `validation/validation_report.json` | Its committed output: 128 pass, 20 fail, 10 skip, every failure classified |
| `ipkg_ui_walk.py` | The rendered-page walk over the rater page |
| `validation/ui/ui_walk.json`, `ui_walk.txt` | Its committed output (the rater page carries no accuracy figure) |
| `ipkg_organ_convention.py` | Convention A (organ with mass, host-attributed), defined once and imported by the builders |
| `build_cohort_columns.py` | Writes the convention-A summary behind Table A.3 (`results/segmentation/organ_convention_a_autonomous_summary_2026-10-04.json`) |
| `build_retrieval_tables.py` | Writes the 1,311-pool retrieval tables behind Table 5 (full-pool block), G.1 and G.2 (`results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json`) |
| `split_pack/` | The shared-cohort split pack: one patient list per collection with per-patient roles and twin exposure (`ipkg_split_pack.json`, `.csv`) and its builder |
| `cutover/README.md` | The cut-over record of the final rating round: the switch to the corrected build and the final-round submission times by rater code |

Re-running these needs the application tree and the delivered case packages, which are not part of this
repository; the single-organ case-study masks are derivative works of their collections and are not
redistributed. The committed outputs are the record. Files the cut-over record links to (checklist, change
note, rollback notes) are internal working notes and are not included.
