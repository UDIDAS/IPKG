# cutover/ - preparing to put the corrected build in front of the oncologists

**Prepared 2026-10-04. Nothing here has been executed, and
the cutover is the advisor's call, not ours.**

The frozen build is still serving: a local server, behind
`<app-host>`, 177 files unchanged since the baseline of
2026-09-29, 5 submissions and 26 answers. `guard_live_app.py` has been read
and **not** changed.

## What is here

| file | what it is |
|:--|:--|
| `../check_cutover.py` | the runnable half. 13 checks, three states (PASS / FAIL / PENDING), writes `check_cutover.json`. It refuses to open port 8505 |
| `CUTOVER_CHECKLIST.md` | the gate and the eleven ordered steps, with the five current blockers itemised |
| `CHANGE_NOTE_FOR_CLINICIANS.md` | the one page the oncologists read. **Draft - do not send**; its first paragraph depends on the advisor's arm decision and both wordings are in the file |
| `TRIPWIRE_REBASELINE.md` | what has to be re-baselined in `guard_live_app.py`, and in what order, so the application is never unguarded |
| `ROLLBACK.md` | the path back, how long it takes, and what happens to answers submitted in between |
| `OUTSTANDING_AT_CUTOVER.md` | what is still owed by the pipeline team and by the advisor, and which of it a clinician can see |
| `GUARD_EXTRA_TREES.txt` | the directories outside the serving tree that the serving app reads. Read by CUT-11; must also go into the guard's `DATA_TREES` |
| `APPROVED_ARM.txt` | does not exist yet. One word, from the advisor: `interactive` or `autonomous` |
| `round1_survey_snapshot/` | does not exist yet. Taken at checklist step 6, with the server stopped |

## Read in this order

1. `CUTOVER_CHECKLIST.md` - what must be true, and the five things that are not.
2. `OUTSTANDING_AT_CUTOVER.md` - the two advisor decisions the rest waits on.
3. `TRIPWIRE_REBASELINE.md` and `ROLLBACK.md` before touching anything.
4. `CHANGE_NOTE_FOR_CLINICIANS.md` last, because its first paragraph is decided
   by (2).

## The three findings worth knowing before reading anything else

1. **No git revision reproduces the frozen build byte for byte.** The closest,
   `ipkg-survey-2026-09-22`, matches 20 of 25 code and config files; the other
   five differ by line endings alone, which is enough to make the tripwire
   report drift forever after a checkout rollback. The rollback source has to
   be a file copy taken before cutover. `ROLLBACK.md` §0.

2. **A re-rating deletes the earlier rating.** `survey_store` upserts on
   `(kind, dataset, case_id, reviewer_key)` - no build, no arm - and deletes
   the submission's answers before rewriting them. "Your previous answers are
   kept" is true only because of the snapshot in checklist step 6.
   `ROLLBACK.md` §3.

3. **The serving build reads two directories in the tree it is replacing** -
   `round2/final_cases_round2` (14 records) and
   `round2/records_reissue/corpora` (4,818 withdrawals, which decide which
   sub-site words the retrieval panel prints). The guard does not watch them.
   `TRIPWIRE_REBASELINE.md` §4.

## Run it

```
cd <app-tree>
set PYTHONIOENCODING=utf-8
<app-workspace>\.venv\Scripts\python.exe check_cutover.py --pixels
```

`--no-sweeps` for the fast version, `--stage post` after the new build is
serving, `--json` for the record. It starts nothing, stops nothing and writes
only `cutover/check_cutover.json`.

---

## The invitation and the change note: what actually happened (2026-10-10)

The study lead asked for this to be recorded here as the sign-off. The honest
entry is that **there is no sign-off, because nothing was sent.**

- **No invitation was sent and the change note has gone to nobody.** The reserved
  domain did not move during the cut-over, so the link each reviewer already had
  from round one still resolves - to the corrected build. R2 used that link.
- **R2's four submissions landed on 2026-10-09 between 17:00:50 and 17:05:37**,
  on `ipkg-corrected-2026-10-09`, store stamp `corrected-2026-10-08`.
- **So R2 rated the corrected build without having been told what changed.** He
  was not told the sub-site words were corrected, that the pool label moved from
  1,334 to 1,311, or that this was a different build from the one he opened a
  draft against on 4 October. Study lead, 2026-10-09: *"If the rater who has
  already submitted did not have the note, I will say so in the paper."* He did
  not have it.
- **The change note is still unsigned.** `CHANGE_NOTE_FOR_CLINICIANS.md` carries
  its own "do not send" banner and this README still labels it a draft. The
  blocker written into that banner - that the corrected build was not yet serving
  - has cleared; the remaining blocker is agreement on the wording, which is the
  study lead's. It cannot be marked sent-today on our initiative: sending it is
  an outward step to three clinicians and it needs his word first.

**What this costs if it is not fixed before the other two start.** R2 is rated
against an un-notified build and R1 and R3 would be rated against a notified one,
which makes the three raters non-comparable on the one variable the round exists
to control. The cheap fix is to send the note to all three today, on the study
lead's word, and to record R2 as having received it after submitting.

Checklist steps 2 and 11 therefore remain not-run, and that is now a statement of
fact rather than a pending task.
