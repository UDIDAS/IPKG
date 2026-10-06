#!/usr/bin/env python3
"""Mechanical draft<->repo cell-diff checker (process fix, 2026-09-23).

Motivation: every number problem the study lead found in draft rounds v30-v35 was one of (a) a stale
value surviving from an earlier freeze, or (b) arithmetic done on rounded numbers instead of
pasting the generated cell.  Nothing automated ever crossed the repo<->manuscript boundary;
this script does.  Run it on EVERY new draft PDF before anyone reads numbers out of it:

  ~/.conda/envs/llmft/bin/python scripts/audit/check_draft_cells.py docs/<draft>.pdf

It checks the PDF text two ways:

  1. STALE SCAN - tokens known to be superseded (each entry says why and what replaced it).
     Any hit is flagged with its page; a hit may be legitimate narrative ("pool 1,334 -> 1,323"),
     so flags mean "a human looks at this page", not "wrong".  Exit code 1 if any flag.
  2. EXPECTED SCAN - every distinctive display cell parsed live from the current cells memo
     (docs/VKG_v12_dedup_table_cells_2026-09-17.md, itself generated from results/retrieval/dedup/
     by scripts/reports/make_v12_dedup_cells.py), plus pool arithmetic from the census JSON,
     plus Table 1 graph counts computed from kg_graphs/*.json at run time.  Missing tokens are
     informational (the draft may not carry every cell yet); found tokens list their pages.

Known limitation (stated, not hidden): this is a token-presence check.  A CURRENT value pasted
into the WRONG cell is invisible to it - that class still needs the human table-by-table read.
Delta cells are matched as ordered triples (delta, ci_lo, ci_hi within a short window), which
survives bracket/dash formatting differences between the memo and the PDF.
"""
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEMO = os.path.join(ROOT, "docs", "VKG_v12_dedup_table_cells_2026-09-17.md")
CENSUS = os.path.join(ROOT, "results", "audit", "axis_normalized_twin_census.json")

# --- tokens that are superseded everywhere a table/abstract cell is concerned -------------
# (regex on normalized text, why it is stale, what replaced it)
STALE = [
    (r"(?<![\d.])1334(?![\d.])", "pool size before the 2026-09-22 second twin pass", "1,323"),
    (r"(?<![\d.])1347(?![\d.])", "pool size before the 2026-09-20 axis-normalized census", "1,323"),
    (r"(?<![\d.])1425(?![\d.])", "pool size before the original 78-twin dedup", "1,323"),
    (r"(?<![\d.])1221(?![\d.])", "FLARE23 corpus size before the second twin pass", "1,210"),
    (r"(?<![\d.])1215(?![\d.])", "four-organ coverage denominator guess (v34 round)", "1,204/1,210"),
    (r"-0\.062(?![\d])", "rounded-subtraction Delta for stress (iii); paired-mean is canonical", "-0.061"),
    (r"(?<![\d.])11762(?![\d.])", "Table 1 node count from the pre-renal-concept graph build", "11,764"),
    (r"(?<![\d.])21248(?![\d.])", "Table 1 edge count from the pre-renal-concept graph build", "21,450"),
    (r"(?<![\d.])113\.1(?![\d])", "retired end-to-end latency (s); current the co-author timings differ", "99.26/89.77 s (the co-author 09-21)"),
    (r"FLARE23[_ ]0286", "superseded case study (0286 = case_00067, a KiTS twin)", "FLARE23_0405"),
    (r"(?<![\d.])691(?![\d.])", "unscanned-residual count as first stated (pre-second-pass miscount)", "687 (= 1,210 - 523)"),
    # --- superseded by the 2026-09-24 third pass (grid census of the no-CT stratum: 14 more twins) ---
    (r"(?<![\d.])1323(?![\d.])", "pool size before the 2026-09-24 third twin pass (grid census)", "1,309"),
    (r"(?<![\d.])1210(?![\d.])", "FLARE23 corpus size before the third twin pass", "1,196"),
    (r"(?<![\d.])1204(?![\d.])", "four-organ coverage numerator at the 1,210 corpus", "1,190/1,196"),
    (r"(?<![\d.])493(?![\d.])", "stratum (c) n at the 1,323 pool", "479"),
    (r"0\.655(?![\d])", "stratum (c) (ii) mAP at the 1,323 pool", "0.661"),
    (r"\+?0\.345(?![\d])", "stratum (c) (iii') Delta at the 1,323 pool", "+0.339"),
    (r"0\.735(?![\d])", "stress (iii') mAP at the 1,323 pool", "0.731"),
    (r"-0\.264(?![\d])", "stress (iii') Delta at the 1,323 pool", "-0.268"),
    (r"(?<![\d.])98/113", "queries-with-twins count at 102 removals", "112/113"),
    (r"(?<![\d.])23/36", "KiTS queries-with-twins at 102 removals", "35/36"),
    (r"(?<![\d.])19/21", "LiTS queries-with-twins at 102 removals", "21/21"),
    (r"(?<![\d.])687(?![\d.])", "no-CT residual phrasing: label-screened only (pre-grid-census)",
     "673 no-CT records remain, label- AND grid-screened (third pass removed 14)"),
    (r"(?<![\d.])102(?![\d.])\s*(?:twin|dupl|remov|same-scan)", "twin-removal total before the third pass", "116"),
]

# floats that are too common to be evidence of anything
NONDISTINCTIVE = {"0.0", "1.0", "0.000", "1.000", "0.00", "1.00"}


def normalize(text):
    text = (text.replace("−", "-").replace("–", "-").replace("—", "-")
                .replace(" ", " ").replace(" ", " ").replace(" ", " ")
                .replace("′", "'").replace("’", "'"))
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def memo_tokens():
    """(token_regex, human_label) for every distinctive cell in the current memo tables."""
    out, heading = [], ""
    for line in open(MEMO):
        if line.startswith("#"):
            heading = line.strip("# \n")
            continue
        if not line.startswith("|") or set(line.strip()) <= {"|", ":", "-", " "}:
            continue
        cells = [c.strip() for c in normalize(line).strip("|\n").split("|")]
        if not cells or cells[0] in ("Configuration", "Stratum"):
            continue
        row = cells[0] or "(cont.)"
        for cell in cells[1:]:
            m = re.match(r"^([+-]?\d\.\d{3})\s*\[([+-]?\d\.\d{3}),\s*([+-]?\d\.\d{3})\]", cell)
            if m:  # delta [ci_lo, ci_hi] -> ordered-triple regex, format-tolerant
                d, lo, hi = (re.escape(x) for x in m.groups())
                out.append((rf"{d}\D{{0,14}}{lo}\D{{0,8}}{hi}", f"{heading} | {row} | Delta+CI {m.group(0)}"))
                continue
            if re.fullmatch(r"[+-]?\d\.\d{2,3}", cell) and cell.lstrip("+-") not in NONDISTINCTIVE:
                out.append((rf"(?<![\d.]){re.escape(cell)}(?![\d.])", f"{heading} | {row} | {cell}"))
    return out


def artifact_tokens():
    out = []
    cen = json.load(open(CENSUS))
    pa = cen.get("pool_arithmetic", {})
    for key, label in [("pool_after_second_pass", "pool size"),
                       ("flare_corpus_after", "FLARE23 corpus size"),
                       ("total_twins_removed", "total twins removed")]:
        if key in pa:
            v = pa[key]
            out.append((rf"(?<![\d.]){v}(?![\d.])", f"census | {label} = {v}"))
    for name, label in [("unified_mmkg_flare23.json", "Table 1 reference graph"),
                        ("unified_mmkg_predicted_flare23.json", "Table 1 predicted FLARE graph")]:
        p = os.path.join(ROOT, "kg_graphs", name)
        if os.path.exists(p):
            g = json.load(open(p))
            n, e = len(g.get("nodes", [])), len(g.get("edges", g.get("links", [])))
            out.append((rf"(?<![\d.]){n}(?![\d.])", f"{label} | nodes = {n:,}"))
            out.append((rf"(?<![\d.]){e}(?![\d.])", f"{label} | edges = {e:,}"))
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    import fitz
    doc = fitz.open(sys.argv[1])
    pages = [normalize(p.get_text()) for p in doc]

    def hits(rx):
        return [i + 1 for i, t in enumerate(pages) if re.search(rx, t)]

    print(f"draft: {sys.argv[1]} ({len(pages)} pages)")
    print(f"memo:  {os.path.relpath(MEMO, ROOT)}\n")

    flags = 0
    print("== STALE SCAN (any hit needs a human look at that page) ==")
    for rx, why, repl in STALE:
        pg = hits(rx)
        if pg:
            flags += 1
            print(f"  FLAG p.{','.join(map(str, pg))}  {rx}  — {why}; current: {repl}")
    if not flags:
        print("  clean — no superseded token found")

    print("\n== EXPECTED SCAN (current cells; MISSING = not transcribed yet or phrased differently) ==")
    found = missing = 0
    for rx, label in memo_tokens() + artifact_tokens():
        pg = hits(rx)
        if pg:
            found += 1
            print(f"  ok   p.{','.join(map(str, pg))}  {label}")
        else:
            missing += 1
            print(f"  MISSING          {label}")

    print(f"\nsummary: {flags} stale flag(s) · {found} expected cell(s) found · {missing} missing")
    print("limitation: token-presence only - a current value in the wrong cell is not detectable here.")
    sys.exit(1 if flags else 0)


if __name__ == "__main__":
    main()
