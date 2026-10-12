#!/usr/bin/env python3
"""Complete replacement rows for Table 6 (tab:retrieval-primary) and the CT sub-pool block of Table 7 (tab:retrieval),
corrected arm, in v47 column order, with ONE Holm correction over the complete F1 family (study lead's 10-07 list,
items 1-4).

Inputs (committed, no recomputation):
  results/retrieval/dedup/baselines_corrected_labels_2026-10-05.json  Table 6: graph rows (the co-author's corrected
      arm, reproduced cell for cell) + rows (v)-(vii'); 113 queries over the 113 cases
  results/retrieval/dedup/baselines_subpool_1311_2026-10-06.json      Table 7 sub-pool: graph rows (ii)-(iv) + (v)-(vii')
      on the 638-case sub-pool (membership identical to ct_subpool_membership_2026-10-07.txt, sha256 f646fd8f...)
Every comparator-vs-(ii) contrast in both files was computed by one procedure: paired query-level bootstrap 95% CI
(B = 5,000) and two-sided sign-flip permutation p (B = 20,000, p = (k + 1) / (B + 1)), seed 20261003.

F1 = 7 contrasts on Table 6 [(i), (iii), (iv), (v), (vi), (vii), (vii')] + 6 on the sub-pool block [(iii), (iv), (v),
(vi), (vii), (vii')]; row (i) is undefined on the sub-pool; (iii') stays outside F1 and is reported raw.

Holm is applied as the step-down procedure: p_adj(k) = max_{j<=k} min(1, (m - j + 1) p_(j)). The shared holm() used
until now (retrieval_tables_789.holm and its verbatim copy in build_retrieval_tables.py) omitted the running maximum,
so tied raw p values received different adjusted values depending on row order. Both values are reported per row.

  python emit_final_rows_2026-10-07.py [--tables-json retrieval_tables_1311.json]
-> results/retrieval/dedup/final_rows_2026-10-07.json, tab_retrieval_primary_2026-10-07.tex,
   tab_retrieval_ct_subpool_rows_2026-10-07.tex [, retrieval_tables_1311_filled_2026-10-07.json]
"""
import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
D = os.path.join(ROOT, "results", "retrieval", "dedup")
LABEL = {"(i)": "Reference phenotypes, support-restricted (upper bound)",
         "(ii)": "Predicted phenotypes, support-restricted (proposed)",
         "(iii)": "Predicted, coverage-blind (one-sided organs $=$ mismatch)",
         "(iii')": "Predicted, imputed absence (one-sided organs may agree)",
         "(iv)": "Predicted, organ-agnostic baseline",
         "(v)": "Radiomics-vector (reference regions)",
         "(vi)": "Organ-conditioned radiomics (reference regions)",
         "(vii)": "Image embedding, organ-agnostic (reference regions)",
         "(vii')": "Image embedding, organ-conditioned (reference regions)"}
ORDER6 = ["(i)", "(ii)", "(iii)", "(iii')", "(iv)", "(v)", "(vi)", "(vii)", "(vii')"]
ORDER7 = ["(ii)", "(iii)", "(iii')", "(iv)", "(v)", "(vi)", "(vii)", "(vii')"]
BASELINES = {"(v)", "(vi)", "(vii)", "(vii')"}   # no reasoning path, so Spur.@10 is n/a (study lead 10-08 item 2)
DATE = os.environ.get("EMIT_DATE", "2026-10-08")


def holm_stepdown(raw):
    """raw: {key: p}. Step-down Holm with the running maximum (ties get the same adjusted value)."""
    items = sorted(raw.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = run
    return out


def ptex(p):
    if p is None:
        return "---"
    mant, exp = f"{p:.1e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def ci(d):
    f = lambda x: "$" + f"{x:+.3f}" + "$"
    return f"{f(d['dmAP'])} [{f(d['ci95'][0])}, {f(d['ci95'][1])}]"


def tag(t):
    return t.replace("'", "$'$").ljust(8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tables-json")
    a = ap.parse_args()
    t6 = json.load(open(os.path.join(D, "baselines_corrected_labels_2026-10-05.json")))["tables"]["corrected"]["rows"]
    t7 = json.load(open(os.path.join(D, "baselines_subpool_1311_2026-10-06.json")))["arms"]["corrected"]["rows"]
    raw = {("T6", r): t6[r]["dmAP_vs_ii"]["p_raw"] for r in ORDER6 if r not in ("(ii)", "(iii')")}
    raw.update({("T7", r): t7[r]["dmAP_vs_ii"]["p_raw"] for r in ORDER7 if r not in ("(ii)", "(iii')")})
    assert len(raw) == 13, len(raw)
    adj = holm_stepdown(raw)
    out = {"note": __doc__.strip(), "family_F1_size": len(raw), "permutation_floor": round(1 / 20001, 8), "tables": {}}
    for name, rows, order in (("tab:retrieval-primary", t6, ORDER6), ("tab:retrieval ct-subpool", t7, ORDER7)):
        key = "T6" if name.startswith("tab:retrieval-primary") else "T7"
        block = {}
        for r in order:
            x = rows[r]; c = x.get("dmAP_vs_ii")
            row = {"label": LABEL[r], "n_queries": x["n_queries"], "P@5": x["P@5"], "P@10": x["P@10"], "mAP": x["mAP"],
                   "nDCG": x["nDCG"], "Spur.@10": None if r in BASELINES else x["spurious@10"], "Mism.@10": x["mismatch@10"]}
            if c:
                row.update({"dmAP": c["delta"], "ci95": c["ci95"], "p_raw": c["p_raw"],
                            "p_holm_F1": None if r == "(iii')" else round(adj[(key, r)], 6),
                            "p_holm_previous_method": c.get("p_holm"), "p_reported": "raw" if r == "(iii')" else "Holm over F1"})
            block[r] = row
        out["tables"][name] = block
    json.dump(out, open(os.path.join(D, f"final_rows_{DATE}.json"), "w"), indent=1)

    def p_of(row):
        return row.get("p_raw") if row.get("p_holm_F1") is None else row["p_holm_F1"]
    L6 = ["% Table 6 (tab:retrieval-primary), corrected arm, 113 queries; rows (i)-(iv) are the co-author's corrected arm",
          "% (reproduced cell for cell), rows (v)-(vii') re-scored under the corrected labels. Holm once over F1 (13 contrasts);",
          "% (iii') raw. Emitted by emit_final_rows_2026-10-07.py; columns: P@10, mAP, nDCG, Mism.@10, dmAP [95% CI], p."]
    for r in ORDER6:
        x = out["tables"]["tab:retrieval-primary"][r]; b = r == "(ii)"
        cells = [f"{x['P@10']:.3f}", f"{x['mAP']:.3f}", f"{x['nDCG']:.3f}", f"{x['Mism.@10']:.2f}"]
        if b:
            cells = [f"\\textbf{{{c}}}" for c in cells] + ["---", "---"]; lab = f"\\textbf{{{x['label']}}}"
        else:
            cells += [ci(x), ptex(p_of(x))]; lab = x["label"]
        L6.append(f" {tag(r)} & {lab} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(D, f"tab_retrieval_primary_{DATE}.tex"), "w").write("\n".join(L6) + "\n")
    L7 = ["% Table 7 (tab:retrieval) CT sub-pool block, corrected arm: 638 = 113 single-organ + 525 FLARE23 with a CT;",
          "% 109 of 113 queries scored. Graph rows (ii)-(iv) as the co-author's emit_ct_subpool_rows.py; rows (v)-(vii') added;",
          "% Spur.@10 is defined on the reasoning path, which the external retrievers do not have: n/a on (v)-(vii').",
          "% Holm once over F1 (13 contrasts); (iii') raw. Columns: P@5, P@10, mAP, nDCG, Spur.@10, Mism.@10, dmAP [95% CI], p."]
    for r in ORDER7:
        x = out["tables"]["tab:retrieval ct-subpool"][r]; b = r == "(ii)"
        cells = [f"{x['P@5']:.3f}", f"{x['P@10']:.3f}", f"{x['mAP']:.3f}", f"{x['nDCG']:.3f}", ("n/a" if x['Spur.@10'] is None else f"{x['Spur.@10']:.2f}"), f"{x['Mism.@10']:.2f}"]
        if b:
            cells = [f"\\textbf{{{c}}}" for c in cells] + ["---", "---"]; lab = f"\\textbf{{{x['label']}}}"
        else:
            cells += [ci(x), ptex(p_of(x))]; lab = x["label"]
        L7.append(f" {tag(r)} & {lab} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(D, f"tab_retrieval_ct_subpool_rows_{DATE}.tex"), "w").write("\n".join(L7) + "\n")

    if a.tables_json:                                   # fill the 20 "not recomputed" Table 6 baseline cells
        tj = json.load(open(a.tables_json)); n = 0
        key = {"P@10": "P@10", "mAP": "mAP", "nDCG": "nDCG", "Mism.@10": "Mism.@10", "dmAP": "dmAP"}
        for e in tj["cell_diff_vs_manuscript"]:
            if e.get("table") == "tab:retrieval-primary" and e.get("new_1311") == "not recomputed" and e["row"] in ("(v)", "(vi)", "(vii)", "(vii')"):
                e["new_1311"] = out["tables"]["tab:retrieval-primary"][e["row"]][key[e["metric"]]]
                e["moves"] = e["new_1311"] != e["published"]
                e["note"] = "corrected labels; baselines_corrected_labels_2026-10-05.json (emit_final_rows_2026-10-07.py)"
                n += 1
        tj.setdefault("meta", {})["filled_by_pipeline_2026-10-07"] = (
            f"{n} Table 6 baseline cells filled; the CT sub-pool block rows (v)-(vii') and the F1 Holm p values are in "
            f"final_rows_{DATE}.json; row (i) of tab:retrieval stays withdrawn")
        json.dump(tj, open(os.path.join(D, f"retrieval_tables_1311_filled_{DATE}.json"), "w"), indent=1)
        print("filled cells:", n)
    for name, block in out["tables"].items():
        print(name)
        for r, x in block.items():
            print(f"  {r:6s} P@5 {x['P@5']} P@10 {x['P@10']} mAP {x['mAP']} nDCG {x['nDCG']} Spur {x['Spur.@10']} Mism {x['Mism.@10']}"
                  + (f"  d {x['dmAP']} {x['ci95']} raw {x['p_raw']:.1e} holmF1 {x['p_holm_F1']} (prev {x['p_holm_previous_method']})" if 'dmAP' in x else ""))


if __name__ == "__main__":
    main()
