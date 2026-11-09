#!/usr/bin/env python3
"""Draft-cell checker: every table cell of the manuscript PDF against the committed artifact it comes from.

  python scripts/audit/check_draft_cells.py <manuscript.pdf> [--table "Table 4"] [--verbose]

For each table, the expected rows are rebuilt at run time from the result files named in TABLE_MAP.md (nothing is
hard-coded except labels, the artifact paths and the display rounding). A row is a short ordered list of cells as the
paper prints them (half-up rounding to the printed precision; signed deltas; "6.5 x 10^-4" p values). A row passes
when its cells occur in that order, separated only by non-digit text, inside the window of the PDF text that follows
the table's caption. A row that is absent, out of order or carries a different value in any cell fails.

Exit code 0 when every row of every table is found, 1 otherwise. Table K.1 (source: survey database, not public) and
the bracketed pilot values of Table 3 (not regenerated) are outside the check, as stated in TABLE_MAP.md.

Limitation, stated: the check is per row inside the caption window, so a row printed in the right table but under the
wrong row label passes if its cells are otherwise identical to another row's; rows are distinctive enough (four or
more cells) that this has not occurred.
"""
import argparse
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
WINDOW_BEFORE, WINDOW_AFTER = 1500, 9000


def J(*parts):
    return json.load(open(os.path.join(ROOT, *parts)))


# ---------------------------------------------------------------- display formatting (as printed in the paper)
def rnd(x, nd):
    """Half-up rounding of the stored value to nd decimals, as a string (the paper's convention: 0.4365 -> 0.437)."""
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def f(x, nd=3):
    return rnd(x, nd)


def s(x, nd=3):
    """Signed value (+0.033 / -0.006)."""
    v = rnd(x, nd)
    return v if v.startswith("-") else "+" + v


def pct(x, nd=1):
    return rnd(x, nd) + "%"


def thou(n):
    return f"{int(n):,}"


class P:
    """A p value printed as m x 10^-e (mantissa to one decimal, trailing .0 dropped)."""
    def __init__(self, p):
        mant, exp = f"{p:.1e}".split("e")
        self.mant, self.exp = mant[:-2] if mant.endswith(".0") else mant, int(exp)

    def rx(self):
        return rf"{re.escape(self.mant)}\s*[x×]\s*10\s*-\s*{abs(self.exp)}(?!\d)"

    def __repr__(self):
        return f"{self.mant}x10^{self.exp}"


def cell_rx(c):
    if isinstance(c, P):
        return c.rx()
    return rf"(?<![\d.]){re.escape(str(c))}(?![\d])"


def row_rx(cells, gap=60):
    return rf"[^\d]{{0,{gap}}}".join(cell_rx(c) for c in cells)


# ---------------------------------------------------------------- PDF text
def pdf_text(path):
    import fitz
    t = " ".join(p.get_text() for p in fitz.open(path))
    for a, b in (("−", "-"), ("–", "-"), ("—", " — "), (" ", " "), (" ", " "),
                 (" ", " "), ("′", "'"), ("’", "'"), ("×", "×")):
        t = t.replace(a, b)
    t = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", t)
    return re.sub(r"\s+", " ", t)


def caption_positions(text, name):
    return [m.start() for m in re.finditer(rf"{re.escape(name)}:", text)]


# ---------------------------------------------------------------- tables: name -> function returning rows
# each row: (label, [cells], source path relative to the repository root)
TABLES = {}
OPTIONAL = {}   # tables checked only when their caption is present in the PDF (analyses added after submission)


def table(name):
    def deco(fn):
        TABLES[name] = fn
        return fn
    return deco


@table("Table 4")
def table_4():
    src = "results/retrieval/dedup/final_rows_2026-10-08.json"
    rows = J(src)["tables"]["tab:retrieval-primary"]
    out = []
    for r, x in rows.items():
        cells = [f(x["P@10"]), f(x["mAP"]), f(x["nDCG"]), f(x["Mism.@10"], 2)]
        if "dmAP" in x:
            p = x["p_raw"] if x.get("p_holm_F1") is None else x["p_holm_F1"]
            cells += [s(x["dmAP"]), s(x["ci95"][0]), s(x["ci95"][1]), P(p)]
        out.append((r, cells, src))
    return out


# ================================================================ Tables 2, 3, 5, 6, E.3
# ---------------------------------------------------------------- part A: Tables 2, 3, 5, 6, E.3
class P1(P):
    """A p value printed with its one-decimal mantissa kept (3.0 x 10^-4), as in Table 6."""
    def __init__(self, p):
        mant, exp = f"{p:.1e}".split("e")
        self.mant, self.exp = mant, int(exp)


class Br(P):
    """A cell followed by an optional bracketed pilot value ('1.000 [1.000]'); the bracket is not checked."""
    def __init__(self, v):
        self.v = str(v)

    def rx(self):
        return rf"(?<![\d.]){re.escape(self.v)}(?![\d])(?:\s*\[[^\]]{{0,12}}\])?"

    def __repr__(self):
        return self.v


class asis(P):
    """A value the artifact stores already rounded (trailing zeros dropped); matched with or without the printed trailing
    zeros (0.67 / 0.670), never with a different value."""
    def __init__(self, x):
        self.v = repr(float(x))

    def rx(self):
        return rf"(?<![\d.]){re.escape(self.v)}0*(?![\d])"

    def __repr__(self):
        return self.v


@table("Table 2")
def table_2():
    s_src = "results/kg/ausam_3d_summary.json"
    o_src = "results/kg/semioracle_flare23_volume_fidelity.json"
    a_src = "results/repair/closed_loop_tab12_13.json"
    nf = J(s_src)["node_fidelity"]
    so = J(o_src)
    au = J(a_src)["tab3_autonomous_volume_fidelity"]
    out = []
    for organ, key in (("liver", "liver"), ("kidney", "kidney (L+R pooled)"), ("pancreas", "pancreas")):
        x = nf[f"{organ}/flare23"]
        cells = [f(x["volume_corr"]), pct(x["volume_MAPE_pct"])]
        for arm in ("raw", "repair"):
            cells += [f(au[arm][key]["r"]), pct(au[arm][key]["MAPE_pct"])]
        out.append((f"{organ} FLARE23 (semi-oracle | autonomous raw | repaired)", cells, f"{s_src}; {a_src}"))
    for organ, ds, lab in (("liver", "lits", "LiTS"), ("kidney", "kits", "KiTS23"), ("pancreas", "msd", "MSD Task07")):
        x = nf[f"{organ}/{ds}"]
        out.append((f"{organ} {lab}", [f(x["volume_corr"]), pct(x["volume_MAPE_pct"])], s_src))
    sp = so["organs"]["spleen"]
    cells = [f(sp["volume_r"]), pct(sp["volume_MAPE_pct"])]
    for arm in ("raw", "repair"):
        cells += [f(au[arm]["spleen"]["r"]), pct(au[arm]["spleen"]["MAPE_pct"])]
    out.append(("spleen FLARE23 (576-case semi-oracle | autonomous raw | repaired)", cells, f"{o_src}; {a_src}"))
    # caption: the 576-case liver / kidney / pancreas values and the cohort sizes
    cap = []
    for key in ("liver", "kidney (L+R pooled)", "pancreas"):
        cap += [f(so["organs"][key]["volume_r"]), pct(so["organs"][key]["volume_MAPE_pct"])]
    out.append(("caption: 576-case liver, kidney, pancreas r/MAPE", cap, o_src))
    out.append(("caption: n per held-out list",
                [f"LiTS n = {nf['liver/lits']['n']}"], s_src))
    out.append(("caption: n per held-out list", [f"KiTS23 n = {nf['kidney/kits']['n']}"], s_src))
    out.append(("caption: n per held-out list", [f"MSD n = {nf['pancreas/msd']['n']}"], s_src))
    out.append(("caption: FLARE23 held-out list", [f"{nf['liver/flare23']['n']}-case FLARE23 held-out"], s_src))
    out.append(("caption: 576-case run, spleen nodes", [f"{so['n_cases']}-case semi-oracle FLARE23 run (n = {sp['n']} spleen"], o_src))
    out.append(("caption: autonomous set", [f"{au['raw']['liver']['n']}-patient FLARE23 closed-loop"], a_src))
    return out


@table("Table 3")
def table_3():
    src = "results/queries/query_suite_tab6.json"
    d = J(src)
    rows = {r["family"].split()[0]: r for r in d["cohort_113"]["rows"]}
    out = []
    for fam in ("Q2", "Q3", "Q4a", "Q4b", "Q5", "Q7a", "Q7b"):
        r = rows[fam]
        n = r["N_T"] + r["N_F"] + r["N_U"]
        cells = [r["N_T"], r["N_F"], r["N_U"]]
        if r["precision"] is not None:
            cells += [Br(f(r["precision"])), Br(f(r["recall"])), Br(f(r["F1"]))]
        cells.append(Br(rnd(100.0 * r["N_U"] / n, 1)))
        out.append((fam, cells, src))
    m = d["cohort_113"]["macro_over_evaluable"]
    out.append(("Macro mean (Q2, Q3, Q7a, Q7b)", [Br(f(m["macro_P"])), Br(f(m["macro_R"])), Br(f(m["macro_F1"]))], src))
    q2 = rows["Q2"]
    out.append(("caption: pooled N", [f"(N = {q2['N_T'] + q2['N_F'] + q2['N_U']})"], src))
    # Q1 row: from the retrieval artifacts behind Tables 4 and 5
    pr_src = "results/retrieval/dedup/final_rows_2026-10-08.json"
    fp_src = "results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json"
    pr = J(pr_src)["tables"]["tab:retrieval-primary"]
    fp = J(fp_src)["tab:retrieval"]["corrected"]["rows"]
    out.append(("Q1 row", [f(pr["(ii)"]["mAP"]),
                           f(-pr["(iii)"]["dmAP"]), f(-pr["(iii')"]["dmAP"]),
                           rnd(-fp["(iii)"]["dmAP_vs_ii"]["delta"], 2), rnd(-fp["(iii')"]["dmAP_vs_ii"]["delta"], 2)],
                f"{pr_src}; {fp_src}"))
    # Q9 row: Table E.3 total
    e3_src = "results/summarization/statement_precision_table_e3.json"
    e3 = J(e3_src)["all_statements"]
    out.append(("Q9 row", [f"statement precision {f(e3['precision'])} over {e3['n']}"], e3_src))
    # Q10 row, SPARQL part (the application timings are app-side; see the report)
    q = d["q10_latency_sparql_ms"]
    per_ds = [q[k]["across_queries"]["median_ms"] for k in ("predicted_kits", "predicted_lits", "predicted_msd")]
    fl = q["flare23 (1,312 cases)"]
    out.append(("Q10 row: SPARQL", [rnd(min(per_ds), 0), rnd(max(per_ds), 0), rnd(fl["across_queries"]["median_ms"], 0),
                                     rnd(fl["across_queries"]["IQR_ms"][0], 0), thou(rnd(fl["across_queries"]["IQR_ms"][1], 0)),
                                     thou(fl["n_triples"])], src))
    return out


@table("Table 5")
def table_5():
    fp_src = "results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json"
    sp_src = "results/retrieval/dedup/final_rows_2026-10-08.json"
    fp = J(fp_src)["tab:retrieval"]["corrected"]["rows"]
    sp = J(sp_src)["tables"]["tab:retrieval ct-subpool"]
    out = []
    for r, x in fp.items():
        cells = [f(x["P@5"]), f(x["P@10"]), f(x["mAP"]), f(x["nDCG"]), f(x["spurious@10"], 2), f(x["mismatch@10"], 2)]
        if "dmAP_vs_ii" in x:
            c = x["dmAP_vs_ii"]
            cells += [s(c["delta"]), s(c["ci95"][0]), s(c["ci95"][1]), P(c["p_raw"])]
        out.append((f"full pool {r}", cells, fp_src))
    for r, x in sp.items():
        spur = "n/a" if x["Spur.@10"] is None else f(x["Spur.@10"], 2)
        cells = [f(x["P@5"]), f(x["P@10"]), f(x["mAP"]), f(x["nDCG"]), spur, f(x["Mism.@10"], 2)]
        if "dmAP" in x:
            p = x["p_raw"] if x.get("p_holm_F1") is None else x["p_holm_F1"]
            cells += [s(x["dmAP"]), s(x["ci95"][0]), s(x["ci95"][1]), P(p)]
        out.append((f"CT sub-pool {r}", cells, sp_src))
    n_full = J(fp_src)["tab:retrieval"]["corrected"]
    out.append(("caption: nq graph rows", [f"nq = {n_full['rows']['(ii)']['n_queries']} for the graph rows"], fp_src))
    out.append(("block header: full pool size", [f"Full pool ({thou(n_full['n_pool'])} cases)"], fp_src))
    return out


@table("Table 6")
def table_6():
    q32_src = "results/llm_expert/llm_expert_study_qwen3_32b.json"
    q7_src = "results/llm_expert/llm_expert_study.json"
    c_src = "results/llm_expert/llm_expert_comparators_qwen3_32b.json"
    h_src = "results/repair/holm_f3_f4_2026-10-08.json"
    a, b, c, h = J(q32_src), J(q7_src), J(c_src), J(h_src)["F4"]["contrasts"]
    both = f"{q32_src}; {q7_src}"
    out = [("Cohen's kappa", [f(a["cohen_kappa_expert_vs_automatic"]), f(b["cohen_kappa_expert_vs_automatic"])], both),
           ("Raw agreement", [f(a["raw_agreement"]), f(b["raw_agreement"])], both),
           ("Judge / rule positive rate", [f(a["diagnostics"]["llm_positive_rate"]), f(a["diagnostics"]["auto_positive_rate"]),
                                           f(b["diagnostics"]["llm_positive_rate"]), f(b["diagnostics"]["auto_positive_rate"])], both)]
    cf = [x["diagnostics"]["confusion"][k] for x in (a, b) for k in ("auto1_llm1", "auto1_llm0", "auto0_llm1", "auto0_llm0")]
    out.append(("Confusion (++ / +- / -+ / --)", cf, both))
    out.append(("mAP of proposed, judge relevance", [asis(a["expert_relevance_mAP"]), asis(b["expert_relevance_mAP"])], both))
    out.append(("mAP of proposed, rule relevance", [asis(a["automatic_relevance_mAP"]), asis(b["automatic_relevance_mAP"])], both))
    rows = c["rows"]
    for key in rows:
        x = rows[key]
        r = key.split()[0]
        cells = [f(x["expert_relevance_mAP"]), x["n_scored_queries_expert"], asis(x["expert_P@10"])]
        if key in h:
            k = h[key]
            cells += [s(k["delta"]), s(k["ci95"][0]), s(k["ci95"][1]), P1(k["p_holm_stepdown"])]
            src = f"{c_src}; {h_src}"
        else:
            src = c_src
        if r == "(iv)":
            n_undef = re.search(r"(\d+)/\d+ queries", c["F4_paired_vs_ii_judge_relevance"]["(iv) organ-agnostic"]).group(1)
            cells.append(f"excluded ({n_undef} undefined)")
        out.append((r, cells, src))
    out.append(("caption: Nq", [f"Nq = {a['n_queries']}"], q32_src))
    out.append(("caption: kappa pairs", [f"on {a['diagnostics']['n_kappa_pairs']} class-balanced pairs"], q32_src))
    out.append(("caption: pairs reused / new", [f"{a['n_pairs']} original pairs reused, {c['n_pairs_new']} new pairs"],
                f"{q32_src}; {c_src}"))
    out.append(("caption: scorable queries", [f"over {c['F4_paired_vs_ii_judge_relevance']['n_queries']} scorable queries"], c_src))
    return out


@table("Table E.3")
def table_e3():
    src = "results/summarization/statement_precision_table_e3.json"
    c_src = "results/kg/containment_reextraction_113_predicted.json"
    e = J(src)
    out = []
    for r in e["rows"]:
        out.append((r["row"], [r["n"], f(r["precision"])], src))
    t = e["all_statements"]
    out.append(("All statements", [t["n"], f(t["precision"])], src))
    out.append(("caption: all statements", [f"All statements: {t['hits']} of {t['n']}"], src))
    site = next(r for r in e["rows"] if r["row"].startswith("Anatomic site"))
    out.append(("caption: anatomic site", [f"({site['hits']} of {site['n']})"], src))
    cv = J(c_src)["table11_containment_precision_v2"]
    pdx = cv["per_dataset"]
    out.append(("caption: containment per dataset",
                [f"KiTS23 {f(pdx['kits']['precision'])}", f"LiTS {f(pdx['lits']['precision'])}", f"MSD {f(pdx['msd']['precision'])}",
                 f(cv["frozen_value_replaced"])], c_src))
    return out

# ================================================================ Tables A.1-A.5, B.1, E.2, L.1
# ---------------------------------------------------------------- Supplementary Appendix A, B, E.2, L
SEG = "results/segmentation"


@table("Table A.1")
def table_a1():
    """Semi-oracle held-out 3D segmentation. Means (4 d.p.) from the per-dataset eval files; CI and n from seg_stats."""
    st_src = f"{SEG}/seg_stats.json"
    st = J(st_src)
    out = []
    for ds, org in (("flare_task2", "liver"), ("flare_task2", "kidney"), ("flare_task2", "pancreas"),
                    ("flare23", "liver"), ("flare23", "kidney"), ("flare23", "pancreas"),
                    ("kits", "kidney"), ("msd", "pancreas"), ("lits", "liver")):
        src = f"{SEG}/ausam_3d_{ds}.json"
        e, ss = J(src), st["organ"][ds][org]
        cells = [f(e["mean_3d_dice"][org]), f(ss["DSC"]["ci95"][0]), f(ss["DSC"]["ci95"][1])]
        if e["mean_nsd_2mm"].get(org) is not None:
            cells += [f(e["mean_nsd_2mm"][org]), f(e["mean_hd95_mm"][org], 2)]
        cells.append(ss["DSC"]["n"])
        out.append((f"organ {ds} {org}", cells, f"{src} + {st_src}"))
    for ds in ("kits", "msd", "lits"):          # FLARE23 tumor row is withdrawn in the paper (prose, no cells)
        src = f"{SEG}/tumor_3d_{ds}.json"
        e, ss = J(src), st["tumor"][ds]
        cells = [f(e["mean_3d_dice"]), f(ss["DSC"]["ci95"][0]), f(ss["DSC"]["ci95"][1])]
        if e["mean_nsd_2mm"] is not None:
            cells += [f(e["mean_nsd_2mm"]), f(e["mean_hd95_mm"], 2)]
        cells.append(e["n_patients"])
        out.append((f"tumor {ds}", cells, f"{src} + {st_src}"))
    return out


@table("Table A.2")
def table_a2():
    osrc, tsrc = f"{SEG}/organ_generic_kg.json", f"{SEG}/tumor_incremental.json"
    org = J(osrc)["per_organ_dataset"]
    s4 = next(x for x in J(tsrc)["stages"] if x["name"] == "s4_flare")["per_dataset_test"]
    src = f"{osrc} (per_organ_dataset) + {tsrc} (s4_flare)"
    return [("LiTS", [f(org["liver/lits"]), f(s4["lits"])], src),
            ("KiTS23", [f(org["kidney/kits"]), f(s4["kits"])], src),
            ("MSD Task07", [f(org["pancreas/msd"]), f(s4["pancreas"])], src),
            ("FLARE23", [f(org["liver/flare23"]), f(org["kidney/flare23"]), f(org["pancreas/flare23"]), f(s4["flare"])], src),
            ("FLARE-Task2", [f(org["liver/flare_task2"]), f(org["kidney/flare_task2"]), f(org["pancreas/flare_task2"])], src)]


@table("Table A.3")
def table_a3():
    csrc = f"{SEG}/cohort157_rescore_2026-10-03.json"
    asrc = f"{SEG}/organ_convention_a_autonomous_summary_2026-10-04.json"
    psrc = f"{SEG}/promptfree_round2_2026-10-01.json"
    c, a = J(csrc), J(asrc)
    fc = a["cohorts"]["full_collection"]

    def po(coll, organ):
        return fc[coll]["per_organ"][organ]

    def par(coll, organ):                     # tumor-excluded (parenchyma) organ Dice, structure mean
        return f(po(coll, organ)["parenchyma_dice"]["mean"])

    def conv_a(coll, organ):
        return f(po(coll, organ)["organ_dice"]["mean"])

    def tumor(block):
        r, g = block["tumor_raw"], block["tumor_gated"]
        return [f(r["dice"]), f(r["dice_ci95"][0]), f(r["dice_ci95"][1]), f(g["dice"]), f(g["dice_ci95"][0]), f(g["dice_ci95"][1])]

    def all_(block):
        return [f(block["tumor_raw"]["dice"]), f(block["tumor_gated"]["dice"])]

    pc = c["per_collection"]
    FP = "FLARE23 tumour-positive"
    out = [
        ("LiTS", [pc["lits"]["secondary_all"]["n"], pc["lits"]["primary_twin_clean"]["n"], par("LiTS", "liver")]
         + tumor(pc["lits"]["primary_twin_clean"]) + all_(pc["lits"]["secondary_all"]), f"{csrc} + {asrc}"),
        ("MSD Task07", [pc["msd"]["secondary_all"]["n"], pc["msd"]["primary_twin_clean"]["n"], par("MSD Pancreas", "pancreas")]
         + tumor(pc["msd"]["primary_twin_clean"]) + all_(pc["msd"]["secondary_all"]), f"{csrc} + {asrc}"),
        ("KiTS23", [pc["kits"]["secondary_all"]["n"], pc["kits"]["primary_twin_clean"]["n"], par("KiTS23", "kidney (pooled)"),
                    par("KiTS23", "right kidney"), par("KiTS23", "left kidney")]
         + tumor(pc["kits"]["primary_twin_clean"]) + all_(pc["kits"]["secondary_all"]), f"{csrc} + {asrc}"),
        # FLARE23 pooled-kidney tumor-excluded cell (printed 0.884) has no public artifact: the row is split around it
        ("FLARE23 (n, liver)", [pc["flare23"]["secondary_all"]["n"], pc["flare23"]["primary_twin_clean"]["n"], par(FP, "liver")],
         f"{csrc} + {asrc}"),
        ("FLARE23 (per side .. tumor)", [par(FP, "right kidney"), par(FP, "left kidney"), par(FP, "spleen"), par(FP, "pancreas")]
         + tumor(pc["flare23"]["primary_twin_clean"]) + all_(pc["flare23"]["secondary_all"]), f"{csrc} + {asrc}"),
        ("Pooled", [c["pooled"]["secondary_all"]["n"], c["pooled"]["primary_twin_clean"]["n"]]
         + tumor(c["pooled"]["primary_twin_clean"]) + all_(c["pooled"]["secondary_all"]), csrc),
    ]
    f54 = c["flare23_54_scan_table"]
    # FLARE23 twin-clean n = 14: the printed intervals are the application-side table's bootstrap
    # (ipkg-v2/substrate/flare23_twinclean_14.json); same 14 cases and means as cohort157_rescore's block, asserted here.
    tsrc = "ipkg-v2/substrate/flare23_twinclean_14.json"
    t14 = J(tsrc)["twin_clean"]
    assert sorted(t14["cases"]) == sorted(f54["twin_clean_cases"]), "twin-clean case lists differ"
    assert f(t14["raw_dice"]) == f(f54["primary_twin_clean"]["tumor_raw"]["dice"])
    assert f(t14["gated_dice"]) == f(f54["primary_twin_clean"]["tumor_gated"]["dice"])
    out += [("FLARE23 (54) n, twin-clean raw", [f54["secondary_all"]["n"], t14["n"], f(t14["raw_dice"]),
                                                 *(Either(v, signed=False) if _tie4(v) else f(v) for v in t14["raw_ci95"])], f"{csrc} + {tsrc}"),
            ("FLARE23 (54) twin-clean gated + all", [f(t14["gated_dice"])] + [Either(v, signed=False) if _tie4(v) else f(v) for v in t14["gated_ci95"]]
             + all_(f54["secondary_all"]), f"{csrc} + {tsrc}")]
    # exposed-through-a-twin subset of the 54-scan list: per-scan values minus the twin-clean list
    tc = set(f54["twin_clean_cases"])
    cases = [x for x in J(psrc)["models"]["sam3_tumor_generic"]["datasets"]["flare23"]["cases"] if x["case"] not in tc]
    mean = lambda arm: sum(x[arm]["dice"] for x in cases) / len(cases)
    out.append(("FLARE23 (54), exposed through a twin", [len(cases), f(mean("tumor_raw")), f(mean("tumor_gated"))],
                f"{psrc} (flare23 cases) minus {csrc} twin_clean_cases"))

    # caption: populations and convention A
    # dataset names carrying digits (KiTS23, FLARE23) interrupt a cell run, so caption runs are split at them
    out.append(("caption: organ populations LiTS, MSD", [fc["LiTS"]["n_cases"], fc["MSD Pancreas"]["n_cases"]], asrc))
    out.append(("caption: organ population KiTS23", [fc["KiTS23"]["n_cases"]], asrc))
    out.append(("caption: organ population FLARE23, records with no organ",
                [fc[FP]["n_cases"], fc[FP]["n_records_with_no_annotated_organ"]], asrc))
    A = fc["ALL"]
    per = {k: v for k, v in A["per_organ"].items() if k != "kidney (pooled)"}   # per-side kidneys, as structures
    nst = sum(v["organ_dice"]["n"] for v in per.values())
    wm = lambda m: sum(v[m]["mean"] * v[m]["n"] for v in per.values()) / sum(v[m]["n"] for v in per.values())
    # pooled convention-A CI [0.867, 0.877] needs per-structure values, which the public summary does not carry
    out.append(("caption: pooled convention A (structure mean)", [f(wm("organ_dice"))], asrc))
    out.append(("caption: parenchyma, recall, structures, records, case mean",
                [f(wm("parenchyma_dice")), f(wm("parenchyma_recall")), thou(nst), thou(A["n_records"]), thou(A["n_cases"]),
                 f(A["case_mean"]["organ_dice"]["mean"])], asrc))
    out.append(("caption: convention A LiTS liver", [conv_a("LiTS", "liver")], asrc))
    out.append(("caption: convention A KiTS23 pooled, per side; MSD pancreas",
                [conv_a("KiTS23", "kidney (pooled)"), conv_a("KiTS23", "right kidney"), conv_a("KiTS23", "left kidney"),
                 conv_a("MSD Pancreas", "pancreas")], asrc))
    out.append(("caption: convention A FLARE23 liver", [conv_a(FP, "liver")], asrc))
    # FLARE23 pooled-kidney convention-A cell (printed 0.893) has no public artifact
    out.append(("caption: convention A FLARE23 per side, spleen, pancreas",
                [conv_a(FP, "right kidney"), conv_a(FP, "left kidney"), conv_a(FP, "spleen"), conv_a(FP, "pancreas")], asrc))
    out.append(("caption: KiTS23 pooled tumor-excluded kidney", [par("KiTS23", "kidney (pooled)"), fc["KiTS23"]["n_cases"]], asrc))
    out.append(("caption: per-side tumor-excluded kidney",
                [par("KiTS23", "right kidney"), par("KiTS23", "left kidney"), par(FP, "right kidney"),
                 po(FP, "right kidney")["parenchyma_dice"]["n"], par(FP, "left kidney"), po(FP, "left kidney")["parenchyma_dice"]["n"]], asrc))
    return out


@table("Table A.4")
def table_a4():
    src = "results/kg/ausam_3d_summary.json"
    nf = J(src)["node_fidelity"]
    return [(o, [f(nf[f"{o}/flare_task2"]["volume_corr"]), pct(nf[f"{o}/flare_task2"]["volume_MAPE_pct"])], f"{src} (node_fidelity/{o}/flare_task2)")
            for o in ("liver", "kidney", "pancreas")]


@table("Table A.5")
def table_a5():
    src = f"{SEG}/tumor_incremental.json"
    out = []
    for st in J(src)["stages"]:
        t = st["per_dataset_test"]
        out.append((st["name"], [f(t["lits"]), f(t["pancreas"]), f(t["kits"]), f(t["flare"])], f"{src} ({st['name']})"))
    return out


@table("Table B.1")
def table_b1():
    out = []
    for ds in ("lits", "kits", "msd", "flare23"):
        src = f"kg_graphs/unified_mmkg_predicted_{ds}.json"
        g = J(src)
        out.append((ds, [thou(len(g["nodes"])), thou(len(g["edges"]))], src))
    pf = J("kg_graphs/unified_mmkg_predicted_flare23.json")
    out.append(("caption: FLARE23 predicted-graph cases", [sum(n["type"] == "ImagingCase" for n in pf["nodes"])],
                "kg_graphs/unified_mmkg_predicted_flare23.json"))
    ref = J("kg_graphs/unified_mmkg_flare23.json")
    n_flare_pool = J("results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json")["meta"]["n_flare23"]
    out.append(("caption: FLARE23 reference graph",
                [thou(sum(n["type"] == "ImagingCase" for n in ref["nodes"])), thou(n_flare_pool), thou(len(ref["nodes"])), thou(len(ref["edges"]))],
                "kg_graphs/unified_mmkg_flare23.json + results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json (meta/n_flare23)"))
    return out


@table("Table E.2")
def table_e2():
    src = "results/kg/lesion_node_fidelity_flare23.json"
    d = J(src)
    ln, t, po = d["lesion_nodes"], d["table4_strict_iou_matched"], d["per_organ_strict"]
    out = [("caption: matched pairs of reference nodes", [t["n_pairs"], ln["reference"]], src),
           ("precision / recall", [f(ln["node_precision"]), f(ln["node_recall"])], src),
           ("volume MAPE (r)", [pct(t["volume_MAPE_pct"]), f(t["volume_pearson_r"])], src),
           ("Feret-diameter MAPE (r)", [pct(t["diameter_MAPE_pct"]), f(t["diameter_pearson_r"])], src),
           ("centroid error mean (median)", [f(t["centroid_error_mm_mean"], 2), f(t["centroid_error_mm_median"], 2)], src),
           ("volume-tercile edges", [f(t["volume_tercile_edges_cm3"][0], 1), f(t["volume_tercile_edges_cm3"][1], 1)], src),
           ("volume-tercile agreement", [f(t["volume_tercile_agreement"])], src),
           ("diameter-tercile edges + agreement", [f(t["diameter_tercile_edges_mm"][0], 1), f(t["diameter_tercile_edges_mm"][1], 1),
                                                    f(t["diameter_tercile_agreement"])], src),
           ("per-organ multiplicity agreement", [f(t["multiplicity_agreement"])], src)]
    for o in ("liver", "kidney", "pancreas"):
        x = po[o]
        out.append((f"by organ: {o}", [f(x["volume_MAPE_pct"], 1), pct(x["diameter_MAPE_pct"]), x["n_pairs"]], f"{src} (per_organ_strict/{o})"))
    return out


@table("Table L.1")
def table_l1():
    src = "results/kg/ontology_mappings_export_check.json"
    g = J(src)["table5_grouping"]
    out = []
    for label, key in (("Anatomical structures and sub-sites", "anatomical incl. sub-sites"), ("Lesion and tumor types", "lesion types"),
                       ("Imaging phenotypes", "categorical phenotype nodes"), ("Overall core concepts", "total_coded")):
        a, b = (int(v) for v in g[key].split("/"))
        out.append((label, [f"{a}/{b}", rnd(100 * a / b, 1)], src))
    st = J("kg_graphs/ontology_mappings.json")["stats"]
    out.append(("text: mapping file keys and codes", [st["concepts"], st["codes"], st["by_system"]["SNOMED CT"], st["by_system"]["NCIt"]],
                "kg_graphs/ontology_mappings.json (stats)"))
    return out

# ================================================================ Tables E.4, G.1-G.3, I.1, I.2
# ---------------------------------------------------------------- part C: Tables E.4, G.1, G.2, G.3, I.1, I.2
class Either(P):
    """A value stored at 4 decimals whose 4th decimal is 5: the printed 3-decimal value is not determined by the
    artifact (the unrounded value is not stored), so either neighbour is accepted. Used only where no per-patient
    values exist to recompute the exact mean."""
    def __init__(self, x, nd=3, signed=True):
        from decimal import ROUND_HALF_DOWN
        q = Decimal(1).scaleb(-nd)
        d = Decimal(repr(float(x)))
        lo, hi = (str(d.quantize(q, rounding=r)) for r in (ROUND_HALF_DOWN, ROUND_HALF_UP))
        fmt = (lambda v: v if v.startswith("-") else "+" + v) if signed else (lambda v: v)
        self.vals = sorted({fmt(lo), fmt(hi)})

    def rx(self):
        return "(?:" + "|".join(rf"(?<![\d.]){re.escape(v)}(?![\d])" for v in self.vals) + ")"

    def __repr__(self):
        return "|".join(self.vals)


def _tie4(x):
    """True when a 4-decimal stored value ends in 5 (rounding to 3 decimals is then undetermined)."""
    return Decimal(repr(float(x))).as_tuple().exponent == -4 and repr(float(x)).endswith("5")


def sd(x, nd=3):
    """Signed delta as printed; an exact zero prints unsigned (0.000)."""
    if abs(float(x)) < 0.5 * 10 ** -nd:
        return rnd(0.0, nd)
    return s(x, nd)


def sd4(x, nd=3):
    return Either(x, nd) if _tie4(x) else sd(x, nd)


def holm_stepdown(ps):
    """Step-down Holm with the running maximum; ps in family order, returns adjusted in the same order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run, m = [None] * len(ps), 0.0, len(ps)
    for j, i in enumerate(order):
        run = max(run, min(1.0, (m - j) * ps[i]))
        out[i] = run
    return out


@table("Table E.4")
def table_e4():
    src = "results/queries/query_suite_flare23.json"
    d = J(src)
    out = []
    for r in d["rows"]:
        fam = r["family"].split()[0]
        nt, nf, nu = r["N_T"], r["N_F"], r["N_U"]
        ref_t = d["reference_counts"][r["family"]]["T"]
        cells = [thou(nt), thou(nf), thou(nu), thou(ref_t)]
        if r["F1"] is not None:
            cells += [f(r["precision"]), f(r["recall"]), f(r["F1"])]
        else:
            # no reference positive: the F1 cell reports specificity over the scored (both-determinate) pairs; with
            # reference T = 0 every predicted T among them is a false positive
            assert ref_t == 0
            cells += [f(1 - nt / r["n_scored_TF_pairs"])]
        cells += [rnd(100 * nu / (nt + nf + nu), 1)]
        out.append((fam, cells, src))
    q2 = d["rows"][0]
    out.append(("caption N (pooled)", [thou(q2["N_T"] + q2["N_F"] + q2["N_U"])], src))
    m = d["macro_over_evaluable"]
    out.append(("Macro mean", [f(m["macro_P"]), f(m["macro_R"]), f(m["macro_F1"])], src))
    return out


@table("Table G.1")
def table_g1():
    src = "results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json"
    st = J(src)["tab:retrieval-strata"]["corrected"]
    meth = (("Proposed", "proposed"), ("(iii)", "coverage_blind"), ("(iii')", "imputed_absent"))

    def base(x, mism=True):
        c = [str(x["n_queries"]), f(x["P@5"]), f(x["P@10"]), f(x["mAP"])]
        return c + ([f(x["mismatch@10"], 2)] if mism and x["mismatch@10"] is not None else [])

    out = []
    a = st["(a) within-dataset control"]["pooled"]
    for lab, k in meth:
        c = base(a[k])
        if k != "proposed":                       # identity check: exact tie, no CI, no p
            c += [sd(a[k]["mAP"] - a["proposed"]["mAP"])]
        out.append((f"(a) {lab}", c, src))
    strata = (("(b)", st["(b) single-organ -> FLARE23"]), ("(c)", st["(c) FLARE23 -> single-organ"]))
    # F2 = the (iii) rows of (b) and (c); step-down Holm over the two raw p
    p_adj = holm_stepdown([x["delta_obs"]["(iii)"]["p_raw"] for _, x in strata])
    for (sl, x), padj in zip(strata, p_adj):
        for lab, k in meth:
            c = base(x["rows"][k])
            if k != "proposed":
                dd = x["delta_obs"]["(iii)" if k == "coverage_blind" else "(iii')"]
                # artifact delta_obs is (ii) minus the ablation; the table prints the ablation minus (ii)
                c += [sd(-dd["delta"]), sd(-dd["ci95"][1]), sd(-dd["ci95"][0])]
                if k == "coverage_blind":
                    c += [rnd(padj, 1)]
            out.append((f"{sl} {lab}", c, src))
    return out


@table("Table G.2")
def table_g2():
    src = "results/retrieval/dedup/retrieval_tables_1311_2026-10-07.json"
    ce = J(src)["tab:cross-examples"]["corrected"]
    out = [("caption pool size", [thou(ce["pool_size"])], src)]
    for q in ce["queries"]:
        cells = [str(q["proposed"]["first_relevant_flare23_rank"]), str(q["proposed"]["tie_width_at_reported_rank"]),
                 str(q["coverage_blind"]["first_relevant_flare23_rank"]), str(q["imputed_absent"]["first_relevant_flare23_rank"])]
        out.append((q["query"], cells, src))
    return out


@table("Table G.3")
def table_g3():
    src = "results/repair/validator_perturbation_tab11.json"
    d = J(src)
    vs = d["volume_scaling_flagged_pct"]["ALL"]
    fs = [k for k in vs if k.startswith("f=")]
    out = [("caption node counts", [str(d["nodes"]["total_scaling_nodes"])], src),
           ("caption FLARE23 nodes", [str(d["mask_level"]["ALL"]["n_nodes"])], src),
           ("Flagged, band (scaling)", [rnd(vs[k], 1) for k in fs], src)]
    ml = d["mask_level"]["ALL"]
    conds = ("unperturbed", "erosion_3mm", "dilation_3mm", "spurious_component_10pct")
    for lab, key in (("Flagged, band (mask)", "volume_band_flagged_pct"),
                     ("Flagged, connectedness", "component_flagged_pct"),
                     ("Flagged, either", "either_flagged_pct")):
        out.append((lab, [rnd(ml[c][key], 1) for c in conds], src))
    return out


@table("Table I.1")
def table_i1():
    src = "results/repair/closed_loop_tab12_13.json"
    td = J(src)["tab12_dice"]
    out = []
    for lab, k in (("Left kidney", "left_kidney"), ("Spleen", "spleen"), ("Pancreas", "pancreas"),
                   ("Right kidney", "right_kidney"), ("Liver", "liver"), ("Tumor", "tumor")):
        x = td[k]
        rr, pr, rp = x["delta_repair_minus_raw"], x["delta_postproc_minus_raw"], x["delta_repair_minus_postproc"]
        p = rp["p_raw"]
        cells = [f(x["raw"]["mean"]), f(x["postproc"]["mean"]), f(x["repair"]["mean"]),
                 sd4(rr["delta"]), sd4(pr["delta"]), sd4(pr["ci95"][0]), sd4(pr["ci95"][1]),
                 sd4(rp["delta"]), sd4(rp["ci95"][0]), sd4(rp["ci95"][1]),
                 P(p) if p < 1e-3 else rnd(p, 2)]
        out.append((lab, cells, src))
    t = td["tumor"]
    v = t["delta_postproc_tumor_exempt_minus_raw"]
    out.append(("caption tumor-exempt variant", [f(t["postproc_tumor_exempt"]["mean"]), sd4(v["delta"]), P(v["p_raw"])], src))
    return out


@table("Table I.2")
def table_i2():
    from statistics import mean
    src = "results/repair/closed_loop_tab12_13.json"
    d = J(src)
    t13, pp, sec = d["tab13"], d["per_patient"], d["secondary_contrasts"]
    hsrc = "results/repair/holm_f3_f4_2026-10-08.json"
    h = J(hsrc)["F3"]["contrasts"]
    out = []
    for lab, k in (("(a) Raw autonomous", "raw"), ("(b) Generic post-processing", "postproc"),
                   ("(c) Ontology-guided repair", "repair"), ("Reference masks", "ref")):
        x = t13[k]
        c1 = [pct(x["volume_MAPE_pct"]), f(x["organ_site_fidelity"]), f(x["triple_F1"])]
        if k != "ref":
            c1 += [f(x["triple_F1_ci95"][0]), f(x["triple_F1_ci95"][1])]
        # Mismatch@10 from the per-patient values (exact mean), half-up to the printed 2 decimals
        c1 += [f(x["mAP"]), f(mean(pp[k]["mism10"].values()), 2), f(x["spurious@10"], 2),
               f(x["mean_lesion_nodes_per_patient"], 2), f(x["lesion_node_precision"], 2)]
        out.append((lab, c1, src))

    def dlt(a, b, key):            # exact paired mean difference from the per-patient values
        return mean(pp[a][key][c] - pp[b][key][c] for c in pp[a][key])

    for lab, (a, b), sk, hk in (("(c)-(b)", ("repair", "postproc"), "(c)-(b)", "(c)-(b) repair - postproc"),
                                ("(c)-(a)", ("repair", "raw"), "(c)-(a)", "(c)-(a) repair - raw"),
                                ("(b)-(a)", ("postproc", "raw"), "(b)-(a)", "(b)-(a) postproc - raw")):
        ms, fi, ap = sec["mape_strict"][sk], sec["fidelity"][sk], sec["ap"][sk]
        tf = h[hk]
        pp100 = lambda v: rnd(float(Decimal(repr(float(v))) * 100), 1)     # percentage points, scaled exactly
        cells = [pp100(dlt(a, b, "mape_strict")), pp100(ms["ci95"][0]), pp100(ms["ci95"][1]),
                 sd(dlt(a, b, "fidelity")), sd(fi["ci95"][0]), sd(fi["ci95"][1]),
                 sd(dlt(a, b, "triple_f1")), sd(tf["ci95"][0]), sd(tf["ci95"][1]), P(tf["p_holm_stepdown"])]
        out.append((f"{lab} MAPE/fidelity/triple F1", cells, src + " + " + hsrc))
        ap_cells = [sd(dlt(a, b, "ap")), sd(ap["ci95"][0]), sd(ap["ci95"][1])]
        if sk == "(c)-(b)":
            cl = d["F3_triple_F1_contrasts"][hk]
            out.append((f"{lab} sign test counts", [str(cl["improved"]), str(cl["worsened"])], src))
        out.append((f"{lab} mAP", ap_cells, src))
    return out


# ---------------------------------------------------------------- revision analyses (not in the submitted manuscript)
# Checked only when their caption is present in the PDF; the caption name is provisional until the table is placed.
OPTIONAL["Table E.5"] = "closed-world false negatives on restricted views (revision run 1)"


@table("Table E.5")
def table_e5():
    src = "results/queries/restricted_views_closed_world.json"
    d = J(src)
    out = []
    for organ, V in d["views"].items():
        for fid, x in V["families"].items():
            i, c = x["pooled"]["ipkg"], x["pooled"]["closed_world"]
            cells = [thou(i["N_T"]), thou(i["N_F"]), thou(i["N_U"])]
            cells += [f(v) for v in (i["precision"], i["recall"], i["F1"]) if v is not None]
            cells += [rnd(i["indet_pct"], 1), thou(c["N_T"]), thou(c["N_F"]), thou(c["false_negatives"])]
            out.append((f"{organ}-only {fid}", [str(v) for v in cells], src))
    return out


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--table", action="append")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    text = pdf_text(a.pdf)
    names = a.table or list(TABLES)
    n_ok = n_bad = 0
    bad_tables = []
    for name in names:
        rows = TABLES[name]()
        pos = caption_positions(text, name)
        if not pos and name in OPTIONAL and not a.table:
            print(f"{name}: not in this draft (optional: {OPTIONAL[name]})")
            continue
        if not pos:
            print(f"{name}: caption not found in the PDF")
            n_bad += len(rows); bad_tables.append(name)
            continue
        windows = [text[max(0, p - WINDOW_BEFORE): p + WINDOW_AFTER] for p in pos]
        fails = []
        for label, cells, src in rows:
            if any(re.search(row_rx(cells), w) for w in windows):
                n_ok += 1
                if a.verbose:
                    print(f"  ok    {name} | {label} | {' '.join(map(str, cells))}")
            else:
                n_bad += 1
                fails.append((label, cells, src))
        status = "ok" if not fails else f"{len(fails)} row(s) NOT FOUND"
        print(f"{name}: {len(rows) - len(fails)}/{len(rows)} rows match   [{status}]")
        for label, cells, src in fails:
            print(f"    MISSING  {label}: {' | '.join(map(str, cells))}   (expected from {src})")
            # locate the nearest partial match to point at the differing cell
            for k in range(len(cells) - 1, 0, -1):
                hit = next((w for w in windows if re.search(row_rx(cells[:k]), w)), None)
                if hit:
                    m = re.search(row_rx(cells[:k]), hit)
                    print(f"             first {k} cell(s) found; PDF continues: '{hit[m.start():m.end() + 60]}'")
                    break
        if fails:
            bad_tables.append(name)
    print(f"\nsummary: {n_ok} rows match, {n_bad} do not" + (f" (tables: {', '.join(bad_tables)})" if bad_tables else ""))
    print("not checked: Table K.1 (survey database, not public); bracketed pilot values of Table 3 (not regenerated).")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
