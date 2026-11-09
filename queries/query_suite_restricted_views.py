#!/usr/bin/env python3
"""Closed-world false negatives on restricted views of the 576 semi-oracle FLARE23 predicted graphs.

Question: when a multi-organ case is seen the way a single-organ collection sees it (LiTS: liver only; KiTS23: kidneys
only; MSD: pancreas only), how many reference-true answers does a closed-world evaluator turn into false negatives,
and what does the three-valued evaluator return instead?

Views. From each predicted FLARE23 record (corpora/corpus_predicted_flare23.json, the graphs behind Table E.4) three
restricted views are built: liver-only, kidney-only, pancreas-only. A view keeps the one organ's phenotype record
unchanged, sets observed_organs to that organ, and drops every other organ, which is exactly how a LiTS / KiTS23 / MSD
record is carried. A case enters a view only if its predicted record observes that organ (liver 575, kidney 576,
pancreas 575 of 576). Phenotypes, burden categories (the FLARE23 cut-points already in the records) and the 65.4 cm3
hepatic threshold are those of the shipped records; nothing is re-derived and no step is stochastic.

Evaluators, on the same view:
  IPKG (three-valued)  the verdict functions of queries/query_suite_eval.py, unchanged: T / F over observed anatomy,
                       U where the condition touches an organ the view does not observe.
  closed-world         every organ the view does not observe is filled with an absent-negative record (no lesion,
                       burden "none", multiplicity "none", volume 0) and the same verdict functions are applied, so
                       every unsupported constraint is scored F. The script asserts that this equals mapping each
                       IPKG U to F, verdict by verdict.

Reference: the verdict on the full four-organ reference record of the same case (corpora/corpus_flare23_kg.json),
scored by the same functions. Reference-indeterminate instantiations (spleen on the 5 records without a spleen
annotation) are counted separately and excluded from P/R/F1 and from the false-negative count.

Families: Q2, Q3, Q7b instantiated per organ over liver, kidney, pancreas, spleen and pooled (as in Table E.4), and
also split into the observed-organ instantiation and the hidden-organ instantiations; Q4a, Q4b, Q5 once per case.

Outputs per family x restricted organ:
  IPKG: N_T, N_F, N_U, precision / recall / F1 over determinate pairs, false negatives (reference T, IPKG F),
        reference-T answered U
  closed-world: N_T, N_F, precision / recall / F1, false negatives (reference T, closed-world F)

  python queries/query_suite_restricted_views.py
-> results/queries/restricted_views_closed_world.json, results/queries/restricted_views_closed_world_table.tex
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, _HERE)
import query_suite_eval as Q                                    # noqa: E402

ROOT = os.path.abspath(os.path.join(_HERE, ".."))
OUT = os.path.join(ROOT, "results", "queries", "restricted_views_closed_world.json")
TEX = os.path.join(ROOT, "results", "queries", "restricted_views_closed_world_table.tex")
ORGANS = ["liver", "kidney", "pancreas", "spleen"]
VIEWS = [("liver", "liver-only (as LiTS)"), ("kidney", "kidney-only (as KiTS23)"), ("pancreas", "pancreas-only (as MSD)")]
ABSENT = {"present": False, "organ_volume_cm3": 0.0, "has_tumor": False, "tumor_volume_cm3": 0.0, "tumor_voxels": 0,
          "burden_cat": "none", "multiplicity": "none", "containment": "none", "anatomic_location": "unknown",
          "size_cat": "unknown"}


def view(rec, organ):
    return {**rec, "observed_organs": [organ], "organs": {organ: rec["organs"][organ]}}


def closed_world(rec):
    return {**rec, "observed_organs": list(ORGANS), "organs": {o: rec["organs"].get(o, dict(ABSENT)) for o in ORGANS}}


FAMILIES = [
    ("Q2", "Organ-specific lookup (pooled)", "per_organ", Q.v_organ_lookup),
    ("Q3", "Largest burden tercile in organ o (pooled)", "per_organ", Q.v_high_bin),
    ("Q4a", "Hepatic volume > 65.4 cm3 AND renal lesion", "single", lambda r: Q.v_compositional(r, "kidney")),
    ("Q4b", "Hepatic volume > 65.4 cm3 AND splenic lesion", "single", lambda r: Q.v_compositional(r, "spleen")),
    ("Q5", "Cross-organ involvement", "single", Q.v_cross_organ),
    ("Q7b", "Multifocal disease in one organ (pooled)", "per_organ", Q.v_multifocal),
]


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else None
    R = tp / (tp + fn) if tp + fn else None
    F1 = 2 * P * R / (P + R) if P and R else (0.0 if P is not None and R is not None else None)
    r = lambda x: None if x is None else round(x, 3)
    return r(P), r(R), r(F1)


def tally(pairs):
    """pairs: list of (system verdict, reference verdict)."""
    c = {"N_T": 0, "N_F": 0, "N_U": 0, "TP": 0, "FP": 0, "FN": 0, "TN": 0, "ref_T_answered_U": 0, "ref_U": 0, "ref_T": 0}
    for vs, vr in pairs:
        c["N_" + vs] += 1
        c["ref_U"] += vr == "U"; c["ref_T"] += vr == "T"
        if vr == "T" and vs == "U":
            c["ref_T_answered_U"] += 1
        if vs in "TF" and vr in "TF":
            k = {("T", "T"): "TP", ("T", "F"): "FP", ("F", "T"): "FN", ("F", "F"): "TN"}[(vs, vr)]
            c[k] += 1
    n = c["N_T"] + c["N_F"] + c["N_U"]
    c["precision"], c["recall"], c["F1"] = prf(c["TP"], c["FP"], c["FN"])
    c["indet_pct"] = round(100 * c["N_U"] / n, 1) if n else None
    c["false_negatives"] = c["FN"]
    return c


def main():
    pred = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_predicted_flare23.json")))["records"]}
    gt = {r["case_id"]: r for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg.json")))["records"]}
    ids = sorted(i for i in pred if i in gt)
    assert len(ids) == 576, len(ids)
    ref_lesion_organs = sum(p["has_tumor"] for c in ids for p in gt[c]["organs"].values())

    out = {"note": __doc__.strip(), "n_cases": len(ids), "reference_tumor_bearing_organ_records": ref_lesion_organs,
           "views": {}}
    n_checked = 0
    for organ, vlabel in VIEWS:
        cids = [c for c in ids if organ in pred[c]["organs"]]
        fam_out = {}
        for fid, flabel, kind, fn in FAMILIES:
            inst = [(o, (lambda r, o=o, fn=fn: fn(r, o))) for o in ORGANS] if kind == "per_organ" else [("", fn)]
            groups = {"pooled": [], "observed_organ": [], "hidden_organs": []}
            groups_cw = {"pooled": [], "observed_organ": [], "hidden_organs": []}
            for c in cids:
                v, ref = view(pred[c], organ), gt[c]
                for o, f in inst:
                    vi, vc, vr = f(v), f(closed_world(v)), f(ref)
                    assert vc == ("F" if vi == "U" else vi), (fid, c, o, vi, vc)
                    n_checked += 1
                    g = "pooled"
                    groups[g].append((vi, vr)); groups_cw[g].append((vc, vr))
                    if kind == "per_organ":
                        g = "observed_organ" if o == organ else "hidden_organs"
                        groups[g].append((vi, vr)); groups_cw[g].append((vc, vr))
            rec = {"family": flabel, "n_cases": len(cids)}
            for g in (("pooled", "observed_organ", "hidden_organs") if kind == "per_organ" else ("pooled",)):
                ip, cw = tally(groups[g]), tally(groups_cw[g])
                rec[g] = {
                    "reference": {"T": ip["ref_T"], "U": ip["ref_U"]},
                    "ipkg": {k: ip[k] for k in ("N_T", "N_F", "N_U", "precision", "recall", "F1", "indet_pct",
                                                 "false_negatives", "ref_T_answered_U")},
                    "closed_world": {k: cw[k] for k in ("N_T", "N_F", "precision", "recall", "F1", "false_negatives")},
                }
                assert cw["N_U"] == 0
                assert cw["false_negatives"] == ip["false_negatives"] + ip["ref_T_answered_U"]
            fam_out[fid] = rec
        out["views"][organ] = {"label": vlabel, "n_cases": len(cids), "families": fam_out}
    out["closed_world_equals_U_to_F_checked_verdicts"] = n_checked

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)

    # Table 3 format, one block per restricted organ: IPKG N_T N_F N_U P R F1 Indet. | closed-world N_T N_F | CW FN
    d = lambda x, nd=3: "---" if x is None else f"{x:.{nd}f}"
    L = ["% Restricted views of the 576 semi-oracle FLARE23 predicted graphs; reference = full four-organ reference verdict.",
         "% Emitted by queries/query_suite_restricted_views.py from results/queries/restricted_views_closed_world.json.",
         "% Columns: ID & Family & IPKG N_T & N_F & N_U & Precision & Recall & F1 & Indet. (%) & closed-world N_T & N_F & "
         "closed-world FN (reference T, closed-world F) & reference T"]
    for organ, vlabel in VIEWS:
        V = out["views"][organ]
        L.append(f"\\multicolumn{{13}}{{l}}{{\\emph{{{vlabel}; {V['n_cases']} cases}}}} \\\\")
        for fid, *_ in FAMILIES:
            x = V["families"][fid]["pooled"]; i, c = x["ipkg"], x["closed_world"]
            fam = V["families"][fid]["family"].replace(">", "$>$").replace("AND", "$\\wedge$").replace("cm3", "cm$^3$")
            L.append(f" {fid} & {fam} & {i['N_T']:,} & {i['N_F']:,} & {i['N_U']:,} & {d(i['precision'])} & {d(i['recall'])} & "
                     f"{d(i['F1'])} & {d(i['indet_pct'], 1)} & {c['N_T']:,} & {c['N_F']:,} & {c['false_negatives']:,} & "
                     f"{x['reference']['T']:,} \\\\")
    open(TEX, "w").write("\n".join(L) + "\n")

    print(f"{len(ids)} cases; reference tumor-bearing organ records: {ref_lesion_organs}; "
          f"closed-world == U->F checked on {n_checked} verdicts")
    hdr = f"  {'':5s} {'refT':>5s} | IPKG {'N_T':>4s} {'N_F':>5s} {'N_U':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'FN':>4s} | CW {'N_T':>4s} {'N_F':>5s} {'R':>6s} {'FN':>4s}"
    for organ, vlabel in VIEWS:
        V = out["views"][organ]
        print(f"\n{vlabel}, n = {V['n_cases']}\n{hdr}")
        for fid, *_ in FAMILIES:
            for g in ("pooled", "observed_organ", "hidden_organs"):
                x = V["families"][fid].get(g)
                if not x:
                    continue
                i, c = x["ipkg"], x["closed_world"]
                tag = fid if g == "pooled" else f"  {'obs' if g == 'observed_organ' else 'hid'}"
                print(f"  {tag:5s} {x['reference']['T']:5d} | IPKG {i['N_T']:4d} {i['N_F']:5d} {i['N_U']:5d} {d(i['precision']):>6s} "
                      f"{d(i['recall']):>6s} {d(i['F1']):>6s} {i['false_negatives']:4d} | CW {c['N_T']:4d} {c['N_F']:5d} "
                      f"{d(c['recall']):>6s} {c['false_negatives']:4d}")
    print(f"\n-> {OUT}\n-> {TEX}")


if __name__ == "__main__":
    main()
