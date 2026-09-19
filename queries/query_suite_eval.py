#!/usr/bin/env python3
"""Ten-family query-suite evaluation (paper Tab. 6, §5.4) from the shipped corpora + KGs.

Verdict semantics (three-valued, observability-aware):
  T / F : the query condition is determinately true / false over the case's OBSERVED anatomy
  U     : the condition touches anatomy the case does not observe -> indeterminate

Two cohorts, reported separately and honestly:
  1. The 113 reference-evaluable single-organ cases: system verdicts from the PREDICTED
     phenotype records, reference labels from the GT records. Precision/recall/F1 are over
     verdicts where both system and reference are determinate; Indet. = N_U/(N_T+N_F+N_U).
  2. The 1,312-case FLARE23 cohort: the shipped FLARE23 graph is GT-derived, so it yields
     reference verdict COUNTS for the multi-organ families (Q4a/Q4b/Q5) but no honest
     system-vs-reference P/R/F1 (that needs predicted FLARE23 graphs -> GPU rerun).

Q6 (spatial subregion) and Q8 (rare-case discovery) are specified but not evaluated in the
draft; Q9 (similar-case retrieval) is the retrieval benchmark (Tabs 7-8); Q10 reports the
SPARQL execution latency of the ten stock queries on the shipped graphs (query stage only —
end-to-end workflow latency additionally includes segmentation + graph build, not measurable
from the bundle).

-> results/queries/query_suite_tab6.json
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "results", "queries")

ORGANS = ("liver", "kidney", "pancreas")
DIAM_5CM_VOL = 65.4  # cm3 of a 5 cm sphere — volume proxy where diameters are unavailable


def load(kind):
    recs = []
    for ds in ("kits", "lits", "msd"):
        recs += json.load(open(os.path.join(ROOT, "corpora", f"corpus_{kind}_{ds}.json")))["records"]
    return {r["case_id"]: r for r in recs}


# ---------- per-case three-valued verdicts on a phenotype record ----------
def v_organ_lookup(rec, o):          # Q2: lesion located in organ o
    if o not in rec["organs"]:
        return "U"
    return "T" if rec["organs"][o]["has_tumor"] else "F"


def v_high_bin(rec, o):              # Q3: lesion in largest (high) burden bin of organ o
    if o not in rec["organs"]:
        return "U"
    p = rec["organs"][o]
    return "T" if (p["has_tumor"] and p["burden_cat"] == "high") else "F"


def v_multifocal(rec, o):            # Q7b: multifocal disease within organ o
    if o not in rec["organs"]:
        return "U"
    return "T" if rec["organs"][o]["multiplicity"] == "multifocal" else "F"


def v_high_burden_any(rec):          # Q7a: high tumor-burden category in observed anatomy
    if any(p["has_tumor"] and p["burden_cat"] == "high" for p in rec["organs"].values()):
        return "T"
    return "F"


def v_compositional(rec, organ_b):   # Q4a/Q4b: hepatic lesion >5cm AND lesion in organ_b
    liver, ob = rec["organs"].get("liver"), rec["organs"].get(organ_b)
    hep = None if liver is None else (liver["has_tumor"] and liver["tumor_volume_cm3"] > DIAM_5CM_VOL)
    other = None if ob is None else ob["has_tumor"]
    if hep is False or other is False:
        return "F"                    # a determinate false conjunct decides the conjunction
    if hep is None or other is None:
        return "U"
    return "T"


def v_cross_organ(rec):              # Q5: lesions in >=2 organs
    n_tumor = sum(1 for p in rec["organs"].values() if p["has_tumor"])
    if n_tumor >= 2:
        return "T"
    # cannot rule out lesions in unobserved anatomy
    return "F" if len(rec["organs"]) >= 2 else "U"


def score(family, instantiations, pred, gt, ids):
    """instantiations: list of (tag, verdict_fn). Pool verdicts; P/R/F1 where both determinate."""
    NT = NF = NU = 0
    tp = fp = fn = tn = 0
    indet_by_ds = {}
    for cid in ids:
        ds = pred[cid]["dataset"]
        for _, fn_v in instantiations:
            vs, vr = fn_v(pred[cid]), fn_v(gt[cid])
            NT += vs == "T"; NF += vs == "F"; NU += vs == "U"
            d = indet_by_ds.setdefault(ds, [0, 0])
            d[1] += 1; d[0] += vs == "U"
            if vs in "TF" and vr in "TF":
                tp += vs == "T" and vr == "T"; fp += vs == "T" and vr == "F"
                fn += vs == "F" and vr == "T"; tn += vs == "F" and vr == "F"
    P = tp / (tp + fp) if tp + fp else None
    R = tp / (tp + fn) if tp + fn else None
    F1 = 2 * P * R / (P + R) if P and R else (0.0 if P is not None and R is not None else None)
    rr = lambda x: round(x, 3) if x is not None else None
    return {"family": family, "N_T": NT, "N_F": NF, "N_U": NU,
            "precision": rr(P), "recall": rr(R), "F1": rr(F1),
            "indet_rate": rr(NU / (NT + NF + NU)) if NT + NF + NU else None,
            "indet_rate_by_dataset": {ds: rr(a / b) for ds, (a, b) in indet_by_ds.items()},
            "n_scored_TF_pairs": tp + fp + fn + tn}


def flare_counts():
    """Reference verdict counts for the multi-organ families on the 1,312-case FLARE23 KG corpus
    (GT-derived graph: counts only, no system-vs-reference P/R/F1)."""
    recs = json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg.json")))["records"]
    fams = {"Q4a hepatic>5cm AND renal lesion": lambda r: v_compositional(r, "kidney"),
            "Q4b hepatic>5cm AND splenic lesion": lambda r: v_compositional(r, "spleen"),
            "Q5 cross-organ involvement (>=2 organs)": v_cross_organ,
            "Q7a high tumor burden": v_high_burden_any}
    out = {}
    for name, f in fams.items():
        c = {"T": 0, "F": 0, "U": 0}
        for r in recs:
            c[f(r)] += 1
        out[name] = {"N_T": c["T"], "N_F": c["F"], "N_U": c["U"],
                     "indet_rate": round(c["U"] / len(recs), 3)}
    return out


def sparql_latency(n_rep=5):
    """Execution latency (ms) of the ten stock SPARQL queries on the shipped graphs."""
    from rdflib import Graph
    sys.path.insert(0, HERE)
    from run_queries import QUERIES, PREFIX
    out = {}
    for tag, ttl in [("flare23 (1,312 cases)", "imaging_kg_flare23.ttl"),
                     ("predicted_kits", "imaging_kg_predicted_kits.ttl"),
                     ("predicted_lits", "imaging_kg_predicted_lits.ttl"),
                     ("predicted_msd", "imaging_kg_predicted_msd.ttl")]:
        g = Graph()
        t0 = time.perf_counter()
        g.parse(os.path.join(ROOT, "kg_graphs", ttl), format="turtle")
        load_ms = (time.perf_counter() - t0) * 1000
        per_q = {}
        for name, q in QUERIES.items():
            times = []
            for _ in range(n_rep):
                t0 = time.perf_counter()
                list(g.query(PREFIX + q))
                times.append((time.perf_counter() - t0) * 1000)
            per_q[name] = round(statistics.median(times), 1)
        vals = sorted(per_q.values())
        q1, q3 = vals[len(vals) // 4], vals[3 * len(vals) // 4]
        out[tag] = {"n_triples": len(g), "graph_load_ms": round(load_ms), "per_query_median_ms": per_q,
                    "across_queries": {"median_ms": round(statistics.median(vals), 1),
                                       "IQR_ms": [q1, q3]}}
        print(f"  {tag}: {len(g):,} triples, median query {statistics.median(vals):.1f} ms "
              f"(IQR {q1}-{q3})")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pred, gt = load("predicted"), load("gt")
    ids = sorted(i for i in pred if i in gt)
    per_org = lambda f: [(o, (lambda r, o=o: f(r, o))) for o in ORGANS]
    rows = [
        score("Q2 organ-specific lookup (pooled liver/kidney/pancreas)", per_org(v_organ_lookup), pred, gt, ids),
        score("Q3 phenotype retrieval: largest burden bin (pooled)", per_org(v_high_bin), pred, gt, ids),
        score("Q4a compositional: hepatic>5cm AND renal lesion", [("", lambda r: v_compositional(r, "kidney"))], pred, gt, ids),
        score("Q4b compositional: hepatic>5cm AND splenic lesion", [("", lambda r: v_compositional(r, "spleen"))], pred, gt, ids),
        score("Q5 cross-organ involvement (>=2 organs)", [("", v_cross_organ)], pred, gt, ids),
        score("Q7a high tumor burden (observed anatomy)", [("", v_high_burden_any)], pred, gt, ids),
        score("Q7b multifocal disease in one organ (pooled)", per_org(v_multifocal), pred, gt, ids),
    ]
    print("Query suite on the 113 reference-evaluable cases (system=predicted, reference=GT):")
    for r in rows:
        print(f"  {r['family']:55s} T/F/U={r['N_T']}/{r['N_F']}/{r['N_U']} "
              f"P={r['precision']} R={r['recall']} F1={r['F1']} indet={r['indet_rate']}")
    det = [r for r in rows if r["F1"] is not None]
    macro = {"macro_P": round(sum(r["precision"] for r in det) / len(det), 3),
             "macro_R": round(sum(r["recall"] for r in det) / len(det), 3),
             "macro_F1": round(sum(r["F1"] for r in det) / len(det), 3),
             "n_evaluable_families": len(det)}
    print(f"  macro over evaluable families: P={macro['macro_P']} R={macro['macro_R']} F1={macro['macro_F1']}")

    print("FLARE23 (GT-derived graph) reference counts for multi-organ families:")
    fc = flare_counts()
    for k, v in fc.items():
        print(f"  {k:45s} T/F/U={v['N_T']}/{v['N_F']}/{v['N_U']}")

    print("SPARQL latency (query stage only):")
    lat = sparql_latency()

    json.dump({"cohort_113": {"n_cases": len(ids), "rows": rows, "macro_over_evaluable": macro},
               "flare23_reference_counts": fc,
               "q9_similar_case_retrieval": "see results/retrieval/retrieval_tab7_primary.json / _tab8_stress.json",
               "q10_latency_sparql_ms": lat,
               "not_evaluated": "Q6 spatial subregion, Q8 rare-case discovery (out of scope per draft); "
                                "P/R/F1 for Q4a/Q4b/Q5 on FLARE23 needs predicted FLARE23 graphs (GPU rerun)."},
              open(os.path.join(OUT, "query_suite_tab6.json"), "w"), indent=1)
    print(f"-> {os.path.join(OUT, 'query_suite_tab6.json')}")


if __name__ == "__main__":
    main()
