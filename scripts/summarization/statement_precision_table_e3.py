#!/usr/bin/env python3
"""Table E.3 (statement-level factual consistency of the generated reports, n = 113) assembled from committed
artifacts, with the all-statements total.

Rows and sources:
  organ present, tumor present in organ, burden category, lesion multiplicity
      results/summarization/summarization_metrics.json (per_type_statement_precision; frozen summ_metrics.py run)
  containment, closed rule (Section 3.2)
      results/kg/containment_reextraction_113_predicted.json (table11_containment_precision_v2); per-dataset
      KiTS23 / LiTS / MSD hit counts recovered from their precisions and summed
  anatomic site, pancreatic sub-site, MSD only
      recomputed here from the delivered corpora: corpora/corpus_predicted_msd.json vs corpora/corpus_gt_msd.json,
      one statement per MSD case whose predicted record asserts a tumor and names a sub-site (the delivered
      agreement; the sub-site words written onto liver and kidney records are void and not scored)

Where an artifact stores precision and n but not the hit count, the count is the unique integer h with
round(h / n, 3) == precision; the script refuses to write if any count is not unique.

  python scripts/summarization/statement_precision_table_e3.py
-> results/summarization/statement_precision_table_e3.json
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
P = lambda *a: os.path.join(ROOT, *a)


def hits(precision, n):
    cand = [h for h in range(n + 1) if round(h / n, 3) == round(precision, 3)]
    if len(cand) != 1:
        raise SystemExit(f"hit count not unique for precision {precision} over {n}: {cand}")
    return cand[0]


def records(path):
    d = json.load(open(path))
    d = d["records"] if isinstance(d, dict) and "records" in d else d
    return d if isinstance(d, dict) else {r["case_id"]: r for r in d}


def main():
    sm = json.load(open(P("results", "summarization", "summarization_metrics.json")))["per_type_statement_precision"]
    rows = []
    for key, label in (("present", "Organ present"), ("has_tumor", "Tumor present in organ"),
                       ("burden_cat", "Burden category"), ("multiplicity", "Lesion multiplicity")):
        n = sm[key]["n"]; rows.append({"row": label, "n": n, "hits": hits(sm[key]["precision"], n),
                                       "source": "results/summarization/summarization_metrics.json"})

    pred, ref = records(P("corpora", "corpus_predicted_msd.json")), records(P("corpora", "corpus_gt_msd.json"))
    n = h = 0
    for cid, rec in pred.items():
        po, ro = rec["organs"]["pancreas"], ref[cid]["organs"]["pancreas"]
        if po.get("has_tumor") and po.get("anatomic_location") not in (None, "na", "unknown", "none"):
            n += 1; h += po["anatomic_location"] == ro.get("anatomic_location")
    rows.append({"row": "Anatomic site (pancreatic sub-site, MSD only)", "n": n, "hits": h,
                 "source": "corpora/corpus_predicted_msd.json vs corpora/corpus_gt_msd.json (delivered corpora)"})

    cv = json.load(open(P("results", "kg", "containment_reextraction_113_predicted.json")))["table11_containment_precision_v2"]
    per = {ds: hits(v["precision"], v["n"]) for ds, v in cv["per_dataset"].items()}
    nc, hc = cv["n_statements"], sum(per.values())
    if hits(cv["precision"], nc) != hc:
        raise SystemExit(f"containment per-dataset hits {per} do not sum to the pooled precision {cv['precision']}")
    rows.append({"row": "Containment (closed rule)", "n": nc, "hits": hc, "per_dataset_hits": per,
                 "source": "results/kg/containment_reextraction_113_predicted.json"})

    for r in rows:
        r["precision"] = round(r["hits"] / r["n"], 3)
    N, H = sum(r["n"] for r in rows), sum(r["hits"] for r in rows)
    out = {"note": __doc__.strip(), "rows": rows, "all_statements": {"n": N, "hits": H, "precision": round(H / N, 3)}}
    json.dump(out, open(P("results", "summarization", "statement_precision_table_e3.json"), "w"), indent=1)
    for r in rows:
        print(f"  {r['row']:48s} {r['n']:4d} {r['hits']:4d} {r['precision']:.3f}")
    print(f"  {'All statements':48s} {N:4d} {H:4d} {H / N:.3f}")


if __name__ == "__main__":
    main()
