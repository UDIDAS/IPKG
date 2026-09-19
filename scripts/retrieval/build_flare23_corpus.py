#!/usr/bin/env python3
"""Build the 1,312-case FLARE23 candidate corpus for the large-pool retrieval stress test
(paper Tab. 8) and the stratified analysis (Tab. 9), from the shipped GT-derived FLARE23 KG
(kg_graphs/unified_mmkg_flare23.json) — no CT data needed.

Record format mirrors corpora/corpus_*_{kits,lits,msd}.json. Decisions (documented for the paper):
  - Organ set: the 5 core KG organs (liver, right_kidney, spleen, pancreas, left_kidney);
    the extended 8-organ annotations present on ~250 cases are not part of the retrieval universe.
  - Kidney laterality is POOLED into a single `kidney` entry (volumes summed, lesion counts summed,
    burden taken from the larger-tumor side) so KiTS's side-agnostic `kidney` phenotype is comparable.
    => |O_FLARE| = 5 in the KG, 4 pooled comparison units in retrieval.
  - burden_cat / multiplicity are taken from the KG lesion nodes as recorded at build time
    (dataset-tercile burden bins, consistent with the single-organ corpora).
  - containment / anatomic_location are not recorded for FLARE23 lesions -> "none"/"unknown"
    (uninformative for the relevance rule, which requires >=2 informative agreements).

-> corpora/corpus_flare23_kg.json
"""
import json
import os
import re
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC = os.path.join(ROOT, "kg_graphs", "unified_mmkg_flare23.json")
OUT = os.path.join(ROOT, "corpora", "corpus_flare23_kg.json")

CORE = {"liver": "liver", "right kidney": "right_kidney", "left kidney": "left_kidney",
        "spleen": "spleen", "pancreas": "pancreas"}


def main():
    u = json.load(open(SRC))
    organs_by_case = defaultdict(dict)      # case -> organ_name -> {volume}
    lesions_by_case = defaultdict(lambda: defaultdict(list))  # case -> organ_name -> [lesion props]

    for n in u["nodes"]:
        if n["type"] == "Organ" and n["label"] in CORE:
            m = re.match(r"organ_(.+?)_(liver|right_kidney|left_kidney|spleen|pancreas)$", n["id"])
            if not m:
                continue
            organs_by_case[m.group(1)][m.group(2)] = n["properties"]
        elif n["type"] == "Lesion":
            m = re.match(r"lesion_(.+?)_(liver|right_kidney|left_kidney|spleen|pancreas)_", n["id"])
            if not m:
                continue
            lesions_by_case[m.group(1)][m.group(2)].append(n["properties"])

    def entry(vol_cm3, les):
        e = {"present": True, "organ_volume_cm3": round(vol_cm3, 2), "has_tumor": bool(les),
             "tumor_volume_cm3": 0.0, "burden_cat": "none", "multiplicity": "none",
             "containment": "none", "anatomic_location": "unknown", "size_cat": "unknown"}
        if les:
            les = sorted(les, key=lambda p: -p.get("gt_volume_cm3", 0.0))
            e["tumor_volume_cm3"] = round(sum(p.get("gt_volume_cm3", 0.0) for p in les), 2)
            e["burden_cat"] = les[0].get("tumorBurden", "none")
            total = sum(p.get("lesion_count", 1) for p in les)
            e["multiplicity"] = "multifocal" if total >= 2 else "solitary"
        return e

    records = []
    for cid in sorted(organs_by_case):
        og, lg = organs_by_case[cid], lesions_by_case.get(cid, {})
        organs = {}
        for side_free in ("liver", "spleen", "pancreas"):
            if side_free in og:
                organs[side_free] = entry(og[side_free]["gt_volume_cm3"], lg.get(side_free, []))
        sides = [s for s in ("right_kidney", "left_kidney") if s in og]
        if sides:
            organs["kidney"] = entry(sum(og[s]["gt_volume_cm3"] for s in sides),
                                     [p for s in sides for p in lg.get(s, [])])
        records.append({"case_id": f"flare23_{cid}", "dataset": "flare23", "granularity": "volume",
                        "observed_organs": sorted(organs), "organs": organs})

    json.dump({"dataset": "flare23", "n": len(records),
               "note": "Derived from the GT-only FLARE23 KG; kidneys pooled; see script docstring.",
               "records": records}, open(OUT, "w"), indent=1)
    n_tumor = sum(1 for r in records if any(o["has_tumor"] for o in r["organs"].values()))
    print(f"FLARE23 corpus: {len(records)} cases ({n_tumor} with tumor) -> {OUT}")


if __name__ == "__main__":
    main()
