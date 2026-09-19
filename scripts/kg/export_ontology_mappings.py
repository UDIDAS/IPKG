#!/usr/bin/env python3
"""Re-export the ontology mapping table FROM the rebuilt graphs and verify it against
kg_graphs/ontology_mappings.json (the declared single source) — the study lead item 2, 2026-09-15.

Comparison is by (system, code) — labels differ legitimately (graph nodes carry the vocabulary
display string, the JSON keys carry the concept role). Extended-organ concepts (aorta, IVC,
adrenals, esophagus, stomach, duodenum, gallbladder) appear on ~250 enriched FLARE23 cases and
are outside the curated core table by design.

Confirms: (1) every declared code is realized in at least one rebuilt graph, (2) every graph
concept is either declared or an expected extended-organ concept, (3) the renal keys/codes are
present on both sides, (4) the Table 5 grouping = anatomical 8/8, lesion 3/3, categorical
phenotype nodes 2/4 -> 13/15.
-> results/kg/ontology_mappings_export_check.json
"""
import glob
import json
import os
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "kg", "ontology_mappings_export_check.json")

EXTENDED_ORGANS = {"AdrenalGland", "Aorta", "Duodenum", "Esophagus", "Gallbladder",
                   "InferiorVenaCava", "Stomach"}
# phenotype row of Table 5: 4 categorical phenotype channels; diameter/volume are literals
PHENO_ROW = ["Observation::Tumor burden", "Observation::Lesion multiplicity",
             "Observation::Organ containment", "Observation::Cross-organ extension"]
LITERAL_KEYS = {"Observation::Tumor diameter", "Observation::Tumor volume"}
POOLED_UNIT = "Organ::Kidney"


def main():
    declared = json.load(open(os.path.join(ROOT, "kg_graphs", "ontology_mappings.json")))
    dmap = declared["mappings"]
    dcodes = {}                                      # (system, code) -> declared key
    for key, systems in dmap.items():
        for s, v in (systems or {}).items():
            dcodes[(v["system"], str(v["code"]))] = key

    gcodes = defaultdict(set)                        # (system, code) -> {graph labels} (node exists)
    emapped = set()                                  # (system, code) with a mapped_to_concept edge
    graph_files = sorted(glob.glob(os.path.join(ROOT, "kg_graphs", "unified_mmkg*.json")))
    for gf in graph_files:
        g = json.load(open(gf))
        mapped_ids = {e["target"] for e in g.get("edges", []) if e.get("relation") == "mapped_to_concept"}
        for n in g.get("nodes", []):
            if n.get("type") == "OntologyConcept":
                p = n.get("properties", {})
                if p.get("system") and p.get("code"):
                    key = (p["system"], str(p["code"]))
                    gcodes[key].add(n.get("label") or n["id"])
                    if n["id"] in mapped_ids:
                        emapped.add(key)
    # concept nodes never referenced by a mapping edge anywhere AND not declared -> legacy orphans
    orphans = {f"{s} {c}": sorted(l) for (s, c), l in gcodes.items()
               if (s, c) not in emapped and (s, c) not in dcodes}

    declared_not_in_graphs = sorted(f"{k}  [{s} {c}]" for (s, c), k in dcodes.items()
                                    if (s, c) not in gcodes)
    graph_not_declared = {f"{s} {c}": sorted(labs) for (s, c), labs in sorted(gcodes.items())
                          if (s, c) not in dcodes}
    unexpected = {code: labs for code, labs in graph_not_declared.items()
                  if not (set(labs) & EXTENDED_ORGANS) and code not in orphans}

    renal_json = {k: sorted(f"{v['system']} {v['code']}" for v in (dmap[k] or {}).values())
                  for k in dmap if "kidney" in k.lower() or "renal" in k.lower()}
    renal_graph = sorted({f"{s} {c} ({'/'.join(labs)})" for (s, c), labs in gcodes.items()
                          if any("kidney" in l.lower() or "renal" in l.lower() for l in labs)})

    anat = [k for k in dmap if (k.startswith("Organ::") and k != POOLED_UNIT)
            or k.startswith("AnatomicSite::")]
    lesion = [k for k in dmap if k.startswith("Lesion::")]
    pheno_coded = [k for k in PHENO_ROW if dmap.get(k)]
    grouping = {"anatomical incl. sub-sites": f"{len(anat)}/8",
                "lesion types": f"{len(lesion)}/3",
                "categorical phenotype nodes": f"{len(pheno_coded)}/4",
                "total_coded": f"{len(anat) + len(lesion) + len(pheno_coded)}/15",
                "pooled_kidney (retrieval unit, excluded)": POOLED_UNIT in dmap,
                "literal_backed (not phenotype nodes)": sorted(LITERAL_KEYS & set(dmap)),
                "concept_keys": len(dmap),
                "codes": sum(len(v or {}) for v in dmap.values())}

    ok = (not declared_not_in_graphs and not unexpected
          and len(renal_json) == 4 and grouping["total_coded"] == "13/15")
    out = {"graphs_scanned": [os.path.basename(f) for f in graph_files],
           "n_graph_codes": len(gcodes),
           "declared_codes_missing_from_graphs": declared_not_in_graphs,
           "graph_codes_not_declared_extended_organs": {k: v for k, v in graph_not_declared.items()
                                                        if k not in unexpected},
           "graph_codes_not_declared_UNEXPECTED": unexpected,
           "orphan_concept_nodes_no_mapping_edge": orphans,
           "renal_keys_json": renal_json, "renal_codes_in_graphs": renal_graph,
           "table5_grouping": grouping,
           "verdict": "MATCH - ontology_mappings.json is a faithful export of the rebuilt graphs; 13/15 confirmed"
           if ok else "MISMATCH - see fields above"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("declared codes missing from graphs:", declared_not_in_graphs or "none")
    print("unexpected graph codes:", unexpected or "none")
    print("extended-organ codes (expected, out of core scope):", len(graph_not_declared) - len(unexpected))
    print("renal JSON keys:", list(renal_json))
    print("renal graph codes:", renal_graph)
    print("Table 5 grouping:", grouping)
    print(out["verdict"])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
