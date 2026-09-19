#!/usr/bin/env python3
"""Ground the kidney lesion nodes of the shipped FLARE23 reference graph (built by the handoff builder, not
reproducible from this bundle) to the renal-tumor concept, exactly as liver / pancreatic lesions are grounded:
one `mapped_to_concept` edge per lesion to SNOMED CT 126880001 "Neoplasm of kidney" (NCIt C3150 "Kidney
Neoplasm" added as a second concept node, as for the other lesion types).  Idempotent.  Edits
kg_graphs/imaging_kg_flare23.ttl and kg_graphs/unified_mmkg_flare23.json in place.
"""
import json
import os

from rdflib import Graph, Namespace, URIRef, Literal, RDF

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
KG = os.path.join(ROOT, "kg_graphs")
MMKG = Namespace("http://example.org/mmkg/schema/"); INST = Namespace("http://example.org/mmkg/instance/")
CONCEPTS = [("SNOMEDCT", "SNOMED CT", "126880001", "Neoplasm of kidney"), ("NCIt", "NCIt", "C3150", "Kidney Neoplasm")]


def main():
    # ---- unified JSON
    p = os.path.join(KG, "unified_mmkg_flare23.json"); u = json.load(open(p))
    ids = {n["id"] for n in u["nodes"]}
    for tag, system, code, disp in CONCEPTS:
        cid = f"concept_{tag}_{code}"
        if cid not in ids:
            u["nodes"].append({"id": cid, "type": "OntologyConcept", "properties": {"system": system, "code": code}, "label": "RenalTumor"})
    have = {(e["source"], e["target"]) for e in u["edges"] if e["relation"] == "mapped_to_concept"}
    kid = [n["id"] for n in u["nodes"] if n["type"] == "Lesion" and "kidney" in n["id"]]
    added = 0
    for l in kid:
        if (l, "concept_SNOMEDCT_126880001") not in have:
            u["edges"].append({"source": l, "relation": "mapped_to_concept", "target": "concept_SNOMEDCT_126880001"}); added += 1
    json.dump(u, open(p, "w"))
    print(f"unified_mmkg_flare23.json: {len(kid)} kidney lesions, {added} concept edges added")

    # ---- Turtle
    p = os.path.join(KG, "imaging_kg_flare23.ttl"); g = Graph(); g.parse(p, format="turtle")
    n0 = len(g)
    for tag, system, code, disp in CONCEPTS:
        cu = INST[f"concept_{tag}_{code}"]
        g.add((cu, RDF.type, MMKG.OntologyConcept)); g.add((cu, MMKG.code, Literal(code)))
        g.add((cu, MMKG.display, Literal("RenalTumor"))); g.add((cu, MMKG.system, Literal(system)))
    for l in kid:
        g.add((INST[l], MMKG.mapped_to_concept, INST["concept_SNOMEDCT_126880001"]))
        g.add((INST[l], RDF.type, MMKG.RenalTumor))
    g.serialize(p, format="turtle")
    print(f"imaging_kg_flare23.ttl: {n0} -> {len(g)} triples")


if __name__ == "__main__":
    main()
