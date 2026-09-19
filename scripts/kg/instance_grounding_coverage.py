#!/usr/bin/env python3
"""Instance-level ontology grounding: fraction of Organ / Lesion nodes in each shipped graph that carry a
mapped_to_concept edge (Tab. 5 reports vocabulary coverage; this is the graph-level complement).
Output: results/kg/instance_grounding_coverage.json
"""
import glob
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
out = {}
for f in sorted(glob.glob(os.path.join(ROOT, "kg_graphs", "unified_mmkg*.json"))):
    u = json.load(open(f)); src = {e["source"] for e in u["edges"] if e["relation"] == "mapped_to_concept"}
    row = {}
    for typ in ("Organ", "Lesion"):
        ids = [n["id"] for n in u["nodes"] if n["type"] == typ]; g = sum(i in src for i in ids)
        row[typ] = {"nodes": len(ids), "grounded": g, "coverage": round(g / len(ids), 3) if ids else None}
    out[os.path.basename(f)[:-5]] = row
    print(f"  {os.path.basename(f)[:-5]:32s} organs {row['Organ']['grounded']}/{row['Organ']['nodes']}  lesions {row['Lesion']['grounded']}/{row['Lesion']['nodes']}")
json.dump({"note": "fraction of organ / lesion instance nodes with a mapped_to_concept edge, per shipped graph (after the renal-tumor / "
           "pooled-kidney grounding of 2026-09-14; before it the KiTS predicted graph had 0/36 organs and 0/35 lesions grounded and the "
           "FLARE23 graph 0/202 kidney lesions)", "graphs": out}, open(os.path.join(ROOT, "results", "kg", "instance_grounding_coverage.json"), "w"), indent=1)
