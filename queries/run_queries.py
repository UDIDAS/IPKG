#!/usr/bin/env python3
"""Execute the FLARE23 MMKG queries. Loads imaging_kg_flare23.ttl with rdflib and runs a set of
named SPARQL queries spanning ImagingCase -> Organ <- Lesion, with ontology grounding.

  pip install rdflib
  python run_queries.py            # run all
  python run_queries.py <name>     # run one
"""
import os, sys
from rdflib import Graph

HERE = os.path.dirname(os.path.abspath(__file__))
TTL = os.path.join(HERE, "imaging_kg_flare23.ttl")
if not os.path.exists(TTL):                                    # git checkout: fall back to the repo copy
    TTL = os.path.join(HERE, "..", "..", "kg", "graph", "imaging_kg_flare23.ttl")
PREFIX = "PREFIX mmkg: <http://example.org/mmkg/schema/>\n"

QUERIES = {
"01_cohort_summary":
"""SELECT ?organ (COUNT(?o) AS ?n) (ROUND(AVG(?v)) AS ?avg_cm3)
WHERE { ?o a mmkg:Organ ; a ?organ ; mmkg:gt_volume_cm3 ?v . FILTER(?organ != mmkg:Organ) }
GROUP BY ?organ ORDER BY DESC(?n)""",

"02_tumor_burden_by_organ":
"""SELECT ?organType ?burden (COUNT(?l) AS ?n)
WHERE { ?l a mmkg:Lesion ; mmkg:tumorBurden ?burden ; mmkg:located_in ?o .
        ?o a ?organType . FILTER(?organType != mmkg:Organ) }
GROUP BY ?organType ?burden ORDER BY ?organType DESC(?n)""",

"03_high_burden_tumors":
"""SELECT ?case ?organType ?vol ?diam
WHERE { ?c a mmkg:ImagingCase ; mmkg:case_id ?case ; mmkg:depicts_organ ?o .
        ?l mmkg:located_in ?o ; mmkg:tumorBurden "high" ;
           mmkg:gt_volume_cm3 ?vol ; mmkg:gt_max_diameter_mm ?diam .
        ?o a ?organType . FILTER(?organType != mmkg:Organ) }
ORDER BY DESC(?vol) LIMIT 15""",

"04_pancreatic_tumors":
"""SELECT ?case ?burden ?mult ?vol
WHERE { ?c a mmkg:ImagingCase ; mmkg:case_id ?case ; mmkg:depicts_organ ?o .
        ?o a mmkg:Pancreas .
        ?l mmkg:located_in ?o ; mmkg:tumorBurden ?burden ;
           mmkg:lesionMultiplicity ?mult ; mmkg:gt_volume_cm3 ?vol . }
ORDER BY DESC(?vol) LIMIT 15""",

"05_kidney_tumors":
"""SELECT ?case ?burden ?mult ?vol
WHERE { ?c a mmkg:ImagingCase ; mmkg:case_id ?case ; mmkg:depicts_organ ?o .
        ?o a mmkg:Kidney .
        ?l mmkg:located_in ?o ; mmkg:tumorBurden ?burden ;
           mmkg:lesionMultiplicity ?mult ; mmkg:gt_volume_cm3 ?vol . }
ORDER BY DESC(?vol) LIMIT 15""",

"06_multifocal_tumors":
"""SELECT ?case ?organType ?count ?vol
WHERE { ?c a mmkg:ImagingCase ; mmkg:case_id ?case ; mmkg:depicts_organ ?o .
        ?l mmkg:located_in ?o ; mmkg:lesionMultiplicity "multifocal" ;
           mmkg:lesion_count ?count ; mmkg:gt_volume_cm3 ?vol .
        ?o a ?organType . FILTER(?organType != mmkg:Organ) }
ORDER BY DESC(?count) LIMIT 15""",

"07_largest_tumors":
"""SELECT ?case ?organType ?vol ?burden
WHERE { ?c a mmkg:ImagingCase ; mmkg:case_id ?case ; mmkg:depicts_organ ?o .
        ?l mmkg:located_in ?o ; mmkg:gt_volume_cm3 ?vol ; mmkg:tumorBurden ?burden .
        ?o a ?organType . FILTER(?organType != mmkg:Organ) }
ORDER BY DESC(?vol) LIMIT 10""",

"08_multi_organ_cases":
"""SELECT ?case (COUNT(?o) AS ?n_organs)
WHERE { ?c a mmkg:ImagingCase ; mmkg:case_id ?case ; mmkg:depicts_organ ?o . }
GROUP BY ?case ORDER BY DESC(?n_organs) LIMIT 10""",

"09_tumor_with_ontology_grounding":
"""SELECT ?case ?organType ?vol ?concept
WHERE { ?c a mmkg:ImagingCase ; mmkg:case_id ?case ; mmkg:depicts_organ ?o .
        ?o a ?organType ; mmkg:mapped_to_concept ?concept .
        ?l mmkg:located_in ?o ; mmkg:gt_volume_cm3 ?vol .
        FILTER(?organType != mmkg:Organ) }
ORDER BY DESC(?vol) LIMIT 10""",

"10_tumor_to_organ_volume_ratio":
"""SELECT ?case ?organType ?tvol ?ovol (ROUND(1000*(?tvol/?ovol))/10 AS ?pct_of_organ)
WHERE { ?c a mmkg:ImagingCase ; mmkg:case_id ?case ; mmkg:depicts_organ ?o .
        ?o a ?organType ; mmkg:gt_volume_cm3 ?ovol .
        ?l mmkg:located_in ?o ; mmkg:gt_volume_cm3 ?tvol .
        FILTER(?organType != mmkg:Organ) }
ORDER BY DESC(?pct_of_organ) LIMIT 12""",
}


def main():
    g = Graph(); g.parse(TTL, format="turtle")
    print(f"loaded {len(g):,} triples from {os.path.basename(TTL)}\n")
    names = [sys.argv[1]] if len(sys.argv) > 1 else list(QUERIES)
    for name in names:
        q = QUERIES[name]
        rows = list(g.query(PREFIX + q))
        print(f"### {name}  ({len(rows)} rows)")
        for r in rows[:12]:
            print("   " + " | ".join(str(x) for x in r))
        print()


if __name__ == "__main__":
    main()
