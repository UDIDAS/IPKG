#!/usr/bin/env python3
"""Where does the application's question box route a tumour-volume question? (study lead v50, Limitations red text)

Oncologist R1 (final round, 11 Oct) reported that asking for pancreatic / kidney TUMOUR volumes returned ORGAN volumes
(pancreas_125, FLARE23_0405). This runs segx_nlq.Engine from the rated build (tag ipkg-corrected-2026-10-09) with the
encoder the application uses (BAAI/bge-small-en-v1.5, CPU) on R1's questions and close variants, and records the matched
intent, its similarity and whether it is a refusal. The graph does record per-organ lesion volumes (ipkg:gt_volume_cm3
on lesions, used by the organ_distribution and lesions_over_size intents); the intent bank has no per-organ tumour-volume
intent, so the question is answered by the organ-volume intent instead of being refused.

  NLQ=<segx_nlq.py from the tag> HF_HOME=... python nlq_tumor_volume_routing.py -> results/app_checks/nlq_tumor_volume_routing_2026-10-12.json
"""
import importlib.util, json, os
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
spec = importlib.util.spec_from_file_location("segx_nlq", os.environ["NLQ"]); nlq = importlib.util.module_from_spec(spec); spec.loader.exec_module(nlq)
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
enc = lambda t: np.asarray(m.encode(t, normalize_embeddings=False, show_progress_bar=False))
E = nlq.Engine().fit(enc)
QS = ["what are the pancreatic tumor volumes in the cohort", "pancreatic tumor volumes", "show me the pancreatic tumor volume",
      "kidney tumor data", "what is the kidney tumor volume", "show kidney tumor volumes in the cohort",
      "what is the normal range of liver volume"]
rows = []
for q in QS:
    sim, phrase, entry, is_ref = E.match(q, enc)[0]
    rows.append({"question": q, "matched_intent": entry["id"], "similarity": round(float(sim), 3), "refusal": bool(is_ref),
                 "nearest_phrasing": phrase, "slots": nlq.extract_slots(q)})
out = {"note": __doc__.strip(), "build": "ipkg-corrected-2026-10-09", "min_sim_for_answer": 0.45,
       "intents": [i["id"] for i in nlq.INTENTS], "results": rows}
json.dump(out, open(os.path.join(ROOT, "results", "app_checks", "nlq_tumor_volume_routing_2026-10-12.json"), "w"), indent=1)
for r in rows: print(f"{r['question']!r:58s} -> {r['matched_intent']:18s} {r['similarity']}")
