#!/usr/bin/env python3
"""Bundle-local retrieval core — vendored from `kg_retrieval_v2.py` (similarity / relevant / _dcg)
with the module-level corpus load and absolute /home/user/... paths removed, so the bundle's
retrieval experiments can be reproduced from `corpora/` on any machine.

The scoring functions are verbatim copies of kg_retrieval_v2 (same constants, same semantics);
`load_corpus()` reads the per-dataset corpora shipped in ../../corpora.
Verified to reproduce results/retrieval/retrieval_on_predicted.json exactly.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CORP = os.path.join(ROOT, "corpora")

# 5-organ universe of the original 3-regime corpus (only liver/kidney/pancreas occur in the
# 113-case corpus; extra entries are inert for coverage_blind because no record carries them).
ORGAN_UNIVERSE = ["liver", "spleen", "pancreas", "kidney", "right_kidney"]
CATS = ["burden_cat", "multiplicity", "containment", "anatomic_location"]
LOC_CATS = {"anatomic_location"}

IC = {"root": 0.0, "abdominal_organ": 0.3,
      "liver": 0.6, "spleen": 0.6, "pancreas": 0.6, "kidney": 0.5,
      "right_kidney": 0.75, "left_kidney": 0.75,
      "head": 0.9, "body": 0.9, "tail": 0.9}
PARENT = {"abdominal_organ": "root",
          "liver": "abdominal_organ", "spleen": "abdominal_organ",
          "pancreas": "abdominal_organ", "kidney": "abdominal_organ",
          "right_kidney": "kidney", "left_kidney": "kidney",
          "head": "pancreas", "body": "pancreas", "tail": "pancreas"}


def load_corpus(kind, datasets=None):
    """kind in {'predicted','gt'} -> {case_id: record} from corpora/corpus_<kind>_*.json."""
    recs = []
    for f in sorted(glob.glob(os.path.join(CORP, f"corpus_{kind}_*.json"))):
        d = json.load(open(f))
        if datasets and d["dataset"] not in datasets:
            continue
        recs += d["records"]
    return {r["case_id"]: r for r in recs}


def ancestors(c):
    out = [c]
    while c in PARENT:
        c = PARENT[c]; out.append(c)
    return out


def onto_sim(a, b, graded=True):
    if a == b:
        return 1.0
    if not graded or a not in IC or b not in IC:
        return 0.0
    Aa, Ab = ancestors(a), ancestors(b)
    lcs = next((x for x in Aa if x in Ab), "root")
    denom = IC.get(a, 0) + IC.get(b, 0)
    return (2 * IC.get(lcs, 0) / denom) if denom > 0 else 0.0


def phen(rec, organ):
    return rec["organs"].get(organ)


def feat_agree(pa, pb, cat, graded):
    va, vb = pa.get(cat, "none"), pb.get(cat, "none")
    if cat in LOC_CATS:
        if va in ("na", "unknown", "none") or vb in ("na", "unknown", "none"):
            return 1.0 if va == vb else 0.0
        return onto_sim(va, vb, graded)
    return 1.0 if va == vb else 0.0


def organ_cats(o):
    return CATS if o == "pancreas" else CATS[:3]


def similarity(A, B, mode, weights=None, gamma_min=1):
    """Similarity in [0,1], or None if incomparable (proposed on <gamma_min shared organs)."""
    oa, ob = set(A["observed_organs"]), set(B["observed_organs"])
    if mode == "base":
        fa, fb = phen(A, sorted(oa)[0]), phen(B, sorted(ob)[0])
        return float(np.mean([1.0 if fa.get(c, "none") == fb.get(c, "none") else 0.0 for c in CATS[:3]]))
    if mode == "proposed":
        shared = oa & ob
        if len(shared) < gamma_min:
            return None
        organs, graded = shared, True
    elif mode == "coverage_blind":
        organs, graded = set(ORGAN_UNIVERSE), True
    elif mode == "imputed_absent":
        # (iii') true Prop.-1 ablation (gamma plan §4, added 2026-09-13): every organ observed on only one
        # side is imputed "no finding" on the other side and COMPARED — agreement with an observed-absent
        # finding (the n term) scores like any agreement, mismatch with an observed-present finding (the p
        # term) like any mismatch. Not a flat penalty. Shared organs are scored as in the other modes.
        organs, graded = oa | ob, True
    elif mode == "graded":
        organs, graded = oa | ob, True
    elif mode in ("typed", "flat_tier"):
        organs, graded = oa | ob, False
    else:
        raise ValueError(mode)

    sims, wts = [], []
    for o in sorted(organs):          # deterministic summation order (set order depends on the hash seed;
                                      # mathematically tied multi-organ candidates otherwise tie-break at random)
        pa, pb = phen(A, o), phen(B, o)
        if pa is None and pb is None:
            continue
        if pa is None or pb is None:
            if mode == "coverage_blind":
                sims.append(0.0); wts.append(1.0)
            elif mode == "imputed_absent":
                imp = {"burden_cat": "none", "multiplicity": "none", "containment": "none", "anatomic_location": "na"}
                pa2, pb2 = (imp if pa is None else pa), (imp if pb is None else pb)
                cats = organ_cats(o)
                w = weights if weights is not None else {c: 1.0 for c in CATS}
                num = sum(w.get(c, 1.0) * feat_agree(pa2, pb2, c, graded) for c in cats)
                den = sum(w.get(c, 1.0) for c in cats)
                sims.append(num / den if den else 0.0); wts.append(1.0)
            continue
        cats = organ_cats(o)
        w = weights if weights is not None else {c: 1.0 for c in CATS}
        num = sum(w.get(c, 1.0) * feat_agree(pa, pb, c, graded) for c in cats)
        den = sum(w.get(c, 1.0) for c in cats)
        sims.append(num / den if den else 0.0); wts.append(1.0)
    if not sims:
        return None if mode == "proposed" else 0.0
    return float(np.average(sims, weights=wts))


def _agree_over(A, B, keys):
    shared = set(A["observed_organs"]) & set(B["observed_organs"])
    for o in shared:
        pa, pb = phen(A, o), phen(B, o)
        n = 0
        for k in keys:
            va = pa.get(k)
            if va == pb.get(k) and va not in ("none", "na", "unknown", None):
                n += 1
        if n >= (2 if len(keys) > 1 else 1):
            return True
    return False


def relevant(A, B):
    return _agree_over(A, B, ["burden_cat", "multiplicity", "containment", "anatomic_location"])


def _dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
