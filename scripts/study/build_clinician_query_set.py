#!/usr/bin/env python3
"""Instantiate the 30-query clinician relevance study set (docs/clinician_study_query_set.md)
on the de-duplicated pool — owner split, 2026-09-15.

Pool: 113 single-organ queries + corpora/corpus_flare23_kg_dedup.json (1,234 after the audit's
58 twins, the confirmed LiTS twin FLARE23_0102, and the 19 voxel-exact LiTS-query twins found
2026-09-16 once the LiTS geometry was recovered — results/audit/lits_extended_audit.json)
= 1,347 cases.

Rankings:
  (ii)/(iii)/(iii') are computed here with the benchmark scorer (retrieval_core_local).
  (vi)/(vii') come from app_handoff/precomputed/rankings_113_queries.json — since 2026-09-16
  the DEPTH-30 DE-DUPLICATED-POOL lists (scripts/retrieval/export_rankings_113.py, features
  recomputed locally and validated: the 113-pool table reproduces the frozen Delta numbers to
  the third decimal). The former 1,425-pool twin-filtered lists are superseded; the twin
  filter below is retained as a guard and is a no-op on the dedup pool.

Known caveats carried into the output (meta.flags):
  * containment is 'boundary' for every 113-cohort record (exclusive-label extraction artifact)
    -> 'contained' cells are unsatisfiable and fall back to (source, burden, multiplicity);
    'boundary-spanning' cells are trivially satisfied and NOT semantically boundary-spanning.
  * D for FLARE23 (block B) queries uses only the |top10(ii) DELTA top10(iii')| term — no
    baseline rankings exist for FLARE23 queries yet.
  * LiTS query twin-status RESOLVED 2026-09-16 (lits_extended_audit.json): 19 of 21 LiTS
    queries had voxel-exact FLARE23 twins; removed from the pool and the twin filter here.

Selection is deterministic: eligible cases sorted by ID, numpy default_rng(12345), rng consumed
in block order A1..A18, B1..B8, C1..C4. Every substitution is recorded.

-> results/study/clinician_study_30q.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "retrieval"))
import retrieval_core_local as core
from retrieval_core_local import load_corpus, relevant

OUT = os.path.join(ROOT, "results", "study", "clinician_study_30q.json")
SEED = 12345
Q4A_CM3 = 65.4
core.ORGAN_UNIVERSE = ["liver", "spleen", "pancreas", "kidney"]


# ---------------- ranking ----------------
def rank_ids(qrec, pool, mode, k=10):
    scored = []
    for cid, rec in pool.items():
        if rec is qrec or rec["case_id"] == qrec["case_id"]:
            continue
        s = core.similarity(qrec, rec, mode)
        if s is not None:
            scored.append((s, cid))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]], scored


def summary(rec):
    parts = []
    for o in rec["observed_organs"]:
        p = rec["organs"][o]
        if p.get("has_tumor"):
            parts.append(f"{o} {p.get('organ_volume_cm3', '?')} cm3 — tumor {p.get('tumor_volume_cm3')} cm3 "
                         f"({p.get('burden_cat')} burden, {p.get('multiplicity')}, {p.get('containment')}"
                         + (f", {p.get('anatomic_location')}" if o == "pancreas"
                            and p.get("anatomic_location") not in (None, "unknown", "na") else "") + ")")
        else:
            parts.append(f"{o} {p.get('organ_volume_cm3', '?')} cm3 — no lesion (observed-absent)")
    return "; ".join(parts)


def main():
    rng = np.random.default_rng(SEED)
    pred = load_corpus("predicted", datasets=["kits", "lits", "msd"])
    gt = load_corpus("gt", datasets=["kits", "lits", "msd"])
    flare = {r["case_id"]: r for r in
             json.load(open(os.path.join(ROOT, "corpora", "corpus_flare23_kg_dedup.json")))["records"]}
    q113 = sorted(i for i in pred if i in gt)
    pool = {**{i: pred[i] for i in q113}, **flare}          # ranking pool (predicted / recorded)
    with_ct_flare = {"flare23_" + r["case_id"] if not r["case_id"].startswith("flare23_") else r["case_id"]
                     for r in json.load(open(os.path.join(ROOT, "corpora", "corpus_predicted_flare23.json")))["records"]}
    subpool_ids = set(q113) | (with_ct_flare & set(flare))

    rk = json.load(open(os.path.join(ROOT, "app_handoff", "precomputed", "rankings_113_queries.json")))
    audit = json.load(open(os.path.join(ROOT, "results", "audit", "duplicate_scan_audit.json")))
    twins = set(audit["twins_confirmed"]) | {m["flare23"] for m in audit.get("manually_confirmed_twins", [])}
    ext = os.path.join(ROOT, "results", "audit", "lits_extended_audit.json")
    if os.path.exists(ext):                              # 2026-09-16: 19 voxel-exact LiTS-query twins
        twins |= set(json.load(open(ext)).get("twins_confirmed", []))

    # tercile edges per collection (from reference tumor volumes; LiTS = pseudo-cm3, flagged)
    edges = {}
    for ds in ("kits", "lits", "msd"):
        vols = sorted(p["tumor_volume_cm3"] for i in q113 if gt[i]["dataset"] == ds
                      for p in gt[i]["organs"].values() if p["has_tumor"])
        by_cat = {c: [p["tumor_volume_cm3"] for i in q113 if gt[i]["dataset"] == ds
                      for p in gt[i]["organs"].values() if p["has_tumor"] and p["burden_cat"] == c]
                  for c in ("low", "medium", "high")}
        edges[ds] = {"low_max": max(by_cat["low"]), "high_min": min(by_cat["high"]),
                     "units": "pseudo-cm3 (no headers)" if ds == "lits" else "cm3"}

    # ---------------- per-query machinery for the 113 ----------------
    print("ranking the 113 queries over the de-duplicated pool ...", flush=True)
    Q = {}
    for qid in q113:
        qrec = pred[qid]
        t_ii, scored_ii = rank_ids(qrec, pool, "proposed")
        t_iii, _ = rank_ids(qrec, pool, "coverage_blind")
        t_iiip, _ = rank_ids(qrec, pool, "imputed_absent")
        rels = [c for _, c in scored_ii if relevant(gt.get(qid, qrec), pool[c])]
        # construction relevance: query reference record vs candidate recorded record
        n_rel = sum(1 for c in pool if c != qid and relevant(gt[qid], pool[c]))
        hit = ap = 0.0
        rl = [1 if relevant(gt[qid], pool[c]) else 0 for _, c in scored_ii]
        for i, r in enumerate(rl):
            if r:
                hit += 1
                ap += hit / (i + 1)
        ap = ap / sum(rl) if sum(rl) else None
        pre = rk["queries"][qid].get("table9_dedup_pool") or rk["queries"][qid]["table9_pool1425"]
        t_vi = [e["case_id"] for e in pre["vi"] if e["case_id"] not in twins][:10]
        t_viip = [e["case_id"] for e in pre["vii_prime"] if e["case_id"] not in twins][:10]
        d_iiip = len(set(t_ii) ^ set(t_iiip))
        d_vi = len(set(t_ii) ^ set(t_vi))
        Q[qid] = {"rec": qrec, "ref": gt[qid], "ds": qrec["dataset"], "n_rel": n_rel, "ap_ii": ap,
                  "top10": {"ii": t_ii, "iii": t_iii, "iii_prime": t_iiip, "vi": t_vi, "vii_prime": t_viip},
                  "D_iii_prime": d_iiip, "D_vi": d_vi, "D": d_iiip + d_vi,
                  "vi_truncated": len(t_vi) < 10, "vii_prime_truncated": len(t_viip) < 10}

    # ---------------- FLARE23 queries (block B / C) ----------------
    print("ranking the FLARE23 queries (ii, iii') ...", flush=True)
    FQ = {}
    for qid, qrec in flare.items():
        t_ii, scored_ii = rank_ids(qrec, pool, "proposed")
        t_iiip, _ = rank_ids(qrec, pool, "imputed_absent")
        n_rel = sum(1 for c in pool if c != qid and relevant(qrec, pool[c]))
        rl = [1 if relevant(qrec, pool[c]) else 0 for _, c in scored_ii]
        hit = ap = 0.0
        for i, r in enumerate(rl):
            if r:
                hit += 1
                ap += hit / (i + 1)
        ap = ap / sum(rl) if sum(rl) else None
        FQ[qid] = {"rec": qrec, "ds": "flare23", "n_rel": n_rel, "ap_ii": ap,
                   "top10": {"ii": t_ii, "iii_prime": t_iiip},
                   "D_iii_prime": len(set(t_ii) ^ set(t_iiip)), "D_vi": None,
                   "D": len(set(t_ii) ^ set(t_iiip))}

    eligible113 = {q: v for q, v in Q.items() if v["n_rel"] >= 3}
    eligibleF = {q: v for q, v in FQ.items() if v["n_rel"] >= 3}
    print(f"eligible: {len(eligible113)}/113 single-organ, {len(eligibleF)}/{len(FQ)} FLARE23")

    selected, records, subs = set(), [], []

    def pick(cands, rule, rng):
        """cands: sorted list of (qid, info). rule in the md's vocabulary."""
        if not cands:
            return None
        if "highest D" in rule:
            return max(cands, key=lambda kv: (kv[1]["D"], kv[0]))[0] if "2nd" not in rule else \
                sorted(cands, key=lambda kv: (-kv[1]["D"], kv[0]))[1][0] if len(cands) > 1 else cands[0][0]
        if "lowest D" in rule:
            return min(cands, key=lambda kv: (kv[1]["D"], kv[0]))[0]
        if rule.startswith("calib"):
            zero = [kv for kv in cands if kv[1]["D"] == 0]
            pool_ = zero or sorted(cands, key=lambda kv: kv[1]["D"])[:3]
            return pool_[int(rng.integers(0, len(pool_)))][0]
        return cands[int(rng.integers(0, len(cands)))][0]

    def phen113(qid):
        ref = Q[qid]["ref"]
        o = ref["observed_organs"][0]
        return ref["organs"][o], o

    # ---------------- Block A ----------------
    A = [
        ("A1", "lits", "high", "multifocal", "contained", "discriminating, highest D"),
        ("A2", "lits", "high", "solitary", "contained", "discriminating"),
        ("A3", "lits", "medium", "multifocal", "contained", "discriminating"),
        ("A4", "lits", "low", "solitary", "contained", "calib"),
        ("A5", "lits", "any", "any", "boundary", "discriminating"),
        ("A6", "kits", "high", "solitary", "contained", "discriminating"),
        ("A7", "kits", "high", "solitary", "contained", "discriminating, 2nd-highest D"),
        ("A8", "kits", "medium", "solitary", "contained", "discriminating"),
        ("A9", "kits", "low", "solitary", "contained", "calib"),
        ("A10", "kits", "any", "multifocal", "contained", "discriminating"),
        ("A11", "kits", "any", "any", "boundary", "discriminating"),
        ("A12", "msd", "high", "solitary", "contained", "discriminating"),
        ("A13", "msd", "high", "solitary", "contained", "discriminating; sub-site=head"),
        ("A14", "msd", "medium", "solitary", "contained", "discriminating; sub-site=body/tail"),
        ("A15", "msd", "low", "solitary", "contained", "calib"),
        ("A16", "msd", "any", "any", "boundary", "discriminating"),
        ("A17", "msd", "any", "any", "any", "fixed=pancreas_041"),
        ("A18", "msd", "high", "any", "any", "fixed=pancreas_015"),
    ]

    def a_eligible(ds, burden, mult, cont, rule):
        out, notes = [], []
        for qid in sorted(eligible113):
            if qid in selected or Q[qid]["ds"] != ds:
                continue
            p, o = phen113(qid)
            if burden != "any" and p["burden_cat"] != burden:
                continue
            if mult != "any" and p["multiplicity"] != mult:
                continue
            # containment: all records are 'boundary' (artifact) -> 'contained' cannot filter
            if "sub-site=head" in rule and p.get("anatomic_location") != "head":
                continue
            if "sub-site=body/tail" in rule and p.get("anatomic_location") not in ("body", "tail"):
                continue
            if "discriminating" in rule and Q[qid]["D"] == 0:
                continue
            out.append((qid, Q[qid]))
        if cont == "contained":
            notes.append("containment='contained' unsatisfiable (all-boundary artifact) — cell relaxed to (source, burden, multiplicity)")
        if cont == "boundary":
            notes.append("containment='boundary-spanning' trivially satisfied (all-boundary artifact) — NOT semantically boundary-spanning")
        return out, notes

    for aid, ds, burden, mult, cont, rule in A:
        notes = []
        if rule.startswith("fixed="):
            qid = rule.split("=")[1]
            if qid not in eligible113 or qid in selected:
                notes.append(f"fixed case {qid} ineligible — substituted per fallback")
                qid = None
        else:
            qid = None
        if qid is None:
            cands, notes2 = a_eligible(ds, burden, mult, cont, rule)
            notes += notes2
            if not cands and burden != "any":            # fallback: burden +-1 tercile
                order = ["low", "medium", "high"]
                for nb in order:
                    if nb == burden:
                        continue
                    cands, _ = a_eligible(ds, nb, mult, cont, rule)
                    if cands:
                        notes.append(f"fallback: burden {burden} -> {nb}")
                        break
            if not cands and mult != "any":              # then multiplicity
                cands, _ = a_eligible(ds, "any", "any", cont, rule)
                if cands:
                    notes.append(f"fallback: multiplicity {mult} -> any, burden -> any")
            qid = pick(cands, rule, rng)
        if qid is None:
            subs.append({"cell": aid, "note": "NO ELIGIBLE CASE — cell left empty; " + "; ".join(notes)})
            continue
        selected.add(qid)
        records.append(("A", aid, qid, rule, notes))

    # ---------------- Block B ----------------
    def fprof(qid):
        return {o: flare[qid]["organs"][o] for o in flare[qid]["observed_organs"]}

    def b_eligible(cond, rule):
        out = []
        for qid in sorted(eligibleF):
            if qid in selected:
                continue
            p = fprof(qid)
            if not cond(p):
                continue
            if "discriminating" in rule and FQ[qid]["D"] == 0:
                continue
            out.append((qid, FQ[qid]))
        return out

    tum = lambda p, o: o in p and p[o]["has_tumor"]
    only = lambda p, o: tum(p, o) and all(not p[x]["has_tumor"] for x in p if x != o)
    B = [
        ("B1", lambda p: only(p, "liver") and p["liver"]["burden_cat"] == "high" and p["liver"]["multiplicity"] == "solitary",
         "discriminating, highest D"),
        ("B2", lambda p: tum(p, "liver") and p["liver"]["multiplicity"] == "multifocal", "discriminating"),
        ("B3", lambda p: only(p, "kidney"), "discriminating"),
        ("B4", lambda p: only(p, "pancreas"), "discriminating"),
        ("B5", lambda p: tum(p, "spleen"), "any"),
        ("B6", lambda p: tum(p, "liver") and p["liver"]["tumor_volume_cm3"] > Q4A_CM3
         and p["liver"]["multiplicity"] == "solitary", "discriminating"),
        ("B7", lambda p: all(not p[o]["has_tumor"] for o in p), "lowest D"),
        ("B8", None, "fixed=flare23_FLARE23_0840"),
    ]
    for bid, cond, rule in B:
        notes = ["D uses only |top10(ii) DELTA top10(iii')| — no baseline rankings for FLARE23 queries yet"]
        if rule.startswith("fixed="):
            qid = rule.split("=")[1]
            if qid not in eligibleF or qid in selected:
                notes.append(f"fixed case {qid} ineligible (n_rel<3, twin, or taken) — recorded, not substituted")
                qid = qid if qid in FQ and qid not in selected else None
        else:
            cands = b_eligible(cond, rule)
            if not cands and "discriminating" in rule:
                cands = b_eligible(cond, "any")
                if cands:
                    notes.append("no discriminating (D>0) query in this cell — relaxed to D=0 (all FLARE23 "
                                 "kidney/pancreas-only cells tie under (ii) vs (iii'); revisit after the "
                                 "baseline rankings add the (vi) term to D)")
            if bid == "B5" and len(cands) == 0:
                cands = b_eligible(lambda p: only(p, "kidney"), "any")
                notes.append("B5: no eligible spleen-lesion query with >=3 relevant — replaced by second kidney query per spec")
            if bid == "B7":
                # negative control: lesion-free queries have no construction-relevant candidates by
                # definition — eligibility waived per spec role
                cands = [(q, FQ[q]) for q in sorted(FQ) if q not in selected
                         and all(not fprof(q)[o]["has_tumor"] for o in fprof(q))]
                notes.append("B7 negative control: >=3-relevant eligibility waived (lesion-free query has no rule-relevant candidates)")
                qid = min(cands, key=lambda kv: (kv[1]["D"], kv[0]))[0] if cands else None
            else:
                qid = pick(cands, rule, rng)
        if qid is None:
            subs.append({"cell": bid, "note": "; ".join(notes)})
            continue
        selected.add(qid)
        records.append(("B", bid, qid, rule, notes))

    # ---------------- Block C ----------------
    allelig = {**eligible113, **eligibleF}
    rest = {q: v for q, v in allelig.items() if q not in selected}
    c_notes = {"C2": "restricted to the 113 single-organ queries (only they have (vi) rankings)",
               "C3": "(vii') is the depth-30 de-duplicated-pool list restricted to the CT sub-pool (656 cases; re-ranked locally 2026-09-16 — no longer provisional)",
               "C4": "AP under (ii), construction relevance, de-duplicated pool (1,347 as of 2026-09-16)"}
    c1 = max((kv for kv in sorted(rest.items())), key=lambda kv: (kv[1]["D_iii_prime"], kv[0]), default=None)
    if c1:
        selected.add(c1[0]); records.append(("C", "C1", c1[0], "largest |t10(ii) DELTA t10(iii')|", []))
    c2 = max((kv for kv in sorted(rest.items()) if kv[0] in Q and kv[0] not in selected),
             key=lambda kv: (kv[1]["D_vi"] or 0, kv[0]), default=None)
    if c2:
        selected.add(c2[0]); records.append(("C", "C2", c2[0], "largest |t10(ii) DELTA t10(vi)|", [c_notes["C2"]]))
    # C3: recompute (ii) on the CT sub-pool (113 + dedup FLARE23-with-CT) for 113 queries, compare with (vii')
    sub_pool = {c: pool[c] for c in pool if c in subpool_ids}
    best = None
    for qid in sorted(Q):
        if qid in selected:
            continue
        t_ii_sub, _ = rank_ids(Q[qid]["rec"], sub_pool, "proposed")
        pre = rk["queries"][qid].get("table9_dedup_pool") or rk["queries"][qid]["table9_pool1425"]
        t_vp = [e["case_id"] for e in pre["vii_prime"] if e["case_id"] not in twins and e["case_id"] in subpool_ids][:10]
        d = len(set(t_ii_sub) ^ set(t_vp))
        if best is None or d > best[1]:
            best = (qid, d)
    if best:
        selected.add(best[0]); records.append(("C", "C3", best[0], f"largest |t10(ii) DELTA t10(vii')| on CT sub-pool (D={best[1]})", [c_notes["C3"]]))
    c4 = min((kv for kv in sorted(allelig.items()) if kv[0] not in selected and kv[1]["ap_ii"] is not None),
             key=lambda kv: (kv[1]["ap_ii"], kv[0]), default=None)
    if c4:
        selected.add(c4[0]); records.append(("C", "C4", c4[0], f"lowest AP under (ii) = {round(c4[1]['ap_ii'], 3)}", [c_notes["C4"]]))

    # ---------------- export ----------------
    out_q = []
    for block, cell, qid, rule, notes in records:
        info = Q.get(qid) or FQ.get(qid)
        rec = info["rec"]
        ref = info.get("ref", rec)
        o0 = ref["observed_organs"][0] if qid in Q else None
        p0 = ref["organs"][o0] if o0 else None
        top10 = info["top10"]
        union = sorted(set().union(*[set(v) for v in top10.values()]))
        out_q.append({
            "block": block, "cell": cell, "query_id": qid,
            "source": info["ds"], "observed_organs": rec["observed_organs"],
            "burden_tercile": p0["burden_cat"] if p0 else None,
            "tercile_edges": edges.get(info["ds"]),
            "multiplicity": p0["multiplicity"] if p0 else None,
            "containment": p0["containment"] if p0 else None,
            "subsite": p0.get("anatomic_location") if (p0 and o0 == "pancreas") else None,
            "n_rule_relevant": info["n_rel"],
            "D": info["D"], "D_iii_prime": info["D_iii_prime"], "D_vi": info.get("D_vi"),
            "selection_rule": rule, "substitution_notes": notes,
            "query_summary": summary(ref),
            "top10": top10,
            "pooled_candidates": [{"case_id": c,
                                   "relevant_construction_rule": relevant(ref if qid in Q else rec, pool[c]),
                                   "summary": summary(pool[c])} for c in union],
            "n_pooled": len(union)})

    meta = {"seed": SEED, "pool": "113 + 1,234 FLARE23 (corpus_flare23_kg_dedup.json, 2026-09-16) = 1,347",
            "spec": "docs/clinician_study_query_set.md",
            "tercile_edges": edges,
            "flags": [
                "(vi)/(vii') RESOLVED 2026-09-16: depth-30 de-duplicated-pool re-rank (export_rankings_113.py; features recomputed locally, 113-pool table reproduces the frozen Delta numbers to the third decimal) — the twin-filter truncation is gone; this was the last provisional gate for the grader packet",
                "containment: fixed for the FLARE23 576 predicted corpus 2026-09-16 (containment_v2/corpus_predicted_flare23_containment_v2.json, organ:=organ∪tumor rule); the 113-cohort records remain 'boundary'-degenerate until their masks are staged for re-extraction — 'contained' cells stay relaxed for the 113 side",
                "block B D omits the (vi) term (no baseline rankings for FLARE23 queries)",
                "LiTS query twin-status RESOLVED 2026-09-16: 19/21 LiTS queries had voxel-exact FLARE23 twins (results/audit/lits_extended_audit.json); twins removed from the pool and the (vi)/(vii') filter, so LiTS query cells no longer contain their trivial twin matches"],
            "n_selected": len(records), "substitutions_or_empty_cells": subs}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"meta": meta, "queries": out_q}, open(OUT, "w"), indent=1)
    print(f"\nselected {len(records)} queries ({len(subs)} substitution/empty-cell notes)")
    for block, cell, qid, rule, notes in records:
        info = Q.get(qid) or FQ.get(qid)
        print(f"  {cell:4s} {qid:28s} D={info['D']:>3} n_rel={info['n_rel']:>4}  {rule}"
              + ("  [" + "; ".join(n for n in notes if "artifact" not in n) + "]" if notes and any("artifact" not in n for n in notes) else ""))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
