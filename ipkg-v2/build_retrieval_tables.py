# -*- coding: utf-8 -*-
"""The four retrieval tables, re-derived on the 1,311 pool.

WHAT THIS PRODUCES
    tab:retrieval-primary   the primary controlled benchmark (reference-derived
                            relevance), 113 queries x 112 candidates
    tab:retrieval           the large-pool stress test (construction-defined
                            relevance), 113 queries against the 1,311 pool
    tab:retrieval-strata    retrieval stratified by the observability relation
                            between query and candidates
    tab:cross-examples      rank of the intended same-organ FLARE23 target for
                            three single-organ queries

plus three things the advisor asked for by name:

  1. a CELL-LEVEL DIFF against what the manuscript currently prints, so a
     co-author can read "old -> new" rather than re-read the table;
  2. the VOID FIX and the SUB-SITE CORRECTION as SEPARATE effects, re-derived
     on 1,311 (they were last measured on 1,196 and the pool changed under
     them);
  3. a circularity analysis: relevant() in the delivered scorer derives the
     relevance ground truth from the same four fields similarity() ranks on,
     so what the published mAP measures is partly the definition. What can be
     defended is produced; where nothing can be, the limit is named.

NOTHING HERE RE-IMPLEMENTS A PUBLISHED SCORER. feat_agree, onto_sim, the
information-content ontology, organ_cats, the configuration map and the
relevance rule are the pipeline team's delivered functions, imported and called. The metric
layer (per_query / agg / paired / holm / mismatch10) is a VERBATIM copy of
the pipeline team's own results/retrieval generator, scripts/retrieval/retrieval_tables_789.py,
because that file lives in the live survey tree which this repository may read
and must never import from (an import writes __pycache__ into a tree whose 177
files are hashed every ten minutes). The copy is checked, not asserted: the
--verify pass re-runs it on the pipeline team's original corpora and the original 1,425-case
pool and must reproduce every published cell exactly before any new number is
written.

THE CONFIGURATION LADDER. Two things changed under the published tables and
they are reported apart. "Sub-site" is the pancreatic head / body / tail axis,
`organs.<organ>.anatomic_location` - the only LOC_CAT, compared only on the
pancreas.

  sub-site vocabulary
    D  delivered   the round-1 word, recovered from each record's own
                   `anatomic_location_delivered`: the axial-slice-axis rule the
                   2026-09-29 audit refuted, FLARE23 spelled "unknown", and a
                   pancreatic word printed on livers and kidneys.
    C  corpus      the pipeline team's 2026-10-03 re-issue as shipped: the corrected
                   left-right axis on the MSD pancreas, "na" on every
                   non-pancreatic organ and on every FLARE23 organ.
    P  resolved    C with ipkg_scorer_fixed.resolve_site()'s precedence rule
                   applied to BOTH sides - round-2 record, then our overlay,
                   then a withdrawal, then the delivered word.

  void policy
    V0 delivered   _scorer_core.feat_agree as delivered: a sub-site that is
                   void on BOTH sides returns 1.0 when the two voids are
                   spelled the same way and 0.0 when they are not.
    V1 void fix    the axis is DROPPED from the organ mean - numerator and
                   denominator - whenever either side is void, and an organ
                   left with no comparable axis contributes nothing to the
                   record-level mean.

  arms           = (vocabulary, void policy)
    published      D + V0   the configuration the manuscript's numbers were
                            computed in, re-run on today's pool
    void_fix       D + V1   the void fix alone
    subsite_corpus C + V0   the re-issued sub-sites alone
    subsite_resolved P+V0   the precedence rule alone
    corpus_voidfix C + V1
    corrected      P + V1   BOTH, which is what the application ranks with

USAGE
    python build_retrieval_tables.py --verify      reproduce the published
                                                   tables and stop
    python build_retrieval_tables.py               verify, then build
    python build_retrieval_tables.py --no-verify   build only (faster)
    python build_retrieval_tables.py --no-heavy    skip the 1,198-query sweep

Writes only into tables/retrieval/. Reads the live tree and both the pipeline team rounds.
CPU only; no GPU, no server, no mask is opened.
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time

sys.dont_write_bytecode = True          # never leave a .pyc in a read-only tree

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(_HERE, "tables", "retrieval")
LIVE = r"<app-workspace>"
PUBLISHED_CORP = os.path.join(LIVE, "pipeline_handoff", "corpora")
PUBLISHED_RES = os.path.join(LIVE, "pipeline_handoff", "results", "retrieval")

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import ipkg_final_cases as FC                               # noqa: E402
import ipkg_scorer_fixed as SF                              # noqa: E402

CORE, IR = SF.CORE, SF.IR
# The corrected 4-unit pooled universe of Tables 8-11. ipkg_retrieval sets it
# on import; it is set again here so a reader of this file does not have to
# know that, and so the value is recorded in the output.
UNIVERSE = ["liver", "spleen", "pancreas", "kidney"]
CORE.ORGAN_UNIVERSE = UNIVERSE
CATS = ["burden_cat", "multiplicity", "containment", "anatomic_location"]

B_BOOT, B_PERM = 5000, 20000
TODAY = datetime.date.today().isoformat()


# ===========================================================================
# METRIC LAYER - verbatim from pipeline_handoff/scripts/retrieval/retrieval_tables_789.py
# (copied rather than imported; see the module docstring). Only two changes,
# both mechanical and both marked: `core.similarity` is replaced by an
# injected `simfn` so an arm can be scored, and `rng` is a parameter instead
# of a module global so a comparison is reproducible on its own.
# ===========================================================================
def organ_agree(qp, cp):
    """>=2 informative feature agreements - the same bar relevance uses."""
    n = 0
    for k in CATS:
        va = qp.get(k)
        if va == cp.get(k) and va not in ("none", "na", "unknown", None):
            n += 1
    return n >= 2


def mismatch10(q_rel, cand_rels):
    """Outcome-based cross-organ mismatch over the top-10 candidates."""
    qo = q_rel["observed_organs"][0]
    qp = q_rel["organs"][qo]
    flags = []
    for c in cand_rels:
        at_qo = qo in c["organs"] and organ_agree(qp, c["organs"][qo])
        elsewhere = any(organ_agree(qp, p)
                        for o, p in c["organs"].items() if o != qo)
        flags.append(1.0 if (not at_qo and elsewhere) else 0.0)
    return float(np.mean(flags)) if flags else 0.0


def per_query(sim, rel, qids, cids, mode, multi_organ_queries=False,
              simfn=None, relfn=None, keep_rels=False):
    """Per-query metrics. sim/rel: case_id -> record for ranking / relevance.
    Ties broken by candidate list order (stable sort), as in the pipeline team's script."""
    simfn = simfn or CORE.similarity
    relfn = relfn or CORE.relevant
    out = {}
    for qi in qids:
        cands = []
        qrec = sim[qi]
        for bi in cids:
            if bi == qi:
                continue
            s = simfn(qrec, sim[bi], mode)
            if s is None:
                continue
            cands.append((s, bi))
        if not cands:
            continue
        cands.sort(key=lambda x: -x[0])
        rels = [1 if relfn(rel[qi], rel[bi]) else 0 for _, bi in cands]
        if sum(rels) == 0:
            continue
        hit = ap = 0.0
        for i, r in enumerate(rels):
            if r:
                hit += 1
                ap += hit / (i + 1)
        top10 = [bi for _, bi in cands[:10]]
        qobs = set(sim[qi]["observed_organs"])
        m = {"p5": float(np.mean(rels[:5])), "p10": float(np.mean(rels[:10])),
             "ap": ap / sum(rels),
             "ndcg": CORE._dcg(rels[:10]) / (CORE._dcg(sorted(rels, reverse=True)[:10]) or 1),
             "spur10": float(np.mean([0.0 if (qobs & set(sim[bi]["observed_organs"]))
                                      else 1.0 for bi in top10]))}
        m["mism10"] = (mismatch10(rel[qi], [rel[bi] for bi in top10])
                       if not multi_organ_queries
                       and len(rel[qi]["observed_organs"]) == 1 else None)
        if keep_rels:
            m["_rels"] = rels
            m["_top10"] = top10
            m["_n_ranked"] = len(cands)
        out[qi] = m
    return out


def agg(pq):
    def r(k):
        vals = [v[k] for v in pq.values() if v.get(k) is not None]
        return round(float(np.mean(vals)), 3) if vals else None
    return {"n_queries": len(pq), "P@5": r("p5"), "P@10": r("p10"),
            "mAP": r("ap"), "nDCG": r("ndcg"), "spurious@10": r("spur10"),
            "mismatch@10": r("mism10")}


def paired(dA, dB, rng, key="ap"):
    keys = sorted(set(dA) & set(dB))
    dif = np.array([dA[k][key] - dB[k][key] for k in keys])
    if not len(dif):
        return None
    obs = float(dif.mean())
    boots = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(B_BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    signs = rng.choice([-1.0, 1.0], size=(B_PERM, len(dif)))
    p = float((np.sum(np.abs((signs * dif).mean(axis=1)) >= abs(obs)) + 1) / (B_PERM + 1))
    return {"n_common": len(keys), "delta": round(obs, 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)], "p_raw": p}


def holm(pdict):
    """Holm-Bonferroni, step-down, WITH the running maximum.

    THE RUNNING MAXIMUM IS THE PROCEDURE, and this function did not have it
    until 2026-10-08 (advisor item the co-author 2). Holm requires the adjusted values to
    be non-decreasing in the raw ones: each is the larger of its own
    (m - rank) * p and the previous one. Without that line an adjusted p can
    FALL as the raw p rises, which is not a conservative approximation of Holm -
    it is not Holm, and it errs in the dangerous direction, understating p and
    so overstating significance.

    It bites hardest on TIES, which is exactly this project's case. Every raw p
    in family F1 sits at the permutation floor, 1/20001 = 4.99975e-05, because
    B = 20,000 sign-flips cannot resolve anything smaller. With m tied values
    the correct answer is m * floor for all of them; without the running maximum
    they fan out as m*p, (m-1)*p, (m-2)*p, so the last contrast in the family is
    reported at the raw floor with no correction at all. On the 13-contrast F1
    the manuscript states, the correct value is 6.5e-4 throughout.

    The reference implementations are kg_retrieval_v2.holm (the pipeline team's, in the
    delivered scripts) and paper/src/JBI/stat.holm; both carry the running
    maximum. It is written out here rather than imported because the first lives
    under pipeline_handoff/ in the OTHER tree, and a cross-tree import of a file
    CUT-11 already has to declare, for seven lines of arithmetic, is a worse
    dependency than the duplication.
    """
    items = sorted([(k, v) for k, v in pdict.items() if v is not None],
                   key=lambda kv: kv[1]["p_raw"])
    m = len(items)
    prev = 0.0
    for rank, (k, v) in enumerate(items):
        adj = max(min(1.0, (m - rank) * v["p_raw"]), prev)
        prev = adj
        v["p_holm"] = round(adj, 5)
    return pdict


# ===========================================================================
# THE ARMS: sub-site vocabulary x void policy
# ===========================================================================
VOID = SF.is_void


def _copy_rec(r):
    out = dict(r)
    out["organs"] = {k: (dict(v) if isinstance(v, dict) else v)
                     for k, v in (r.get("organs") or {}).items()}
    return out


def vocab_delivered(rec):
    """Round-1 sub-site words, recovered from the record's own
    `anatomic_location_delivered`. A record without that key never had its
    sub-site re-issued, so its current word IS the delivered word."""
    changed = False
    out = _copy_rec(rec)
    for o, p in (out.get("organs") or {}).items():
        d = p.get("anatomic_location_delivered")
        if d is not None and d != p.get("anatomic_location"):
            p["anatomic_location"] = d
            changed = True
    return out if changed else rec


def vocab_corpus(rec):
    return rec


def vocab_resolved(rec):
    return SF.resolve_record(rec)


def vocab_resolved_msd_only(rec):
    """The precedence rule applied to the MSD records ONLY, everything else
    left at the delivered word. This is EXACTLY the perturbation the published
    limitation describes - "correcting 52 of the 56 MSD sub-site labels moves
    mAP by 0.0000" - and it is kept separate from the full correction because
    the full correction also withdraws a pancreatic word from 56 kidney and
    liver entries, which is a different change with a different consequence."""
    if str(rec.get("dataset") or "").lower() == "msd":
        return vocab_resolved(rec)
    return vocab_delivered(rec)


VOCAB = {"D": vocab_delivered, "C": vocab_corpus, "P": vocab_resolved,
         "Pmsd": vocab_resolved_msd_only}

ARMS = {
    #  name              vocab  void-fix  what it is
    "published":        ("D", False),
    "void_fix":         ("D", True),
    "subsite_corpus":   ("C", False),
    "subsite_resolved": ("P", False),
    "corpus_voidfix":   ("C", True),
    "corrected":        ("P", True),
    "subsite_msd_only": ("Pmsd", False),
}
ARM_LABEL = {
    "published": "delivered sub-sites + delivered void (as published)",
    "void_fix": "delivered sub-sites + VOID FIX",
    "subsite_corpus": "re-issued sub-sites + delivered void",
    "subsite_resolved": "precedence-resolved sub-sites + delivered void",
    "corpus_voidfix": "re-issued sub-sites + VOID FIX",
    "corrected": "precedence-resolved sub-sites + VOID FIX (shipped)",
    "subsite_msd_only": ("MSD sub-sites corrected, every other record left at "
                         "the delivered word + delivered void - the published "
                         "limitation's exact perturbation"),
}
HEADLINE = "corrected"


def arm_pool(arm, pool):
    vk, _vf = ARMS[arm]
    f = VOCAB[vk]
    return {c: f(r) for c, r in pool.items()}


# ---------------------------------------------------------------------------
# MEMOISED SIMILARITY
#
# similarity() over this pool is a function of two RECORD SIGNATURES and
# nothing else: 43 distinct (organ, phenotype) tuples occur in 1,311 records,
# so the same pair of signatures is scored tens of thousands of times. The
# cache is keyed on those signatures and FILLED by the delivered function
# (CORE.similarity) or by SF.similarity_fixed - it never computes a score
# itself. _selftest_fast() checks the collapse on a 200,000-pair sample.
# ---------------------------------------------------------------------------
def signature(rec):
    organs = rec.get("organs") or {}
    return (tuple(sorted(rec["observed_organs"])),
            tuple(sorted((o, p.get("burden_cat"), p.get("multiplicity"),
                          p.get("containment"), p.get("anatomic_location"))
                         for o, p in organs.items())))


_MISS = object()          # a cache miss, distinct from a cached None


class Sim:
    """similarity for one arm, memoised on record signatures."""

    def __init__(self, pool, void_fix):
        self.void_fix = void_fix
        self._base = SF.similarity_fixed if void_fix else CORE.similarity
        # Keyed on the RECORD OBJECT, never on case_id: the primary benchmark
        # scores a predicted record and a reference record that carry the SAME
        # case_id and different phenotypes, and a case_id key silently hands
        # one row the other row's score. `_hold` keeps every keyed record
        # alive so an id() can never be reused under us.
        self._sig = {}
        self._hold = []
        for r in pool.values():
            self._sigof(r)
        self._cache = {}
        self.calls = 0
        self.misses = 0

    def _sigof(self, rec):
        k = id(rec)
        s = self._sig.get(k)
        if s is None:
            s = signature(rec)
            self._sig[k] = s
            self._hold.append(rec)
        return s

    def __call__(self, A, B, mode):
        self.calls += 1
        ka = self._sigof(A)
        kb = self._sigof(B)
        key = (mode, ka, kb)
        v = self._cache.get(key, _MISS)
        if v is not _MISS:
            return v
        self.misses += 1
        v = self._base(A, B, mode)
        self._cache[key] = v
        return v


def _selftest_fast(pool, refmap=None, n=4000):
    """The cache must be transparent. Scores a sample of pairs with and
    without it, in every mode and both void policies, and requires bitwise
    equality.

    `refmap` is passed deliberately: a predicted record and its reference
    record share a case_id and differ in phenotype, which is exactly the
    collision an identifier-keyed cache gets wrong and a sample drawn from one
    corpus alone can never see."""
    ids = sorted(pool)
    rng = np.random.default_rng(7)
    pairs = [(ids[i], ids[j]) for i, j in rng.integers(0, len(ids), size=(n, 2))]
    bad, mixed = 0, 0
    for vf in (False, True):
        everything = dict(pool)
        if refmap:
            everything.update({"__ref__" + k: v for k, v in refmap.items()})
        s = Sim(everything, vf)
        base = SF.similarity_fixed if vf else CORE.similarity
        for a, b in pairs:
            if a == b:
                continue
            for mode in ("proposed", "coverage_blind", "imputed_absent", "base"):
                x, y = s(pool[a], pool[b], mode), base(pool[a], pool[b], mode)
                if x != y and not (x is None and y is None):
                    bad += 1
        if refmap:
            rids = sorted(refmap)
            for i in rng.integers(0, len(rids), size=min(n, 1500)):
                for j in rng.integers(0, len(rids), size=3):
                    a, b = rids[int(i)], rids[int(j)]
                    if a == b:
                        continue
                    for mode in ("proposed", "coverage_blind", "imputed_absent",
                                 "base"):
                        x = s(refmap[a], refmap[b], mode)
                        y = base(refmap[a], refmap[b], mode)
                        if x != y and not (x is None and y is None):
                            bad += 1
                        mixed += 1
    return {"pairs_sampled": n, "same_case_id_cross_corpus_checks": mixed,
            "modes": 4, "void_policies": 2, "mismatches": bad, "ok": bad == 0}


# ===========================================================================
# POOLS
# ===========================================================================
def pool_1311():
    """The deployment pool exactly as ipkg_final_cases assembles it: the 113
    single-organ PREDICTED records in kits, lits, msd order, then the pipeline team's
    round-3 de-duplicated FLARE23 corpus. The order is load-bearing - ties at
    S = 1.0 break by pool order under the stable sort."""
    return FC.pool_map()


def reference_1311():
    """case_id -> reference record. The 113 have reference-mask records; a
    FLARE23 pool member IS reference-derived, so it is its own reference."""
    path = os.path.join(FC.CORPORA_DIR, "corpus_reference_113.json")
    ref = {r["case_id"]: r
           for r in json.load(open(path, encoding="utf-8"))["records"]}
    return ref


def pool_published():
    """the pipeline team's original corpora and the original 1,425-case pool, for --verify."""
    def recs(p):
        return json.load(open(p, encoding="utf-8"))["records"]
    pred, gt = {}, {}
    for ds in ("kits", "lits", "msd"):
        for r in recs(os.path.join(PUBLISHED_CORP, "corpus_predicted_%s.json" % ds)):
            pred[r["case_id"]] = r
        for r in recs(os.path.join(PUBLISHED_CORP, "corpus_gt_%s.json" % ds)):
            gt[r["case_id"]] = r
    flare = {r["case_id"]: r
             for r in recs(os.path.join(PUBLISHED_CORP, "corpus_flare23_kg.json"))}
    return pred, gt, flare


# ===========================================================================
# VERIFICATION: reproduce the published tables, cell for cell
# ===========================================================================
def verify():
    """Run the vendored metric layer and the delivered scorer on the pipeline team's original
    corpora, in the pipeline team's call order with the pipeline team's seed, and compare every published
    cell. Returns (ok, report)."""
    CORE.ORGAN_UNIVERSE = UNIVERSE
    rng = np.random.default_rng(12345)                       # the pipeline team's seed
    pred, gt, flare = pool_published()
    q113 = [i for i in pred if i in gt]
    rep = {"n_q113": len(q113), "n_flare": len(flare),
           "pool": len(q113) + len(flare), "checks": [], "mismatches": []}

    # ---- Table 7 (primary) ------------------------------------------------
    t7_pq = {
        "(i) Reference phenotypes + gamma (upper bound)": per_query(gt, gt, q113, q113, "proposed"),
        "(ii) Predicted phenotypes + gamma (proposed)": per_query(pred, gt, q113, q113, "proposed"),
        "(iii) Predicted, coverage-blind (no gamma)": per_query(pred, gt, q113, q113, "coverage_blind"),
        "(iv) Predicted, organ-agnostic baseline": per_query(pred, gt, q113, q113, "base"),
    }
    t7 = {k: agg(v) for k, v in t7_pq.items()}
    ref = "(ii) Predicted phenotypes + gamma (proposed)"
    f1 = {k: paired(t7_pq[k], t7_pq[ref], rng) for k in t7_pq if k != ref}
    holm(f1)
    for k, v in f1.items():
        t7[k]["dmAP_vs_ii"] = v

    # ---- Table 8 (stress) -------------------------------------------------
    pool_pred = {**pred, **flare}
    pool_gt = {**gt, **flare}
    cids = q113 + sorted(flare)
    t8_pq = {
        "(i) Reference phenotypes + gamma (upper bound)": per_query(pool_gt, pool_gt, q113, cids, "proposed"),
        "(ii) Predicted phenotypes + gamma (proposed)": per_query(pool_pred, pool_pred, q113, cids, "proposed"),
        "(iii) Predicted, coverage-blind (no gamma)": per_query(pool_pred, pool_pred, q113, cids, "coverage_blind"),
        "(iv) Predicted, organ-agnostic baseline": per_query(pool_pred, pool_pred, q113, cids, "base"),
    }
    t8 = {k: agg(v) for k, v in t8_pq.items()}
    f1s = {k: paired(t8_pq[k], t8_pq[ref], rng) for k in t8_pq if k != ref}
    holm(f1s)
    for k, v in f1s.items():
        t8[k]["dmAP_vs_ii"] = v

    # ---- compare ----------------------------------------------------------
    for fname, got in (("retrieval_tab7_primary.json", t7),
                       ("retrieval_tab8_stress.json", t8)):
        want = json.load(open(os.path.join(PUBLISHED_RES, fname), encoding="utf-8"))["rows"]
        for row, cells in want.items():
            for metric, v in cells.items():
                g = got.get(row, {}).get(metric)
                if metric == "dmAP_vs_ii":
                    same = (isinstance(g, dict) and g.get("delta") == v.get("delta")
                            and g.get("ci95") == v.get("ci95")
                            and round(g.get("p_holm", -1), 5) == round(v.get("p_holm", -2), 5))
                else:
                    same = (g == v)
                rep["checks"].append([fname, row, metric, same])
                if not same:
                    rep["mismatches"].append([fname, row, metric, v, g])
    ok = not rep["mismatches"]
    rep["ok"] = ok
    rep["n_checks"] = len(rep["checks"])
    # Kept so the de-duplication step of the ladder can be read off the same
    # run that validated the metric layer: these ARE the manuscript's rows,
    # recomputed, on the 1,425-case pool the manuscript used.
    rep["rows_1425"] = {
        "primary": {k: {m: v for m, v in r.items() if m != "dmAP_vs_ii"}
                    for k, r in t7.items()},
        "stress": {k: {m: v for m, v in r.items() if m != "dmAP_vs_ii"}
                   for k, r in t8.items()}}
    return ok, rep


# ===========================================================================
# THE FOUR TABLES ON THE 1,311 POOL
# ===========================================================================
ROWS_GRAPH = [
    ("(i)", "Reference phenotypes, support-restricted (upper bound)", "ref", "proposed"),
    ("(ii)", "Predicted phenotypes, support-restricted (proposed)", "pred", "proposed"),
    ("(iii)", "Predicted, coverage-blind (one-sided organs $=$ mismatch)", "pred", "coverage_blind"),
    ("(iii')", "Predicted, imputed absence (one-sided organs may agree)", "pred", "imputed_absent"),
    ("(iv)", "Predicted, organ-agnostic baseline", "pred", "base"),
]


def table_primary(pool, refmap, arms, log=print):
    """tab:retrieval-primary. 113 queries over the 113 reference-evaluable
    cases (112 candidates per query). Ranking on PREDICTED phenotypes,
    relevance from REFERENCE-derived phenotypes, so the representation under
    test plays no part in defining the gold standard."""
    out = {}
    q113 = [c for c, r in pool.items() if r.get("dataset") != "flare23"]
    for arm in arms:
        vk, vf = ARMS[arm]
        p = arm_pool(arm, pool)
        g = {c: VOCAB[vk](r) for c, r in refmap.items()}
        sim = Sim({**p, **{("__ref__" + k): v for k, v in g.items()}}, vf)
        rows, pq = {}, {}
        for tag, label, side, mode in ROWS_GRAPH:
            s = g if side == "ref" else p
            pq[tag] = per_query(s, g, q113, q113, mode, simfn=sim, keep_rels=True)
            rows[tag] = dict(agg(pq[tag]), label=label, ranks_on=side, mode=mode)
        rng = np.random.default_rng(20261003)
        cmp_ = {tag: paired(pq[tag], pq["(ii)"], rng)
                for tag, _l, _s, _m in ROWS_GRAPH if tag != "(ii)"}
        holm(cmp_)
        for tag, v in cmp_.items():
            rows[tag]["dmAP_vs_ii"] = v
        # per-dataset breakdown: the MSD rows the advisor flagged
        byds = {}
        for ds in ("kits", "lits", "msd"):
            ids = [c for c in q113 if pool[c].get("dataset") == ds]
            byds[ds] = {tag: agg({k: v for k, v in pq[tag].items() if k in ids})
                        for tag, _l, _s, _m in ROWS_GRAPH}
        out[arm] = {"rows": rows, "by_dataset": byds,
                    "n_queries": len(q113), "n_candidates": len(q113) - 1,
                    "_pq": pq}
        log("  primary [%-16s] (ii) mAP=%s P@10=%s  msd (ii) mAP=%s"
            % (arm, rows["(ii)"]["mAP"], rows["(ii)"]["P@10"],
               byds["msd"]["(ii)"]["mAP"]))
    return out


def table_stress(pool, arms, log=print):
    """tab:retrieval. 113 single-organ queries against the whole 1,311-case
    pool, relevance construction-defined from each candidate's own record."""
    out = {}
    q113 = [c for c, r in pool.items() if r.get("dataset") != "flare23"]
    cids = list(pool)
    for arm in arms:
        _vk, vf = ARMS[arm]
        p = arm_pool(arm, pool)
        sim = Sim(p, vf)
        rows, pq = {}, {}
        for tag, label, _side, mode in ROWS_GRAPH:
            if tag == "(i)":
                continue            # no reference record for a FLARE23 pool member
            pq[tag] = per_query(p, p, q113, cids, mode, simfn=sim, keep_rels=True)
            rows[tag] = dict(agg(pq[tag]), label=label, ranks_on="pred", mode=mode)
        rng = np.random.default_rng(20261003)
        cmp_ = {tag: paired(pq[tag], pq["(ii)"], rng)
                for tag in pq if tag != "(ii)"}
        holm(cmp_)
        for tag, v in cmp_.items():
            rows[tag]["dmAP_vs_ii"] = v
        byds = {}
        for ds in ("kits", "lits", "msd"):
            ids = [c for c in q113 if pool[c].get("dataset") == ds]
            byds[ds] = {tag: agg({k: v for k, v in pq[tag].items() if k in ids})
                        for tag in pq}
        out[arm] = {"rows": rows, "by_dataset": byds, "n_queries": len(q113),
                    "n_pool": len(cids), "_pq": pq}
        log("  stress  [%-16s] (ii) mAP=%s P@10=%s n_q=%s  msd (ii) mAP=%s"
            % (arm, rows["(ii)"]["mAP"], rows["(ii)"]["P@10"],
               rows["(ii)"]["n_queries"], byds["msd"]["(ii)"]["mAP"]))
    return out


STRATA_MODES = (("proposed", "Proposed IPKG"),
                ("coverage_blind", "(iii) Coverage-blind"),
                ("imputed_absent", "(iii') Imputed absence"))


def table_strata(pool, arms, log=print, flare_within=False):
    """tab:retrieval-strata. (a) within-dataset control (identity check),
    (b) single-organ -> FLARE23, (c) FLARE23 -> single-organ. In every
    stratum the candidate pool is coverage-homogeneous."""
    out = {}
    q113 = [c for c, r in pool.items() if r.get("dataset") != "flare23"]
    fids = [c for c, r in pool.items() if r.get("dataset") == "flare23"]
    for arm in arms:
        _vk, vf = ARMS[arm]
        p = arm_pool(arm, pool)
        sim = Sim(p, vf)
        st = {}
        # (a) within-dataset
        wa, pooled = {}, {m: {} for m, _ in STRATA_MODES}
        groups = [("kits",), ("lits",), ("msd",)]
        if flare_within:
            groups.append(("flare23",))
        for (ds,) in groups:
            ids = [c for c in pool if pool[c].get("dataset") == ds]
            pqs = {m: per_query(p, p, ids, ids, m, simfn=sim,
                                multi_organ_queries=(ds == "flare23"))
                   for m, _l in STRATA_MODES}
            ident = {m: all(abs(pqs[m][k]["ap"] - pqs["proposed"][k]["ap"]) < 1e-12
                            and abs(pqs[m][k]["ndcg"] - pqs["proposed"][k]["ndcg"]) < 1e-12
                            for k in pqs["proposed"] if k in pqs[m])
                     and set(pqs[m]) == set(pqs["proposed"])
                     for m, _l in STRATA_MODES}
            wa[ds] = {m: agg(v) for m, v in pqs.items()}
            wa[ds]["rankings_identical"] = {m: bool(ident[m]) for m in ident}
            if ds != "flare23":
                for m, _l in STRATA_MODES:
                    pooled[m].update(pqs[m])
        st["(a) within-dataset control"] = {
            "per_dataset": wa,
            "pooled": {m: agg(v) for m, v in pooled.items()},
            "identity_check_passed": all(
                wa[ds]["rankings_identical"]["coverage_blind"]
                for ds in ("kits", "lits", "msd")),
            "note": "identity check, no p-value"}
        # (b) single-organ -> FLARE23
        b = {m: per_query(p, p, q113, fids, m, simfn=sim) for m, _l in STRATA_MODES}
        # (c) FLARE23 -> single-organ
        c = {m: per_query(p, p, fids, q113, m, simfn=sim, multi_organ_queries=True)
             for m, _l in STRATA_MODES}
        rng = np.random.default_rng(20261003)
        f2 = {}
        for tag, d in (("(b) single-organ -> FLARE23", b),
                       ("(c) FLARE23 -> single-organ", c)):
            f2[tag + " | (iii)"] = paired(d["proposed"], d["coverage_blind"], rng)
            f2[tag + " | (iii')"] = paired(d["proposed"], d["imputed_absent"], rng)
        holm(f2)
        st["(b) single-organ -> FLARE23"] = {
            "rows": {m: agg(v) for m, v in b.items()},
            "delta_obs": {"(iii)": f2["(b) single-organ -> FLARE23 | (iii)"],
                          "(iii')": f2["(b) single-organ -> FLARE23 | (iii')"]}}
        st["(c) FLARE23 -> single-organ"] = {
            "rows": {m: agg(v) for m, v in c.items()},
            "delta_obs": {"(iii)": f2["(c) FLARE23 -> single-organ | (iii)"],
                          "(iii')": f2["(c) FLARE23 -> single-organ | (iii')"]}}
        out[arm] = st
        log("  strata  [%-16s] (a) mAP=%s  (b) mAP=%s n=%s  (c) mAP=%s n=%s"
            % (arm, st["(a) within-dataset control"]["pooled"]["proposed"]["mAP"],
               st["(b) single-organ -> FLARE23"]["rows"]["proposed"]["mAP"],
               st["(b) single-organ -> FLARE23"]["rows"]["proposed"]["n_queries"],
               st["(c) FLARE23 -> single-organ"]["rows"]["proposed"]["mAP"],
               st["(c) FLARE23 -> single-organ"]["rows"]["proposed"]["n_queries"]))
    return out


CROSS_QUERIES = [("Pancreas -> FLARE23, contained tumor", "pancreas_041"),
                 ("LiTS -> FLARE23, multifocal liver", "volume-101"),
                 ("Pancreas -> FLARE23, high burden", "pancreas_015")]


def table_cross(pool, arms, log=print):
    """tab:cross-examples. Rank of the intended same-organ FLARE23 target for
    three single-organ queries within the full pool. Definitions are the pipeline team's
    tab10_example_ranks.py: the rank of the first FLARE23 candidate, and of
    the first FLARE23 candidate RELEVANT under construction-defined
    relevance."""
    out = {}
    cids = list(pool)
    flare = {c for c, r in pool.items() if r.get("dataset") == "flare23"}
    for arm in arms:
        _vk, vf = ARMS[arm]
        p = arm_pool(arm, pool)
        sim = Sim(p, vf)
        qrows = []
        for title, q in CROSS_QUERIES:
            if q not in p:
                qrows.append({"title": title, "query": q,
                              "error": "not in the 1,311 pool"})
                continue
            qo = p[q]["observed_organs"][0]
            ph = p[q]["organs"][qo]
            row = {"title": title, "query": q, "organ": qo,
                   "query_phenotype": {k: ph.get(k) for k in CATS}}
            for label, mode in (("proposed", "proposed"),
                                ("coverage_blind", "coverage_blind"),
                                ("imputed_absent", "imputed_absent")):
                cands = [(sim(p[q], p[b], mode), b) for b in cids if b != q]
                cands = [(s, b) for s, b in cands if s is not None]
                cands.sort(key=lambda x: -x[0])
                ranked = [b for _s, b in cands]
                first_fl = next((i + 1 for i, b in enumerate(ranked) if b in flare), None)
                first_rel = next((i + 1 for i, b in enumerate(ranked)
                                  if b in flare and CORE.relevant(p[q], p[b])), None)
                n_rel = sum(1 for b in flare if CORE.relevant(p[q], p[b]))
                fl_only = [b for b in ranked if b in flare]
                within = next((i + 1 for i, b in enumerate(fl_only)
                               if CORE.relevant(p[q], p[b])), None)
                # how wide the tie is at the rank being reported
                tie = None
                if first_rel:
                    s_at = cands[first_rel - 1][0]
                    tie = sum(1 for s, _b in cands if abs(s - s_at) < 1e-12)
                row[label] = {"first_flare23_rank": first_fl,
                              "first_relevant_flare23_rank": first_rel,
                              "rank_within_flare23_only": within,
                              "single_organ_candidates_above":
                                  (first_fl - 1) if first_fl else None,
                              "n_relevant_flare23_candidates": n_rel,
                              "n_ranked": len(ranked),
                              "tie_width_at_reported_rank": tie}
            qrows.append(row)
        out[arm] = {"pool_size": len(cids), "queries": qrows}
        log("  cross   [%-16s] %s" % (arm, "  ".join(
            "%s:#%s" % (r["query"], (r.get("proposed") or {}).get("first_relevant_flare23_rank"))
            for r in qrows)))
    return out


# ===========================================================================
# THE TWO EFFECTS, SEPARATELY (advisor item 2)
# ===========================================================================
def effects(pool, log=print, heavy=True, k=10):
    """How many QUERIES each correction moves, re-derived on 1,311.

    A query "moves" when its top-10 changes: as a SET (different cases shown)
    or as an ORDER (same cases, different order). Reported apart because a
    re-ordering inside an exact tie is not the same event as a different case
    appearing on screen. Configuration (ii), the proposed scorer, which is
    what the application ranks with."""
    res = {"configuration": "(ii) proposed, gamma_min=1", "k": k,
           "pool_size": len(pool), "arms": {}}
    base_arm = "published"
    cids = list(pool)
    pools = {a: arm_pool(a, pool) for a in
             (base_arm, "void_fix", "subsite_corpus", "subsite_resolved", "corrected")}
    sims = {a: Sim(pools[a], ARMS[a][1]) for a in pools}

    def top10(arm, q):
        p, sim = pools[arm], sims[arm]
        cands = []
        qr = p[q]
        for b in cids:
            if b == q:
                continue
            s = sim(qr, p[b], "proposed")
            if s is None:
                continue
            cands.append((s, b))
        cands.sort(key=lambda x: -x[0])
        return [b for _s, b in cands[:k]], (cands[0][0] if cands else None)

    which = {"single_organ": [c for c, r in pool.items() if r.get("dataset") != "flare23"]}
    if heavy:
        which["flare23"] = [c for c, r in pool.items() if r.get("dataset") == "flare23"]
    for grp, ids in which.items():
        base = {}
        t0 = time.time()
        for q in ids:
            base[q] = top10(base_arm, q)
        log("    effects/%s base arm done in %.1fs (%d queries)"
            % (grp, time.time() - t0, len(ids)))
        for arm in ("void_fix", "subsite_corpus", "subsite_resolved", "corrected"):
            t0 = time.time()
            set_moved = order_moved = 0
            moved_ids, ds_moved = [], {}
            s1d = []
            enters, leaves = {}, {}
            for q in ids:
                a, s1a = base[q]
                b, s1b = top10(arm, q)
                if set(a) != set(b):
                    set_moved += 1
                    moved_ids.append(q)
                    d = pool[q].get("dataset")
                    ds_moved[d] = ds_moved.get(d, 0) + 1
                    for x in set(b) - set(a):
                        enters[x] = enters.get(x, 0) + 1
                    for x in set(a) - set(b):
                        leaves[x] = leaves.get(x, 0) + 1
                if a != b:
                    order_moved += 1
                if s1a is not None and s1b is not None:
                    s1d.append(s1b - s1a)
            # WHICH CASES move in and out, not only how many queries moved.
            # A correction that promotes three cases into a thousand lists is
            # a different event from one that reshuffles a thousand lists, and
            # the count of moved queries cannot tell them apart.
            # TIES BREAK BY CASE ID, NOT BY WHATEVER ORDER THE DICT HAPPENED
            # TO FILL IN. These two lists are printed into a manuscript table
            # (the six most-entering cases), and the counts tie hard: on the
            # corrected FLARE23 arm the top three all enter exactly 973
            # top-10s. Sorting on the count alone left the order to insertion,
            # so two consecutive runs of this script over identical inputs
            # named different cases in the table, which is a reproducibility
            # defect in the artefact even though no metric moves. The second
            # key makes the list a function of the data.
            top_in = sorted(enters.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            top_out = sorted(leaves.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            res["arms"].setdefault(arm, {})[grp] = {
                "n_queries": len(ids),
                "top10_set_changed": set_moved,
                "top10_order_changed": order_moved,
                "by_dataset_set_changed": ds_moved,
                "mean_S1_delta": round(float(np.mean(s1d)), 5) if s1d else None,
                "distinct_cases_entering_a_top10": len(enters),
                "distinct_cases_leaving_a_top10": len(leaves),
                "cases_entering_most_top10s":
                    [{"case_id": c, "dataset": pool[c].get("dataset"),
                      "n_top10s_entered": n,
                      "pancreas_site_before":
                          ((pools[base_arm][c].get("organs") or {})
                           .get("pancreas") or {}).get("anatomic_location"),
                      "pancreas_site_after":
                          ((pools[arm][c].get("organs") or {})
                           .get("pancreas") or {}).get("anatomic_location")}
                     for c, n in top_in],
                "cases_leaving_most_top10s":
                    [{"case_id": c, "dataset": pool[c].get("dataset"),
                      "n_top10s_left": n} for c, n in top_out],
                "example_moved": moved_ids[:5],
                "seconds": round(time.time() - t0, 1)}
            log("      %-16s %-12s set-changed %4d / %4d   order-changed %4d"
                % (arm, grp, set_moved, len(ids), order_moved))
    return res


# ===========================================================================
# WHICH SIDE MOVED: the ranking, or the gold standard?
#
# This separation is not cosmetic. `similarity()` compares the sub-site on the
# PANCREAS ONLY - organ_cats() gives the other organs three categories - but
# `relevant()` reads all four categories on EVERY shared organ. Withdrawing
# the spurious pancreatic word from a kidney or a liver therefore cannot
# change a single score and does change the relevance labels, because a
# free informative agreement disappears from the >=2 bar. A table that
# reports only the net mAP move hides the fact that part of it is the gold
# standard moving under a ranking that did not.
# ===========================================================================
def ranking_vs_relevance(pool, refmap, log=print):
    """2x2: ranking arm x relevance vocabulary, configuration (ii).

    primary: relevance comes from the reference corpus, so its sub-site
             vocabulary can be varied independently of the ranking arm.
    stress:  relevance is construction-defined from the candidate's own
             record, so the two are varied together and the cross cells are
             produced by scoring one vocabulary and judging on the other.
    """
    q113 = [c for c, r in pool.items() if r.get("dataset") != "flare23"]
    cids = list(pool)
    out = {"note": ("similarity() compares anatomic_location on the pancreas "
                    "only; relevant() reads it on every shared organ, so a "
                    "sub-site change on a kidney or a liver moves the gold "
                    "standard without moving any score"),
           "primary": {}, "stress": {}, "by_dataset": {}}
    pools = {a: arm_pool(a, pool) for a in ("published", "corrected")}
    refs = {v: {c: VOCAB[v](r) for c, r in refmap.items()} for v in ("D", "P")}
    sims = {a: Sim(dict(list(pools[a].items())
                        + [("__r" + v + "__" + c, rr)
                           for v in refs for c, rr in refs[v].items()]),
                   ARMS[a][1]) for a in pools}
    for a in pools:
        for v in refs:
            pq = per_query(pools[a], refs[v], q113, q113, "proposed",
                           simfn=sims[a])
            out["primary"]["rank=%s | relevance=%s" % (a, v)] = agg(pq)
            out["by_dataset"].setdefault("rank=%s | relevance=%s" % (a, v), {})
            for ds in ("kits", "lits", "msd"):
                ids = [c for c in q113 if pool[c].get("dataset") == ds]
                out["by_dataset"]["rank=%s | relevance=%s" % (a, v)][ds] =                     agg({k: x for k, x in pq.items() if k in ids})
            log("    rank=%-10s relevance=%s  primary mAP=%s P@10=%s"
                % (a, v, out["primary"]["rank=%s | relevance=%s" % (a, v)]["mAP"],
                   out["primary"]["rank=%s | relevance=%s" % (a, v)]["P@10"]))
    for a in pools:
        for v in ("published", "corrected"):
            pq = per_query(pools[a], pools[v], q113, cids, "proposed",
                           simfn=sims[a])
            out["stress"]["rank=%s | relevance=%s" % (a, v)] = agg(pq)
            log("    rank=%-10s relevance=%-10s stress  mAP=%s n_q=%s"
                % (a, v, out["stress"]["rank=%s | relevance=%s" % (a, v)]["mAP"],
                   out["stress"]["rank=%s | relevance=%s" % (a, v)]["n_queries"]))
    # how many relevance LABELS the sub-site change flips, counted directly
    flips = {"primary_pairs_checked": 0, "primary_labels_flipped": 0,
             "by_dataset": {}}
    for q in q113:
        ds = pool[q].get("dataset")
        d = flips["by_dataset"].setdefault(ds, {"pairs": 0, "flipped": 0,
                                                "to_relevant": 0,
                                                "to_irrelevant": 0})
        for c in q113:
            if c == q:
                continue
            a = bool(CORE.relevant(refs["D"][q], refs["D"][c]))
            b = bool(CORE.relevant(refs["P"][q], refs["P"][c]))
            flips["primary_pairs_checked"] += 1
            d["pairs"] += 1
            if a != b:
                flips["primary_labels_flipped"] += 1
                d["flipped"] += 1
                d["to_relevant" if b else "to_irrelevant"] += 1
    out["relevance_label_flips"] = flips
    log("    relevance labels flipped by the sub-site correction: %d of %d "
        "ordered query-candidate pairs"
        % (flips["primary_labels_flipped"], flips["primary_pairs_checked"]))
    return out


# ===========================================================================
# CIRCULARITY (advisor item 3)
# ===========================================================================
def _random_order_ap(rels, rng, reps=20):
    """Expected AP of a RANDOM ordering of the same candidate set - the floor
    any ranker clears for free, given how many candidates the relevance rule
    calls relevant."""
    r = np.asarray(rels, dtype=float)
    n, pos = len(r), float(r.sum())
    if pos == 0:
        return None
    acc = 0.0
    for _ in range(reps):
        p = rng.permutation(r)
        hit = ap = 0.0
        for i, x in enumerate(p):
            if x:
                hit += 1
                ap += hit / (i + 1)
        acc += ap / pos
    return acc / reps, pos / n


def held_out_relevance(axis):
    """Relevance on ONE axis the ranker was not allowed to see.

    The delivered rule reads all four axes and asks for >=2 informative
    agreements on a shared organ; similarity() ranks on the same four. A
    relevance defined on an axis the score never saw is not circular - it is
    weaker (one agreement, one axis), and that is stated rather than hidden."""
    def rel(A, B):
        shared = set(A["observed_organs"]) & set(B["observed_organs"])
        for o in shared:
            pa = (A.get("organs") or {}).get(o) or {}
            pb = (B.get("organs") or {}).get(o) or {}
            va = pa.get(axis)
            if va == pb.get(axis) and va not in ("none", "na", "unknown", None):
                return True
        return False
    return rel


def circularity(pool, refmap, log=print):
    """What the headline mAP is and is not evidence of."""
    out = {"note": ("relevant() derives the gold standard from the same four "
                    "fields similarity() ranks on, so the benchmark tests "
                    "whether the ranker behaves as designed and cannot test "
                    "whether the phenotypes are correct")}
    q113 = [c for c, r in pool.items() if r.get("dataset") != "flare23"]
    cids = list(pool)
    arm = HEADLINE
    _vk, vf = ARMS[arm]
    p = arm_pool(arm, pool)
    g = {c: VOCAB[_vk](r) for c, r in refmap.items()}
    sim = Sim(p, vf)
    rng = np.random.default_rng(99)

    # --- 1. prevalence and the random-order floor --------------------------
    floors = {}
    for tag, (simrecs, relrecs, qids, cands, multi) in {
            "primary (113 x 112, reference-derived relevance)":
                (p, g, q113, q113, False),
            "stress (113 x 1,311, construction-defined relevance)":
                (p, p, q113, cids, False)}.items():
        pq = per_query(simrecs, relrecs, qids, cands, "proposed", simfn=sim,
                       multi_organ_queries=multi, keep_rels=True)
        rows = []
        for q, m in pq.items():
            v = _random_order_ap(m["_rels"], rng)
            if v:
                rows.append((v[0], v[1], m["ap"]))
        floors[tag] = {
            "n_queries_scored": len(rows),
            "mean_relevance_prevalence": round(float(np.mean([r[1] for r in rows])), 4),
            "mAP_random_order": round(float(np.mean([r[0] for r in rows])), 4),
            "mAP_proposed": round(float(np.mean([r[2] for r in rows])), 4),
        }
        floors[tag]["headroom_above_chance"] = round(
            floors[tag]["mAP_proposed"] - floors[tag]["mAP_random_order"], 4)
        log("    circularity floor  %-58s chance %.4f -> proposed %.4f"
            % (tag, floors[tag]["mAP_random_order"], floors[tag]["mAP_proposed"]))
    out["random_order_floor"] = floors

    # --- 2. held-out-axis relevance ----------------------------------------
    ho = {}
    for axis in CATS:
        relfn = held_out_relevance(axis)
        a = agg(per_query(p, g, q113, q113, "proposed", simfn=sim, relfn=relfn))
        b = agg(per_query(p, p, q113, cids, "proposed", simfn=sim, relfn=relfn))
        ho[axis] = {"primary": a, "stress": b}
        log("    held-out axis %-18s primary mAP=%s (n=%s)  stress mAP=%s (n=%s)"
            % (axis, a["mAP"], a["n_queries"], b["mAP"], b["n_queries"]))
    out["held_out_axis_relevance"] = {
        "definition": ("relevance = agreement on ONE axis on a shared organ "
                       "(the delivered rule needs >=2 agreements over all "
                       "four); the ranker still scores all four axes, so this "
                       "weakens rather than removes the dependence - the only "
                       "fully disjoint variant would also have to withhold the "
                       "axis from the score, which changes the system under "
                       "test"),
        "rows": ho}

    # --- 3. the systematic sub-site perturbation, re-derived ---------------
    # The published claim is that correcting 52 of the 56 MSD sub-sites moved
    # mAP by exactly 0.0000. Re-derived here as the delivered-vs-corrected
    # sub-site contrast on 1,311, under BOTH void policies, because under the
    # delivered void semantics a re-spelling can move a score and under the
    # fix it cannot.
    pert = {}
    for a1, a2, label in (("published", "subsite_msd_only",
                           "MSD sub-sites only (the published perturbation), "
                           "delivered void (V0)"),
                          ("published", "subsite_resolved",
                           "full sub-site correction, delivered void (V0)"),
                          ("void_fix", "corrected",
                           "full sub-site correction, void fix (V1)")):
        r = {}
        for nm, a in ((a1, a1), (a2, a2)):
            _v, vfx = ARMS[a]
            pa = arm_pool(a, pool)
            ga = {c: VOCAB[ARMS[a][0]](rr) for c, rr in refmap.items()}
            sa = Sim(pa, vfx)
            r[nm] = {"primary": agg(per_query(pa, ga, q113, q113, "proposed", simfn=sa)),
                     "stress": agg(per_query(pa, pa, q113, list(pa), "proposed", simfn=sa))}
        pert[label] = {
            "from": a1, "to": a2,
            "primary_mAP": [r[a1]["primary"]["mAP"], r[a2]["primary"]["mAP"]],
            "primary_dmAP": round((r[a2]["primary"]["mAP"] or 0)
                                  - (r[a1]["primary"]["mAP"] or 0), 4),
            "stress_mAP": [r[a1]["stress"]["mAP"], r[a2]["stress"]["mAP"]],
            "stress_dmAP": round((r[a2]["stress"]["mAP"] or 0)
                                 - (r[a1]["stress"]["mAP"] or 0), 4)}
        log("    subsite sensitivity %-44s primary %s  stress %s"
            % (label, pert[label]["primary_dmAP"], pert[label]["stress_dmAP"]))
    out["subsite_perturbation"] = pert

    # --- 4. what cannot be produced ----------------------------------------
    cen = os.path.join(LIVE, "round3", "dedup_audit",
                       "flare23_twin_census_complete_2026-10-03.json")
    twin = {"census": cen, "usable_pairs_in_pool": None}
    if os.path.isfile(cen):
        d = json.load(open(cen, encoding="utf-8"))
        pairs = d.get("pairs") or []
        fl = set(c for c, r in pool.items() if r.get("dataset") == "flare23")

        def norm(f):
            return f if f in fl else ("flare23_" + f if "flare23_" + f in fl else None)
        both = [(q["public_case"], norm(q["flare23"])) for q in pairs
                if q.get("public_case") in pool and norm(q.get("flare23"))]
        twin = {"census": os.path.basename(cen), "n_pairs_in_census": len(pairs),
                "flare23_side_in_pool": sum(1 for q in pairs if norm(q.get("flare23"))),
                "usable_pairs_in_pool": len(both),
                "why": ("scan identity is the one relevance signal in hand that "
                        "is independent of the ranked fields, and the pool has "
                        "none left: de-duplication removed exactly the FLARE23 "
                        "twins of the query set, so no twin pair has both "
                        "members in the 1,311 pool")}
    out["external_relevance_unavailable"] = twin
    out["cannot_produce"] = [
        "a clinician-adjudicated relevance set: none exists for these 1,311 cases",
        ("a scan-identity (twin) relevance set: 0 of the %d census pairs have "
         "both members in the de-duplicated pool"
         % (twin.get("n_pairs_in_census") or 0)),
        ("a relevance definition fully disjoint from the score: every field the "
         "records carry that relevance could be built from is a field the score "
         "already reads; withholding one from the score changes the system "
         "under test rather than the evaluation of it"),
    ]
    return out


# ===========================================================================
# WHAT THE MANUSCRIPT CURRENTLY PRINTS  (advisor item 1)
# ===========================================================================
# Transcribed from paper/src/JBI/JBI-VKG-Final.tex (the submitted draft) and
# cross-checked against pipeline_handoff/results/retrieval/*.json by verify().
# TBD marks a cell the draft prints as \TBD.
PUBLISHED = {
  "tab:retrieval-primary": {
    "caption_pool": "113 reference-evaluable cases, 112 candidates per query",
    "cells": {
      "(i)":    {"P@10": 0.996, "mAP": 0.992, "nDCG": 0.999, "Mism.@10": 0.00, "dmAP": +0.020},
      "(ii)":   {"P@10": 0.992, "mAP": 0.972, "nDCG": 0.996, "Mism.@10": 0.00, "dmAP": None},
      "(iii)":  {"P@10": 0.969, "mAP": 0.933, "nDCG": 0.977, "Mism.@10": 0.02, "dmAP": -0.039},
      "(iii')": {"P@10": "TBD", "mAP": 0.911, "nDCG": "TBD", "Mism.@10": "TBD", "dmAP": -0.061},
      "(iv)":   {"P@10": 0.436, "mAP": 0.493, "nDCG": 0.439, "Mism.@10": 0.55, "dmAP": -0.479},
      "(v)":    {"P@10": 0.804, "mAP": 0.715, "nDCG": 0.819, "Mism.@10": 0.14, "dmAP": -0.257},
      "(vi)":   {"P@10": 0.953, "mAP": 0.954, "nDCG": 0.951, "Mism.@10": 0.00, "dmAP": -0.018},
      "(vii)":  {"P@10": 0.840, "mAP": 0.709, "nDCG": 0.855, "Mism.@10": 0.06, "dmAP": -0.262},
      "(vii')": {"P@10": 0.913, "mAP": 0.925, "nDCG": 0.914, "Mism.@10": 0.00, "dmAP": -0.047},
    }},
  "tab:retrieval": {
    "caption_pool": "1,425-case pool (113 single-organ + 1,312 FLARE23)",
    "cells": {
      "(i)":    {"P@5": 1.000, "P@10": 1.000, "mAP": 0.998, "nDCG": 1.000, "Spur.@10": 0.00, "Mism.@10": 0.00, "dmAP": -0.001},
      "(ii)":   {"P@5": 1.000, "P@10": 1.000, "mAP": 0.999, "nDCG": 1.000, "Spur.@10": 0.00, "Mism.@10": 0.00, "dmAP": None},
      "(iii)":  {"P@5": 1.000, "P@10": 0.989, "mAP": 0.938, "nDCG": 0.993, "Spur.@10": 0.00, "Mism.@10": 0.00, "dmAP": -0.061},
      "(iii')": {"P@5": "TBD", "P@10": "TBD", "mAP": 0.757, "nDCG": "TBD", "Spur.@10": "TBD", "Mism.@10": "TBD", "dmAP": -0.242},
      "(iv)":   {"P@5": 0.481, "P@10": 0.440, "mAP": 0.327, "nDCG": 0.444, "Spur.@10": 0.557, "Mism.@10": 0.56, "dmAP": -0.672},
    }},
  "tab:retrieval-strata": {
    "caption_pool": "1,425-case pool; strata as defined in the caption",
    "cells": {
      "(a) Proposed":  {"n_q": 109, "P@10": 0.987, "mAP": 0.994, "Mism.@10": 0.00},
      "(a) (iii)":     {"n_q": 109, "P@10": 0.987, "mAP": 0.994, "Mism.@10": 0.00},
      "(a) (iii')":    {"n_q": 109, "P@10": 0.987, "mAP": 0.994, "Mism.@10": 0.00},
      "(b) Proposed":  {"n_q": 109, "P@10": 0.997, "mAP": 1.000, "Mism.@10": 0.00},
      "(b) (iii)":     {"n_q": 109, "P@10": 0.997, "mAP": 1.000, "Mism.@10": 0.00},
      "(b) (iii')":    {"n_q": 109, "P@10": "TBD", "mAP": 1.000, "Mism.@10": "TBD"},
      "(c) Proposed":  {"n_q": 592, "P@10": 0.710, "mAP": 0.660, "Mism.@10": "n/a"},
      "(c) (iii)":     {"n_q": 592, "P@10": 0.710, "mAP": 0.660, "Mism.@10": "n/a"},
      "(c) (iii')":    {"n_q": 592, "P@10": "TBD", "mAP": 1.000, "Mism.@10": "n/a"},
    }},
  "tab:cross-examples": {
    "caption_pool": "1,425-case pool",
    "cells": {
      "pancreas_041": {"Rank (proposed)": 40, "Rank (blind)": 53},
      "volume-101":   {"Rank (proposed)": 16, "Rank (blind)": 20},
      "pancreas_015": {"Rank (proposed)": 45, "Rank (blind)": 53},
    }},
}


def cell_diff(primary, stress, strata, cross, arm=HEADLINE):
    """Every cell of the four tables, as the manuscript prints it and as it
    comes out on the 1,311 pool under `arm`. One row per cell; `moves` is
    True when the printed value and the new value differ."""
    rows = []

    def add(table, row, metric, old, new, note=""):
        if isinstance(old, float) and isinstance(new, float):
            moves = abs(old - new) >= 5e-4
        else:
            moves = (str(old) != str(new))
        rows.append({"table": table, "row": row, "metric": metric,
                     "published": old, "new_1311": new,
                     "moves": bool(moves), "note": note})

    P = primary[arm]["rows"]
    for tag, cells in PUBLISHED["tab:retrieval-primary"]["cells"].items():
        got = P.get(tag)
        for metric, old in cells.items():
            if got is None:
                add("tab:retrieval-primary", tag, metric, old, "not recomputed",
                    "radiomics / embedding baseline: needs CT features, not on disk")
                continue
            new = {"P@5": got.get("P@5"), "P@10": got.get("P@10"),
                   "mAP": got.get("mAP"), "nDCG": got.get("nDCG"),
                   "Mism.@10": got.get("mismatch@10"),
                   "Spur.@10": got.get("spurious@10"),
                   "dmAP": (got.get("dmAP_vs_ii") or {}).get("delta")
                   if got.get("dmAP_vs_ii") else None}.get(metric)
            add("tab:retrieval-primary", tag, metric, old, new)
    S = stress[arm]["rows"]
    for tag, cells in PUBLISHED["tab:retrieval"]["cells"].items():
        got = S.get(tag)
        for metric, old in cells.items():
            if got is None:
                add("tab:retrieval", tag, metric, old, "not recomputed",
                    "no reference record exists for a FLARE23 pool member")
                continue
            new = {"P@5": got.get("P@5"), "P@10": got.get("P@10"),
                   "mAP": got.get("mAP"), "nDCG": got.get("nDCG"),
                   "Mism.@10": got.get("mismatch@10"),
                   "Spur.@10": got.get("spurious@10"),
                   "dmAP": (got.get("dmAP_vs_ii") or {}).get("delta")
                   if got.get("dmAP_vs_ii") else None}.get(metric)
            add("tab:retrieval", tag, metric, old, new)
    ST = strata[arm]
    smap = {"(a) Proposed": ("(a) within-dataset control", "pooled", "proposed"),
            "(a) (iii)": ("(a) within-dataset control", "pooled", "coverage_blind"),
            "(a) (iii')": ("(a) within-dataset control", "pooled", "imputed_absent"),
            "(b) Proposed": ("(b) single-organ -> FLARE23", "rows", "proposed"),
            "(b) (iii)": ("(b) single-organ -> FLARE23", "rows", "coverage_blind"),
            "(b) (iii')": ("(b) single-organ -> FLARE23", "rows", "imputed_absent"),
            "(c) Proposed": ("(c) FLARE23 -> single-organ", "rows", "proposed"),
            "(c) (iii)": ("(c) FLARE23 -> single-organ", "rows", "coverage_blind"),
            "(c) (iii')": ("(c) FLARE23 -> single-organ", "rows", "imputed_absent")}
    for tag, cells in PUBLISHED["tab:retrieval-strata"]["cells"].items():
        st, key, mode = smap[tag]
        got = ST[st][key][mode]
        for metric, old in cells.items():
            new = {"n_q": got.get("n_queries"), "P@10": got.get("P@10"),
                   "mAP": got.get("mAP"),
                   "Mism.@10": ("n/a" if got.get("mismatch@10") is None
                                else got.get("mismatch@10"))}.get(metric)
            add("tab:retrieval-strata", tag, metric, old, new)
    C = {r["query"]: r for r in cross[arm]["queries"]}
    for q, cells in PUBLISHED["tab:cross-examples"]["cells"].items():
        got = C.get(q, {})
        for metric, old in cells.items():
            key = "proposed" if "proposed" in metric else "coverage_blind"
            new = (got.get(key) or {}).get("first_relevant_flare23_rank",
                                           "not in pool")
            add("tab:cross-examples", q, metric, old, new)
    return rows


# ===========================================================================
# THE LADDER: manuscript -> today, one step at a time
# ===========================================================================
def meta_ladder(ver, primary, stress):
    """Four numbers a co-author can read in order, each one step from the
    one above it. Step 1 is the manuscript's own pool; it comes from the
    --verify pass, which reproduced every published cell exactly, so the
    first rung is a measurement and not a transcription."""
    r1425 = (ver or {}).get("rows_1425") or {}
    k_ii_p = "(ii) Predicted phenotypes + gamma (proposed)"
    out = {"configuration": "(ii) proposed, gamma_min=1",
           "steps": []}

    def step(name, pool_n, pr, st, what):
        out["steps"].append({
            "step": name, "pool": pool_n, "what_changed": what,
            "primary_mAP": pr.get("mAP") if pr else None,
            "primary_P@10": pr.get("P@10") if pr else None,
            "stress_mAP": st.get("mAP") if st else None,
            "stress_P@10": st.get("P@10") if st else None,
            "stress_n_q": st.get("n_queries") if st else None})
    if r1425:
        step("1. as published", 1425,
             r1425["primary"].get(k_ii_p), r1425["stress"].get(k_ii_p),
             "the manuscript's own pool, corpora and scorer, recomputed")
    for nm, arm, what in (
            ("2. de-duplicated pool", "published",
             "114 FLARE23 twins of the query set removed; records, scorer and "
             "sub-site words unchanged"),
            ("3. + void fix", "void_fix",
             "a void sub-site axis dropped from the organ mean"),
            ("4. + corrected sub-sites (shipped)", "corrected",
             "precedence-resolved sub-site on both sides")):
        if arm not in primary:
            continue
        step(nm, len(primary[arm]["_pq"]["(ii)"]) and None or None, None, None, what)
        out["steps"][-1].update({
            "pool": primary[arm]["n_queries"] and stress[arm]["n_pool"],
            "primary_mAP": primary[arm]["rows"]["(ii)"]["mAP"],
            "primary_P@10": primary[arm]["rows"]["(ii)"]["P@10"],
            "stress_mAP": stress[arm]["rows"]["(ii)"]["mAP"],
            "stress_P@10": stress[arm]["rows"]["(ii)"]["P@10"],
            "stress_n_q": stress[arm]["rows"]["(ii)"]["n_queries"]})
    return out


# ===========================================================================
# WRITERS
# ===========================================================================
def _n(v):
    return "{:,}".format(int(v))


def _plain(s):
    """A row label for Markdown: the LaTeX math delimiters removed."""
    return str(s).replace("$=$", "=").replace("$'$", "'")


def _f(v, nd=3):
    if v is None:
        return "---"
    if isinstance(v, float):
        return ("%." + str(nd) + "f") % v
    return str(v)


def _d(v):
    if not isinstance(v, dict):
        return "---"
    return "%+.3f [%+.3f, %+.3f] p=%s" % (v["delta"], v["ci95"][0], v["ci95"][1],
                                          ("%.1e" % v.get("p_holm", v["p_raw"])))


def write_md(path, meta, primary, stress, strata, cross, eff, circ, diff,
             sides, ladder):
    L = []
    w = L.append
    arm = HEADLINE
    w("# IPKG retrieval tables on the 1,311 pool")
    w("")
    w("Generated %s by `build_retrieval_tables.py`. Every number below carries "
      "its artefact, its protocol, its convention, its cohort and its n." % TODAY)
    w("")
    w("Start at **Section 6c** for the one-step-at-a-time ladder from the "
      "manuscript's numbers to today's, at **Section 5** for the cell-level "
      "diff a co-author can apply, and at **Section 7** for what these tables "
      "are and are not evidence of.")
    w("")
    w("## 0. The configuration every table below is computed in")
    w("")
    w("| | |")
    w("|:--|:--|")
    w("| Pool | **%s** = 113 single-organ predicted records (kits 36, lits 21, "
      "msd 56) + %s FLARE23 reference-derived records |"
      % (_n(meta["pool_size"]), _n(meta["n_flare23"])))
    w("| Corpora | `ipkg_delivered/corpora/corpus_single_organ_113.json` + "
      "`corpus_flare23_kg_dedup.json`, both the pipeline team round 3 (2026-10-03); "
      "reference side `corpus_reference_113.json` |")
    w("| Assembly order | kits, lits, msd, then the dedup corpus - load-bearing, "
      "because ties at S = 1.0 break by pool order under the stable sort |")
    w("| Scorer | the pipeline team's delivered `_scorer_core` / `ipkg_retrieval`, imported "
      "and called; organ universe `%s` |" % ", ".join(UNIVERSE))
    w("| Void policy | **V1, the void fix**: a sub-site axis void on either "
      "side is DROPPED from the organ mean, numerator and denominator; an "
      "organ left with no comparable axis contributes nothing to the "
      "record-level mean |")
    w("| Sub-site convention | **P, precedence-resolved, applied to BOTH "
      "sides**: the pipeline team round-2 record, then our 2026-09-29 overlay, then a "
      "withdrawal, then the delivered word |")
    w("| Organ convention | Convention A where an organ figure is quoted "
      "(`ipkg_organ_convention.py`); no organ Dice appears in these tables |")
    w("| Statistics | paired query-level bootstrap 95%% CI (B=%d) + two-sided "
      "sign-flip permutation p (B=%d), Holm within the family shown; "
      "point estimates are deterministic |" % (B_BOOT, B_PERM))
    w("| Metric code | verbatim copy of the pipeline team's "
      "`pipeline_handoff/scripts/retrieval/retrieval_tables_789.py`; the copy "
      "reproduces **%d of %d** published cells exactly (`--verify`) |"
      % (meta["verify"]["n_ok"], meta["verify"]["n_checks"]))
    w("")
    w("Arms used below, all on the same 1,311 pool:")
    w("")
    w("| Arm | Sub-site vocabulary | Void policy |")
    w("|:--|:--|:--|")
    for a in ("published", "void_fix", "subsite_corpus", "subsite_resolved",
              "corpus_voidfix", "corrected"):
        vk, vf = ARMS[a]
        w("| `%s` | %s | %s |" % (a, {"D": "D delivered (round-1 word)",
                                      "C": "C re-issued corpus",
                                      "P": "P precedence-resolved"}[vk],
                                  "V1 void fix" if vf else "V0 delivered"))
    w("")

    # ---- Table 1 ----------------------------------------------------------
    w("## 1. tab:retrieval-primary")
    w("")
    w("**Caption.** Primary controlled retrieval benchmark on the %s-case pool. "
      "n_q = %d queries over the %d reference-evaluable cases (%d candidates "
      "per query); ranking on predicted phenotypes, relevance from "
      "reference-derived phenotypes, so the representation under test plays no "
      "part in defining the gold standard. Configuration (ii) is the proposed "
      "system. **Void policy: V1, a void sub-site axis is dropped from the "
      "organ mean. Sub-site convention: P, precedence-resolved, applied to "
      "both sides.** Mism.@10 is the outcome-based cross-organ mismatch rate; "
      "Spur.@10 is path-based. dmAP is against (ii) with paired query-level "
      "bootstrap 95%% CI and Holm-corrected permutation p over the four-member "
      "family. Rows (v)-(vii') of the published table are radiomics and "
      "image-embedding baselines over reference organ regions; they are NOT "
      "recomputed here - they need CT-derived feature vectors that are not on "
      "this machine - and the published values stand unchanged and so labelled."
      % (_n(meta["pool_size"]), primary[arm]["n_queries"],
         primary[arm]["n_queries"], primary[arm]["n_candidates"]))
    w("")
    w("| | Configuration | n_q | P@5 | P@10 | mAP | nDCG | Spur.@10 | Mism.@10 | dmAP vs (ii) [95% CI] |")
    w("|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
    for tag, label, _s, _m in ROWS_GRAPH:
        r = primary[arm]["rows"][tag]
        w("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |"
          % (tag, _plain(label), r["n_queries"], _f(r["P@5"]), _f(r["P@10"]),
             _f(r["mAP"]), _f(r["nDCG"]), _f(r["spurious@10"]),
             _f(r["mismatch@10"], 2), _d(r.get("dmAP_vs_ii"))))
    w("")
    w("Per dataset, configuration (ii) and the ablations. **These are the MSD "
      "rows the advisor warned about**; n is the number of queries of that "
      "collection that have at least one relevant candidate under the "
      "reference-derived rule.")
    w("")
    w("| Collection | Config | n_q | P@10 | mAP | nDCG |")
    w("|:--|:--|--:|--:|--:|--:|")
    for ds in ("kits", "lits", "msd"):
        for tag, _l, _s, _m in ROWS_GRAPH:
            r = primary[arm]["by_dataset"][ds][tag]
            w("| %s | %s | %d | %s | %s | %s |"
              % (ds, tag, r["n_queries"], _f(r["P@10"]), _f(r["mAP"]), _f(r["nDCG"])))
    w("")

    # ---- Table 2 ----------------------------------------------------------
    w("## 2. tab:retrieval")
    w("")
    w("**Caption.** Large-pool stress test: %d single-organ queries against the "
      "%s-case cross-dataset pool (113 single-organ cases with predicted "
      "records + %s FLARE23 candidates from the de-duplicated FLARE23 graph, "
      "reference-derived) with construction-defined relevance; n_q per row is "
      "the number of queries with at least one relevant candidate under that "
      "definition. **Void policy: V1. Sub-site convention: P, both sides.** "
      "Row (i) of the published table (reference phenotypes) is NOT produced: "
      "a FLARE23 pool member has no separate reference record - it IS the "
      "reference - so a reference-phenotype arm does not exist over this pool. "
      "The published lower block (the %s-case sub-pool with radiomics and "
      "embedding baselines) is not recomputed for the same reason as Table 1."
      % (stress[arm]["n_queries"], _n(meta["pool_size"]),
         _n(meta["n_flare23"]), "689"))
    w("")
    w("| | Configuration | n_q | P@5 | P@10 | mAP | nDCG | Spur.@10 | Mism.@10 | dmAP vs (ii) [95% CI] |")
    w("|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|")
    for tag, label, _s, _m in ROWS_GRAPH:
        r = stress[arm]["rows"].get(tag)
        if r is None:
            w("| %s | %s | --- | --- | --- | --- | --- | --- | --- | not produced on this pool |"
              % (tag, _plain(label)))
            continue
        w("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |"
          % (tag, _plain(label), r["n_queries"], _f(r["P@5"]), _f(r["P@10"]),
             _f(r["mAP"]), _f(r["nDCG"]), _f(r["spurious@10"]),
             _f(r["mismatch@10"], 2), _d(r.get("dmAP_vs_ii"))))
    w("")
    w("| Collection | Config | n_q | P@10 | mAP | nDCG |")
    w("|:--|:--|--:|--:|--:|--:|")
    for ds in ("kits", "lits", "msd"):
        for tag in [t for t, _l, _s, _m in ROWS_GRAPH if t in stress[arm]["rows"]]:
            r = stress[arm]["by_dataset"][ds][tag]
            w("| %s | %s | %d | %s | %s | %s |"
              % (ds, tag, r["n_queries"], _f(r["P@10"]), _f(r["mAP"]), _f(r["nDCG"])))
    w("")

    # ---- Table 3 ----------------------------------------------------------
    w("## 3. tab:retrieval-strata")
    w("")
    w("**Caption.** Retrieval stratified by the observability relation between "
      "query and candidates, on the %s-case pool, construction-defined "
      "relevance. **Void policy: V1. Sub-site convention: P, both sides.** "
      "(a) restricts both query and candidate pool to one collection, so the "
      "configurations must coincide (identity check, no p). (b) single-organ "
      "queries against the %s FLARE23 candidates only. (c) FLARE23 queries "
      "against the 113 single-organ candidates only; Mism.@10 is defined for "
      "single-organ queries and is n/a there. In every stratum the candidate "
      "pool is coverage-homogeneous. delta_obs is proposed-minus-ablation mAP "
      "with paired bootstrap 95%% CI, Holm within the four-member family over "
      "(b) and (c)." % (_n(meta["pool_size"]), _n(meta["n_flare23"])))
    w("")
    w("| Stratum | Method | n_q | P@5 | P@10 | mAP | nDCG | Mism.@10 | delta_obs [95% CI] |")
    w("|:--|:--|--:|--:|--:|--:|--:|--:|:--|")
    A = strata[arm]["(a) within-dataset control"]
    for m, lab in STRATA_MODES:
        r = A["pooled"][m]
        w("| (a) Within-dataset control | %s | %d | %s | %s | %s | %s | %s | %s |"
          % (lab, r["n_queries"], _f(r["P@5"]), _f(r["P@10"]), _f(r["mAP"]),
             _f(r["nDCG"]), _f(r["mismatch@10"], 2),
             "0.000 (exact tie)" if m != "proposed" else "---"))
    for st, nm in (("(b) single-organ -> FLARE23", "(b) Cross-dataset (single-organ to FLARE23)"),
                   ("(c) FLARE23 -> single-organ", "(c) Reverse (FLARE23 to single-organ)")):
        for m, lab in STRATA_MODES:
            r = strata[arm][st]["rows"][m]
            dd = strata[arm][st]["delta_obs"].get(
                {"coverage_blind": "(iii)", "imputed_absent": "(iii')"}.get(m, ""))
            w("| %s | %s | %d | %s | %s | %s | %s | %s | %s |"
              % (nm, lab, r["n_queries"], _f(r["P@5"]), _f(r["P@10"]),
                 _f(r["mAP"]), _f(r["nDCG"]),
                 "n/a" if r["mismatch@10"] is None else _f(r["mismatch@10"], 2),
                 _d(dd) if m != "proposed" else "---"))
    w("")
    w("Identity check, per collection (a): ")
    for ds, v in A["per_dataset"].items():
        if ds == "flare23":
            continue
        w("- `%s` n_q=%d, proposed mAP %s; coverage-blind rankings identical: "
          "**%s**; imputed-absent rankings identical: **%s**"
          % (ds, v["proposed"]["n_queries"], _f(v["proposed"]["mAP"]),
             v["rankings_identical"]["coverage_blind"],
             v["rankings_identical"]["imputed_absent"]))
    w("")

    # ---- Table 4 ----------------------------------------------------------
    w("## 4. tab:cross-examples")
    w("")
    w("**Caption.** Rank of the intended same-organ FLARE23 target within the "
      "%s-case pool for three single-organ queries (lower is better); the rank "
      "is of the first FLARE23 candidate RELEVANT to the query under the "
      "construction-defined relevance of tab:retrieval. **Void policy: V1. "
      "Sub-site convention: P, both sides.** `tie width` is how many "
      "candidates sit at exactly the score of the reported rank - where it is "
      "large, the rank is decided by pool order, not by the score, and the "
      "number should not be read as a measurement." % _n(meta["pool_size"]))
    w("")
    w("| Query (organ, phenotype) | Query case | n ranked | Rank (proposed) | tie width | Rank (blind) | Rank (imputed) | relevant FLARE23 candidates |")
    w("|:--|:--|--:|--:|--:|--:|--:|--:|")
    for r in cross[arm]["queries"]:
        if "error" in r:
            w("| %s | %s | --- | --- | --- | --- | --- | %s |"
              % (r["title"], r["query"], r["error"]))
            continue
        pr, cb, ia = r["proposed"], r["coverage_blind"], r["imputed_absent"]
        w("| %s | %s | %d | %s | %s | %s | %s | %s |"
          % (r["title"], r["query"], pr["n_ranked"],
             pr["first_relevant_flare23_rank"], pr["tie_width_at_reported_rank"],
             cb["first_relevant_flare23_rank"], ia["first_relevant_flare23_rank"],
             pr["n_relevant_flare23_candidates"]))
    w("")
    for r in cross[arm]["queries"]:
        if "error" in r:
            continue
        w("- `%s` query phenotype (%s): %s" % (r["query"], r["organ"],
                                               ", ".join("%s=%s" % (k, v) for k, v in
                                                         r["query_phenotype"].items())))
    w("")

    # ---- item 1: the diff -------------------------------------------------
    moved = [d for d in diff if d["moves"]]
    w("## 5. Every cell that moves against the manuscript (advisor item 1)")
    w("")
    w("Old = the value printed in `paper/src/JBI/JBI-VKG-Final.tex`. New = the "
      "same cell on the 1,311 pool, arm `%s`. %d of %d cells move. A cell "
      "printed as \\TBD in the draft counts as moving because it now has a "
      "value." % (arm, len(moved), len(diff)))
    w("")
    w("| Table | Row | Metric | Published | New (1,311) | Note |")
    w("|:--|:--|:--|--:|--:|:--|")
    for d in moved:
        w("| %s | %s | %s | %s | %s | %s |"
          % (d["table"], d["row"], d["metric"], d["published"], d["new_1311"],
             d["note"]))
    w("")
    unchanged = [d for d in diff if not d["moves"]]
    w("Cells that do NOT move (%d): %s" % (len(unchanged), ", ".join(
        "%s/%s/%s" % (d["table"].replace("tab:", ""), d["row"], d["metric"])
        for d in unchanged) or "none"))
    w("")

    # ---- item 2: the two effects -----------------------------------------
    w("## 6. The void fix and the sub-site correction, as separate effects "
      "(advisor item 2)")
    w("")
    w("Configuration (ii), k=%d, on the %s-case pool. The baseline arm is "
      "`published` (delivered sub-site words, delivered void semantics) and "
      "each row is ONE change against it. A query counts as moved when its "
      "top-10 changes as a SET; the ORDER column additionally counts a "
      "re-ordering of the same ten, which on this pool is usually a tie being "
      "broken differently rather than a different case being shown."
      % (eff["k"], _n(eff["pool_size"])))
    w("")
    w("| Effect | Query group | n_q | top-10 set changed | top-10 order changed | mean S1 delta |")
    w("|:--|:--|--:|--:|--:|--:|")
    names = {"void_fix": "void fix ALONE (V0 -> V1, sub-sites unchanged)",
             "subsite_corpus": "re-issued sub-sites ALONE (D -> C, void unchanged)",
             "subsite_resolved": "precedence-resolved sub-sites ALONE (D -> P, void unchanged)",
             "corrected": "BOTH (D+V0 -> P+V1)"}
    for a in ("void_fix", "subsite_corpus", "subsite_resolved", "corrected"):
        for grp in ("single_organ", "flare23"):
            v = (eff["arms"].get(a) or {}).get(grp)
            if not v:
                continue
            w("| %s | %s | %d | %d | %d | %s |"
              % (names[a], grp, v["n_queries"], v["top10_set_changed"],
                 v["top10_order_changed"], _f(v["mean_S1_delta"], 5)))
    w("")
    w("By collection, single-organ queries whose top-10 set changed:")
    w("")
    for a in ("void_fix", "subsite_corpus", "subsite_resolved", "corrected"):
        v = (eff["arms"].get(a) or {}).get("single_organ")
        if v:
            w("- `%s`: %s" % (a, v["by_dataset_set_changed"] or "none"))
    w("")
    w("### 6a. Which CASES move, and the mechanism")
    w("")
    w("A count of moved queries cannot distinguish a correction that promotes "
      "three cases into a thousand lists from one that reshuffles a thousand "
      "lists. The cases entering the most top-10s, with their pancreatic "
      "sub-site before and after, say which it is.")
    w("")
    for a in ("void_fix", "subsite_corpus", "subsite_resolved", "corrected"):
        for grp in ("single_organ", "flare23"):
            v = (eff["arms"].get(a) or {}).get(grp)
            if not v or not v.get("cases_entering_most_top10s"):
                continue
            w("**`%s`, %s queries** - %d distinct cases enter at least one "
              "top-10, %d leave at least one."
              % (a, grp, v["distinct_cases_entering_a_top10"],
                 v["distinct_cases_leaving_a_top10"]))
            w("")
            w("| Case entering | Dataset | top-10s entered | pancreas sub-site before | after |")
            w("|:--|:--|--:|:--|:--|")
            for e in v["cases_entering_most_top10s"][:6]:
                w("| %s | %s | %d | %s | %s |"
                  % (e["case_id"], e["dataset"], e["n_top10s_entered"],
                     e["pancreas_site_before"], e["pancreas_site_after"]))
            w("")

    # ---- which side moved -------------------------------------------------
    w("## 6b. Which side moved: the ranking, or the gold standard?")
    w("")
    w("`similarity()` compares `anatomic_location` on the **pancreas only** "
      "(`organ_cats` gives every other organ three categories). `relevant()` "
      "reads it on **every shared organ**. Withdrawing the spurious "
      "pancreatic word from a kidney or a liver entry therefore cannot change "
      "one score and does change the relevance labels, because a free "
      "informative agreement drops out of the two-agreement bar. The net mAP "
      "move on the primary benchmark is partly the gold standard moving under "
      "a ranking that did not move at all, and the two are reported apart.")
    w("")
    w("Configuration (ii). Relevance vocabulary D = delivered sub-site words, "
      "P = precedence-resolved.")
    w("")
    w("| Ranking arm | Relevance vocabulary | primary n_q | primary mAP | primary P@10 | primary nDCG |")
    w("|:--|:--|--:|--:|--:|--:|")
    for k, v in sides["primary"].items():
        a, b = [x.strip() for x in k.split("|")]
        w("| %s | %s | %d | %s | %s | %s |"
          % (a.split("=")[1], b.split("=")[1], v["n_queries"], _f(v["mAP"]),
             _f(v["P@10"]), _f(v["nDCG"])))
    w("")
    w("| Ranking arm | Relevance from pool | stress n_q | stress mAP | stress P@10 |")
    w("|:--|:--|--:|--:|--:|")
    for k, v in sides["stress"].items():
        a, b = [x.strip() for x in k.split("|")]
        w("| %s | %s | %d | %s | %s |"
          % (a.split("=")[1], b.split("=")[1], v["n_queries"], _f(v["mAP"]),
             _f(v["P@10"])))
    w("")
    fl = sides["relevance_label_flips"]
    w("Relevance labels flipped by the sub-site correction: **%d of %d** "
      "ordered query-candidate pairs on the primary benchmark."
      % (fl["primary_labels_flipped"], fl["primary_pairs_checked"]))
    w("")
    w("| Collection | pairs | labels flipped | -> relevant | -> irrelevant | top-10 set changed by the ranking |")
    w("|:--|--:|--:|--:|--:|--:|")
    for ds, v in fl["by_dataset"].items():
        moved = ((eff["arms"].get("corrected") or {}).get("single_organ") or {})
        moved = (moved.get("by_dataset_set_changed") or {}).get(ds, 0)
        w("| %s | %d | %d | %d | %d | %d |"
          % (ds, v["pairs"], v["flipped"], v["to_relevant"],
             v["to_irrelevant"], moved))
    w("")
    w("Read the two right-hand columns together: on `kits` and `lits` the "
      "ranking does not move a single query and the gold standard moves "
      "anyway; on `msd` both move.")
    w("")

    # ---- the ladder --------------------------------------------------------
    w("## 6c. The ladder from the manuscript to today, one step at a time")
    w("")
    w("Configuration (ii). Step 1 is the manuscript's own pool, corpora and "
      "scorer, RECOMPUTED - the --verify pass reproduced all %d published "
      "cells exactly - so the first rung is a measurement, not a "
      "transcription." % meta["verify"]["n_checks"])
    w("")
    w("| Step | Pool | What changed | primary mAP | primary P@10 | stress mAP | stress P@10 | stress n_q |")
    w("|:--|--:|:--|--:|--:|--:|--:|--:|")
    for st in ladder["steps"]:
        w("| %s | %s | %s | %s | %s | %s | %s | %s |"
          % (st["step"], _n(st["pool"]), st["what_changed"], _f(st["primary_mAP"]),
             _f(st["primary_P@10"]), _f(st["stress_mAP"]),
             _f(st["stress_P@10"]), st["stress_n_q"]))
    w("")

    # ---- item 3: circularity ---------------------------------------------
    w("## 7. The evaluation is circular (advisor item 3)")
    w("")
    w("`relevant(A,B)` in `_scorer_core` returns true when a shared organ "
      "carries at least two informative agreements over burden_cat, "
      "multiplicity, containment and anatomic_location. `similarity(A,B)` "
      "ranks on exactly those four fields on exactly those organs. The gold "
      "standard is therefore a thresholded version of the score. The benchmark "
      "can show that the ranking mechanism behaves as designed; it cannot show "
      "that the phenotypes are correct, and these tables must not be read as "
      "if it could.")
    w("")
    w("### 7.1 How much of the headline is the definition")
    w("")
    w("| Benchmark | n scored | mean share of candidates the rule calls relevant | mAP of a RANDOM ordering | mAP, proposed | headroom |")
    w("|:--|--:|--:|--:|--:|--:|")
    for tag, v in circ["random_order_floor"].items():
        w("| %s | %d | %s | %s | %s | %s |"
          % (tag, v["n_queries_scored"], _f(v["mean_relevance_prevalence"], 4),
             _f(v["mAP_random_order"], 4), _f(v["mAP_proposed"], 4),
             _f(v["headroom_above_chance"], 4)))
    w("")
    w("A random permutation of the same candidate list, scored under the same "
      "relevance rule, is the floor the ranker clears for free. Where the "
      "floor is already high the headline mAP is mostly the prevalence of "
      "relevance in the pool and not the quality of the ranking.")
    w("")
    w("### 7.2 Held-out-axis relevance")
    w("")
    w("Relevance redefined as agreement on ONE axis on a shared organ, with "
      "the ranker unchanged. This weakens the dependence - the gold standard "
      "now rests on a quarter of the evidence the score uses - but does not "
      "remove it: a definition fully disjoint from the score would have to "
      "withhold the axis from the score too, which changes the system under "
      "test rather than the evaluation of it. The spread across axes is the "
      "readable quantity: a headline that survives only on the axis the pool "
      "is most degenerate in is not a result.")
    w("")
    w("| Relevance axis | primary n_q | primary mAP | primary P@10 | stress n_q | stress mAP | stress P@10 |")
    w("|:--|--:|--:|--:|--:|--:|--:|")
    for axis, v in circ["held_out_axis_relevance"]["rows"].items():
        a, b = v["primary"], v["stress"]
        w("| %s | %d | %s | %s | %d | %s | %s |"
          % (axis, a["n_queries"], _f(a["mAP"]), _f(a["P@10"]),
             b["n_queries"], _f(b["mAP"]), _f(b["P@10"])))
    w("")
    w("### 7.3 The sub-site sensitivity, re-derived on 1,311")
    w("")
    w("The published limitation states that correcting 52 of the 56 MSD "
      "sub-site labels moved mAP by exactly 0.0000. Re-derived here on this "
      "pool in three contrasts, because the correction that actually shipped "
      "is wider than the one the limitation describes. The first row is the "
      "published perturbation exactly - MSD sub-sites corrected, every other "
      "record left at the delivered word. The second adds what the re-issue "
      "also did: withdraw a pancreatic word from 56 kidney and liver entries. "
      "The third repeats the second under the void fix.")
    w("")
    w("| Contrast | primary mAP (from -> to) | d mAP | stress mAP (from -> to) | d mAP |")
    w("|:--|:--|--:|:--|--:|")
    for label, v in circ["subsite_perturbation"].items():
        w("| %s | %s -> %s | %s | %s -> %s | %s |"
          % (label, _f(v["primary_mAP"][0]), _f(v["primary_mAP"][1]),
             _f(v["primary_dmAP"], 4), _f(v["stress_mAP"][0]),
             _f(v["stress_mAP"][1]), _f(v["stress_dmAP"], 4)))
    w("")
    w("The published claim survives in its own terms and must not be widened. "
      "The MSD-only perturbation is the one the limitation describes and it "
      "moves mAP by very nearly nothing. The wider correction does move mAP, "
      "and Section 6b shows why: it moves the GOLD STANDARD, not the ranking. "
      "That is the circularity itself - a label correction that the score "
      "cannot see (`organ_cats` compares a sub-site on the pancreas only) "
      "changes the benchmark because `relevant()` does see it.")
    w("")
    w("### 7.4 What cannot be produced, and why")
    w("")
    for s in circ["cannot_produce"]:
        w("- %s" % s)
    t = circ["external_relevance_unavailable"]
    if t.get("usable_pairs_in_pool") is not None:
        w("")
        w("Scan-identity detail: the complete twin census carries %d pairs; %d "
          "have their FLARE23 side in the de-duplicated pool, and **%d** have "
          "BOTH members in it. De-duplication removed exactly the twins of the "
          "query set, which is correct for the retrieval benchmark and leaves "
          "no externally-grounded relevance signal behind it."
          % (t.get("n_pairs_in_census", 0), t.get("flare23_side_in_pool", 0),
             t.get("usable_pairs_in_pool", 0)))
    w("")

    # ---- arm sweep --------------------------------------------------------
    w("## 8. All arms side by side")
    w("")
    w("Configuration (ii), the proposed system, on the %s-case pool."
      % _n(meta["pool_size"]))
    w("")
    w("| Arm | sub-site | void | primary mAP | primary P@10 | stress mAP | stress P@10 | stress n_q |")
    w("|:--|:--|:--|--:|--:|--:|--:|--:|")
    for a in primary:
        vk, vf = ARMS[a]
        pr = primary[a]["rows"]["(ii)"]
        sr = stress[a]["rows"]["(ii)"]
        w("| `%s` | %s | %s | %s | %s | %s | %s | %d |"
          % (a, vk, "V1" if vf else "V0", _f(pr["mAP"]), _f(pr["P@10"]),
             _f(sr["mAP"]), _f(sr["P@10"]), sr["n_queries"]))
    w("")
    w("## 9. Provenance")
    w("")
    for k, v in meta["provenance"].items():
        w("- **%s** - %s" % (k, v))
    w("")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def write_tex(outdir, meta, primary, stress, strata, cross, arm=HEADLINE):
    pool = meta["pool_size"]
    nfl = meta["n_flare23"]
    # --- primary ---
    t = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Primary controlled retrieval benchmark on the $%s$-case"
         r" pool: $n_q=%d$ queries over the $113$ reference-evaluable cases"
         r" ($112$ candidates per query); ranking on predicted phenotypes,"
         r" relevance from reference-derived phenotypes."
         r" \textbf{Void policy: a sub-site axis void on either side is dropped"
         r" from the organ mean (numerator and denominator); an organ left with"
         r" no comparable axis contributes nothing. Sub-site convention:"
         r" precedence-resolved, applied to both sides.}"
         r" \emph{Mism.@10} is the outcome-based cross-organ mismatch rate."
         r" $\Delta$mAP versus the proposed configuration with paired"
         r" query-level bootstrap 95\%% CI and Holm-corrected permutation $p$"
         r" ($B_{\mathrm{boot}}=%s$, $B_{\mathrm{perm}}=%s$)."
         r" Rows (v)--(vii$'$) of the previous version are radiomics and"
         r" image-embedding baselines over reference organ regions; they are"
         r" not recomputed on this pool and are omitted rather than carried"
         r" forward at values measured on a different pool.}"
         % ("{:,}".format(pool).replace(",", "{,}"), primary[arm]["n_queries"],
            "5{,}000", "20{,}000"),
         r"\label{tab:retrieval-primary}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{@{}l l c c c c c c@{}}", r"\toprule",
         r"& Configuration & $n_q$ & P@10 & mAP & nDCG & Mism.@10 & "
         r"$\Delta$mAP vs.\ (ii) [95\% CI] \\", r"\midrule"]
    for tag, label, _s, _m in ROWS_GRAPH:
        r = primary[arm]["rows"][tag]
        tg = tag.replace("'", "$'$")
        lab = ("\\textbf{%s}" % label) if tag == "(ii)" else label
        t.append("%s & %s & %d & %s & %s & %s & %s & %s \\\\"
                 % (tg, lab, r["n_queries"], _f(r["P@10"]), _f(r["mAP"]),
                    _f(r["nDCG"]), _f(r["mismatch@10"], 2),
                    _tex_delta(r.get("dmAP_vs_ii"))))
    t += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(outdir, "tab_retrieval_primary.tex"), "w",
         encoding="utf-8").write("\n".join(t) + "\n")

    # --- stress ---
    t = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Large-pool stress test: $%d$ single-organ queries against"
         r" the $%s$-case cross-dataset pool ($113$ single-organ cases with"
         r" predicted records $+$ $%s$ FLARE23 candidates from the"
         r" de-duplicated FLARE23 graph, reference-derived) with"
         r" construction-defined relevance; $n_q$ per row is the number of"
         r" queries with at least one relevant candidate under that definition."
         r" \textbf{Void policy: void sub-site axis dropped from the organ"
         r" mean. Sub-site convention: precedence-resolved, both sides.}"
         r" \emph{Spur.@10} is the proportion of top-10 candidates sharing no"
         r" observed organ with the query; \emph{Mism.@10} is the"
         r" outcome-based cross-organ mismatch rate. A reference-phenotype"
         r" upper bound (row (i) of the previous version) does not exist over"
         r" this pool: a FLARE23 member is itself reference-derived.}"
         % (stress[arm]["n_queries"],
            "{:,}".format(pool).replace(",", "{,}"),
            "{:,}".format(nfl).replace(",", "{,}")),
         r"\label{tab:retrieval}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{@{}l l c c c c c c c c@{}}", r"\toprule",
         r"& Configuration & $n_q$ & P@5 & P@10 & mAP & nDCG & Spur.@10 & "
         r"Mism.@10 & $\Delta$mAP vs.\ (ii) [95\% CI] \\", r"\midrule"]
    for tag, label, _s, _m in ROWS_GRAPH:
        r = stress[arm]["rows"].get(tag)
        if r is None:
            continue
        tg = tag.replace("'", "$'$")
        lab = ("\\textbf{%s}" % label) if tag == "(ii)" else label
        t.append("%s & %s & %d & %s & %s & %s & %s & %s & %s & %s \\\\"
                 % (tg, lab, r["n_queries"], _f(r["P@5"]), _f(r["P@10"]),
                    _f(r["mAP"]), _f(r["nDCG"]), _f(r["spurious@10"]),
                    _f(r["mismatch@10"], 2), _tex_delta(r.get("dmAP_vs_ii"))))
    t += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(outdir, "tab_retrieval.tex"), "w",
         encoding="utf-8").write("\n".join(t) + "\n")

    # --- strata ---
    t = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Retrieval stratified by the observability relation between"
         r" query and candidates on the $%s$-case pool, construction-defined"
         r" relevance ($\Delta_{\mathrm{obs}}$ is proposed-minus-ablation mAP"
         r" with paired bootstrap 95\%% CI; Holm over the four contrasts of"
         r" (b) and (c))."
         r" \textbf{Void policy: void sub-site axis dropped from the organ"
         r" mean. Sub-site convention: precedence-resolved, both sides.}"
         r" (a)~restricts both query and candidate pool to one collection, so"
         r" the configurations must coincide (identity check, no $p$)."
         r" (b)~single-organ queries against the $%s$ FLARE23 candidates only."
         r" (c)~FLARE23 queries against the $113$ single-organ candidates only;"
         r" Mism.@10 is defined for single-organ queries and is n/a there."
         r" In every stratum the candidate pool is coverage-homogeneous.}"
         % ("{:,}".format(pool).replace(",", "{,}"),
            "{:,}".format(nfl).replace(",", "{,}")),
         r"\label{tab:retrieval-strata}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{@{}l l c c c c c c@{}}", r"\toprule",
         r"Stratum & Method & $n_q$ & P@10 & mAP & nDCG & Mism.@10 & "
         r"$\Delta_{\mathrm{obs}}$ [95\% CI] \\", r"\midrule"]
    A = strata[arm]["(a) within-dataset control"]
    t.append(r"\multirow{3}{*}{(a) Within-dataset control}")
    for m, lab in STRATA_MODES:
        r = A["pooled"][m]
        t.append(" & %s & %d & %s & %s & %s & %s & %s \\\\"
                 % (lab.replace("'", "$'$"), r["n_queries"], _f(r["P@10"]),
                    _f(r["mAP"]), _f(r["nDCG"]), _f(r["mismatch@10"], 2),
                    "---" if m == "proposed" else "0.000 (exact tie)"))
    for st, nm in (("(b) single-organ -> FLARE23",
                    r"(b) Cross-dataset (single-organ $\to$ FLARE23)"),
                   ("(c) FLARE23 -> single-organ",
                    r"(c) Reverse (FLARE23 $\to$ single-organ)")):
        t.append(r"\midrule")
        t.append(r"\multirow{3}{*}{%s}" % nm)
        for m, lab in STRATA_MODES:
            r = strata[arm][st]["rows"][m]
            dd = strata[arm][st]["delta_obs"].get(
                {"coverage_blind": "(iii)", "imputed_absent": "(iii')"}.get(m, ""))
            t.append(" & %s & %d & %s & %s & %s & %s & %s \\\\"
                     % (lab.replace("'", "$'$"), r["n_queries"], _f(r["P@10"]),
                        _f(r["mAP"]), _f(r["nDCG"]),
                        "n/a" if r["mismatch@10"] is None else _f(r["mismatch@10"], 2),
                        "---" if m == "proposed" else _tex_delta(dd)))
    t += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    open(os.path.join(outdir, "tab_retrieval_strata.tex"), "w",
         encoding="utf-8").write("\n".join(t) + "\n")

    # --- cross ---
    t = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Rank of the intended same-organ FLARE23 target within the"
         r" $%s$-case pool for three single-organ queries (lower is better):"
         r" the first FLARE23 candidate relevant to the query under the"
         r" construction-defined relevance of Table~\ref{tab:retrieval}."
         r" \textbf{Void policy: void sub-site axis dropped from the organ"
         r" mean. Sub-site convention: precedence-resolved, both sides.}"
         r" \emph{Tie} is the number of candidates at exactly the score of the"
         r" reported rank; where it is large the rank is settled by pool order"
         r" rather than by the score.}"
         % "{:,}".format(pool).replace(",", "{,}"),
         r"\label{tab:cross-examples}", r"\resizebox{\columnwidth}{!}{%",
         r"\begin{tabular}{@{}l l c c c c@{}}", r"\toprule",
         r"Query (organ, phenotype) & Query case & Rank (proposed) & Tie & "
         r"Rank (blind) & Rank (imputed) \\", r"\midrule"]
    for r in cross[arm]["queries"]:
        if "error" in r:
            continue
        t.append("%s & %s & %s & %s & %s & %s \\\\"
                 % (r["title"].replace("->", r"$\to$"),
                    r["query"].replace("_", r"\_"),
                    r["proposed"]["first_relevant_flare23_rank"],
                    r["proposed"]["tie_width_at_reported_rank"],
                    r["coverage_blind"]["first_relevant_flare23_rank"],
                    r["imputed_absent"]["first_relevant_flare23_rank"]))
    t += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(os.path.join(outdir, "tab_cross_examples.tex"), "w",
         encoding="utf-8").write("\n".join(t) + "\n")


def _tex_delta(v):
    if not isinstance(v, dict):
        return "---"
    return r"$%+.3f$ [$%+.3f$, $%+.3f$]" % (v["delta"], v["ci95"][0], v["ci95"][1])


def _strip(obj):
    """Drop the per-query payloads before serialising."""
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="reproduce the published tables and stop")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--no-heavy", action="store_true",
                    help="skip the 1,198-query FLARE23 effect sweep")
    ap.add_argument("--flare-within", action="store_true",
                    help="add a FLARE23-within-FLARE23 stratum to (a)")
    ap.add_argument("--arms", default=",".join(ARMS))
    a = ap.parse_args()
    t_start = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    ver = {"ran": False}
    if not a.no_verify:
        print("[verify] reproducing the published tables on the pipeline team's original corpora")
        ok, rep = verify()
        n_ok = sum(1 for c in rep["checks"] if c[3])
        ver = {"ran": True, "ok": ok, "n_checks": rep["n_checks"], "n_ok": n_ok,
               "mismatches": rep["mismatches"], "pool": rep["pool"],
               "rows_1425": rep["rows_1425"]}
        print("[verify] %d / %d published cells reproduced exactly (pool %d)"
              % (n_ok, rep["n_checks"], rep["pool"]))
        for m in rep["mismatches"]:
            print("   MISMATCH", m)
        json.dump(ver, open(os.path.join(OUTDIR, "verification_published.json"),
                            "w", encoding="utf-8"), indent=1)
        if a.verify:
            return 0 if ok else 2
        if not ok:
            print("[verify] REFUSING to build new tables: the metric layer does "
                  "not reproduce the published numbers.")
            return 2
    ver.setdefault("n_ok", 0)
    ver.setdefault("n_checks", 0)

    pool = pool_1311()
    refmap = reference_1311()
    arms = [x for x in a.arms.split(",") if x in ARMS]
    nfl = sum(1 for r in pool.values() if r.get("dataset") == "flare23")
    print("[pool] %d records (%d FLARE23), arms: %s" % (len(pool), nfl, ", ".join(arms)))

    st = _selftest_fast(pool, refmap)
    print("[selftest] memoised similarity transparent on %d sampled pairs + %d "
          "same-case_id cross-corpus pairs x 4 modes x 2 void policies: %s"
          % (st["pairs_sampled"], st["same_case_id_cross_corpus_checks"], st["ok"]))
    if not st["ok"]:
        print("[selftest] REFUSING to continue: %d mismatches" % st["mismatches"])
        return 2

    print("[tables]")
    primary = table_primary(pool, refmap, arms)
    stress = table_stress(pool, arms)
    strata = table_strata(pool, arms, flare_within=a.flare_within)
    cross = table_cross(pool, arms)
    print("[effects]")
    eff = effects(pool, heavy=not a.no_heavy)
    print("[which side moved]")
    sides = ranking_vs_relevance(pool, refmap)
    print("[circularity]")
    circ = circularity(pool, refmap)
    diff = cell_diff(primary, stress, strata, cross)

    meta = {
        "generated": TODAY,
        "pool_size": len(pool),
        "n_flare23": nfl,
        "n_single_organ": len(pool) - nfl,
        "universe": UNIVERSE,
        "headline_arm": HEADLINE,
        "headline_arm_label": ARM_LABEL[HEADLINE],
        "arms": {k: {"subsite_vocabulary": v[0], "void_fix": v[1],
                     "label": ARM_LABEL[k]} for k, v in ARMS.items()},
        "B_boot": B_BOOT, "B_perm": B_PERM,
        "selftest": st,
        "verify": ver,
        "provenance": {
            "pool corpora": "ipkg_delivered/corpora/corpus_single_organ_113.json + "
                            "corpus_flare23_kg_dedup.json (the pipeline team round 3, 2026-10-03)",
            "reference corpus": "ipkg_delivered/corpora/corpus_reference_113.json "
                                "(the pipeline team round 3)",
            "scorer": "ipkg_delivered/code/_scorer_core.py + ipkg_retrieval.py, "
                      "imported, never re-implemented",
            "void fix": "ipkg_scorer_fixed.similarity_fixed / comparable_cats",
            "sub-site precedence": "ipkg_scorer_fixed.resolve_site (round-2 record, "
                                   "then overlay, then withdrawal, then delivered)",
            "delivered sub-site words": "each record's own anatomic_location_delivered",
            "metric layer": "verbatim copy of pipeline_handoff/scripts/retrieval/"
                            "retrieval_tables_789.py (per_query / agg / paired / "
                            "holm / mismatch10), checked by --verify",
            "published values diffed against": "paper/src/JBI/JBI-VKG-Final.tex, "
                                               "cross-checked against "
                                               "pipeline_handoff/results/retrieval/*.json",
            "hardware": "CPU only; no GPU run, no server touched, no mask opened",
        },
    }

    payload = {"meta": meta,
               "tab:retrieval-primary": _strip(primary),
               "tab:retrieval": _strip(stress),
               "tab:retrieval-strata": _strip(strata),
               "tab:cross-examples": _strip(cross),
               "effects_separately": eff,
               "ranking_vs_relevance": sides,
               "ladder": meta_ladder(ver, primary, stress),
               "circularity": circ,
               "cell_diff_vs_manuscript": diff}
    json.dump(payload, open(os.path.join(OUTDIR, "retrieval_tables_1311.json"),
                            "w", encoding="utf-8"), indent=1)

    with open(os.path.join(OUTDIR, "cell_diff_vs_manuscript.csv"), "w",
              newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["table", "row", "metric",
                                             "published", "new_1311", "moves",
                                             "note"])
        wtr.writeheader()
        for d in diff:
            wtr.writerow(d)

    write_md(os.path.join(OUTDIR, "RETRIEVAL_TABLES_1311.md"), meta, primary,
             stress, strata, cross, eff, circ, diff, sides,
             meta_ladder(ver, primary, stress))
    write_tex(OUTDIR, meta, primary, stress, strata, cross)

    moved = sum(1 for d in diff if d["moves"])
    print("[done] %d of %d manuscript cells move; wrote %s  (%.1fs)"
          % (moved, len(diff), OUTDIR, time.time() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
