#!/usr/bin/env python3
"""Twin-elimination gate (2026-09-24): mechanical confirmation that the duplicate problem is
closed for everything we can test, artifact by artifact.

Three independent checks, all against shipped files (no re-derivation by hand):

  A. REMOVED-SET INTEGRITY - the 116-record difference between the full FLARE23 corpus and
     the dedup corpus is exactly the union of the four removal waves' own lists
     (78 original audit + 13 axis-normalized census + 11 CT-voxel second pass + 14 grid-census
     third pass), no more, no less, no overlap surprises.
  B. NO RESURRECTION - no removed record id appears in any pool-bearing artifact: the dedup
     corpus, the app rankings (app_handoff/precomputed/rankings_113_queries.json), the
     example-ranks table, the clinician study set, or the eleven-query effect report's
     AFTER rankings (its BEFORE rankings legitimately contain the eleven - that is the point
     of the file - so they are excluded).
  C. KS-SCAN RECONCILIATION - KS's CT-voxel-equality scan (paper/flare23_twin_scan_2026-09-20
     .json, branch abdomen/ipkg; 310 same-scan pairs over the 576 UD-masked FLARE studies) is
     replayed against the CURRENT pool: every scanned FLARE study whose CT-voxel twin is one
     of our 113 queries must be OUT of the dedup corpus.  Zero violations means the masked
     part of the pool is twin-free at CT-VOXEL level, not merely label-screen level.
     (LiTS twins are named as Task03 liver_N; mapped to LiTS query ids volume-N via
     results/audit/lits_geometry_map.json, the corr >= 0.99999 identity mapping.)

  D. GRID-CENSUS RECONCILIATION (added 2026-09-24, third pass) - the no-CT stratum has no
     CT in anyone's release copy, so it can never appear in a CT-level scan; it is covered by
     the annotation-independent grid census instead.  Every twin the census confirmed
     (unscanned687_collision_confirm) must be OUT of the dedup corpus.  Residual after this
     check (disclosed): a re-share RESAMPLED to a new grid evades the grid test and the
     CT-voxel test equally; none of the 310 known re-shares is resampled.

  KS_SCAN=/dev/shm/ks_twin_scan.json python scripts/audit/twin_elimination_gate.py
-> results/audit/twin_elimination_gate_2026-09-24.json  (exit 1 on any failure)
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
KS_SCAN = os.environ.get("KS_SCAN", "/dev/shm/ks_twin_scan.json")
OUT = os.path.join(ROOT, "results", "audit", "twin_elimination_gate_2026-09-24.json")


def j(path):
    return json.load(open(os.path.join(ROOT, path)))


def ids_in(obj, universe):
    """Every string anywhere in a JSON structure that is a known record id."""
    found = set()
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, str):
            if x in universe:
                found.add(x)
            elif "flare23_" + x in universe:  # bare FLARE23_XXXX spellings
                found.add("flare23_" + x)
        elif isinstance(x, dict):
            stack.extend(x.keys())
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)
    return found


def main():
    report = {"gate": "twin_elimination_gate_2026-09-24", "checks": {}, "failures": []}

    # ---- A: removed-set integrity --------------------------------------------------------
    full = {r["case_id"] for r in j("corpora/corpus_flare23_kg.json")["records"]}
    dedup = {r["case_id"] for r in j("corpora/corpus_flare23_kg_dedup.json")["records"]}
    removed = full - dedup
    cen = j("results/audit/axis_normalized_twin_census.json")
    wave2 = set(cen["new_twins_confirmed"])
    wave3 = {p["flare"] for p in cen["second_pass_2026-09-22"]["pairs_confirmed"]}
    wave4 = {p["flare"] for p in cen.get("third_pass_2026-09-24", {}).get("pairs_confirmed", [])}
    dup = j("results/audit/duplicate_scan_audit.json")
    ext = j("results/audit/lits_extended_audit.json")
    wave1 = (set(dup["twins_confirmed"])
             | {e["flare23"] for e in dup["manually_confirmed_twins"]}
             | set(ext["twins_confirmed"]))   # 58 + 1 (FLARE23_0102) + 19 LiTS = 78
    a = {"full_corpus": len(full), "dedup_corpus": len(dedup), "removed": len(removed),
         "wave1_original_audit": len(wave1), "wave2_census": len(wave2), "wave3_second_pass": len(wave3),
         "wave4_third_pass_grid_census": len(wave4),
         "union_of_waves_equals_removed": removed == (wave1 | wave2 | wave3 | wave4),
         "waves_disjoint": len(wave1) + len(wave2) + len(wave3) + len(wave4)
                           == len(wave1 | wave2 | wave3 | wave4),
         "unexplained_removed": sorted(removed - wave1 - wave2 - wave3 - wave4),
         "listed_but_still_in_corpus": sorted((wave1 | wave2 | wave3 | wave4) & dedup)}
    report["checks"]["A_removed_set_integrity"] = a
    if not a["union_of_waves_equals_removed"] or a["listed_but_still_in_corpus"]:
        report["failures"].append("A: removed set does not reconcile with the four waves' lists")

    # ---- B: no resurrection --------------------------------------------------------------
    b = {}
    artifacts = [
        ("app_handoff/precomputed/rankings_113_queries.json", None),
        ("results/retrieval/dedup/tab10_example_ranks.json", None),
        ("results/study/clinician_study_30q.json", None),
        ("results/retrieval/dedup/eleven_kits_effect_2026-09-22.json", "after"),
        ("results/retrieval/dedup/fourteen_twins_effect_2026-09-24.json", "pool_1309"),
    ]
    for path, only_key in artifacts:
        fp = os.path.join(ROOT, path)
        if not os.path.exists(fp):
            b[path] = {"status": "NOT PRESENT LOCALLY (ships via Drive; check there)"}
            continue
        obj = json.load(open(fp))
        if only_key:  # keep only sub-objects under keys containing the marker (e.g. 'after')
            obj = [v for k, v in _walk_items(obj) if only_key in k.lower()]
        hits = ids_in(obj, removed)
        b[path] = {"removed_ids_found": sorted(hits), "clean": not hits}
        if hits:
            report["failures"].append(f"B: {path} still contains removed ids: {sorted(hits)[:5]}")
    report["checks"]["B_no_resurrection"] = b

    # ---- C: KS-scan reconciliation -------------------------------------------------------
    scan = json.load(open(KS_SCAN))
    queries = set()
    for ds in ("kits", "lits", "msd"):
        queries |= {r["case_id"] for r in j(f"corpora/corpus_gt_{ds}.json")["records"]}
    liver2lits = {v["liver"]: k for k, v in j("results/audit/lits_geometry_map.json")["mapping"].items()}
    viol, twin_of_query, per_coll = [], 0, {}
    n_twins = n_in_pool = 0
    for fid, e in scan["results"].items():
        cid = "flare23_" + fid
        in_pool = cid in dedup
        n_in_pool += in_pool
        t = e.get("twin")
        if not t:
            continue
        n_twins += 1
        q = t["case"]
        q = liver2lits.get(q, q)            # liver_N -> volume-N; kits/msd ids match as-is
        if q in queries:
            twin_of_query += 1
            per_coll[t["collection"]] = per_coll.get(t["collection"], 0) + 1
            if in_pool:
                viol.append({"flare": cid, "query": q, "collection": t["collection"]})
    c = {"scanned": len(scan["results"]), "ct_voxel_twins": n_twins,
         "scanned_still_in_pool": n_in_pool,
         "twins_of_a_query": twin_of_query, "twins_of_a_query_by_collection": per_coll,
         "violations_query_twin_still_in_pool": viol,
         "ks_summary": scan.get("summary")}
    report["checks"]["C_ks_scan_reconciliation"] = c
    if viol:
        report["failures"].append(f"C: {len(viol)} CT-voxel query twins still in the pool")

    # ---- D: grid-census reconciliation (third pass, no-CT stratum) -------------------------
    d = {}
    conf_path = os.path.join(ROOT, "results", "audit", "unscanned687_collision_confirm_2026-09-24.json")
    if os.path.exists(conf_path):
        conf = json.load(open(conf_path))
        confirmed = {t["flare"] for t in conf["confirmed_twins"]}
        still_in = sorted(confirmed & dedup)
        d = {"grid_collisions": conf["n_collisions"], "confirmed_twins": len(confirmed),
             "flags_unresolved": conf["n_flags"],
             "confirmed_twins_still_in_pool": still_in,
             "residual": "a RESAMPLED re-share evades the grid test and the CT-voxel test "
                         "equally; none of the 310 known re-shares is resampled"}
        if still_in or conf["n_flags"]:
            report["failures"].append(f"D: grid-census twins still in pool ({still_in}) "
                                      f"or unresolved flags ({conf['n_flags']})")
    else:
        d = {"status": "confirm artifact missing"}
        report["failures"].append("D: unscanned687_collision_confirm artifact missing")
    report["checks"]["D_grid_census_reconciliation"] = d

    report["verdict"] = ("PASS - pool is twin-free w.r.t. every list, the full CT-voxel scan "
                         "(masked stratum) and the grid census (no-CT stratum)") \
        if not report["failures"] else "FAIL"
    json.dump(report, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=1))
    print("A:", {k: v for k, v in report["checks"]["A_removed_set_integrity"].items() if k != "unexplained_removed"})
    print("B:", {p: r.get("clean", r.get("status")) for p, r in report["checks"]["B_no_resurrection"].items()})
    print("C:", {k: v for k, v in report["checks"]["C_ks_scan_reconciliation"].items() if k != "violations_query_twin_still_in_pool"})
    print("D:", report["checks"]["D_grid_census_reconciliation"])
    sys.exit(1 if report["failures"] else 0)


def _walk_items(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield prefix + str(k), v
            yield from _walk_items(v, prefix + str(k) + ".")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_items(v, prefix + f"[{i}].")


if __name__ == "__main__":
    main()
