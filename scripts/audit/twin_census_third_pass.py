#!/usr/bin/env python3
"""Third pass of the twin census (2026-09-24): fold the FOURTEEN same-scan twins found by the
grid census of the 687 no-CT pool records into the census artifact and remove them from the
deduplicated corpus.  Pool 1,323 -> 1,309, corpus 1,210 -> 1,196, twins 102 -> 116.

Provenance: `scripts/audit/unscanned687_grid_prescreen.py` (voxel-grid matching on GT-label
NIfTI headers streamed from Metadata.zip — annotation-independent: every one of KS's 310
CT-voxel twins is grid-identical, and the GT label carries the CT's grid) found 17 exact-grid
collisions between unmasked pool records and query volumes; `confirm_grid_collisions.py`
confirmed 14 by mask overlap / exact tumor-voxel equality (12 KiTS: organ IoU 0.87-0.955 or
tumor IoU 0.9746 on a 542K-voxel tumor; 2 LiTS: tumor GT voxel counts EXACTLY equal, the LiTS
census's T2 criterion) and closed 3 as coincidence/benign.  These 687 records have NO CT in
any release copy (the zip's 950 images exclude all of them — verified in the prescreen), so
no CT-voxel scan, ours or KS's, could ever have seen them: the grid census is the strongest
physically possible test for this stratum, caveat (disclosed in the prescreen artifact) that
a RESAMPLED re-share would evade it and the CT-voxel test equally.

This script performs no new measurement — the evidence is the two committed artifacts above.
It (1) adds a `third_pass_2026-09-24` section to axis_normalized_twin_census.json, (2) updates
`pool_arithmetic`, and (3) removes the 14 from corpora/corpus_flare23_kg_dedup.json with an
appended provenance note.  Idempotent: re-running after removal is a no-op with a warning.

  python twin_census_third_pass.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CENSUS = os.path.join(ROOT, "results", "audit", "axis_normalized_twin_census.json")
CONFIRM = os.path.join(ROOT, "results", "audit", "unscanned687_collision_confirm_2026-09-24.json")
CORPUS = os.path.join(ROOT, "corpora", "corpus_flare23_kg_dedup.json")


def main():
    conf = json.load(open(CONFIRM))
    twins = conf["confirmed_twins"]
    assert len(twins) == 14 and conf["n_flags"] == 0, "confirm artifact changed - re-review"
    ids = [t["flare"] for t in twins]

    cen = json.load(open(CENSUS))
    cen["third_pass_2026-09-24"] = {
        "note": __doc__.strip(),
        "source": "grid census of the 687 unmasked pool records "
                  "(unscanned687_grid_prescreen_2026-09-24.json: 17 exact-grid collisions; "
                  "unscanned687_collision_confirm_2026-09-24.json: 14 confirmed, 3 benign)",
        "prior_blind_spot": "this stratum was label-screened only (first pass); it has no CTs in "
                            "any release copy, so the CT-voxel scans that caught the second-pass "
                            "eleven could never cover it — grid identity + mask overlap / exact "
                            "tumor-voxel equality is the strongest test the released data permits",
        "pairs_confirmed": twins, "n_confirmed": len(twins),
        "residual": "after this pass the 673 remaining no-CT pool records are label-screened AND "
                    "grid-screened (annotation-independent): none shares a voxel grid with any "
                    "query, so none can be an identical-grid same-scan re-share of a query. "
                    "Disclosed caveat: a re-share RESAMPLED to a new grid would evade this test "
                    "and the CT-voxel test equally; none of the 310 known re-shares is resampled."}
    cen["pool_arithmetic"] = {"pool_before": 1347, "pool_after_first_pass": 1334,
                              "pool_after_second_pass": 1323, "pool_after_third_pass": 1309,
                              "flare_corpus_after": 1196, "total_twins_removed": 116}
    json.dump(cen, open(CENSUS, "w"), indent=1)

    cor = json.load(open(CORPUS))
    before = len(cor["records"])
    cor["records"] = [r for r in cor["records"] if r["case_id"] not in ids]
    removed = before - len(cor["records"])
    if removed == 0:
        print(f"corpus already at {before} records - census refreshed, corpus untouched")
        return
    assert removed == 14, f"expected to remove 14, removed {removed}"
    cor["n"] = len(cor["records"])
    cor["note"] += (" | 2026-09-24: 14 further FLARE23 cases removed - same-scan twins of 12 KiTS "
                    "+ 2 LiTS queries found by the annotation-independent GRID census of the 687 "
                    "no-CT pool records (results/audit/unscanned687_grid_prescreen_2026-09-24.json "
                    "+ unscanned687_collision_confirm_2026-09-24.json; organ IoU 0.87-0.955 / "
                    "tumor IoU up to 0.98 / LiTS tumor GT voxel counts exactly equal). These "
                    "records have no CT in any release copy, so voxel-level testing was "
                    "impossible for anyone; the remaining 673 no-CT records are grid-screened "
                    "clean. Pool 1,323 -> 1,309, corpus 1,196, twins 116.")
    json.dump(cor, open(CORPUS, "w"), indent=1)
    print(f"census third_pass written; corpus {before} -> {len(cor['records'])}; "
          f"pool 1,323 -> 1,309; twins 116")


if __name__ == "__main__":
    main()
