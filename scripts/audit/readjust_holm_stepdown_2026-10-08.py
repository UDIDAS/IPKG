#!/usr/bin/env python3
"""Re-adjust Holm p values with the step-down maximum in committed result files whose generators cannot be re-run here
(closed-loop masks lived on the retired cluster; the baseline feature cache is gone). Raw p values are kept; each
family is the one its generator adjusted together. Gate: only `p_holm` values may change; the script refuses to write
otherwise. The retrieval tables 7/8/9 are NOT handled here: they were re-emitted by the fixed retrieval_tables_789.py.

  python readjust_holm_stepdown_2026-10-08.py -> rewrites the files below + results/audit/holm_readjust_log_2026-10-08.json
"""
import copy
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
R = lambda *p: os.path.join(ROOT, "results", *p)


def rows_family(path, skip=("(iii')",)):
    """family = dmAP_vs_ii of every row under `path` that carries a p_holm (rows in `skip` stay raw)."""
    return lambda d: [r["dmAP_vs_ii"] for k, r in get(d, path).items() if k not in skip and r.get("dmAP_vs_ii")
                      and r["dmAP_vs_ii"].get("p_holm") is not None]


def get(d, path):
    for k in path:
        d = d[k]
    return d


FILES = {
    R("repair", "closed_loop_tab12_13.json"): [lambda d: list(d["F3_triple_F1_contrasts"].values())],
    R("llm_expert", "llm_expert_comparators_qwen3_32b.json"): [lambda d: list(d["F4_paired_vs_ii_judge_relevance"]["contrasts"].values())],
    R("llm_expert", "llm_judge_clinician_packet_Qwen3_32B.json"): [lambda d: [v for v in d["f5_contrasts_vs_ii"].values() if v.get("p_holm") is not None]],
    R("retrieval", "dedup", "baselines_tab7_tab8.json"): [rows_family(("tab7", "rows"), ()), rows_family(("tab8_subpool", "rows"), ())],
    R("retrieval", "dedup", "baselines_corrected_labels_2026-10-05.json"): [rows_family(("tables", "published", "rows")), rows_family(("tables", "corrected", "rows"))],
    R("retrieval", "dedup", "baselines_subpool_1311_2026-10-06.json"): [rows_family(("arms", "published", "rows")), rows_family(("arms", "corrected", "rows"))],
}


def stepdown(fam):
    items = sorted(fam, key=lambda v: v["p_raw"])
    run, m = 0.0, len(items)
    for j, v in enumerate(items):
        run = max(run, min(1.0, (m - j) * v["p_raw"]))
        v["p_holm"] = round(run, 5)


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "p_holm"}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def main():
    log = {"note": __doc__.strip(), "files": {}}
    for f, fams in FILES.items():
        d = json.load(open(f)); before = copy.deepcopy(d); changes = []
        for fam in fams:
            members = fam(d)
            old = [v.get("p_holm") for v in members]
            stepdown(members)
            changes += [{"p_raw": v["p_raw"], "old": o, "new": v["p_holm"]} for v, o in zip(members, old) if o != v["p_holm"]]
        assert strip(before) == strip(d), f"gate: a non-Holm value would change in {f}"
        lines = open(f).read().splitlines()
        ind = len(lines[1]) - len(lines[1].lstrip(" ")) if len(lines) > 1 else None   # keep the file's own indentation
        json.dump(d, open(f, "w"), indent=ind or None)
        log["files"][os.path.relpath(f, ROOT)] = {"families": [len(fam(d)) for fam in fams], "changed": changes}
        print(os.path.relpath(f, ROOT), "families", [len(fam(d)) for fam in fams], "changed", [(c["old"], c["new"]) for c in changes])
    json.dump(log, open(R("audit", "holm_readjust_log_2026-10-08.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
