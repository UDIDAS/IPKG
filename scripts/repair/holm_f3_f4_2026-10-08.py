#!/usr/bin/env python3
"""Holm with the step-down maximum for families F3 (closed loop, 3 contrasts) and F4 (judge table, 4 contrasts)
(study lead 10-08, item 1). Raw p values are read from the committed results, not recomputed:
  F3  results/repair/closed_loop_tab12_13.json            F3_triple_F1_contrasts            (n = 40 patients)
  F4  results/llm_expert/llm_expert_comparators_qwen3_32b.json  F4_paired_vs_ii_judge_relevance  (n = 19 queries)
Both were computed as paired bootstrap 95 % CI + two-sided sign-flip permutation p (B = 20,000; p = (k + 1)/(B + 1)).
For information, F5 (judge over the clinician packet, llm_judge_clinician_packet_Qwen3_32B.json) is re-adjusted too.

  python holm_f3_f4_2026-10-08.py -> results/repair/holm_f3_f4_2026-10-08.json
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "results", "repair", "holm_f3_f4_2026-10-08.json")


def holm_stepdown(raw):
    items = sorted(raw.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - j) * p))
        out[k] = run
    return out


def family(src, block):
    raw = {k: v["p_raw"] for k, v in block.items()}
    adj = holm_stepdown(raw)
    return {"source": src, "m": len(raw),
            "contrasts": {k: {"delta": v.get("delta"), "ci95": v.get("ci95"), "n": v.get("n_common") or v.get("n"),
                              "p_raw": v["p_raw"], "p_holm_stepdown": round(adj[k], 6),
                              "p_holm_previously_printed": v.get("p_holm")} for k, v in block.items()}}


def main():
    cl = json.load(open(os.path.join(ROOT, "results", "repair", "closed_loop_tab12_13.json")))
    jd = json.load(open(os.path.join(ROOT, "results", "llm_expert", "llm_expert_comparators_qwen3_32b.json")))
    pk = json.load(open(os.path.join(ROOT, "results", "llm_expert", "llm_judge_clinician_packet_Qwen3_32B.json")))
    out = {"note": __doc__.strip(),
           "F3": family("closed_loop_tab12_13.json:F3_triple_F1_contrasts", cl["F3_triple_F1_contrasts"]),
           "F4": family("llm_expert_comparators_qwen3_32b.json:F4_paired_vs_ii_judge_relevance.contrasts",
                        jd["F4_paired_vs_ii_judge_relevance"]["contrasts"]),
           "F5_for_information": family("llm_judge_clinician_packet_Qwen3_32B.json:f5_contrasts_vs_ii", pk["f5_contrasts_vs_ii"])}
    json.dump(out, open(OUT, "w"), indent=1)
    for f in ("F3", "F4", "F5_for_information"):
        print(f, "m =", out[f]["m"])
        for k, c in out[f]["contrasts"].items():
            print(f"   {k:42s} raw {c['p_raw']:.1e}  step-down {c['p_holm_stepdown']:.1e}  (printed {c['p_holm_previously_printed']})")
    print("->", OUT)


if __name__ == "__main__":
    main()
