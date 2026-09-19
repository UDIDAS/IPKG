#!/usr/bin/env python3
"""Q10 JOINT timing pass (queue item 6 / gating #8): segmentation + pipeline on the SAME studies.

The stage-wise proxy (results/queries/q10_stage_timing_proxy.json) timed each stage on different
runs/hardware; this script runs the full workflow once, per case, on the 36 KiTS test volumes:

  load CT+GT -> preprocess (256^2 RGB windowing) -> semi-oracle segmentation (GT-box prompts,
  kidney pass + tumor pass, batch 8) -> phenotype extraction (the DELIVERED
  app_handoff/extract_phenotypes.py, kits preset) -> graph build (kg_build_graph.py, once for
  the cohort) -> the 10 stock SPARQL queries (queries/run_queries.py, 5 reps, rdflib).

End-to-end per case = load + preprocess + segmentation + extraction + graph_build/n + median
SPARQL query; reported with median + IQR. Segmentation dominates the spread (slice count varies).

Hardware/weights caveat (recorded in the JSON): one dedicated RTX A6000; organ pass = base
facebook/sam3, tumor pass = base + the FLARE23-only tumor expert overlay
(~/hmmkg_ckpts/sam3_tumor_flare_only.pth). The per-dataset AUSAM checkpoints of the shipped
corpora live on the NCSA Delta store (unreachable from this host); latency is bound by the
SAM3 architecture + code path, not the fine-tuned weights, so the timing transfers — the Dice
recorded per case is a sanity check, NOT a paper number. Burden terciles are passed as fixed
config (voxel terciles of the frozen corpus_predicted_kits.json), as the delivered extractor
expects.

  HF_TOKEN=... python q10_joint_timing_kits.py [--cases DIR] [--limit N] [--batch 8]
-> results/queries/q10_joint_timing_kits.json
"""
import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
from skimage.transform import resize

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "segmentation"))
sys.path.insert(0, os.path.join(ROOT, "app_handoff"))
sys.path.insert(0, os.path.join(ROOT, "queries"))
os.environ.setdefault("VKG_DATA", "/path/to/staging/acm_data")
import sam3_autonomous_local as IE                      # noqa: E402  hu_to_rgb, WIN, _load_sam3_ckpt
from extract_phenotypes import extract                  # noqa: E402  the delivered extractor
from run_queries import QUERIES, PREFIX                 # noqa: E402  the 10 stock queries

CASES_DIR = "/path/to/staging/acm_data/kits_vol"
TUMOR_OVERLAY = os.path.expanduser("~/hmmkg_ckpts/sam3_tumor_flare_only.pth")
OUT = os.path.join(ROOT, "results", "queries", "q10_joint_timing_kits.json")
MINPX = 50                                              # as eval_ausam_3d / build_predicted_corpus


def bbox_from_mask(m, pad=3):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    H, W = m.shape
    return [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(W - 1, int(xs.max()) + pad), min(H - 1, int(ys.max()) + pad)]


def predict_batch(model, proc, rgbs, boxes, text):
    import torch
    from torch.amp import autocast
    inp = proc(images=rgbs, text=[text] * len(rgbs), input_boxes=[[b] for b in boxes],
               input_boxes_labels=[[1]] * len(rgbs), return_tensors="pt")
    kw = {"pixel_values": inp["pixel_values"].to("cuda")}
    for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels"):
        if inp.get(k) is not None:
            kw[k] = inp[k].to("cuda")
    with torch.no_grad(), autocast("cuda"):
        out = model(**kw)
        idx = out.pred_logits.sigmoid().argmax(dim=1)
        pm = out.pred_masks[torch.arange(len(rgbs)), idx].float().unsqueeze(1)
        pm = torch.nn.functional.interpolate(pm, size=(256, 256), mode="bilinear", align_corners=False)
    return (pm.sigmoid().squeeze(1).cpu().numpy() > 0.5)


def predict_volume(model, proc, shape, seg, labs, text, rgb_cache, batch):
    """Semi-oracle over axis-0 slices where the GT structure has >= MINPX at 256^2; GT box prompt."""
    pred = np.zeros(shape, bool)
    todo = []
    for z in range(shape[0]):
        gm256 = resize(np.isin(seg[z], labs).astype(float), (256, 256), order=0, preserve_range=True) > 0.5
        if gm256.sum() < MINPX:
            continue
        todo.append((z, bbox_from_mask(gm256.astype(np.uint8), pad=3) or [0, 0, 255, 255]))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        prs = predict_batch(model, proc, [rgb_cache[z] for z, _ in chunk], [b for _, b in chunk], text)
        for (z, _), pr in zip(chunk, prs):
            pred[z] = resize(pr.astype(float), shape[1:], order=0, preserve_range=True) > 0.5
    return pred, len(todo)


def dice(a, b):
    s = a.sum() + b.sum()
    return round(float(2 * (a & b).sum() / s), 4) if s else None


def med_iqr(vals):
    q1, q3 = np.percentile(vals, [25, 75])
    return {"median": round(float(np.median(vals)), 2), "IQR": [round(float(q1), 2), round(float(q3), 2)], "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=CASES_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    import torch
    from transformers import Sam3Processor
    proc = Sam3Processor.from_pretrained(IE.SAM3_MODEL_ID, token=IE.HF_TOKEN)
    t0 = time.time()
    om = IE._load_sam3_ckpt("/nonexistent", "cuda")                 # base SAM3 (no overlay exists locally)
    tm = IE._load_sam3_ckpt(TUMOR_OVERLAY, "cuda")                  # base + FLARE tumor expert
    model_load_s = round(time.time() - t0, 1)
    print(f"models loaded in {model_load_s}s", flush=True)

    # fixed burden-tercile config from the frozen predicted-KiTS corpus (voxel counts)
    frozen = json.load(open(os.path.join(ROOT, "corpora", "corpus_predicted_kits.json")))["records"]
    tvols = sorted(r["organs"]["kidney"]["tumor_voxels"] for r in frozen
                   if r["organs"].get("kidney", {}).get("tumor_voxels", 0) > 0)
    terciles = (tvols[len(tvols) // 3], tvols[2 * len(tvols) // 3])

    cases = sorted(glob.glob(f"{a.cases}/case_*"))[: a.limit]
    per_case, records = {}, []
    for i, cdir in enumerate(cases):
        cid = os.path.basename(cdir)
        t = {}
        s0 = time.time()
        nii = nib.load(f"{cdir}/imaging.nii.gz")
        ct = nii.get_fdata()
        seg = np.asarray(nib.load(f"{cdir}/segmentation.nii.gz").dataobj).astype(np.uint8)
        sp = [float(z) for z in nii.header.get_zooms()[:3]]
        t["load_s"] = time.time() - s0

        s0 = time.time()
        rgb = [IE.hu_to_rgb(resize(ct[z], (256, 256), preserve_range=True, anti_aliasing=True), *IE.WIN).astype(np.uint8)
               for z in range(ct.shape[0])]
        t["preprocess_s"] = time.time() - s0

        s0 = time.time()
        omask, n_org = predict_volume(om, proc, ct.shape, seg, [1], "kidney", rgb, a.batch)
        t["seg_organ_s"] = time.time() - s0
        s0 = time.time()
        tmask, n_tum = predict_volume(tm, proc, ct.shape, seg, [2], "tumor", rgb, a.batch)
        t["seg_tumor_s"] = time.time() - s0

        s0 = time.time()
        m = np.zeros(ct.shape, np.uint8)
        m[omask] = 1
        m[tmask] = 2
        rec = extract(m, sp, cid, dataset="kits", terciles_vox=terciles)
        t["extract_s"] = time.time() - s0
        records.append(rec)

        per_case[cid] = {**{k: round(v, 2) for k, v in t.items()},
                         "n_slices_organ": n_org, "n_slices_tumor": n_tum, "n_slices_vol": ct.shape[0],
                         "dice_organ_sanity": dice(omask, seg == 1), "dice_tumor_sanity": dice(tmask, seg == 2)}
        print(f"[{i+1}/{len(cases)}] {cid} " + " ".join(f"{k}={v}" for k, v in per_case[cid].items()
              if k.endswith("_s")) + f" dice o/t {per_case[cid]['dice_organ_sanity']}/{per_case[cid]['dice_tumor_sanity']}",
              flush=True)
        torch.cuda.empty_cache()

    # ---- graph build (once, timed): kg_build_graph.py in an isolated VKG_KG dir ----
    scratch = os.environ.get("Q10_SCRATCH", "/dev/shm/q10_joint")
    kgdir = os.path.join(scratch, "kg")
    os.makedirs(kgdir, exist_ok=True)
    for f in ("schema.ttl", "ontology_mappings.json"):
        shutil.copy(os.path.join(ROOT, "kg_graphs", f), kgdir)
    corpus_fp = os.path.join(scratch, "corpus_q10_joint_kits.json")
    json.dump({"dataset": "kits_q10_joint", "n": len(records), "records": records}, open(corpus_fp, "w"))
    env = {**os.environ, "VKG_KG": kgdir, "VKG_CORPORA": scratch}
    s0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "kg", "kg_build_graph.py"), corpus_fp],
                       env=env, capture_output=True, text=True)
    graph_build_s = time.time() - s0
    if r.returncode != 0:
        sys.exit(f"kg_build_graph failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    ttl = glob.glob(f"{kgdir}/imaging_kg_q10_joint_kits.ttl") or glob.glob(f"{kgdir}/imaging_kg*.ttl")
    ttl = ttl[0]

    # ---- SPARQL stage: the 10 stock queries, 5 reps, median per query (query_suite_eval protocol) ----
    from rdflib import Graph
    g = Graph()
    s0 = time.time()
    g.parse(ttl)
    graph_load_ms = round((time.time() - s0) * 1000)
    per_q = {}
    for name, q in QUERIES.items():
        times = []
        for _ in range(5):
            s0 = time.time()
            list(g.query(PREFIX + q))
            times.append((time.time() - s0) * 1000)
        per_q[name] = round(statistics.median(times), 1)
    vals = sorted(per_q.values())
    q1, q3 = np.percentile(vals, [25, 75])
    sparql_median_ms = round(statistics.median(vals), 1)

    # ---- joint end-to-end per case ----
    stages = ("load_s", "preprocess_s", "seg_organ_s", "seg_tumor_s", "extract_s")
    e2e = {c: round(sum(v[s] for s in stages) + graph_build_s / len(per_case) + sparql_median_ms / 1000, 2)
           for c, v in per_case.items()}
    seg = [v["seg_organ_s"] + v["seg_tumor_s"] for v in per_case.values()]

    out = {
        "note": ("JOINT single-pass Q10 workflow timing on the 36 KiTS test volumes: per case load -> preprocess -> "
                 "semi-oracle SAM3 segmentation (GT-box prompts, kidney + tumor passes, batch 8) -> delivered "
                 "extractor (kits preset, fixed voxel terciles) -> kg_build_graph.py (once) -> 10 stock SPARQL "
                 "queries (5 reps, rdflib). End-to-end per case = stages + graph_build/n + median query. "
                 "Hardware: one dedicated NVIDIA RTX A6000 (48 GB), single worker. Weights: organ = base "
                 "facebook/sam3, tumor = base + FLARE23-only expert overlay - the per-dataset AUSAM checkpoints "
                 "live on the Delta store (unreachable); latency is architecture-bound so the timing transfers, "
                 "but per-case Dice here is a sanity check only, NOT a paper number."),
        "n_cases": len(per_case),
        "model_load_s_once": model_load_s,
        "per_case": per_case,
        "stage_summary_s": {s: med_iqr([v[s] for v in per_case.values()]) for s in stages},
        "segmentation_total_s": med_iqr(seg),
        "graph_build_s_cohort": round(graph_build_s, 2),
        "graph_load_ms": graph_load_ms,
        "n_triples": len(g),
        "sparql_per_query_median_ms": per_q,
        "sparql_across_queries_ms": {"median": sparql_median_ms, "IQR": [round(float(q1), 1), round(float(q3), 1)]},
        "end_to_end_per_case_s": med_iqr(list(e2e.values())),
        "end_to_end_by_case_s": e2e,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nend-to-end per case: {out['end_to_end_per_case_s']}  (seg {out['segmentation_total_s']})")
    print(f"graph build {out['graph_build_s_cohort']}s, {out['n_triples']} triples, "
          f"sparql median {sparql_median_ms} ms -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
