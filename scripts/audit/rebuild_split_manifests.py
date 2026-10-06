#!/usr/bin/env python3
"""Rebuild the training/eval split manifests of the AUSAM (SAM 3 fine-tune) pools — no GPU, no pool .npy needed.

The original pool manifests (<pool>/meta.json) were lost with the compute-node scratch storage. The case lists
and splits are rebuilt here from the documented build recipes plus small per-case label statistics, then the
shipped split function is applied in BOTH modes in which the delivered code calls it:

  filtered   pool filtered to one dataset, then patient_split   (train_organ_generic.py --dataset,
             train_tumor_ausam.py, eval_ausam_3d.py)  -> the split the per-dataset models were TRAINED on
  global     patient_split on the whole multi-dataset pool        (eval_crossdataset_organ.py,
             build_predicted_corpus.py, eval_tumor_3d.py, train_tumor_incremental.py / sam3_tumor_generic)

patient_split (train_organ_generic.py:33-48 == train_tumor_incremental.py:54-70): RandomState(42), per dataset in
meta insertion order: sorted(cases) -> shuffle -> first int(0.2n) test, next int(0.1n) val, rest train.  One RNG
runs across datasets, and numpy's shuffle consumes draws as a function of list LENGTH only, so a dataset's global
split depends on its own case set plus the SIZES of the datasets before it.

Recipes (scripts on the SWOG AUSAM-MMKG branch, src/scripts/):
  organ_pool_lkp   build_organ_pool_lkp.py  (CAP=5000 slices per (source, organ), MINPX=50, sources in order
                   lits, msd, flare_task2, flare23, kits).  The delivered models used this CAPPED pool.
     lits        sorted volume ids, liver = label 1 only, axis 0, slices with >=50 px, budget 5000
     msd         sorted labelsTr, pancreas = label 1, axis 2, >=50 px, budget 5000
     flare_task2 FLARE_Task2 train + Validation-Public labels (data NOT local -> not rebuilt; size 100 from log)
     flare23     flare_organ_pool from extract_flare_organ_slices.py: sorted(CT n label) Metadata.zip cases,
                 run interrupted after 750 cases (log '[750/950]'); a case enters if any of liver[1],
                 kidney[2,13], pancreas[4] has a slice >=50 px; SPP=8 slices per organ per case
     kits        kits_kidney_pool (extract_kits_kidney_slices.py, KITS_LIMIT=100): case_00000..00099
  tumor pools  (train_tumor_incremental.load_all order: tumor_pool [pancreas, lits], flare_tumor_pool, kits_tumor_pool)
     pancreas    MSD label 2, case kept if >=15 tumor voxels; slices with >=15 px   (build_tumor_pool.py)
     lits        label 2, slices with >=50 px, case kept if >=1 slice   (rebuild_lits_pool_patientlevel.py)
     flare       label 14, slices >=15 px over the case list tumor_imaged_cases.json (LOST).  Reconstructed as
                 the tumor-bearing cases of the 576-case FLARE23 corpus (corpus_predicted_flare23.json);
                 accepted only because it reproduces the logged 270 cases / 7269 slices / 5041-615-1613 exactly.
     kits        extract_kits_tumor_slices.py, KITS_LIMIT=180: case_00000..00179 (log: 180 patients; KiTS data
                 not local -> taken from recipe + log, not data-verified)

Stages:
  python rebuild_split_manifests.py extract --lits-seg DIR --msd-labels DIR
         [--flare-label-region FILE --flare-zip-entries FILE]       -> results/audit/inputs/*.json
         (LiTS seg npy + MSD labelsTr: drive 'data/VKG datasets/{LITs/seg,Pancreas/labelsTr}';
          FLARE23 labels: the contiguous label byte range of 'data/VKG datasets/Metadata.zip', see --help)
  python rebuild_split_manifests.py [build]                       -> results/audit/split_manifests_2026-10-01.json
"""
import argparse
import glob
import json
import os
import struct
import sys
import zlib
import gzip
from collections import OrderedDict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
INP = os.path.join(ROOT, "results", "audit", "inputs")
OUT = os.path.join(ROOT, "results", "audit", "split_manifests_2026-10-01.json")
SEG = os.path.join(ROOT, "results", "segmentation")
J = lambda p: json.load(open(p))
CAP, MINPX_ORGAN, SPP = 5000, 50, 8


# ----------------------------------------------------------------------------------------------- split function
def patient_split_cases(case_lists):
    """Exact re-statement of the shipped patient_split on case sets (it sorts per dataset, so only the set and the
    dataset order matter).  case_lists: [(dataset, cases), ...] in meta insertion order."""
    rng = np.random.RandomState(42)
    out = OrderedDict()
    for d, cs in case_lists:
        cs = sorted(set(cs)); rng.shuffle(cs)
        n = len(cs); nte = max(1, int(0.2 * n)); nva = max(1, int(0.1 * n))
        out[d] = {"train": sorted(cs[nte + nva:]), "val": sorted(cs[nte:nte + nva]), "test": sorted(cs[:nte])}
    return out


def role(c, sp):
    for r in ("test", "val", "train"):
        if c in sp[r]:
            return r
    return "absent"


# ----------------------------------------------------------------------------------------------------- extract
def extract(a):
    import nibabel as nib
    os.makedirs(INP, exist_ok=True)
    # LiTS: per volume, slices (axis 0) with >=50 px of label 1 (liver) and of label 2 (tumor)
    lits = {}
    for f in sorted(glob.glob(f"{a.lits_seg}/segmentation-*.npy")):
        vid = int(os.path.basename(f)[13:-4])
        s = np.load(f, mmap_mode="r")
        h = np.stack([np.bincount(np.asarray(s[z]).astype(np.int64).ravel(), minlength=3)[:3] for z in range(s.shape[0])])
        lits[f"volume-{vid}"] = {"vid": vid, "shape": list(s.shape), "liver_sl50": int((h[:, 1] >= 50).sum()),
                                 "tumor_sl50": int((h[:, 2] >= 50).sum())}
    json.dump(lits, open(f"{INP}/lits_label_stats.json", "w"), indent=0)
    # MSD: per case, slices (axis 2) with >=50 px label 1; tumor (label 2) voxels and slices >=15 px
    msd = {}
    for f in sorted(glob.glob(f"{a.msd_labels}/pancreas_*.nii.gz")):
        lab = np.asarray(nib.load(f).dataobj).astype(np.uint8)
        p1 = (lab == 1).sum(axis=(0, 1)); p2 = (lab == 2).sum(axis=(0, 1))
        msd[os.path.basename(f)[:-7]] = {"shape": list(lab.shape), "pancreas_sl50": int((p1 >= 50).sum()),
                                         "tumor_vox": int(p2.sum()), "tumor_sl15": int((p2 >= 15).sum())}
    json.dump(msd, open(f"{INP}/msd_label_stats.json", "w"), indent=0)
    print(f"extract: lits {len(lits)}  msd {len(msd)}")
    if a.flare_label_region:
        E = J(a.flare_zip_entries)
        imgs = {os.path.basename(k)[:-12] for k in E if k.startswith("Metadata/images/") and k.endswith("_0000.nii.gz")}
        labs = {os.path.basename(k)[:-7] for k in E if k.startswith("Metadata/labels/") and k.endswith(".nii.gz")}
        C = sorted(imgs & labs)
        base = min(E[f"Metadata/labels/{c}.nii.gz"][0] for c in labs)  # region starts at the first label entry
        base = a.flare_region_offset if a.flare_region_offset is not None else base
        st = {}
        with open(a.flare_label_region, "rb") as fh:
            for c in C:
                off, cs = E[f"Metadata/labels/{c}.nii.gz"]
                fh.seek(off - base); b = fh.read(cs + 400)
                nl, el = struct.unpack("<HH", b[26:30])
                lab = np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(zlib.decompress(b[30 + nl + el:], -15))).dataobj).astype(np.uint8)
                H = np.stack([np.bincount(lab[:, :, z].ravel(), minlength=16)[:16] for z in range(lab.shape[2])])
                st[c] = {"shape": list(lab.shape), "liver": int((H[:, 1] >= 50).sum()),
                         "kidney": int(((H[:, 2] + H[:, 13]) >= 50).sum()), "pancreas": int((H[:, 4] >= 50).sum()),
                         "tumor_vox": int(H[:, 14].sum()), "tumor_sl15": int((H[:, 14] >= 15).sum())}
        json.dump(C, open(f"{INP}/flare23_ct_label_cases_sorted.json", "w"))
        json.dump(st, open(f"{INP}/flare23_label_stats.json", "w"), indent=0)
        print(f"extract: flare23 {len(C)} CT+label cases")


# ------------------------------------------------------------------------------------------------------- build
def check(name, got, want):
    return {"check": name, "rebuilt": got, "logged": want, "match": got == want}


def build():
    L = J(f"{INP}/lits_label_stats.json"); M = J(f"{INP}/msd_label_stats.json")
    FC = J(f"{INP}/flare23_ct_label_cases_sorted.json"); FS = J(f"{INP}/flare23_label_stats.json")
    LOG = J(f"{INP}/logged_counts.json")
    checks, notes = [], []

    # ---------------- organ_pool_lkp (capped, as delivered) ----------------
    def capped(cases, nsl, cap=CAP):
        b, kept, contrib = cap, [], {}
        for c in cases:
            if b <= 0:
                break
            k = min(b, nsl[c])
            if k > 0:
                kept.append(c); contrib[c] = k; b -= k
        return kept, contrib
    lits_ids = sorted(L, key=lambda c: L[c]["vid"])
    o_lits, o_lits_sl = capped(lits_ids, {c: L[c]["liver_sl50"] for c in L})
    o_msd, o_msd_sl = capped(sorted(M), {c: M[c]["pancreas_sl50"] for c in M})
    f750 = FC[:750]
    o_f23 = [c for c in f750 if any(FS[c][o] > 0 for o in ("liver", "kidney", "pancreas"))]
    o_f23_sl = {c: sum(min(SPP, FS[c][o]) for o in ("liver", "kidney", "pancreas")) for c in o_f23}
    o_kits = [f"case_{i:05d}" for i in range(100)]
    N_FT2 = LOG["organ"]["flare_task2"]["patients"]
    ft2_placeholder = [f"__flare_task2_{i:03d}" for i in range(N_FT2)]   # sizes only (RNG state); not real ids

    organ_sets = OrderedDict([("lits", o_lits), ("msd", o_msd), ("flare_task2", None), ("flare23", o_f23), ("kits", o_kits)])
    organ_filtered = {d: patient_split_cases([(d, cs)])[d] for d, cs in organ_sets.items() if cs is not None}
    organ_global = patient_split_cases([(d, cs if cs is not None else ft2_placeholder) for d, cs in organ_sets.items()])
    organ_global.pop("flare_task2")
    slc = {"lits": o_lits_sl, "msd": o_msd_sl, "flare23": o_f23_sl}

    lg = LOG["organ"]
    checks += [check("organ lits slices/cases", [sum(o_lits_sl.values()), len(o_lits)], lg["lits"]["slices_cases"]),
               check("organ msd slices/cases", [sum(o_msd_sl.values()), len(o_msd)], lg["msd"]["slices_cases"]),
               check("organ flare23 cases", len(o_f23), lg["flare23"]["patients"]),
               check("organ flare23 slices liver/kidney/pancreas",
                     [sum(min(SPP, FS[c][o]) for c in o_f23) for o in ("liver", "kidney", "pancreas")],
                     lg["flare23"]["slices_per_organ"])]
    for d in ("lits", "msd", "flare23", "kits"):
        sp = organ_filtered[d]
        checks.append(check(f"organ {d} filtered patients tr/va/te", [len(sp[r]) for r in ("train", "val", "test")],
                            lg[d]["patients_tr_va_te"]))
        if d in slc:
            checks.append(check(f"organ {d} filtered slices tr/va/te",
                                [sum(slc[d][c] for c in sp[r]) for r in ("train", "val", "test")], lg[d]["slices_tr_va_te"]))
        ref = f"{SEG}/ausam_3d_{d}.json"
        if os.path.exists(ref):
            evald = sorted(x["case"] for x in J(ref)["cases"])
            if d == "flare23":   # only the locally available test volumes were scored
                checks.append(check("organ flare23 ausam_3d test cases subset of rebuilt test", set(evald) <= set(sp["test"]), True))
            else:
                checks.append(check(f"organ {d} test list == results/segmentation/ausam_3d_{d}.json", sp["test"] == evald, True))
    # uncapped (non-delivered) variant: the 2026-08-07 experiment, reverted; reproduces the logged slice counts
    u_lits, u_lits_sl = capped(lits_ids, {c: L[c]["liver_sl50"] for c in L}, cap=10 ** 9)
    u_msd, u_msd_sl = capped(sorted(M), {c: M[c]["pancreas_sl50"] for c in M}, cap=10 ** 9)
    checks += [check("UNCAPPED (not delivered) lits slices/cases", [sum(u_lits_sl.values()), len(u_lits)], lg["lits"]["uncapped_slices_cases"]),
               check("UNCAPPED (not delivered) msd slices", sum(u_msd_sl.values()), lg["msd"]["uncapped_slices"])]

    cross = {}
    for d in ("lits", "msd", "flare23", "kits"):
        f, g = organ_filtered[d], organ_global[d]
        cross[d] = {"n_test": len(f["test"]), "test_in_common": len(set(f["test"]) & set(g["test"])),
                    "global_test_in_model_TRAIN": len(set(g["test"]) & set(f["train"])),
                    "global_test_in_model_VAL": len(set(g["test"]) & set(f["val"]))}
    ug = patient_split_cases([("lits", u_lits), ("msd", u_msd), ("flare_task2", ft2_placeholder), ("flare23", o_f23), ("kits", o_kits)])
    uf = patient_split_cases([("kits", o_kits)])["kits"]
    cross["kits_if_pool_uncapped"] = {"test_in_common": len(set(uf["test"]) & set(ug["kits"]["test"])), "n_test": len(uf["test"])}

    # ---------------- tumor pools ----------------
    t_pan = [c for c in sorted(M) if M[c]["tumor_vox"] >= 15 and M[c]["tumor_sl15"] > 0]
    t_lits = [c for c in lits_ids if L[c]["tumor_sl50"] > 0]
    F576 = [r["case_id"].replace("flare23_", "") for r in J(f"{ROOT}/corpora/corpus_predicted_flare23.json")["records"]]
    s576 = set(F576)
    t_flare = [c for c in FC if c in s576 and FS[c]["tumor_vox"] >= 15 and FS[c]["tumor_sl15"] > 0]
    t_kits = [f"case_{i:05d}" for i in range(180)]
    tsl = {"pancreas": {c: M[c]["tumor_sl15"] for c in t_pan}, "lits": {c: L[c]["tumor_sl50"] for c in t_lits},
           "flare": {c: FS[c]["tumor_sl15"] for c in t_flare}}
    tumor_sets = OrderedDict([("pancreas", t_pan), ("lits", t_lits), ("flare", t_flare), ("kits", t_kits)])
    tumor_filtered = {d: patient_split_cases([(d, cs)])[d] for d, cs in tumor_sets.items()}
    tumor_global = patient_split_cases(list(tumor_sets.items()))
    lt = LOG["tumor"]
    for d in ("pancreas", "lits", "flare"):
        checks.append(check(f"tumor {d} slices/cases", [sum(tsl[d].values()), len(tumor_sets[d])], lt[d]["slices_cases"]))
        checks.append(check(f"tumor {d} filtered slices tr/va/te",
                            [sum(tsl[d][c] for c in tumor_filtered[d][r]) for r in ("train", "val", "test")], lt[d]["slices_tr_va_te"]))
    for d in tumor_sets:
        checks.append(check(f"tumor {d} filtered patients tr/va/te", [len(tumor_filtered[d][r]) for r in ("train", "val", "test")],
                            lt[d]["patients_tr_va_te"]))
    for ds, td in (("msd", "pancreas"), ("lits", "lits"), ("kits", "kits")):
        corp = sorted(r["case_id"] for r in J(f"{ROOT}/corpora/corpus_predicted_{ds}.json")["records"])
        checks.append(check(f"tumor GLOBAL test[{td}] == corpora/corpus_predicted_{ds}.json case list", tumor_global[td]["test"] == corp, True))
    # sam3_tumor_flare_only (separate training script): all flare tumor cases, 90/10 train/val, seed 42, no test
    rng = np.random.RandomState(42); cs = sorted(t_flare); rng.shuffle(cs)
    fo_val = sorted(cs[:max(1, int(0.10 * len(cs)))])
    flare_only = {"train": sorted(set(t_flare) - set(fo_val)), "val": fo_val, "test": []}
    fo_sl = [sum(tsl["flare"][c] for c in flare_only[r]) for r in ("train", "val")]
    checks.append(check("sam3_tumor_flare_only slices train/val", fo_sl, LOG["flare_only"]["slices_train_val"]))

    tumor_3d = {}
    for ds, td in (("lits", "lits"), ("kits", "kits"), ("msd", "pancreas"), ("flare23", "flare")):
        p = f"{SEG}/tumor_3d_{ds}.json"
        if os.path.exists(p):
            cs3 = [x["case"] for x in J(p)["cases"]]
            tumor_3d[ds] = {"n": len(cs3), **{f"in_model_{r}": len(set(cs3) & set(tumor_filtered[td][r])) for r in ("train", "val", "test")}}

    # ---------------- (b) exposure of the four displayed cases ----------------
    twins = {"volume-76": ("FLARE23_1027", "results/audit/lits_extended_audit.json"),
             "case_00067": ("FLARE23_0286", "results/audit/axis_normalized_twin_census.json"),
             "FLARE23_0405": ("case_00078", "results/audit/flare0405_kits00078_twin_check.json"),
             "pancreas_125": (None, "results/audit/axis_normalized_twin_census.json; results/audit/lits_extended_audit.json")}
    for q, (t, src) in twins.items():
        if t:
            txt = "".join(open(os.path.join(ROOT, s.strip())).read() for s in src.split(";"))
            assert q in txt and t in txt, (q, t, src)
    BASE = {"model": "facebook/sam3 (base, no fine-tuning)", "role": "not_finetuned",
            "note": "no AUSAM training; pre-training data composition not audited here"}

    def fl_roles(c):
        return {"sam3_organ_generic_ausam_flare23": role(c, organ_filtered["flare23"]),
                "sam3_tumor_ausam_flare": role(c, tumor_filtered["flare"]),
                "sam3_tumor_generic (global split)": role(c, tumor_global["flare"]),
                "sam3_tumor_flare_only": role(c, flare_only)}
    exposure = {
        "volume-76": {"organ_model": {"model": "sam3_organ_generic_ausam_lits", "role": role("volume-76", organ_filtered["lits"])},
                      "tumor_model": {"model": "sam3_tumor_ausam_lits (deleted 2026-08-22)", "role": role("volume-76", tumor_filtered["lits"]),
                                      "role_in_global_split_used_by_corpus_and_tumor_3d": role("volume-76", tumor_global["lits"])},
                      "sam3_tumor_flare_only": {"role": "absent (not FLARE)", "via_twin": {"FLARE23_1027": role("FLARE23_1027", flare_only)}},
                      "base_sam3": BASE, "twin": {"FLARE23_1027": fl_roles("FLARE23_1027"), "source": twins["volume-76"][1]}},
        "case_00067": {"organ_model": {"model": "sam3_organ_generic_ausam_kits", "role": role("case_00067", organ_filtered["kits"])},
                       "tumor_model": {"model": "sam3_tumor_ausam_kits", "role": role("case_00067", tumor_filtered["kits"]),
                                       "role_in_global_split_used_by_corpus_and_tumor_3d": role("case_00067", tumor_global["kits"])},
                       "sam3_tumor_flare_only": {"role": "absent (not FLARE)", "via_twin": {"FLARE23_0286": role("FLARE23_0286", flare_only)}},
                       "base_sam3": BASE, "twin": {"FLARE23_0286": fl_roles("FLARE23_0286"), "source": twins["case_00067"][1]}},
        "pancreas_125": {"organ_model": {"model": "sam3_organ_generic_ausam_msd", "role": role("pancreas_125", organ_filtered["msd"])},
                         "tumor_model": {"model": "sam3_tumor_ausam_pancreas (deleted 2026-08-22)", "role": role("pancreas_125", tumor_filtered["pancreas"]),
                                         "role_in_global_split_used_by_corpus_and_tumor_3d": role("pancreas_125", tumor_global["pancreas"])},
                         "sam3_tumor_flare_only": {"role": "absent (not FLARE)", "via_twin": "no FLARE23 twin listed in the census files"},
                         "base_sam3": BASE, "twin": None},
        "FLARE23_0405": {"organ_model": {"model": "sam3_organ_generic_ausam_flare23", "role": role("FLARE23_0405", organ_filtered["flare23"])},
                         "tumor_model": {"model": "sam3_tumor_ausam_flare", "role": role("FLARE23_0405", tumor_filtered["flare"]),
                                         "role_in_global_split": role("FLARE23_0405", tumor_global["flare"])},
                         "sam3_tumor_flare_only": {"role": role("FLARE23_0405", flare_only)},
                         "base_sam3": BASE,
                         "twin": {"case_00078": {"sam3_organ_generic_ausam_kits": role("case_00078", organ_filtered["kits"]),
                                                 "sam3_tumor_ausam_kits": role("case_00078", tumor_filtered["kits"]),
                                                 "sam3_tumor_generic (global split)": role("case_00078", tumor_global["kits"])},
                                  "source": twins["FLARE23_0405"][1]}},
    }
    weights_used = {
        "frozen_records_and_manuscript_dice": "per-dataset organ + tumor AUSAM pair (build_predicted_corpus.py:35-42; FLARE23: build_predicted_corpus_flare23.py:38-39)",
        "displayed_masks_volume-76_case_00067_pancreas_125": "base facebook/sam3 organ pass + sam3_tumor_flare_only tumor pass (regen_predicted_masks_113.py header; scripts/segmentation/rerun_final_cases.py arm so_displayed)",
        "displayed_mask_FLARE23_0405": "masks_predicted_flare23/FLARE23_0405.nii.gz = sam3_organ_generic_ausam_flare23 + sam3_tumor_ausam_flare"}

    # ---------------- (c) per-record roles ----------------
    corp_roles = {}
    for ds, od, td in (("kits", "kits", "kits"), ("lits", "lits", "lits"), ("msd", "msd", "pancreas")):
        rows = []
        for r in J(f"{ROOT}/corpora/corpus_predicted_{ds}.json")["records"]:
            c = r["case_id"]
            rows.append({"case_id": c, "organ_model_role": role(c, organ_filtered[od]) if c in organ_sets[od] else "absent",
                         "tumor_model_role": role(c, tumor_filtered[td]), "global_tumor_split_role": role(c, tumor_global[td])})
        corp_roles[ds] = {"organ_model": f"sam3_organ_generic_ausam_{od}", "tumor_model": f"sam3_tumor_ausam_{td}",
                          "summary": {k: {x: sum(1 for w in rows if w[k] == x) for x in ("train", "val", "test", "absent")}
                                      for k in ("organ_model_role", "tumor_model_role")}, "records": rows}
    rows = []
    for c in F576:
        rows.append({"case_id": f"flare23_{c}", "organ_model_role": role(c, organ_filtered["flare23"]) if c in o_f23 else "absent",
                     "tumor_model_role": role(c, tumor_filtered["flare"]) if c in t_flare else "absent"})
    corp_roles["flare23_576"] = {"organ_model": "sam3_organ_generic_ausam_flare23", "tumor_model": "sam3_tumor_ausam_flare",
                                 "summary": {k: {x: sum(1 for w in rows if w[k] == x) for x in ("train", "val", "test", "absent")}
                                             for k in ("organ_model_role", "tumor_model_role")},
                                 "note": "the 576 batch is not split-filtered (build_predicted_corpus_flare23.py globs the whole pool)",
                                 "records": rows}

    notes += [
        "flare_task2 organ case list NOT rebuilt (FLARE_Task2 data not local). Its filtered split is unknown here; for the global "
        "organ split only its size (100 patients, from the training log) is needed and is used via placeholders.",
        "flare tumor case list file tumor_imaged_cases.json is LOST; the 270-case set is reconstructed as the tumor-bearing cases of "
        "the 576-case FLARE23 corpus and accepted only because it reproduces the logged cases/slices/split exactly.",
        "kits organ (case_00000..00099) and kits tumor (case_00000..00179) sets come from the extractor recipes + logged patient "
        "counts; KiTS labels were not re-read (not local). The organ set is confirmed by the 20-case ausam_3d_kits test list and the "
        "tumor set by the 36-case corpus list (both reproduced exactly).",
        "kits pools were built with a thread pool, so meta ORDER was nondeterministic; the split sorts cases, so membership is not affected.",
        "the CT/label shape-equality guard of the organ builder is not re-run here (CT not read); an earlier run with the CT present "
        "reproduced the same LiTS/MSD counts.",
        "flare23 organ pool depends on the interrupted 750-case extraction (no resume); recipe encodes that cutoff.",
        "the delivered organ checkpoints were trained on the CAPPED pool; the uncapped variant is reported only as a count check."]
    out = OrderedDict([
        ("generated_by", "scripts/audit/rebuild_split_manifests.py"),
        ("split_function", "patient_split, RandomState(42), per dataset: sorted -> shuffle -> 20% test / 10% val / rest train"),
        ("split_modes", {"filtered": "pool filtered to one dataset first (per-dataset model training, eval_ausam_3d)",
                         "global": "whole multi-dataset pool (eval_crossdataset_organ, build_predicted_corpus, eval_tumor_3d, sam3_tumor_generic)"}),
        ("count_checks", checks),
        ("all_checks_match", all(c["match"] for c in checks)),
        ("not_rebuildable_or_assumed", notes),
        ("organ_pool_lkp", {"dataset_order": list(organ_sets),
                            "case_lists": {d: (cs if cs is not None else "NOT REBUILT (data not local)") for d, cs in organ_sets.items()},
                            "filtered": organ_filtered, "global": organ_global,
                            "global_note": "flare_task2 absent from 'global' (placeholders only)"}),
        ("crossdataset_vs_training_test_sets", cross),
        ("tumor_pools", {"dataset_order": list(tumor_sets), "pool_files": {"tumor_pool": ["pancreas", "lits"], "flare_tumor_pool": ["flare"],
                                                                           "kits_tumor_pool": ["kits"]},
                         "case_lists": tumor_sets, "filtered": tumor_filtered, "global": tumor_global,
                         "sam3_tumor_flare_only_split": flare_only}),
        ("tumor_3d_eval_cases_vs_training_split", tumor_3d),
        ("displayed_cases_exposure", {"weights_used": weights_used, "cases": exposure}),
        ("corpus_record_roles", corp_roles),
    ])
    json.dump(out, open(OUT, "w"), indent=1)
    bad = [c for c in checks if not c["match"]]
    print(f"{len(checks)} count checks, {len(bad)} mismatches -> {OUT}")
    for c in checks:
        print(("  OK   " if c["match"] else "  FAIL ") + f"{c['check']}: rebuilt={c['rebuilt']} logged={c['logged']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="build", choices=["build", "extract"])
    ap.add_argument("--lits-seg"); ap.add_argument("--msd-labels")
    ap.add_argument("--flare-label-region", help="bytes of Metadata.zip from the first label entry to the end of the labels")
    ap.add_argument("--flare-zip-entries", help="JSON {name: [local_header_offset, compressed_size]} from the zip central directory")
    ap.add_argument("--flare-region-offset", type=int, default=None, help="zip offset at which --flare-label-region starts")
    a = ap.parse_args()
    if a.stage == "extract":
        extract(a)
    else:
        sys.exit(build())
