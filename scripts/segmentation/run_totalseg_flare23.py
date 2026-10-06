"""TotalSegmentator v2 --fast (3 mm), same call as selfprompt/experiments/prep/run_totalseg.py, on the FLARE23
tumor TEST list (global split) + FLARE23_0405 -> /path/to/cluster/work/vkg_data/totalseg_fast/flare23/.

  CUDA_VISIBLE_DEVICES=0 TOTALSEG_WEIGHTS_PATH=<ts_weights> python run_totalseg_flare23.py   (TotalSegmentator v2 env)
"""
import json, os, time
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
OUT = "/path/to/cluster/work/vkg_data/totalseg_fast/flare23"
IMG = "/path/to/cluster/projects/VKG_data/flare23_pool/images"
ROI = ["liver", "kidney_left", "kidney_right", "pancreas", "spleen", "stomach", "colon"]


def main():
    m = json.load(open("~/Desktop/VKG/results/audit/split_manifests_2026-10-01.json"))["tumor_pools"]["global"]["flare"]["test"]
    for c in ["FLARE23_0405"] + m:
        out = f"{OUT}/{c}.nii.gz"
        if os.path.isfile(out):
            continue
        t0 = time.time()
        seg = totalsegmentator(nib.load(f"{IMG}/{c}_0000.nii.gz"), None, fast=True, ml=True, roi_subset=ROI, quiet=True, verbose=False)
        nib.save(seg, out + ".tmp.nii.gz"); os.rename(out + ".tmp.nii.gz", out)
        print(c, f"{time.time()-t0:.0f}s", flush=True)
    print("TS_DONE")


if __name__ == "__main__":
    main()
