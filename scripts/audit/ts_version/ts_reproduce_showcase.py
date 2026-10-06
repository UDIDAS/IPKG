import sys, nibabel as nib, numpy as np
from totalsegmentator.python_api import totalsegmentator
ROI = ["liver", "kidney_left", "kidney_right", "pancreas", "spleen", "stomach", "colon"]
F = "/path/to/staging/fc_verify/fc"
JOBS = {"pancreas_125": (f"{F}/pancreas_125_image.nii.gz", [7], ROI),
        "FLARE23_0405": (f"{F}/ct_FLARE23_0405_0000.nii.gz", [1, 2, 3, 5, 6, 7, 20], ROI)}
def main():
    for c in sys.argv[1:]:
        img, host, roi = JOBS[c]
        seg = totalsegmentator(nib.load(img), None, fast=True, ml=True, roi_subset=roi, quiet=True, verbose=False)
        ts = np.asarray(seg.dataobj).astype(np.uint8)
        np.save(f"/path/to/staging/ts_repro/{c}_ts.npy", ts)
        sh = np.asarray(nib.load(f"/path/to/staging/ts_repro/shipped/{c}.nii.gz").dataobj).astype(np.uint8)
        vals = sorted(set(np.unique(sh).tolist()) - {0}); tum = sh == max(vals)
        po = np.isin(ts, host) & ~tum; so = sh == 1
        print(c, "shape", ts.shape, sh.shape, "shipped labels", vals, "| organ re-run", int(po.sum()),
              "shipped", int(so.sum()), "differing", int((po ^ so).sum()), flush=True)
if __name__ == "__main__":
    main()
