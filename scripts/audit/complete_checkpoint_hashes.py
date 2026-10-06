#!/usr/bin/env python3
"""Complete the checkpoint manifest with full SHA-256 for every surviving checkpoint (IPKG round 2).

Inputs: `sha256sum` over fresh rclone copies of every checkpoint (organ/*.pth, tumor/*.pth), and
`rclone md5sum` of the Drive originals. A hash is filled in only if it is consistent with what the manifest
already recorded (full SHA-256, else the 12-character prefix); any mismatch aborts without writing.

  python complete_checkpoint_hashes.py <sha256_all.txt> [<md5_drive.txt>]
-> results/segmentation/checkpoint_manifest_2026-10-01.json (updated in place)
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MAN = os.path.join(ROOT, "results", "segmentation", "checkpoint_manifest_2026-10-01.json")


def read(p):
    out = {}
    for line in open(p):
        parts = line.split()
        if len(parts) == 2:
            a, b = parts
            k, v = (b, a) if len(a) == 64 else (a, b)        # sha256sum: "<hash>  <file>"; md5 list: "<file> <md5>"
            out[k] = v
    return out


def main():
    sha = read(sys.argv[1])
    md5 = read(sys.argv[2]) if len(sys.argv) > 2 else {}
    m = json.load(open(MAN))
    log = []
    for c in m["checkpoints"]:
        f = c["file"] if "/" in c["file"] else "tumor/" + c["file"]
        h = sha.get(f)
        if h is None:
            log.append(f"{c['file']}: not re-hashed"); continue
        if c.get("sha256"):
            assert c["sha256"] == h, (c["file"], "SHA-256 MISMATCH", c["sha256"], h)
            log.append(f"{c['file']}: full SHA-256 re-verified")
        else:
            assert h.startswith(c["sha256_prefix_only"]), (c["file"], "PREFIX MISMATCH", c["sha256_prefix_only"], h)
            c["sha256"], c["sha256_prefix_only"] = h, None
            log.append(f"{c['file']}: full SHA-256 filled (prefix matched)")
        if c.get("md5_drive") and md5.get(f):
            assert c["md5_drive"] == md5[f], (c["file"], "DRIVE MD5 MISMATCH")
        if "NOT on Drive" in c.get("location", ""):
            c["location"] = "drive_UD:VKG_app_handoff_2026-09-14/response_2026-10-01/checkpoints/tumor/sam3_tumor_flare_only.pth (copied 2026-10-01)"
    m["note"] = ("Checkpoint inventory for the co-author 09-29 §6 item 4, completed 2026-10-01 (round 2): full SHA-256 for every surviving "
                 "checkpoint, each computed on a fresh copy from Drive and checked against the hash or prefix recorded earlier "
                 "(scripts/audit/complete_checkpoint_hashes.py). Two checkpoints are not recoverable (below).")
    json.dump(m, open(MAN, "w"), indent=1)
    print("\n".join(log))


if __name__ == "__main__":
    main()
