#!/usr/bin/env python3
"""
extract_pose.py
---------------
Turn arena-frame motion CSVs into the egocentric pose CSVs the model trains on.

    dataset/<cohort>/motion/{preS,preR}/*.csv   ->  dataset/<cohort>/pose/{preS,preR}/*.csv
    27 columns, 9 keypoints                          21 columns, 7 keypoints

Per frame:
  1. Anchor  : subtract tail_base from every keypoint, so tail_base is the origin.
  2. Heading : theta = atan2(body_center_y - tail_base_y, body_center_x - tail_base_x).
  3. Rotate  : rotate every keypoint's (x, y) by -theta, so the spine lies on +X.
  4. Height  : z is shifted by tail_base_z only -- never rotated.
  5. Drop    : tail_base (now always 0,0,0) and tail_end.

The result is translation- and rotation-invariant: where the animal was in the
arena, and which way it faced, carry no information. Only posture and its change
over time survive.

Columns are looked up BY NAME, so your CSVs need the right column names, not a
particular column order.

Usage
-----
    python data_prep/extract_pose.py --cohort male
    python data_prep/extract_pose.py --input my/motion --output my/pose
    python data_prep/extract_pose.py --cohort male --verify     # compare, write nothing
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Keypoints the transform needs to exist in the input.
ANCHOR = "tail_base"       # becomes the origin
HEADING = "body_center"    # together with the anchor, defines the spine axis

# Keypoints kept in the output, in this order.
OUTPUT_KEYPOINTS = [
    "nose", "head", "body_center",
    "right_hindpaw", "left_hindpaw",
    "right_forepaw", "left_forepaw",
]
AXES = ("x", "y", "z")
OUTPUT_COLUMNS = [f"{kp}_{ax}" for kp in OUTPUT_KEYPOINTS for ax in AXES]

SPLITS = ("preS", "preR")


def required_columns():
    needed = set()
    for kp in set(OUTPUT_KEYPOINTS) | {ANCHOR, HEADING}:
        needed.update(f"{kp}_{ax}" for ax in AXES)
    return sorted(needed)


def extract_pose(motion):
    """Egocentric pose DataFrame from an arena-frame motion DataFrame."""
    missing = [c for c in required_columns() if c not in motion.columns]
    if missing:
        raise ValueError(
            f"motion CSV is missing {len(missing)} required column(s): {missing}\n"
            f"Columns must be named <keypoint>_<x|y|z>; see data_prep/KEYPOINTS.md."
        )

    col = lambda kp, ax: motion[f"{kp}_{ax}"].to_numpy(dtype=np.float32)
    anchor_x, anchor_y, anchor_z = (col(ANCHOR, ax) for ax in AXES)
    heading = np.arctan2(col(HEADING, "y") - anchor_y,
                         col(HEADING, "x") - anchor_x)
    cos_a, sin_a = np.cos(-heading), np.sin(-heading)

    pose = {}
    for kp in OUTPUT_KEYPOINTS:
        shifted_x = col(kp, "x") - anchor_x
        shifted_y = col(kp, "y") - anchor_y
        pose[f"{kp}_x"] = cos_a * shifted_x - sin_a * shifted_y
        pose[f"{kp}_y"] = sin_a * shifted_x + cos_a * shifted_y
        pose[f"{kp}_z"] = col(kp, "z") - anchor_z

    return pd.DataFrame(pose, columns=OUTPUT_COLUMNS)


def process_split(in_dir, out_dir, verify):
    """Convert (or verify) one split. Returns (files, mismatches)."""
    if not os.path.isdir(in_dir):
        print(f"  {in_dir}: not found -- skipped")
        return 0, 0
    if not verify:
        os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".csv"))
    mismatches = 0
    for name in files:
        pose = extract_pose(pd.read_csv(os.path.join(in_dir, name)))
        target = os.path.join(out_dir, name)
        if verify:
            if not os.path.exists(target):
                print(f"  {name}: no existing pose file to compare")
                mismatches += 1
                continue
            existing = pd.read_csv(target)
            if list(existing.columns) != OUTPUT_COLUMNS:
                print(f"  {name}: column names differ")
                mismatches += 1
            elif not np.array_equal(existing.to_numpy(np.float32),
                                    pose.to_numpy(np.float32)):
                diff = np.abs(existing.to_numpy(np.float32) - pose.to_numpy(np.float32)).max()
                print(f"  {name}: values differ, max |diff| = {diff:.3e}")
                mismatches += 1
        else:
            pose.to_csv(target, index=False)
    verb = "checked" if verify else "written"
    print(f"  {os.path.basename(in_dir):<6} {len(files):>3} files {verb} -> {out_dir}")
    return len(files), mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=str, help="Use dataset/<cohort>/{motion,pose}")
    ap.add_argument("--input", type=str, help="Directory of motion CSVs (overrides --cohort)")
    ap.add_argument("--output", type=str, help="Where to write pose CSVs")
    ap.add_argument("--verify", action="store_true",
                    help="Compare against existing pose CSVs instead of writing")
    opt = ap.parse_args()

    if opt.input:
        pairs = [(opt.input, opt.output or opt.input.replace("motion", "pose"))]
    elif opt.cohort:
        base = os.path.join("dataset", opt.cohort)
        pairs = [(os.path.join(base, "motion", s), os.path.join(base, "pose", s))
                 for s in SPLITS]
    else:
        ap.error("give --cohort, or --input (with --output)")

    print(f"{'Verifying' if opt.verify else 'Extracting'} egocentric pose "
          f"({len(OUTPUT_KEYPOINTS)} keypoints, {len(OUTPUT_COLUMNS)} columns)")
    total = mismatches = 0
    for in_dir, out_dir in pairs:
        n, bad = process_split(in_dir, out_dir, opt.verify)
        total += n
        mismatches += bad

    if opt.verify:
        print(f"\n{total} file(s) checked, {mismatches} mismatch(es)")
        return 1 if mismatches else 0
    print(f"\n{total} file(s) written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
