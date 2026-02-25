#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_eyetracking_with_segments.py

Overlay segmentation boundaries (from cached .npz) on top of eye-tracking CSV time-series plots.

- Reads every .npz in --npz_dir (or filtered by --test)
- For each .npz:
    - loads seg_start_s / seg_end_s
    - loads the per-file metadata stored in "files" (JSON)
    - for each referenced eye-tracking.csv:
        - loads the CSV and plots x(t), y(t)
        - overlays vertical lines at segment boundaries
        - optionally shades each segment interval
- Saves PNGs to --out_dir

Notes:
- This script assumes seg_start_s / seg_end_s are *relative seconds within each file*,
  and uses the per-file "n_segments" metadata (in the npz "files" field) to slice segments per file.
- If your .npz was generated from a single file per student, it still works.

Example:
  python plot_eyetracking_with_segments.py \
    --npz_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/valley_dnb2" \
    --out_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/plot_cp" \
    --test dnb \
    --max_participants 5 \
    --shade_segments
"""

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sanitize_filename(s: str) -> str:
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:220]


def infer_time_seconds_from_timestamp(ts: np.ndarray) -> np.ndarray:
    """
    Convert absolute timestamps to relative seconds.
    Heuristic:
      - if median dt >= 1, treat as ms
      - else treat as seconds
    """
    ts = ts.astype(float)
    if ts.size <= 1:
        return np.zeros_like(ts, dtype=float)

    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    med = float(np.median(diffs)) if diffs.size else 0.0

    if med >= 1.0:
        t_s = (ts - ts[0]) / 1000.0
    else:
        t_s = (ts - ts[0])
    return np.maximum.accumulate(t_s)


def load_eye_csv(csv_path: Path) -> pd.DataFrame:
    """
    Loads a CSV with either header or no header.
    Expected columns:
      x, y, timestamp, timeStamp, filtered_x, filtered_y

    Returns DataFrame with: timestamp, x, y (numeric, NaNs dropped).
    Uses filtered_x/y if present else x/y.
    """
    # detect header by reading first row as strings
    df0 = pd.read_csv(csv_path, header=None, nrows=1)
    first_row = df0.iloc[0].astype(str).tolist()
    has_header = any(tok.strip().lower() in {"x", "y", "timestamp", "timestamp_ms", "filtered_x", "filtered_y", "timestamp"} for tok in first_row)

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] >= 6:
            df.columns = ["x", "y", "timestamp", "timeStamp", "filtered_x", "filtered_y"][: df.shape[1]]
        else:
            raise ValueError(f"CSV has no header and unexpected column count={df.shape[1]}: {csv_path}")

    df.columns = [str(c).strip() for c in df.columns]

    # choose x/y source
    if "filtered_x" in df.columns and "filtered_y" in df.columns:
        x = pd.to_numeric(df["filtered_x"], errors="coerce")
        y = pd.to_numeric(df["filtered_y"], errors="coerce")
    else:
        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError(f"Missing x/y columns in {csv_path}")
        x = pd.to_numeric(df["x"], errors="coerce")
        y = pd.to_numeric(df["y"], errors="coerce")

    # timestamp
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
    elif "timeStamp" in df.columns:
        ts_dt = pd.to_datetime(df["timeStamp"], errors="coerce")
        ts = ts_dt.astype("int64") / 1e6  # ms
    else:
        raise ValueError(f"No timestamp column found in {csv_path}")

    out = pd.DataFrame({"timestamp": ts, "x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def load_npz_segments(npz_path: Path):
    """
    Returns:
      client_id, test_type, seg_start_s, seg_end_s, file_meta(list of dict)
    """
    z = np.load(npz_path, allow_pickle=True)
    seg_start_s = z["seg_start_s"].astype(float)
    seg_end_s = z["seg_end_s"].astype(float)

    client_id = str(z["client_id"]) if "client_id" in z else npz_path.name.split("_")[0]
    test_type = str(z["test_type"]) if "test_type" in z else None

    file_meta = []
    if "files" in z:
        try:
            file_meta = json.loads(str(z["files"]))
        except Exception:
            file_meta = []
    return client_id, test_type, seg_start_s, seg_end_s, file_meta


def plot_csv_with_segments(
    csv_path: Path,
    seg_start_s: np.ndarray,
    seg_end_s: np.ndarray,
    out_path: Path,
    title: str,
    shade_segments: bool,
    max_lines: int = 500,
):
    """
    Plot x(t), y(t) and overlay segment boundaries.
    """
    df = load_eye_csv(csv_path)
    if df.empty:
        return False

    t_s = infer_time_seconds_from_timestamp(df["timestamp"].to_numpy(dtype=float))
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    # Collect boundaries: starts and ends (avoid duplicates, avoid 0)
    boundaries = []
    for a, b in zip(seg_start_s, seg_end_s):
        if np.isfinite(a) and a > 1e-9:
            boundaries.append(float(a))
        if np.isfinite(b):
            boundaries.append(float(b))
    boundaries = sorted(set(boundaries))

    # Optionally reduce line count if there are too many boundaries
    if len(boundaries) > max_lines:
        step = int(np.ceil(len(boundaries) / max_lines))
        boundaries = boundaries[::step]

    plt.figure(figsize=(12, 5))
    plt.plot(t_s, x, label="x")
    plt.plot(t_s, y, label="y")
    plt.xlabel("time (s)")
    plt.ylabel("position")
    plt.title(title)

    # Shade segments (optional)
    if shade_segments:
        # Shade every segment interval lightly (alternating)
        # NOTE: matplotlib defaults color; we won't hardcode colors.
        for i, (a, b) in enumerate(zip(seg_start_s, seg_end_s)):
            if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
                continue
            if i % 2 == 0:
                plt.axvspan(a, b, alpha=0.08)

    # Vertical lines for boundaries
    for t in boundaries:
        plt.axvline(t, linestyle="--", linewidth=0.8, alpha=0.8)

    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_dir", type=str, required=True, help="Directory containing cached .npz files")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/plot_cp",
        help="Where to save plots",
    )
    ap.add_argument("--test", type=str, default=None, help="Filter .npz by test_type (e.g. dnb/vst/...)")
    ap.add_argument("--max_participants", type=int, default=3, help="Max number of participants (.npz files) to plot")
    ap.add_argument("--shade_segments", action="store_true", help="Shade segment spans in addition to boundary lines")
    ap.add_argument("--max_lines", type=int, default=500, help="Max number of boundary lines to draw (downsample if more)")
    return ap.parse_args()


def main():
    args = parse_args()
    npz_dir = Path(args.npz_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not npz_dir.exists():
        print(f"[ERROR] npz_dir not found: {npz_dir}", file=sys.stderr)
        sys.exit(1)

    npz_files = sorted(npz_dir.glob("*.npz"))
    if args.test is not None:
        # filter by filename suffix _{test}.npz or by test_type inside file (slower)
        filt = []
        for p in npz_files:
            if p.name.endswith(f"_{args.test}.npz"):
                filt.append(p)
            else:
                # optionally check inside
                try:
                    z = np.load(p, allow_pickle=True)
                    if "test_type" in z and str(z["test_type"]) == args.test:
                        filt.append(p)
                except Exception:
                    pass
        npz_files = filt

    print(f"Found {len(npz_files)} npz files in {npz_dir} (test={args.test})")
    n_done = 0

    for npz_path in npz_files:
        if n_done >= args.max_participants:
            break

        try:
            client_id, test_type, ss, ee, file_meta = load_npz_segments(npz_path)
            test_label = args.test if args.test is not None else (test_type if test_type else "unknown")

            if not file_meta:
                print(f"[WARN] {npz_path.name}: no file metadata in 'files' field; cannot map segments to csv reliably.")
                continue

            # Use file_meta's n_segments to slice ss/ee per csv file (important!)
            seg_cursor = 0
            for fi, fm in enumerate(file_meta, start=1):
                csv_path = Path(fm["file"])
                nseg = int(fm.get("n_segments", 0))
                if nseg <= 0:
                    continue

                ss_i = ss[seg_cursor: seg_cursor + nseg]
                ee_i = ee[seg_cursor: seg_cursor + nseg]
                seg_cursor += nseg

                if not csv_path.exists():
                    print(f"[WARN] missing csv: {csv_path}")
                    continue

                rel_name = sanitize_filename(f"{client_id}_{test_label}_file{fi}")
                out_path = out_dir / f"{rel_name}.png"

                title = f"{client_id} | {test_label} | file{fi} | segments={len(ss_i)}"
                ok = plot_csv_with_segments(
                    csv_path=csv_path,
                    seg_start_s=ss_i,
                    seg_end_s=ee_i,
                    out_path=out_path,
                    title=title,
                    shade_segments=args.shade_segments,
                    max_lines=args.max_lines,
                )
                if ok:
                    print(f"[OK] saved: {out_path}")

            n_done += 1

        except Exception as e:
            print(f"[ERROR] failed on {npz_path.name}: {e}", file=sys.stderr)
            continue

    print("Done.")


if __name__ == "__main__":
    main()