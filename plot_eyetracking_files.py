#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_eyetracking_files.py

Iterate through:
  /data_248/pdss/hospital_data_real/{date}/{client_id}/{test}/eye-tracking/{id}/eye-tracking.csv

and SAVE plots (PNG) to:
  /data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/plot/

For each CSV:
- saves a time-series plot of x(t), y(t)
- optionally saves a scatter plot of x vs y

Examples:
  python plot_eyetracking_files.py --test vst --max_files 3
  python plot_eyetracking_files.py --test dnb --max_files 10 --save_scatter
"""

import argparse
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT_DEFAULT = "/data_248/pdss/hospital_data_real"
OUT_DIR_DEFAULT = "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/plot"


def glob_eye_tracking_csv(root_dir: Path, test: str | None) -> list[Path]:
    if test is None:
        return sorted(root_dir.rglob("eye-tracking.csv"))
    hits = []
    for p in root_dir.rglob("eye-tracking.csv"):
        parts = p.parts
        if len(parts) >= 4 and parts[-4] == test and parts[-3] == "eye-tracking":
            hits.append(p)
    return sorted(hits)


def load_eye_csv(csv_path: Path) -> pd.DataFrame:
    # detect header
    df0 = pd.read_csv(csv_path, header=None, nrows=1)
    first_row = df0.iloc[0].astype(str).tolist()
    has_header = any(tok.strip().lower() in {"x", "y", "timestamp", "timestamp_ms", "filtered_x", "filtered_y", "timestamp"} for tok in first_row)

    if has_header:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_csv(csv_path, header=None)
        # assume the sample format: x,y,timestamp,timeStamp,filtered_x,filtered_y
        if df.shape[1] >= 6:
            df.columns = ["x", "y", "timestamp", "timeStamp", "filtered_x", "filtered_y"][: df.shape[1]]
        else:
            raise ValueError(f"CSV has no header and unexpected column count={df.shape[1]}: {csv_path}")

    df.columns = [str(c).strip() for c in df.columns]

    # Prefer filtered_x/y if present
    if "filtered_x" in df.columns and "filtered_y" in df.columns:
        x = pd.to_numeric(df["filtered_x"], errors="coerce")
        y = pd.to_numeric(df["filtered_y"], errors="coerce")
    else:
        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError(f"Missing x/y columns in {csv_path}")
        x = pd.to_numeric(df["x"], errors="coerce")
        y = pd.to_numeric(df["y"], errors="coerce")

    # Timestamp
    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
    elif "timeStamp" in df.columns:
        ts_dt = pd.to_datetime(df["timeStamp"], errors="coerce")
        ts = ts_dt.astype("int64") / 1e6  # ms
    else:
        raise ValueError(f"No timestamp column found in {csv_path}")

    out = pd.DataFrame({"timestamp": ts, "x": x, "y": y}).dropna()
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def infer_time_seconds(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(float)
    if ts.size <= 1:
        return np.zeros_like(ts, dtype=float)

    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs)]
    med = float(np.median(diffs)) if diffs.size else 1.0
    scale = 1000.0 if med >= 1.0 else 1.0  # ms vs s heuristic
    t_s = (ts - ts[0]) / scale
    t_s = np.maximum.accumulate(t_s)
    return t_s


def sanitize_filename(s: str) -> str:
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s[:200]


def save_plots(df: pd.DataFrame, title: str, out_dir: Path, stem: str, save_scatter: bool) -> None:
    ts = df["timestamp"].to_numpy(dtype=float)
    t_s = infer_time_seconds(ts)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    # Time series plot
    plt.figure()
    plt.plot(t_s, x, label="x")
    plt.plot(t_s, y, label="y")
    plt.xlabel("time (s)")
    plt.ylabel("position")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"{stem}_timeseries.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    # Optional scatter plot
    if save_scatter:
        plt.figure()
        plt.scatter(x, y, s=5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title + " (x vs y)")
        plt.tight_layout()

        out_path2 = out_dir / f"{stem}_scatter.png"
        plt.savefig(out_path2, dpi=150)
        plt.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", type=str, default=ROOT_DEFAULT)
    ap.add_argument("--test", type=str, default=None, help="ast/dnb/flanker/gng/vst (omit to scan all)")
    ap.add_argument("--max_files", type=int, default=3, help="Max number of csv files to process")
    ap.add_argument("--out_dir", type=str, default=OUT_DIR_DEFAULT, help="Directory to save PNG plots")
    ap.add_argument("--save_scatter", action="store_true", help="Also save x-y scatter plot")
    return ap.parse_args()


def main():
    args = parse_args()
    root_dir = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root_dir.exists():
        print(f"[ERROR] root_dir does not exist: {root_dir}", file=sys.stderr)
        sys.exit(1)

    files = glob_eye_tracking_csv(root_dir, args.test)
    print(f"Found {len(files)} eye-tracking.csv files under {root_dir} (test={args.test})")
    print(f"Saving plots to: {out_dir}")

    n = 0
    for p in files:
        if n >= args.max_files:
            break

        try:
            df = load_eye_csv(p)

            rel = str(p.relative_to(root_dir))
            title = f"{n+1}: {rel}"

            # Make a stable filename stem based on path
            stem = sanitize_filename(rel.replace(".csv", ""))

            save_plots(df, title=title, out_dir=out_dir, stem=stem, save_scatter=args.save_scatter)

            print(f"[{n+1}] saved: {stem}_timeseries.png" + (" + scatter" if args.save_scatter else ""))
            n += 1
        except Exception as e:
            print(f"[WARN] failed on {p}: {e}", file=sys.stderr)
            n += 1

    print("Done.")


if __name__ == "__main__":
    main()