from __future__ import annotations
# load raw → segment (Win+C3) → PI per segment → save cached .npz/.pt
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Details:
	•	Load raw gaze per student
	•	Sliding window scores (Win) using C3 cost
	•	Peak picking + min distance + min segment length
	•	Segment the series
	•	Compute PI vector (258)) per segment
	•	Save cached file per student

"""
"""
preprocess.py
- Load raw eye-tracking CSVs
- Dynamic segmentation using Win + C3 cost
- Extract PI features per segment (target: 258 features/segment)
- Save cached per-student arrays to .npz

Expected directory structure (root_dir):
  {date}/{client_id}/{test_type}/eye-tracking/{other_id}/eye-tracking.csv

Labels CSV (example):
  /data_248/pdss/primitive_indicator_scripts/scripts/test_se/client_demographics.csv

PI extractor is fixed at:
  /data_248/pdss/primitive_indicator_scripts/scripts/intern/extract_eye_tracking_pi_window.py

"""
"""

CHANGES (per your request):
  1) Remove rows with NaN x or y (done per file, but we KEEP the student)
  2) Segment on cleaned data (done)
  3) Compute PI using nan-safe statistics by ensuring PI module only sees finite rows
     (we pass dfw with NO NaNs; also use nan-safe checks)
  4) Skip segments with too few valid samples (min_valid_points)
  5) Never drop whole student unless ratio of valid data < 80% (per-client across all files)

Usage example:
    python preprocess.py \
  --root_dir "/data_248/pdss/hospital_data_real" \
  --label_csv "/data_248/pdss/primitive_indicator_scripts/scripts/test_se/client_demographics.csv" \
  --test_type dnb \
  --out_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/dnb_filtered" \
  --pi_excel "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/Primitive Indicator Lists_filtered.xlsx" \
  --pi_sheet "eye-tracking" \
  --half_window_s 2.5 \
  --step_s 1.0 \
  --min_cp_distance_s 1.0 \
  --min_segment_len_s 1.0 \
  --min_valid_points 10 \
  --min_student_valid_ratio 0.80
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------
# Fixed path (per your request)
# -----------------------------
PI_PY_FIXED = "/data_248/pdss/primitive_indicator_scripts/scripts/intern/extract_eye_tracking_pi_window.py"

# -----------------------------
# Label mapping
# -----------------------------
LABEL_MAP = {
    "Non-ADHD": 0, #no adhd
    "Non_adhd": 0,
    "Non-adhd": 0,
    "non-adhd": 0,
    "Non_ADHD": 0,
    "Inattentive": 1, #adhd
    "inattentive": 1,
    "Combined": 1,
    "combined": 1, #
    "Subclinical": 1,
    "subclinical": 1,
}

TEST_TYPE_DEFAULT = "vst"

# -----------------------------
# NEW: Missing-data policy knobs
# -----------------------------
MIN_STUDENT_VALID_RATIO = 0.80  # (5) drop whole student only if overall valid ratio < 80%
MIN_VALID_POINTS = 10           # (4) skip segments/windows if too few valid samples for stable covariance / PI


# -----------------------------
# Config container
# -----------------------------
@dataclass
class SegConfig:
    half_window_s: float = 2.5          # w
    step_s: float = 2.0                  # sliding step between centers
    min_cp_distance_s: float = 1.0       # min distance between change points
    min_segment_len_s: float = 1.0       # min segment length
    beta: Optional[float] = None         # if None -> adaptive per file
    cov_eps_base: float = 1e-6           # base eps added to covariance
    cov_eps_scale: float = 1e-3          # scale * avg_var for eps
    local_maxima_only: bool = True
    min_valid_points: int = MIN_VALID_POINTS  # NEW


# -----------------------------
# Utilities
# -----------------------------
def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_pi_module(pi_py_path: str):
    """Dynamically import the PI extraction module (safe for @dataclass)."""
    pi_py_path = str(Path(pi_py_path).resolve())
    spec = importlib.util.spec_from_file_location("pi_module_runtime", pi_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load PI module spec from: {pi_py_path}")

    mod = importlib.util.module_from_spec(spec)
    # register in sys.modules BEFORE exec_module (dataclass needs it)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize_columns_with_valid_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Produce clean ET DataFrame with UNIQUE columns ['timestamp','x','y'].

    (1) We DO NOT drop whole student here.
        We remove invalid rows (NaN x/y or NaN timestamp) for segmentation/PI,
        but we also return (n_valid, n_total) so caller can decide whether to drop student.

    Returns:
      cleaned_df, n_valid_rows, n_total_rows
    """
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    n_total = int(len(df))

    if n_total == 0:
        return pd.DataFrame(columns=["timestamp", "x", "y"]), 0, 0

    if "timestamp" not in df.columns:
        raise ValueError("Missing required column 'timestamp' in eye-tracking CSV.")

    ts = pd.to_numeric(df["timestamp"], errors="coerce")

    # choose signal
    if "filtered_x" in df.columns and "filtered_y" in df.columns:
        x = pd.to_numeric(df["filtered_x"], errors="coerce")
        y = pd.to_numeric(df["filtered_y"], errors="coerce")
    else:
        if "x" not in df.columns or "y" not in df.columns:
            raise ValueError("Missing required columns 'x'/'y' in eye-tracking CSV.")
        x = pd.to_numeric(df["x"], errors="coerce")
        y = pd.to_numeric(df["y"], errors="coerce")

    out = pd.DataFrame({"timestamp": ts, "x": x, "y": y})

    # (1) Remove rows with NaN x or y (and timestamp, because segmentation needs time)
    valid_mask = np.isfinite(out["timestamp"].to_numpy()) & np.isfinite(out["x"].to_numpy()) & np.isfinite(out["y"].to_numpy())
    n_valid = int(valid_mask.sum())

    cleaned = out.loc[valid_mask].copy()
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)
    return cleaned, n_valid, n_total


def infer_time_unit_and_make_seconds(ts: np.ndarray) -> np.ndarray:
    """
    Convert absolute timestamps to relative seconds.
    Heuristic:
      - most ET logs use ms with diffs ~8-34
      - if median diff >= 1 -> treat as ms
      - else treat as seconds
    """
    ts = ts.astype(float)
    if ts.size < 2:
        return np.zeros_like(ts)

    diffs = np.diff(ts)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    med = float(np.median(diffs)) if diffs.size else 0.0

    is_ms = med >= 1.0
    if is_ms:
        t_s = (ts - ts[0]) / 1000.0
    else:
        t_s = (ts - ts[0])
    return t_s


# -----------------------------
# C3 cost (logdet + mahalanobis)
# -----------------------------
def c3_cost(Y: np.ndarray, eps: float, min_points: int) -> float:
    """
    C3 cost for multivariate Gaussian with unknown mean/cov (regularized):
      cost = n * logdet(S) + sum_i (x_i - mu)^T S^{-1} (x_i - mu)
    (dropping constants)

    (4) Returns NaN if fewer than min_points.
    """
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        return float("nan")
    n, d = Y.shape
    if n < max(min_points, d + 1):
        return float("nan")

    mu = Y.mean(axis=0, keepdims=True)
    Xc = Y - mu

    S = (Xc.T @ Xc) / max(n, 1)
    S = S + eps * np.eye(d)

    sign, logdet = np.linalg.slogdet(S)
    if not np.isfinite(logdet) or sign <= 0:
        return float("nan")

    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return float("nan")

    mahal = float(np.sum((Xc @ Sinv) * Xc))
    return float(n * logdet + mahal)


def compute_eps_from_data(Y: np.ndarray, base: float, scale: float) -> float:
    """
    eps = base + scale * avg_var
    """
    Y = np.asarray(Y, dtype=float)
    if Y.size == 0:
        return base
    v = np.var(Y, axis=0)
    avg_var = float(np.mean(v)) if np.all(np.isfinite(v)) else 0.0
    return float(base + scale * max(avg_var, 0.0))


# -----------------------------
# Win algorithm (offline)
# -----------------------------
def compute_Z_scores(
    t_s: np.ndarray,
    XY: np.ndarray,
    cfg: SegConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Z at a grid of center times:
      centers = [w, ..., T-w] with step cfg.step_s
      Z(tc) = c(r) - c(p) - c(q)
        r: [tc-w, tc+w]
        p: [tc-w, tc]
        q: [tc, tc+w]
    """
    if t_s.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    T_total = float(t_s[-1] - t_s[0])
    w = float(cfg.half_window_s)
    step = float(cfg.step_s)

    start = w
    end = max(w, T_total - w)
    if end <= start:
        return np.array([], dtype=float), np.array([], dtype=float)

    centers = np.arange(start, end + 1e-9, step, dtype=float)
    Z = np.full_like(centers, np.nan, dtype=float)

    for i, tc in enumerate(centers):
        p0, p1 = tc - w, tc
        q0, q1 = tc, tc + w
        r0, r1 = tc - w, tc + w

        ip0 = int(np.searchsorted(t_s, p0, side="left"))
        ip1 = int(np.searchsorted(t_s, p1, side="right"))
        iq0 = int(np.searchsorted(t_s, q0, side="left"))
        iq1 = int(np.searchsorted(t_s, q1, side="right"))
        ir0 = int(np.searchsorted(t_s, r0, side="left"))
        ir1 = int(np.searchsorted(t_s, r1, side="right"))

        Yp = XY[ip0:ip1]
        Yq = XY[iq0:iq1]
        Yr = XY[ir0:ir1]

        # (4) skip if too few valid samples
        if (Yp.shape[0] < cfg.min_valid_points) or (Yq.shape[0] < cfg.min_valid_points) or (Yr.shape[0] < cfg.min_valid_points):
            continue

        eps = compute_eps_from_data(Yr, base=cfg.cov_eps_base, scale=cfg.cov_eps_scale)

        cr = c3_cost(Yr, eps, min_points=cfg.min_valid_points)
        cp = c3_cost(Yp, eps, min_points=cfg.min_valid_points)
        cq = c3_cost(Yq, eps, min_points=cfg.min_valid_points)

        if not (np.isfinite(cr) and np.isfinite(cp) and np.isfinite(cq)):
            continue

        Z[i] = cr - cp - cq

    return centers, Z


def estimate_beta_from_Z(Z: np.ndarray) -> float:
    """
    Adaptive beta:
      beta = max(median(Z)+1.5*MAD(Z), percentile(Z, 85), 0)
    """
    Zf = Z[np.isfinite(Z)]
    if Zf.size == 0:
        return 0.0
    med = float(np.median(Zf))
    mad = robust_mad(Zf)
    beta1 = med + 1.5 * mad
    beta2 = float(np.percentile(Zf, 85))
    return float(max(beta1, beta2, 0.0))


def pick_peaks_greedy(
    centers_s: np.ndarray,
    Z: np.ndarray,
    beta: float,
    min_dist_s: float,
    local_maxima_only: bool = True,
) -> List[float]:
    """
    Peak detection:
      - candidate peaks: local maxima (optional) and Z >= beta
      - enforce min distance by greedy selection in descending Z
    """
    centers_s = np.asarray(centers_s, dtype=float)
    Z = np.asarray(Z, dtype=float)
    good = np.isfinite(Z) & np.isfinite(centers_s)
    centers_s = centers_s[good]
    Z = Z[good]
    if Z.size == 0:
        return []

    candidates: List[int] = []
    for i in range(Z.size):
        if Z[i] < beta:
            continue
        if local_maxima_only:
            left = Z[i - 1] if i - 1 >= 0 else -np.inf
            right = Z[i + 1] if i + 1 < Z.size else -np.inf
            if not (Z[i] > left and Z[i] >= right):
                continue
        candidates.append(i)

    if not candidates:
        return []

    candidates = np.array(candidates, dtype=int)
    order = candidates[np.argsort(Z[candidates])[::-1]]

    chosen: List[float] = []
    for idx in order:
        tc = float(centers_s[idx])
        if all(abs(tc - prev) >= min_dist_s for prev in chosen):
            chosen.append(tc)

    chosen.sort()
    return chosen


def build_segments_from_cps(
    t_s: np.ndarray,
    cps_s: List[float],
    min_seg_len_s: float
) -> List[Tuple[float, float]]:
    """
    Turn change-points into segments [start,end], enforce min segment length.
    """
    if t_s.size == 0:
        return []
    start0 = 0.0
    endT = float(t_s[-1])

    cps = [cp for cp in cps_s if (cp > start0 and cp < endT)]
    cps.sort()

    cleaned: List[float] = []
    prev = start0
    for cp in cps:
        if cp - prev >= min_seg_len_s:
            cleaned.append(cp)
            prev = cp

    while cleaned and (endT - cleaned[-1] < min_seg_len_s):
        cleaned.pop()

    bounds = [start0] + cleaned + [endT]
    segs: List[Tuple[float, float]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= min_seg_len_s:
            segs.append((float(a), float(b)))
    return segs


# -----------------------------
# PI extraction per segment
# -----------------------------
def extract_pi_for_segment(
    pi_mod,
    df_seg: pd.DataFrame,
    fullnames: List[str],
    screen_w: float,
    screen_h: float,
    min_valid_points: int,
) -> Optional[np.ndarray]:
    """
    (3) "nan-safe": we pass ONLY finite rows to PI module.
    (4) skip segment if too few valid samples.
    """
    if df_seg.empty:
        return None

    dfw = df_seg[["timestamp", "x", "y"]].copy()
    # remove non-finite rows (should already be clean, but keep it robust)
    dfw = dfw.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "x", "y"])
    if len(dfw) < min_valid_points:
        return None

    w_start_ms = int(dfw["timestamp"].iloc[0])
    w_end_ms = int(dfw["timestamp"].iloc[-1])

    base_vecs = pi_mod.compute_window_base_vectors(
        dfw=dfw,
        w_start_ms=w_start_ms,
        w_end_ms=w_end_ms,
        screen_w=float(screen_w),
        screen_h=float(screen_h),
    )
    pi_dict = pi_mod.build_pi_from_fullnames(base_vecs=base_vecs, fullnames=fullnames)

    out = np.empty((len(fullnames),), dtype=float)
    for i, name in enumerate(fullnames):
        v = pi_dict.get(name, None)
        try:
            out[i] = float(v) if v is not None else np.nan
        except Exception:
            out[i] = np.nan
    return out


# -----------------------------
# Main preprocessing per file
# -----------------------------
def process_one_eyetracking_file(
    csv_path: Path,
    pi_mod,
    fullnames: List[str],
    cfg: SegConfig,
    screen_w: float,
    screen_h: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """
    Returns:
      X: (num_segments, num_features)
      seg_start_s: (num_segments,)
      seg_end_s: (num_segments,)
      beta_used: float
      n_valid_rows: int
      n_total_rows: int
    """
    df_raw = pd.read_csv(csv_path)

    # (1) Remove rows with NaN x or y (and timestamp), but keep ratio info
    df, n_valid, n_total = normalize_columns_with_valid_ratio(df_raw)

    if df.empty:
        # No usable samples in this file
        X = np.zeros((0, len(fullnames)), dtype=float)
        return X, np.zeros((0,), float), np.zeros((0,), float), 0.0, n_valid, n_total

    ts = df["timestamp"].to_numpy(dtype=float)
    t_s = infer_time_unit_and_make_seconds(ts)
    XY = df[["x", "y"]].to_numpy(dtype=float)

    # (2) Segment on cleaned data
    centers_s, Z = compute_Z_scores(t_s=t_s, XY=XY, cfg=cfg)
    beta_used = cfg.beta if cfg.beta is not None else estimate_beta_from_Z(Z)

    cps_s = pick_peaks_greedy(
        centers_s=centers_s,
        Z=Z,
        beta=float(beta_used),
        min_dist_s=float(cfg.min_cp_distance_s),
        local_maxima_only=bool(cfg.local_maxima_only),
    )

    segments = build_segments_from_cps(
        t_s=t_s,
        cps_s=cps_s,
        min_seg_len_s=float(cfg.min_segment_len_s),
    )

    feats: List[np.ndarray] = []
    seg_starts: List[float] = []
    seg_ends: List[float] = []

    for (a, b) in segments:
        ia = int(np.searchsorted(t_s, a, side="left"))
        ib = int(np.searchsorted(t_s, b, side="right"))
        df_seg = df.iloc[ia:ib].copy()
        if df_seg.empty:
            continue

        # (3) nan-safe PI (finite rows only)
        # (4) skip segment if too few valid samples
        f = extract_pi_for_segment(
            pi_mod=pi_mod,
            df_seg=df_seg,
            fullnames=fullnames,
            screen_w=screen_w,
            screen_h=screen_h,
            min_valid_points=cfg.min_valid_points,
        )
        if f is None:
            continue

        feats.append(f)
        seg_starts.append(a)
        seg_ends.append(b)

    if not feats:
        X = np.zeros((0, len(fullnames)), dtype=float)
        seg_start_s = np.zeros((0,), dtype=float)
        seg_end_s = np.zeros((0,), dtype=float)
    else:
        X = np.stack(feats, axis=0)
        seg_start_s = np.asarray(seg_starts, dtype=float)
        seg_end_s = np.asarray(seg_ends, dtype=float)

    return X, seg_start_s, seg_end_s, float(beta_used), n_valid, n_total


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root_dir", type=str, required=True,
                    help="Root directory of raw hospital data (e.g., /data_248/pdss/hospital_data_real)")
    ap.add_argument("--label_csv", type=str, required=True,
                    help="Path to client_demographics.csv")
    ap.add_argument("--test_type", type=str, default=TEST_TYPE_DEFAULT,
                    help="Which test_type to use (default: vst)")

    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output cache directory for per-student .npz files")

    ap.add_argument("--pi_excel", type=str, required=True,
                    help='Excel file path containing PI list (e.g. "/data_248/.../Primitive Indicator Lists.xlsx")')
    ap.add_argument("--pi_sheet", type=str, default="eye-tracking",
                    help="Sheet name in the PI excel file (default: eye-tracking)")

    ap.add_argument("--screen_w", type=float, default=1920.0)
    ap.add_argument("--screen_h", type=float, default=1080.0)

    # segmentation config
    ap.add_argument("--half_window_s", type=float, default=2.5)
    ap.add_argument("--step_s", type=float, default=1.0)
    ap.add_argument("--min_cp_distance_s", type=float, default=1.0)
    ap.add_argument("--min_segment_len_s", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=None,
                    help="If provided, fixed threshold beta; else adaptive per file.")
    ap.add_argument("--cov_eps_base", type=float, default=1e-6)
    ap.add_argument("--cov_eps_scale", type=float, default=1e-3)

    # NEW
    ap.add_argument("--min_valid_points", type=int, default=MIN_VALID_POINTS,
                    help=f"(4) Skip windows/segments with < this many valid points (default {MIN_VALID_POINTS})")
    ap.add_argument("--min_student_valid_ratio", type=float, default=MIN_STUDENT_VALID_RATIO,
                    help=f"(5) Drop whole student only if overall valid ratio < this (default {MIN_STUDENT_VALID_RATIO})")

    ap.add_argument("--max_clients", type=int, default=None,
                    help="Optional: limit number of clients processed (debug)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing cached .npz")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir)
    label_csv = Path(args.label_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {root_dir}")
    if not label_csv.exists():
        raise FileNotFoundError(f"label_csv not found: {label_csv}")

    pi_py_path = Path(PI_PY_FIXED)
    if not pi_py_path.exists():
        raise FileNotFoundError(f"PI python file not found at fixed path: {pi_py_path}")
    pi_mod = load_pi_module(str(pi_py_path))

    # Load PI fullnames from excel (defines feature order)
    pi_excel = Path(args.pi_excel)
    if not pi_excel.exists():
        raise FileNotFoundError(f"pi_excel not found: {pi_excel}")
    fullnames = pi_mod.load_pi_fullnames_from_excel(str(pi_excel), sheet_name=args.pi_sheet)

    if not isinstance(fullnames, list) or len(fullnames) == 0:
        raise RuntimeError("Failed to load PI fullnames list from excel.")
    if len(fullnames) != 258:
        print(f"[WARN] PI fullname count is {len(fullnames)} (expected 258). Proceeding anyway.")

    # Load labels
    df_lab = pd.read_csv(label_csv)
    required = {"client_id", "group_label"}
    if not required.issubset(df_lab.columns):
        raise ValueError(f"Label CSV must contain columns {required}, got {set(df_lab.columns)}")

    df_lab["group_label_norm"] = df_lab["group_label"].astype(str).str.strip()
    df_lab["y"] = df_lab["group_label_norm"].map(LABEL_MAP)
    df_lab = df_lab.dropna(subset=["client_id", "y"]).copy()
    df_lab["y"] = df_lab["y"].astype(int)

    cfg = SegConfig(
        half_window_s=float(args.half_window_s),
        step_s=float(args.step_s),
        min_cp_distance_s=float(args.min_cp_distance_s),
        min_segment_len_s=float(args.min_segment_len_s),
        beta=float(args.beta) if args.beta is not None else None,
        cov_eps_base=float(args.cov_eps_base),
        cov_eps_scale=float(args.cov_eps_scale),
        local_maxima_only=True,
        min_valid_points=int(args.min_valid_points),
    )

    test_type = str(args.test_type)
    min_student_valid_ratio = float(args.min_student_valid_ratio)

    rows = df_lab[["client_id", "group_label_norm", "y"]].drop_duplicates().to_dict("records")
    if args.max_clients is not None:
        rows = rows[: int(args.max_clients)]

    summary = {
        "n_clients_cached": 0,
        "n_clients_written": 0,
        "n_skipped_no_files": 0,
        "n_skipped_low_valid_ratio": 0,
        "n_failed": 0,
        "avg_segments": None,
        "beta_stats": {},
    }
    seg_counts = []
    betas = []

    import glob

    for r in tqdm(rows, desc="clients"):
        client_id = str(r["client_id"])
        group_label = str(r["group_label_norm"])
        y = int(r["y"])

        out_file = out_dir / f"{client_id}_{test_type}.npz"
        if out_file.exists() and not args.overwrite:
            summary["n_clients_cached"] += 1
            continue

        pattern = str(root_dir / "*" / client_id / test_type / "eye-tracking" / "*" / "eye-tracking.csv")
        files = [Path(p) for p in sorted(glob.glob(pattern))]

        if not files:
            summary["n_skipped_no_files"] += 1
            continue

        all_X, all_ss, all_ee = [], [], []
        file_meta = []

        # (5) track valid ratio across ALL files for this student
        total_valid_rows = 0
        total_rows = 0

        try:
            for f in files:
                X, ss, ee, beta_used, n_valid, n_total = process_one_eyetracking_file(
                    csv_path=f,
                    pi_mod=pi_mod,
                    fullnames=fullnames,
                    cfg=cfg,
                    screen_w=float(args.screen_w),
                    screen_h=float(args.screen_h),
                )
                total_valid_rows += int(n_valid)
                total_rows += int(n_total)

                if X.shape[0] == 0:
                    continue

                all_X.append(X)
                all_ss.append(ss)
                all_ee.append(ee)
                file_meta.append({
                    "file": str(f),
                    "n_segments": int(X.shape[0]),
                    "beta_used": float(beta_used),
                    "n_valid_rows": int(n_valid),
                    "n_total_rows": int(n_total),
                })
                betas.append(float(beta_used))

            # Decide whether to drop whole student based on overall valid ratio
            valid_ratio = (total_valid_rows / total_rows) if total_rows > 0 else 0.0
            if valid_ratio < min_student_valid_ratio:
                summary["n_skipped_low_valid_ratio"] += 1
                continue  # (5) ONLY here do we drop the student

            if not all_X:
                # keep student (not dropped), but nothing usable after segment filtering
                continue

            X_cat = np.concatenate(all_X, axis=0)
            ss_cat = np.concatenate(all_ss, axis=0)
            ee_cat = np.concatenate(all_ee, axis=0)

            np.savez_compressed(
                out_file,
                X=X_cat.astype(np.float32),
                seg_start_s=ss_cat.astype(np.float32),
                seg_end_s=ee_cat.astype(np.float32),
                y=np.int64(y),
                client_id=client_id,
                group_label=group_label,
                test_type=test_type,
                pi_fullnames=np.array(fullnames, dtype=object),
                files=json.dumps(file_meta, ensure_ascii=False),
                seg_config=json.dumps(cfg.__dict__, ensure_ascii=False),
                screen_w=float(args.screen_w),
                screen_h=float(args.screen_h),
                student_valid_ratio=float(valid_ratio),
                total_rows=int(total_rows),
                total_valid_rows=int(total_valid_rows),
            )

            summary["n_clients_written"] += 1
            seg_counts.append(int(X_cat.shape[0]))

        except Exception as e:
            summary["n_failed"] += 1
            import traceback
            print(f"\n[ERROR] client_id={client_id} failed: {e}", file=sys.stderr)
            traceback.print_exc()
            continue

    if seg_counts:
        summary["avg_segments"] = float(np.mean(seg_counts))
    if betas:
        b = np.asarray(betas, dtype=float)
        summary["beta_stats"] = {
            "mean": float(np.mean(b)),
            "median": float(np.median(b)),
            "p90": float(np.percentile(b, 90)),
            "p95": float(np.percentile(b, 95)),
        }

    summary_path = out_dir / "preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Cache dir: {out_dir}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()