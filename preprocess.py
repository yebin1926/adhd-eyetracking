#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess.py
- Load raw eye-tracking CSVs
- Compute Win + C3 score curve Z(t)
- Peak picking + min distance + min segment length
- Segmentation conversion (full coverage, non-overlapping):
    "event windows around peaks + background segments"
    - For each peak, find closest local minimum (valley) on the left/right of the peak on the Z-grid
    - Create an event window [valley_left, valley_right]
    - Merge overlapping/nearby event windows (also merge if gap < min_segment_len_s)
    - Add background segments for the gaps between event windows
    - Edge cases: clip to [0, T]
- Compute PI vector per segment (expects 258 fullnames from your Excel)
- Save cached file per student (.npz)

Expected directory structure (root_dir):
  {date}/{client_id}/{test_type}/eye-tracking/{other_id}/eye-tracking.csv

PI extractor is fixed at:
  /data_248/pdss/primitive_indicator_scripts/scripts/intern/extract_eye_tracking_pi_window.py

Usage example:
  python preprocess.py \
    --root_dir "/data_248/pdss/hospital_data_real" \
    --label_csv "/data_248/pdss/primitive_indicator_scripts/scripts/test_se/client_demographics.csv" \
    --test_type dnb \
    --out_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/valley_ast" \
    --pi_excel "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/Primitive Indicator Lists_filtered.xlsx" \
    --pi_sheet "eye-tracking" \
    --half_window_s 4 \
    --step_s 1.0 \
    --min_cp_distance_s 1 \
    --min_segment_len_s 0.1 \
    --min_valid_points 5 \
    --overwrite \
    --debug
"""

import argparse
import glob
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
# Label mapping (binary)
# 0 = Non_ADHD, 1 = ADHD, Subclinical -> 0 (per your instruction)
# -----------------------------
LABEL_MAP = {
    "Non-ADHD": 0,
    "Non_adhd": 0,
    "Non-adhd": 0,
    "non-adhd": 0,
    "Non_ADHD": 0,
    "Inattentive": 1,
    "inattentive": 1,
    "Combined": 1,
    "combined": 1,
    "Subclinical": 0,
    "subclinical": 0,
}

TEST_TYPE_DEFAULT = "vst"

# -----------------------------
# Missing-data policy knobs
# -----------------------------
MIN_VALID_POINTS = 10  # skip windows/segments if too few valid samples


# -----------------------------
# Config container
# -----------------------------
@dataclass
class SegConfig:
    half_window_s: float = 2.5          # w
    step_s: float = 1.0                 # step between centers
    min_cp_distance_s: float = 1.0      # min distance between peaks
    min_segment_len_s: float = 1.0      # min segment length
    beta: Optional[float] = None        # if None -> adaptive per file
    cov_eps_base: float = 1e-6
    cov_eps_scale: float = 1e-3
    local_maxima_only: bool = True
    min_valid_points: int = MIN_VALID_POINTS


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


def dprint(debug: bool, msg: str) -> None:
    if debug:
        print(msg)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def summarize_arr(x: np.ndarray, name: str, max_head: int = 10) -> dict:
    """Small numeric summary for debug logging."""
    x = np.asarray(x)
    out = {"name": name, "shape": list(x.shape), "dtype": str(x.dtype), "n": int(x.size)}
    if x.size == 0:
        return out
    if np.issubdtype(x.dtype, np.number):
        xf = x[np.isfinite(x)]
        out["finite_n"] = int(xf.size)
        if xf.size:
            out.update(
                {
                    "min": float(np.min(xf)),
                    "max": float(np.max(xf)),
                    "mean": float(np.mean(xf)),
                    "median": float(np.median(xf)),
                    "p90": float(np.percentile(xf, 90)),
                }
            )
    try:
        out["head"] = [float(v) for v in x.flatten()[:max_head]]
    except Exception:
        out["head"] = [str(v) for v in x.flatten()[:max_head]]
    return out


def write_debug_json(path: Path, payload: dict, debug: bool) -> None:
    if not debug:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[DBG] failed to write debug json to {path}: {e}", file=sys.stderr)


def load_pi_module(pi_py_path: str):
    """Dynamically import the PI extraction module (safe for @dataclass)."""
    pi_py_path = str(Path(pi_py_path).resolve())
    spec = importlib.util.spec_from_file_location("pi_module_runtime", pi_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load PI module spec from: {pi_py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (dataclass safety)
    spec.loader.exec_module(mod)
    return mod


def normalize_columns_with_valid_ratio(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Produce clean ET DataFrame with columns ['timestamp','x','y'].

    Removes invalid rows (NaN x/y/timestamp) for segmentation/PI.
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

    valid_mask = (
        np.isfinite(out["timestamp"].to_numpy())
        & np.isfinite(out["x"].to_numpy())
        & np.isfinite(out["y"].to_numpy())
    )
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
    C3 cost (dropping constants).
    Returns NaN if fewer than min_points.
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
    """eps = base + scale * avg_var"""
    Y = np.asarray(Y, dtype=float)
    if Y.size == 0:
        return base
    v = np.var(Y, axis=0)
    avg_var = float(np.mean(v)) if np.all(np.isfinite(v)) else 0.0
    return float(base + scale * max(avg_var, 0.0))


# -----------------------------
# Win algorithm (offline) -> score curve Z(t)
# -----------------------------
def compute_Z_scores(t_s: np.ndarray, XY: np.ndarray, cfg: SegConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    centers = [w, ..., T-w] step cfg.step_s
    Z(tc) = c(r) - c(p) - c(q)
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
    Slightly relaxed adaptive beta:
      beta = max(median(Z) + 0.5*MAD(Z), percentile(Z, 70), 0)
    """
    Zf = Z[np.isfinite(Z)]
    if Zf.size == 0:
        return 0.0
    med = float(np.median(Zf))
    mad = robust_mad(Zf)
    beta1 = med + 0.5 * mad
    beta2 = float(np.percentile(Zf, 70))
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

    Relaxed peak condition (plateaus allowed):
      Z[i] >= left and Z[i] >= right
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
            if not (Z[i] >= left and Z[i] >= right):
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


# -----------------------------
# Event windows around peaks + background segments
# -----------------------------
def _local_minima_indices(Z: np.ndarray) -> np.ndarray:
    """
    Robust local minima on Z-grid.
    - Works with NaNs by considering only finite points
    - Includes endpoints (first/last finite) as minima candidates
    """
    Z = np.asarray(Z, dtype=float)
    if Z.size == 0:
        return np.array([], dtype=int)

    finite_idx = np.where(np.isfinite(Z))[0]
    if finite_idx.size == 0:
        return np.array([], dtype=int)

    mins = []
    mins.append(int(finite_idx[0]))
    mins.append(int(finite_idx[-1]))

    if finite_idx.size < 3:
        return np.array(sorted(set(mins)), dtype=int)

    for k in range(1, finite_idx.size - 1):
        i = int(finite_idx[k])
        il = int(finite_idx[k - 1])
        ir = int(finite_idx[k + 1])
        if Z[i] <= Z[il] and Z[i] <= Z[ir]:
            mins.append(i)

    return np.array(sorted(set(mins)), dtype=int)


def _merge_intervals(intervals: List[Tuple[float, float]], merge_gap_s: float) -> List[Tuple[float, float]]:
    """Merge overlapping intervals and also merge if gap < merge_gap_s."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for a, b in intervals[1:]:
        a0, b0 = merged[-1]
        if a < b0 or (a - b0) < merge_gap_s:
            merged[-1] = (float(a0), float(max(b0, b)))
        else:
            merged.append((float(a), float(b)))
    return merged


def build_segments_event_background(
    centers_s: np.ndarray,
    Z: np.ndarray,
    peaks_s: List[float],
    T_end_s: float,
    min_seg_len_s: float,
    debug: bool = False,
) -> List[Tuple[float, float]]:
    """
    Full coverage non-overlapping segmentation:
      1) for each peak: event window [valley_left, valley_right]
      2) merge overlaps and merge close intervals if gap < min_seg_len_s
      3) background segments are gaps between event windows
      4) clip to [0, T_end_s]
    """
    start0 = 0.0
    endT = float(max(T_end_s, 0.0))
    if endT <= 0:
        return []

    centers_s = np.asarray(centers_s, dtype=float)
    Z = np.asarray(Z, dtype=float)

    if centers_s.size == 0 or Z.size == 0 or not peaks_s:
        dprint(debug, f"[DBG] fallback: no peaks or empty Z/centers -> 1 segment [{start0:.2f},{endT:.2f}]")
        return [(start0, endT)] if (endT - start0) >= min_seg_len_s else []

    mins = _local_minima_indices(Z)
    dprint(debug, f"[DBG] local minima count={mins.size}")
    if mins.size > 0:
        dprint(debug, f"[DBG] minima times (first 10): {[float(centers_s[i]) for i in mins[:10]]}")

    if mins.size == 0:
        dprint(debug, "[DBG] fallback: no minima -> 1 segment whole recording")
        return [(start0, endT)] if (endT - start0) >= min_seg_len_s else []

    event_intervals: List[Tuple[float, float]] = []
    for p in peaks_s:
        idx = int(np.argmin(np.abs(centers_s - float(p))))
        left_candidates = mins[mins < idx]
        right_candidates = mins[mins > idx]

        left_t = start0
        right_t = endT
        if left_candidates.size > 0:
            left_t = float(centers_s[int(left_candidates[-1])])
        if right_candidates.size > 0:
            right_t = float(centers_s[int(right_candidates[0])])

        left_t = max(start0, min(left_t, endT))
        right_t = max(start0, min(right_t, endT))
        if right_t < left_t:
            left_t, right_t = right_t, left_t

        if (right_t - left_t) >= min_seg_len_s:
            event_intervals.append((left_t, right_t))

    dprint(debug, f"[DBG] raw event_intervals={len(event_intervals)}")
    if event_intervals[:5]:
        dprint(debug, f"[DBG] first 5 event_intervals={event_intervals[:5]}")

    merged_events = _merge_intervals(event_intervals, merge_gap_s=min_seg_len_s)
    dprint(debug, f"[DBG] merged_events={len(merged_events)}")
    if merged_events[:5]:
        dprint(debug, f"[DBG] first 5 merged_events={merged_events[:5]}")

    segments: List[Tuple[float, float]] = []
    cur = start0

    for (ea, eb) in merged_events:
        ea = max(start0, min(float(ea), endT))
        eb = max(start0, min(float(eb), endT))
        if eb < ea:
            ea, eb = eb, ea

        gap = ea - cur
        if gap >= min_seg_len_s:
            segments.append((float(cur), float(ea)))
            cur = ea
        elif gap > 0:
            # tiny gap: swallow into event by moving event start backward
            ea = float(cur)

        if (eb - ea) >= min_seg_len_s:
            segments.append((float(ea), float(eb)))
            cur = eb
        else:
            # tiny event -> ignore
            cur = max(cur, eb)

    if (endT - cur) >= min_seg_len_s:
        segments.append((float(cur), float(endT)))
    elif (endT - cur) > 0 and segments:
        # swallow trailing tiny remainder into last segment
        a0, _b0 = segments[-1]
        segments[-1] = (float(a0), float(endT))

    # Ensure sorted, non-overlapping, min length, and merge tiny gaps again
    out: List[Tuple[float, float]] = []
    for a, b in sorted(segments, key=lambda x: (x[0], x[1])):
        if (b - a) < min_seg_len_s:
            continue
        if not out:
            out.append((a, b))
        else:
            pa, pb = out[-1]
            if a < pb:
                out[-1] = (pa, max(pb, b))
            else:
                out.append((a, b))

    if not out:
        dprint(debug, "[DBG] WARNING: build_segments produced empty -> fallback whole recording")
        return [(start0, endT)] if (endT - start0) >= min_seg_len_s else []

    return out


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
    nan-safe: pass ONLY finite rows to PI module.
    Skip segment if too few valid samples.
    """
    if df_seg.empty:
        return None

    dfw = df_seg[["timestamp", "x", "y"]].copy()
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
    debug: bool = False,
    debug_json_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      X: (num_segments, num_features)
      seg_start_s: (num_segments,)
      seg_end_s: (num_segments,)
      beta_used: float
    """
    df_raw = pd.read_csv(csv_path)

    df, _n_valid, _n_total = normalize_columns_with_valid_ratio(df_raw)
    if df.empty:
        X = np.zeros((0, len(fullnames)), dtype=float)
        return X, np.zeros((0,), float), np.zeros((0,), float), 0.0

    ts = df["timestamp"].to_numpy(dtype=float)
    t_s = infer_time_unit_and_make_seconds(ts)
    XY = df[["x", "y"]].to_numpy(dtype=float)

    centers_s, Z = compute_Z_scores(t_s=t_s, XY=XY, cfg=cfg)

    if debug:
        dprint(True, f"[DBG] file={csv_path}")
        dprint(True, f"[DBG] n_valid_samples={len(df)}  T_end_s={float(t_s[-1]) if t_s.size else 0.0:.4f}")
        if t_s.size >= 2:
            dt = np.diff(t_s)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                dprint(True, f"[DBG] dt_s: median={float(np.median(dt)):.6f}  p10={float(np.percentile(dt,10)):.6f}  p90={float(np.percentile(dt,90)):.6f}")
        dprint(True, f"[DBG] centers_s: n={centers_s.size}  step_s={cfg.step_s}  half_window_s={cfg.half_window_s}")
        dprint(True, f"[DBG] Z summary: {summarize_arr(Z, 'Z')}  finite_ratio={float(np.isfinite(Z).mean()) if Z.size else 0.0:.3f}")

    beta_used = cfg.beta if cfg.beta is not None else estimate_beta_from_Z(Z)
    if debug:
        dprint(True, f"[DBG] beta_used={beta_used:.6f} (cfg.beta={'fixed' if cfg.beta is not None else 'adaptive'})")

    peaks_s = pick_peaks_greedy(
        centers_s=centers_s,
        Z=Z,
        beta=float(beta_used),
        min_dist_s=float(cfg.min_cp_distance_s),
        local_maxima_only=bool(cfg.local_maxima_only),
    )

    if debug:
        dprint(True, f"[DBG] peaks: n={len(peaks_s)}  min_cp_distance_s={cfg.min_cp_distance_s}")
        if peaks_s[:10]:
            dprint(True, f"[DBG] peaks_s (first 10)={peaks_s[:10]}")
        if Z.size:
            above = int(np.sum(np.isfinite(Z) & (Z >= float(beta_used))))
            dprint(True, f"[DBG] Z>=beta count={above} / {Z.size}")

    T_end_s = float(t_s[-1]) if t_s.size else 0.0
    segments = build_segments_event_background(
        centers_s=centers_s,
        Z=Z,
        peaks_s=peaks_s,
        T_end_s=T_end_s,
        min_seg_len_s=float(cfg.min_segment_len_s),
        debug=debug,
    )

    if debug:
        dprint(True, f"[DBG] segments built: n={len(segments)}  min_segment_len_s={cfg.min_segment_len_s}")
        if segments[:10]:
            dprint(True, f"[DBG] segments (first 10)={segments[:10]}")
        if len(segments) == 1:
            dprint(True, "[DBG] WARNING: only 1 segment -> likely (a) no peaks passed threshold, (b) minima detection collapsed, or (c) Z mostly NaN")

        if debug_json_path is not None:
            payload = {
                "file": str(csv_path),
                "n_valid_samples": int(len(df)),
                "T_end_s": float(T_end_s),
                "beta_used": float(beta_used),
                "cfg": cfg.__dict__,
                "centers_s": summarize_arr(centers_s, "centers_s"),
                "Z": summarize_arr(Z, "Z"),
                "n_peaks": int(len(peaks_s)),
                "peaks_s_first10": [float(x) for x in peaks_s[:10]],
                "n_segments": int(len(segments)),
                "segments_first10": [(float(a), float(b)) for (a, b) in segments[:10]],
            }
            write_debug_json(Path(debug_json_path), payload, debug=True)

    feats: List[np.ndarray] = []
    seg_starts: List[float] = []
    seg_ends: List[float] = []

    for (a, b) in segments:
        ia = int(np.searchsorted(t_s, a, side="left"))
        ib = int(np.searchsorted(t_s, b, side="right"))
        df_seg = df.iloc[ia:ib].copy()
        f = extract_pi_for_segment(
            pi_mod=pi_mod,
            df_seg=df_seg,
            fullnames=fullnames,
            screen_w=float(screen_w),
            screen_h=float(screen_h),
            min_valid_points=int(cfg.min_valid_points),
        )
        if f is None:
            continue
        feats.append(f)
        seg_starts.append(float(a))
        seg_ends.append(float(b))

    if not feats:
        X = np.zeros((0, len(fullnames)), dtype=float)
        return X, np.zeros((0,), float), np.zeros((0,), float), float(beta_used)

    X = np.stack(feats, axis=0).astype(np.float32)
    ss = np.asarray(seg_starts, dtype=np.float32)
    ee = np.asarray(seg_ends, dtype=np.float32)
    return X, ss, ee, float(beta_used)


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
                    help="Excel file path containing PI fullnames (target: 258).")
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

    ap.add_argument("--min_valid_points", type=int, default=MIN_VALID_POINTS,
                    help="Skip segments/windows if too few valid samples (default: 10)")
    ap.add_argument("--max_clients", type=int, default=None,
                    help="Optional: limit number of clients processed (debug)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing cached .npz")

    ap.add_argument("--debug", action="store_true",
                    help="Verbose debug prints + write per-file debug JSONs.")
    ap.add_argument("--debug_dir", type=str, default=None,
                    help="Where to write debug JSONs (default: <out_dir>/debug)")
    ap.add_argument("--debug_max_print", type=int, default=3,
                    help="Max number of eye-tracking files per student to write debug JSON for (default: 3)")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir)
    label_csv = Path(args.label_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    debug_dir = Path(args.debug_dir) if args.debug_dir else (out_dir / "debug")
    if args.debug:
        ensure_dir(debug_dir)

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

    rows = df_lab[["client_id", "group_label_norm", "y"]].drop_duplicates().to_dict("records")
    if args.max_clients is not None:
        rows = rows[: int(args.max_clients)]

    summary = {
        "n_clients": 0,
        "n_cached": 0,
        "n_skipped_no_files": 0,
        "n_failed": 0,
        "avg_segments": None,
        "beta_stats": {},
    }
    seg_counts: List[int] = []
    betas: List[float] = []

    for r in tqdm(rows, desc="clients"):
        client_id = str(r["client_id"])
        group_label = str(r["group_label_norm"])
        y = int(r["y"])

        out_file = out_dir / f"{client_id}_{test_type}.npz"
        if args.debug:
            dprint(True, f"\n[DBG] ===== client_id={client_id} y={y} label={group_label} =====")

        if out_file.exists() and not args.overwrite:
            summary["n_cached"] += 1
            continue

        pattern = str(root_dir / "*" / client_id / test_type / "eye-tracking" / "*" / "eye-tracking.csv")
        files = [Path(p) for p in sorted(glob.glob(pattern))]

        if not files:
            summary["n_skipped_no_files"] += 1
            continue

        all_X, all_ss, all_ee = [], [], []
        file_meta = []

        try:
            for f in files:
                dbg_path = None
                if args.debug and (len(file_meta) < int(args.debug_max_print)):
                    dbg_path = debug_dir / f"{client_id}_{test_type}__file{len(file_meta)+1}.json"

                X, ss, ee, beta_used = process_one_eyetracking_file(
                    csv_path=f,
                    pi_mod=pi_mod,
                    fullnames=fullnames,
                    cfg=cfg,
                    screen_w=float(args.screen_w),
                    screen_h=float(args.screen_h),
                    debug=bool(args.debug),
                    debug_json_path=dbg_path,
                )
                if X.shape[0] == 0:
                    continue

                all_X.append(X)
                all_ss.append(ss)
                all_ee.append(ee)
                file_meta.append({
                    "file": str(f),
                    "n_segments": int(X.shape[0]),
                    "beta_used": float(beta_used),
                })
                betas.append(float(beta_used))

            if not all_X:
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
            )

            summary["n_clients"] += 1
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
    if args.debug:
        print(f"[DBG] debug JSON dir: {debug_dir}")


if __name__ == "__main__":
    main()