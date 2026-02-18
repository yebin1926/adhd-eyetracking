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
	•	Compute PI vector (444) per segment
	•	Save cached file per student

"""
"""
preprocess.py
- Load raw eye-tracking CSVs
- Dynamic segmentation using Win + C3 cost
- Extract PI feature vectors per segment (444)
- Save per-student cached .npz containing:
    X: (num_segments, 444)
    y: int in {0,1,2,3}
    seg_start_s, seg_end_s: segment boundaries in seconds (relative)
    metadata: client_id, group_label, test_type, screen_w, screen_h, etc.

Changes requested:
1) Excel path is passed as argument (--pi_excel)
2) PI extraction module path is fixed (no argument)
3) Added fixed-window segmentation option with minimal changes:
   --segmentation {dynamic,fixed}
   --fixed_win_s, --fixed_stride_s, --fixed_drop_last

Usage example for dynamic:
  python preprocess.py \
    --root_dir /data_248/pdss/hospital_data_real \
    --label_csv /data_248/pdss/primitive_indicator_scripts/scripts/test_se/client_demographics.csv \
    --test_type ast \
    --out_dir /data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output \
    --pi_excel "/data_248/pdss/primitive_indicator_scripts/scripts/Primitive Indicator Lists.xlsx" \
    --pi_sheet eye-tracking

Usage example for fixed:
python preprocess.py \
  --root_dir "/data_248/pdss/hospital_data_real" \
  --label_csv "/data_248/pdss/primitive_indicator_scripts/scripts/test_se/client_demographics.csv" \
  --test_type dnb \
  --out_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/dnb_fixed_w6_s2" \
  --pi_excel "/data_248/pdss/primitive_indicator_scripts/scripts/Primitive Indicator Lists.xlsx" \
  --pi_sheet "eye-tracking" \
  --segmentation fixed \
  --fixed_win_s 6 \
  --fixed_stride_s 2 \
  --min_segment_len_s 1
"""

import argparse
import glob
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# Fixed PI extractor path (do NOT change; per your requirement)
PI_EXTRACTOR_PATH = Path("/data_248/pdss/primitive_indicator_scripts/scripts/intern/extract_eye_tracking_pi_window.py")

# Default test type
TEST_TYPE_DEFAULT = "vst"


# -----------------------------
# Config dataclasses
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
    local_maxima_only: bool = True       # only local maxima peaks


# -----------------------------
# Utility: dynamically import PI module from fixed path
# -----------------------------
def load_pi_module(py_path: str):
    spec = importlib.util.spec_from_file_location("pi_mod", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    # IMPORTANT: register in sys.modules so @dataclass works inside imported module
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# -----------------------------
# Utility: normalize columns
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has columns: x, y, timestamp
    Handles cases where raw has filtered_x/filtered_y.
    """
    df = df.copy()

    # If file has headers but sometimes spaced/cased differently
    df.columns = [str(c).strip() for c in df.columns]

    # Prefer filtered_x/y if present, else x/y
    if "filtered_x" in df.columns and "filtered_y" in df.columns:
        df["x"] = pd.to_numeric(df["filtered_x"], errors="coerce")
        df["y"] = pd.to_numeric(df["filtered_y"], errors="coerce")
    else:
        if "x" not in df.columns or "y" not in df.columns:
            # Sometimes files might not have headers (rare). Try to read as 6 columns.
            # If you hit this, your CSV likely didn't have a header row.
            raise KeyError("Missing required columns 'x'/'y' (and no filtered_x/filtered_y).")
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # timestamp column
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    else:
        # fallback to timeStamp (string ISO) if numeric timestamp missing
        if "timeStamp" in df.columns:
            # convert ISO to epoch ms
            ts = pd.to_datetime(df["timeStamp"], errors="coerce")
            df["timestamp"] = (ts.astype("int64") / 1e6)  # ms
        else:
            raise KeyError("Missing required timestamp column ('timestamp' or 'timeStamp').")

    # Drop rows with NaNs in essential columns
    df = df.dropna(subset=["x", "y", "timestamp"]).reset_index(drop=True)

    # Sort by timestamp (critical for stable time axis)
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


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
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return np.zeros_like(ts)

    med = float(np.median(diffs))
    # If median diff is large, assume ms
    if med >= 1.0:
        scale = 1000.0
    else:
        scale = 1.0

    t_s = (ts - float(ts[0])) / scale
    # ensure nondecreasing (some logs have duplicates)
    t_s = np.maximum.accumulate(t_s)
    return t_s


# -----------------------------
# C3 cost + Z-score
# -----------------------------
def c3_cost(Y: np.ndarray, eps: float) -> float:
    """
    C3 cost for multivariate Gaussian with unknown mean/cov (regularized):
      cost = n * logdet(S) + sum_i (x_i - mu)^T S^{-1} (x_i - mu)
    (dropping constants)
    """
    Y = np.asarray(Y, dtype=float)
    n, d = Y.shape
    if n < max(3, d + 1):
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

    quad = float(np.einsum("ni,ij,nj->", Xc, Sinv, Xc))
    return float(n * logdet + quad)


def compute_cov_eps(Y: np.ndarray, cfg: SegConfig) -> float:
    """
    eps = cfg.cov_eps_base + cfg.cov_eps_scale * avg_var
    """
    Y = np.asarray(Y, dtype=float)
    if Y.size == 0:
        return float(cfg.cov_eps_base)
    var = np.nanvar(Y, axis=0)
    avg_var = float(np.nanmean(var)) if np.isfinite(np.nanmean(var)) else 0.0
    eps = float(cfg.cov_eps_base + cfg.cov_eps_scale * max(avg_var, 0.0))
    return eps


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
        a = tc - w
        b = tc + w
        # indices for r, p, q
        ir0 = int(np.searchsorted(t_s, a, side="left"))
        ir1 = int(np.searchsorted(t_s, b, side="right"))
        ip0 = ir0
        ip1 = int(np.searchsorted(t_s, tc, side="right"))
        iq0 = int(np.searchsorted(t_s, tc, side="left"))
        iq1 = ir1

        Yr = XY[ir0:ir1]
        Yp = XY[ip0:ip1]
        Yq = XY[iq0:iq1]

        eps_r = compute_cov_eps(Yr, cfg)
        eps_p = compute_cov_eps(Yp, cfg)
        eps_q = compute_cov_eps(Yq, cfg)

        cr = c3_cost(Yr, eps=eps_r)
        cp = c3_cost(Yp, eps=eps_p)
        cq = c3_cost(Yq, eps=eps_q)

        if np.isfinite(cr) and np.isfinite(cp) and np.isfinite(cq):
            Z[i] = float(cr - cp - cq)

    return centers, Z


# -----------------------------
# Peak selection
# -----------------------------
def robust_mad(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(mad * 1.4826)  # approx std


def estimate_beta_from_Z(Z: np.ndarray) -> float:
    """
    Adaptive beta threshold. Tweak here to get more/fewer segments.
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
    local_maxima_only: bool = True
) -> List[float]:
    """
    Pick peaks where Z >= beta.
    If local_maxima_only: restrict to local maxima.
    Greedy selection by descending Z, enforcing min distance.
    Returns chosen change points (seconds).
    """
    centers_s = np.asarray(centers_s, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if centers_s.size == 0 or Z.size == 0:
        return []

    finite = np.isfinite(Z)
    centers_s = centers_s[finite]
    Z = Z[finite]
    if centers_s.size == 0:
        return []

    cand = np.where(Z >= float(beta))[0]
    if cand.size == 0:
        return []

    if local_maxima_only and cand.size > 0:
        # keep only local maxima among candidates
        keep = []
        for i in cand:
            left = Z[i - 1] if i - 1 >= 0 else -np.inf
            right = Z[i + 1] if i + 1 < Z.size else -np.inf
            if Z[i] >= left and Z[i] >= right:
                keep.append(i)
        cand = np.array(keep, dtype=int)
        if cand.size == 0:
            return []

    # greedy by descending Z
    order = cand[np.argsort(Z[cand])[::-1]]
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
    Turn change-points into segments [start,end], enforce min
    t_s is relative seconds starting at 0.
    """
    t_s = np.asarray(t_s, dtype=float)
    if t_s.size == 0:
        return []

    bounds = [0.0] + [float(c) for c in cps_s] + [float(t_s[-1])]
    bounds = sorted(bounds)

    segs: List[Tuple[float, float]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= min_seg_len_s:
            segs.append((float(a), float(b)))
    return segs


def build_fixed_segments(
    t_s: np.ndarray,
    win_s: float,
    stride_s: float,
    min_seg_len_s: float,
    drop_last: bool = False
) -> List[Tuple[float, float]]:
    """
    Build fixed-length sliding window segments over the recording.
    Segments are expressed in seconds relative to t_s[0] (same coordinate system as t_s).

    - Start at 0.0s
    - Each window is [start, start+win_s]
    - Advance by stride_s
    - Enforce min_seg_len_s based on actual available samples inside the window
    - If drop_last=True, discard the last window if it would extend beyond t_s[-1]
      Otherwise, the last window end is clipped to t_s[-1].

    Returns a list of (start_s, end_s) with end_s > start_s.
    """
    t_s = np.asarray(t_s, dtype=float)
    if t_s.size == 0:
        return []

    total = float(t_s[-1] - t_s[0])
    if not np.isfinite(total) or total <= 0:
        return []

    win_s = float(win_s)
    stride_s = float(stride_s)
    if win_s <= 0 or stride_s <= 0:
        return []

    segs: List[Tuple[float, float]] = []
    cur = 0.0

    while cur < total - 1e-9:
        end = cur + win_s
        if drop_last and end > total:
            break
        end_clip = min(end, total)

        # check actual duration in samples (avoid windows with no samples)
        ia = int(np.searchsorted(t_s, t_s[0] + cur, side="left"))
        ib = int(np.searchsorted(t_s, t_s[0] + end_clip, side="right"))
        if ib <= ia:
            cur += stride_s
            continue

        seg_len = float(t_s[ib - 1] - t_s[ia])
        if seg_len >= float(min_seg_len_s):
            segs.append((float(cur), float(end_clip)))

        cur += stride_s

    return segs


# -----------------------------
# PI extraction per segment
# -----------------------------
import inspect

def extract_pi_for_segment(pi_mod, df_seg: pd.DataFrame, fullnames: List[str], screen_w: float, screen_h: float) -> np.ndarray:
    """
    Compute PI vector for one segment using the PI module WITHOUT modifying it.

    Supports PI module variants:
      - compute_window_base_vectors(dfw, w_start_ms, w_end_ms, screen_w, screen_h)
      - build_pi_from_fullnames(base_vecs, fullnames) -> dict {fullname: value/None}
    """
    dfw = df_seg.copy()

    # required columns
    for col in ("x", "y", "timestamp"):
        if col not in dfw.columns:
            raise KeyError(f"df_seg missing required column: {col}")

    # numeric + clean
    dfw["x"] = pd.to_numeric(dfw["x"], errors="coerce")
    dfw["y"] = pd.to_numeric(dfw["y"], errors="coerce")
    dfw["timestamp"] = pd.to_numeric(dfw["timestamp"], errors="coerce")
    dfw = dfw.dropna(subset=["x", "y", "timestamp"]).reset_index(drop=True)
    if dfw.empty:
        raise ValueError("Empty segment after dropping NaNs")

    # segment bounds in ms (PI module expects ms)
    w_start_ms = int(dfw["timestamp"].iloc[0])
    w_end_ms = int(dfw["timestamp"].iloc[-1])
    if w_end_ms < w_start_ms:
        w_start_ms, w_end_ms = w_end_ms, w_start_ms

    sw = float(screen_w)
    sh = float(screen_h)

    # --- compute base vectors ---
    cbv = pi_mod.compute_window_base_vectors
    sig_cbv = inspect.signature(cbv)
    if len(sig_cbv.parameters) >= 5:
        base_vecs = cbv(dfw, w_start_ms, w_end_ms, sw, sh)
    else:
        base_vecs = cbv(dfw)

    # --- build PI dict ---
    if not hasattr(pi_mod, "build_pi_from_fullnames"):
        raise AttributeError("PI module has no build_pi_from_fullnames(). Cannot compute PI values.")
    pi_dict = pi_mod.build_pi_from_fullnames(base_vecs, fullnames)

    # --- dict -> vector aligned with fullnames ---
    vec = np.empty((len(fullnames),), dtype=np.float32)
    for i, name in enumerate(fullnames):
        v = pi_dict.get(name, None)
        if v is None:
            vec[i] = np.nan
        else:
            try:
                vf = float(v)
                vec[i] = vf if np.isfinite(vf) else np.nan
            except Exception:
                vec[i] = np.nan

    return vec


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
    segmentation: str = "dynamic",
    fixed_win_s: float = 10.0,
    fixed_stride_s: float = 2.0,
    fixed_drop_last: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
      X: (num_segments, num_features)
      seg_start_s: (num_segments,)
      seg_end_s: (num_segments,)
      beta_used: float
    """
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    ts = df["timestamp"].to_numpy(dtype=float)
    t_s = infer_time_unit_and_make_seconds(ts)
    XY = df[["x", "y"]].to_numpy(dtype=float)

    if segmentation == "fixed":
        segments = build_fixed_segments(
            t_s=t_s,
            win_s=float(fixed_win_s),
            stride_s=float(fixed_stride_s),
            min_seg_len_s=float(cfg.min_segment_len_s),
            drop_last=bool(fixed_drop_last),
        )
        beta_used = float("nan")
    else:
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

        f = extract_pi_for_segment(
            pi_mod=pi_mod,
            df_seg=df_seg,
            fullnames=fullnames,
            screen_w=screen_w,
            screen_h=screen_h,
        )

        feats.append(f)
        seg_starts.append(float(a))
        seg_ends.append(float(b))

    if len(feats) == 0:
        X = np.zeros((0, len(fullnames)), dtype=np.float32)
        ss = np.zeros((0,), dtype=np.float32)
        ee = np.zeros((0,), dtype=np.float32)
    else:
        X = np.stack(feats, axis=0).astype(np.float32)
        ss = np.asarray(seg_starts, dtype=np.float32)
        ee = np.asarray(seg_ends, dtype=np.float32)

    return X, ss, ee, float(beta_used)


# -----------------------------
# PI list loading from Excel
# -----------------------------
def load_pi_fullnames_from_excel(excel_path: Path, sheet_name: str) -> List[str]:
    """
    Loads PI names (fullnames) from the given Excel sheet.
    Expects a column that contains PI keys/names. This code tries common variants.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    # Common column candidates
    candidates = [
        "pi_fullname", "pi_fullnames", "fullnames", "fullname",
        "pi_key", "pi_keys", "key", "keys",
        "PI", "PI_key", "PI_KEY"
    ]

    col = None
    for c in candidates:
        if c in df.columns:
            col = c
            break

    if col is None:
        # fallback: use first column
        col = df.columns[0]

    fullnames = [str(v).strip() for v in df[col].dropna().tolist()]
    # remove empties
    fullnames = [x for x in fullnames if x != ""]
    if len(fullnames) == 0:
        raise RuntimeError(f"No PI names found in excel={excel_path} sheet={sheet_name} col={col}")
    return fullnames


# -----------------------------
# Label mapping
# -----------------------------
LABEL_MAP = {
    "non-adhd": 0,
    "nonadhd": 0,
    "non_adhd": 0,
    "control": 0,
    "inattentive": 1,
    "combined": 2,
    "subclinical": 3,
}


def normalize_group_label(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "").replace("-", "").replace("_", "")
    return s


def load_labels(label_csv: Path) -> Dict[str, Dict]:
    """
    Load label CSV into dict keyed by client_id.
    Returns mapping: client_id -> {group_label, y, ...}
    """
    df = pd.read_csv(label_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if "client_id" not in df.columns or "group_label" not in df.columns:
        raise KeyError("label_csv missing required columns: client_id, group_label")

    out: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        cid = str(row["client_id"]).strip()
        gl = str(row["group_label"]).strip()
        key = normalize_group_label(gl)
        if key not in LABEL_MAP:
            # try raw lower
            key2 = str(gl).strip().lower()
            y = LABEL_MAP.get(key2, None)
        else:
            y = LABEL_MAP[key]

        if y is None:
            # skip unknown
            continue
        out[cid] = {
            "client_id": cid,
            "group_label": gl,
            "y": int(y),
        }
    return out


# -----------------------------
# Locate raw eye-tracking files
# -----------------------------
def find_eyetracking_files_for_client(root_dir: Path, client_id: str, test_type: str) -> List[Path]:
    """
    Search pattern:
      {date}/{client_id}/{test_type}/eye-tracking/*/eye-tracking.csv
    """
    pattern = str(root_dir / "*" / client_id / test_type / "eye-tracking" / "*" / "eye-tracking.csv")
    files = [Path(p) for p in glob.glob(pattern)]
    files = sorted([p for p in files if p.exists()])
    return files


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

    # CHANGE #1: Excel path is an argument you will pass at runtime
    ap.add_argument("--pi_excel", type=str, required=True,
                    help='Excel file path containing PI list (you will pass: "/data_248/pdss/primitive_indicator_scripts/scripts/Primitive Indicator Lists.xlsx")')
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

    # segmentation mode
    ap.add_argument("--segmentation", type=str, default="dynamic", choices=["dynamic", "fixed"],
                    help="Segmentation method: dynamic (Win+C3) or fixed (sliding fixed windows)")
    ap.add_argument("--fixed_win_s", type=float, default=10.0,
                    help="Fixed window length in seconds (used when --segmentation fixed)")
    ap.add_argument("--fixed_stride_s", type=float, default=2.0,
                    help="Fixed window stride in seconds (used when --segmentation fixed)")
    ap.add_argument("--fixed_drop_last", action="store_true",
                    help="If set, drop the last window if it would exceed the recording end time")

    ap.add_argument("--max_clients", type=int, default=None,
                    help="Optional: limit number of clients processed (debug)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing cached .npz")

    return ap.parse_args()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    root_dir = Path(args.root_dir)
    label_csv = Path(args.label_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load PI module (fixed path)
    pi_mod = load_pi_module(str(PI_EXTRACTOR_PATH))

    # Load PI names from Excel
    pi_excel_path = Path(args.pi_excel)
    fullnames = load_pi_fullnames_from_excel(pi_excel_path, sheet_name=str(args.pi_sheet))
    n_feat = len(fullnames)

    # Load labels
    labels = load_labels(label_csv)

    # Config
    cfg = SegConfig(
        half_window_s=float(args.half_window_s),
        step_s=float(args.step_s),
        min_cp_distance_s=float(args.min_cp_distance_s),
        min_segment_len_s=float(args.min_segment_len_s),
        beta=args.beta if args.beta is None else float(args.beta),
        cov_eps_base=float(args.cov_eps_base),
        cov_eps_scale=float(args.cov_eps_scale),
        local_maxima_only=True,
    )

    # Process clients
    client_ids = sorted(labels.keys())
    if args.max_clients is not None:
        client_ids = client_ids[: int(args.max_clients)]

    summary = {
        "n_clients": 0,
        "n_cached": 0,
        "n_skipped_no_files": 0,
        "n_failed": 0,
        "avg_segments": None,
        "beta_stats": {},
    }

    seg_counts: List[int] = []
    betas_used: List[float] = []

    for cid in tqdm(client_ids, desc="clients"):
        info = labels[cid]
        y = int(info["y"])
        group_label = str(info["group_label"])
        test_type = str(args.test_type)

        # Output file name includes test type
        out_path = out_dir / f"{cid}_{test_type}.npz"
        if out_path.exists() and not args.overwrite:
            summary["n_cached"] += 1
            continue

        files = find_eyetracking_files_for_client(root_dir, cid, test_type=test_type)

        if not files:
            summary["n_skipped_no_files"] += 1
            continue

        all_X, all_ss, all_ee = [], [], []
        file_meta = []

        try:
            for f in files:
                X, ss, ee, beta_used = process_one_eyetracking_file(
                    csv_path=f,
                    pi_mod=pi_mod,
                    fullnames=fullnames,
                    cfg=cfg,
                    screen_w=float(args.screen_w),
                    screen_h=float(args.screen_h),
                    segmentation=str(args.segmentation),
                    fixed_win_s=float(args.fixed_win_s),
                    fixed_stride_s=float(args.fixed_stride_s),
                    fixed_drop_last=bool(args.fixed_drop_last),
                )
                if X.shape[0] == 0:
                    continue
                all_X.append(X)
                all_ss.append(ss)
                all_ee.append(ee)
                betas_used.append(beta_used)
                file_meta.append(str(f))

            if len(all_X) == 0:
                # still save empty? skip
                summary["n_failed"] += 1
                print(f"[ERROR] client_id={cid} failed: no usable segments")
                continue

            X_all = np.concatenate(all_X, axis=0).astype(np.float32)
            ss_all = np.concatenate(all_ss, axis=0).astype(np.float32)
            ee_all = np.concatenate(all_ee, axis=0).astype(np.float32)

            # Basic sanity: enforce monotonic starts
            order = np.argsort(ss_all)
            X_all = X_all[order]
            ss_all = ss_all[order]
            ee_all = ee_all[order]

            # Save
            np.savez_compressed(
                out_path,
                X=X_all,
                y=np.int64(y),
                seg_start_s=ss_all,
                seg_end_s=ee_all,
                client_id=cid,
                group_label=group_label,
                test_type=test_type,
                screen_w=float(args.screen_w),
                screen_h=float(args.screen_h),
                files=np.array(file_meta, dtype=object),
                pi_fullnames=np.array(fullnames, dtype=object),
                segmentation=str(args.segmentation),
                fixed_win_s=float(args.fixed_win_s),
                fixed_stride_s=float(args.fixed_stride_s),
                half_window_s=float(cfg.half_window_s),
                step_s=float(cfg.step_s),
                min_cp_distance_s=float(cfg.min_cp_distance_s),
                min_segment_len_s=float(cfg.min_segment_len_s),
            )

            summary["n_clients"] += 1
            seg_counts.append(int(X_all.shape[0]))

        except Exception as e:
            summary["n_failed"] += 1
            print(f"[ERROR] client_id={cid} failed: {e}")
            import traceback
            traceback.print_exc()

    if len(seg_counts) > 0:
        summary["avg_segments"] = float(np.mean(seg_counts))
    else:
        summary["avg_segments"] = None

    # beta stats (only meaningful for dynamic; fixed will be NaN)
    bu = np.array(betas_used, dtype=float)
    bu = bu[np.isfinite(bu)]
    if bu.size > 0:
        summary["beta_stats"] = {
            "count": int(bu.size),
            "mean": float(np.mean(bu)),
            "std": float(np.std(bu)),
            "min": float(np.min(bu)),
            "max": float(np.max(bu)),
        }
    else:
        summary["beta_stats"] = {}

    # Save summary
    summary_path = out_dir / "preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Cache dir:", str(out_dir))
    print("Summary: ", str(summary_path))


if __name__ == "__main__":
    main()