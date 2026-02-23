from __future__ import annotations
# load raw → FIXED segment → PI per segment → save cached .npz/.pt
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Details:
	•	Load raw gaze per student
	•	FIXED segmentation (window/stride in seconds)
	•	Segment the series
	•	Compute PI vector (444) per segment
	•	Save cached file per student

Minimal changes from the current (dynamic Win+C3) preprocess.py:
- Added fixed window args: --fixed_win_s (default 6), --fixed_stride_s (default 2)
- Added build_fixed_segments()
- Replaced dynamic segmentation step in process_one_eyetracking_file() with fixed windows
- Kept old dynamic CLI args for compatibility (unused in fixed mode)

Usage Example:
    python preprocess_fixed.py \
  --root_dir "/data_248/pdss/hospital_data_real" \
  --label_csv "/data_248/pdss/primitive_indicator_scripts/scripts/test_se/client_demographics.csv" \
  --test_type vst \
  --out_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/vst_w6_s2" \
  --pi_excel "/data_248/pdss/primitive_indicator_scripts/scripts/Primitive Indicator Lists.xlsx" \
  --pi_sheet "eye-tracking"

  (Optional)
  --fixed_win_s 6 --fixed_stride_s 2

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
    "Non-ADHD": 0,
    "NonADHD": 0,
    "Non_adhd": 0,
    "Non-adhd": 0,
    "non-adhd": 0,
    "Non_ADHD": 0,
    "Inattentive": 1,
    "inattentive": 1,
    "Combined": 2,
    "combined": 2,
    "Subclinical": 3,
    "subclinical": 3,
}

TEST_TYPE_DEFAULT = "dnb"

# -----------------------------
# Config container
# -----------------------------
@dataclass
class SegConfig:
    # FIXED segmentation parameters (seconds)
    fixed_win_s: float = 6.0
    fixed_stride_s: float = 2.0

    # keep min segment length (seconds)
    min_segment_len_s: float = 1.0

    # (optional) keep dynamic fields for backward compatibility (unused in fixed mode)
    half_window_s: float = 2.5          # w (unused)
    step_s: float = 2.0                 # (unused)
    min_cp_distance_s: float = 1.0      # (unused)
    beta: Optional[float] = None        # (unused)
    cov_eps_base: float = 1e-6          # (unused)
    cov_eps_scale: float = 1e-3         # (unused)
    local_maxima_only: bool = True      # (unused)


# -----------------------------
# Utilities
# -----------------------------
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean eye-tracking DataFrame with UNIQUE columns:
      ['timestamp', 'x', 'y']
    Prefer filtered_x/filtered_y if present, else raw x/y.
    Drops missing rows and sorts by timestamp.
    """
    # strip column names
    df = df.rename(columns={c: str(c).strip() for c in df.columns})

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

    # IMPORTANT: build a NEW df so there are NO duplicate column names
    out = pd.DataFrame({"timestamp": ts, "x": x, "y": y}).dropna()

    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def infer_time_unit_and_make_seconds(ts: np.ndarray) -> np.ndarray:
    """
    Convert absolute timestamps to relative seconds.
    Heuristic:
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
# FIXED segmentation
# -----------------------------
def build_fixed_segments(
    t_s: np.ndarray,
    win_s: float,
    stride_s: float,
    min_seg_len_s: float,
) -> List[Tuple[float, float]]:
    """
    Build fixed windows [start, start+win] stepping by stride, within [0, T].
    - Only full windows are produced (start+win <= T).
    - If T < win but T >= min_seg_len_s, produce one segment [0, T].
    """
    if t_s.size == 0:
        return []

    T = float(t_s[-1])
    win_s = float(win_s)
    stride_s = float(stride_s)

    if win_s <= 0 or stride_s <= 0:
        raise ValueError("fixed_win_s and fixed_stride_s must be > 0.")

    segs: List[Tuple[float, float]] = []

    if T < win_s:
        if T >= float(min_seg_len_s):
            segs.append((0.0, T))
        return segs

    start = 0.0
    while start + win_s <= T + 1e-9:
        end = start + win_s
        if end - start >= float(min_seg_len_s):
            segs.append((start, end))
        start += stride_s

    return segs


# -----------------------------
# Dynamic-segmentation functions kept (unused in fixed mode)
# -----------------------------
def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


def c3_cost(Y: np.ndarray, eps: float) -> float:
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        return float("nan")
    n, d = Y.shape
    if n < d + 1:
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
    Y = np.asarray(Y, dtype=float)
    if Y.size == 0:
        return base
    v = np.var(Y, axis=0)
    avg_var = float(np.mean(v)) if np.all(np.isfinite(v)) else 0.0
    return float(base + scale * max(avg_var, 0.0))


def compute_Z_scores(
    t_s: np.ndarray,
    XY: np.ndarray,
    cfg: SegConfig
) -> Tuple[np.ndarray, np.ndarray]:
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

        eps = compute_eps_from_data(Yr, base=cfg.cov_eps_base, scale=cfg.cov_eps_scale)

        cr = c3_cost(Yr, eps)
        cp = c3_cost(Yp, eps)
        cq = c3_cost(Yq, eps)

        if not (np.isfinite(cr) and np.isfinite(cp) and np.isfinite(cq)):
            continue

        Z[i] = cr - cp - cq

    return centers, Z


def estimate_beta_from_Z(Z: np.ndarray) -> float:
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
) -> Optional[np.ndarray]:
    if df_seg.empty:
        return None

    dfw = df_seg[["timestamp", "x", "y"]].copy()
    dfw = dfw.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp", "x", "y"])
    if dfw.empty:
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
    XY = df[["x", "y"]].to_numpy(dtype=float)  # kept (unused) for minimal change

        # FIXED segmentation on cleaned data
    segments = build_fixed_segments(
        t_s=t_s,
        win_s=float(cfg.fixed_win_s),
        stride_s=float(cfg.fixed_stride_s),
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

    beta_used = 0.0  # fixed segmentation has no beta/threshold
    return X, seg_start_s, seg_end_s, beta_used


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

    # Excel path is an argument you will pass at runtime
    ap.add_argument("--pi_excel", type=str, required=True,
                    help='Excel file path containing PI list (e.g. "/data_248/.../Primitive Indicator Lists.xlsx")')
    ap.add_argument("--pi_sheet", type=str, default="eye-tracking",
                    help="Sheet name in the PI excel file (default: eye-tracking)")

    ap.add_argument("--screen_w", type=float, default=1920.0)
    ap.add_argument("--screen_h", type=float, default=1080.0)

    # FIXED segmentation config (defaults per your request)
    ap.add_argument("--fixed_win_s", type=float, default=6.0,
                    help="Fixed window size in seconds (default: 6.0)")
    ap.add_argument("--fixed_stride_s", type=float, default=2.0,
                    help="Fixed window stride in seconds (default: 2.0)")
    ap.add_argument("--min_segment_len_s", type=float, default=1.0,
                    help="Minimum segment length in seconds (default: 1.0)")

    # (kept for backward compatibility; unused in fixed segmentation)
    ap.add_argument("--half_window_s", type=float, default=2.5)
    ap.add_argument("--step_s", type=float, default=1.0)
    ap.add_argument("--min_cp_distance_s", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=None,
                    help="(unused in fixed segmentation) kept for compatibility")
    ap.add_argument("--cov_eps_base", type=float, default=1e-6)
    ap.add_argument("--cov_eps_scale", type=float, default=1e-3)

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
    if len(fullnames) != 444:
        print(f"[WARN] PI fullname count is {len(fullnames)} (expected 444). Proceeding anyway.")

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
        fixed_win_s=float(args.fixed_win_s),
        fixed_stride_s=float(args.fixed_stride_s),
        min_segment_len_s=float(args.min_segment_len_s),
        # unused dynamic args are kept in cfg defaults
    )

    test_type = str(args.test_type)

    rows = df_lab[["client_id", "group_label_norm", "y"]].drop_duplicates().to_dict("records")
    if args.max_clients is not None:
        rows = rows[: int(args.max_clients)]

    summary = {
        "n_clients_cached": 0,
        "n_clients_written": 0,
        "n_skipped_no_files": 0,
        "n_failed": 0,
        "avg_segments": None,
        "fixed_win_s": cfg.fixed_win_s,
        "fixed_stride_s": cfg.fixed_stride_s,
    }
    seg_counts = []

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

        try:
            for f in files:
                X, ss, ee, beta_used = process_one_eyetracking_file(
                    csv_path=f,
                    pi_mod=pi_mod,
                    fullnames=fullnames,
                    cfg=cfg,
                    screen_w=float(args.screen_w),
                    screen_h=float(args.screen_h),
                )
                if X.shape[0] == 0:
                    continue

                all_X.append(X)
                all_ss.append(ss)
                all_ee.append(ee)
                file_meta.append({
                    "file": str(f),
                    "n_segments": int(X.shape[0]),
                    "beta_used": float(beta_used),  # always 0.0 for fixed segmentation
                })

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

    summary_path = out_dir / "preprocess_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Cache dir: {out_dir}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()