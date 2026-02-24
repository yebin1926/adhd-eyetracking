from pathlib import Path
import argparse
import numpy as np
"""
Example Usage:
python summarize_npz.py --cache_dir "/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/dnb_filtered" --print_per_student

"""


def text_hist(values, bins, name):
    """Print a simple text histogram using numpy histogram counts."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"{name}: (no finite values)")
        return
    counts, edges = np.histogram(values, bins=bins)
    total = counts.sum()
    print(f"\n{name} histogram (n={total}):")
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        c = int(counts[i])
        pct = (c / total * 100.0) if total else 0.0
        bar = "#" * min(60, int(round(pct)))  # cap bar length
        print(f"  [{lo:6.3f}, {hi:6.3f}) : {c:5d} ({pct:6.2f}%) {bar}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/dnb_nan_92",
        help="Directory containing per-student .npz files",
    )
    parser.add_argument(
        "--print_per_student",
        action="store_true",
        help="Print per-student line: file, segments, nan_ratio, zero_ratio",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="How many worst files to show for nan/zero ratios",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    files = sorted(cache_dir.glob("*.npz"))
    print("cache_dir:", str(cache_dir))
    print("num npz:", len(files))

    bad = []
    seg_counts = []
    dims = set()

    per_nan = []   # nan ratio per file
    per_zero = []  # zero ratio per file (after NaN->0)
    names = []

    for f in files:
        d = np.load(f, allow_pickle=True)
        X = d["X"]
        ss = d["seg_start_s"]
        ee = d["seg_end_s"]

        # feature dim check (expects [num_segments, num_features])
        dims.add(X.shape[1] if X.ndim == 2 else None)
        seg_counts.append(int(X.shape[0]) if X.ndim >= 1 else 0)

        # --- your requested checks ---
        # NaN ratio in X (raw)
        nan_ratio = float(np.isnan(X).mean()) if X.size else 0.0

        # Zero ratio after your typical pipeline behavior (NaN -> 0)
        X0 = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) if X.size else X
        zero_ratio = float((X0 == 0).mean()) if X0.size else 0.0

        per_nan.append(nan_ratio)
        per_zero.append(zero_ratio)
        names.append(f.name)

        if args.print_per_student:
            print(f"{f.name}\tsegments={seg_counts[-1]}\tnan_ratio={nan_ratio:.4f}\tzero_ratio={zero_ratio:.4f}")

        # --- existing integrity checks (kept) ---
        ok = True
        if X.ndim != 2:
            ok = False
        if X.shape[0] != len(ss) or X.shape[0] != len(ee):
            ok = False
        if len(ss) > 1:
            if not (np.all(np.diff(ss) >= 0) and np.all(ee >= ss)):
                ok = False

        # keep your original "finite mean" check, but make it robust to NaNs
        if X.size:
            finite_frac = float(np.isfinite(X).mean())
            if finite_frac < 0.5:
                ok = False

        if not ok:
            bad.append(f.name)

    # --- existing summary (kept) ---
    print("\nfeature dims found:", dims)
    print(
        "segments: min/mean/max =",
        min(seg_counts, default=0),
        (sum(seg_counts) / len(seg_counts)) if seg_counts else 0,
        max(seg_counts, default=0),
    )
    print("bad files:", len(bad))
    if bad[:20]:
        print("examples:", bad[:20])

    # --- new summaries requested ---
    per_nan_arr = np.asarray(per_nan, dtype=float)
    per_zero_arr = np.asarray(per_zero, dtype=float)

    def safe_stats(arr):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return {
            "min": float(arr.min()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(arr.max()),
        }

    nan_stats = safe_stats(per_nan_arr)
    zero_stats = safe_stats(per_zero_arr)

    print("\nNaN ratio stats:", nan_stats)
    print("Zero ratio stats (after NaN->0):", zero_stats)

    # histograms (text bins)
    text_hist(per_nan_arr, bins=[0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0000001], name="nan_ratio")
    text_hist(per_zero_arr, bins=[0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0000001], name="zero_ratio")

    # show worst offenders
    top_k = max(0, int(args.top_k))
    if top_k > 0 and len(names) > 0:
        idx_nan = np.argsort(-per_nan_arr)  # descending
        idx_zero = np.argsort(-per_zero_arr)
        print(f"\nTop {min(top_k, len(names))} highest nan_ratio files:")
        for i in idx_nan[:top_k]:
            print(f"  {names[i]}  segments={seg_counts[i]}  nan_ratio={per_nan_arr[i]:.4f}  zero_ratio={per_zero_arr[i]:.4f}")

        print(f"\nTop {min(top_k, len(names))} highest zero_ratio files:")
        for i in idx_zero[:top_k]:
            print(f"  {names[i]}  segments={seg_counts[i]}  nan_ratio={per_nan_arr[i]:.4f}  zero_ratio={per_zero_arr[i]:.4f}")

    # quick diagnosis
    # (Heuristic thresholds: adjust later if needed)
    med_nan = nan_stats["median"] if nan_stats else 0.0
    med_zero = zero_stats["median"] if zero_stats else 0.0
    p95_zero = zero_stats["p95"] if zero_stats else 0.0

    print("\n=== Diagnosis (heuristic) ===")
    if med_nan > 0.2:
        print(f"- Median NaN ratio is high ({med_nan:.3f}). Many PI features are missing → model may see near-empty inputs.")
    else:
        print(f"- Median NaN ratio is not extremely high ({med_nan:.3f}).")

    if med_zero > 0.7 or p95_zero > 0.9:
        print(f"- Zero ratio is very high (median={med_zero:.3f}, p95={p95_zero:.3f}) after NaN->0.")
        print("  This strongly suggests feature collapse (most values become 0), which can cause the model to default to one class.")
    else:
        print(f"- Zero ratio is not extremely high (median={med_zero:.3f}, p95={p95_zero:.3f}).")
        print("  If collapse still happens, suspect class imbalance, label noise, or train setup (loss/weights/sampler) rather than missing/zero features alone.")

    print("=== End ===")


if __name__ == "__main__":
    main()