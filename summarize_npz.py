from pathlib import Path
import numpy as np

cache_dir = Path("/data_248/pdss/primitive_indicator_scripts/scripts/intern/dynamic_segmentation/preprocessed_output/dnb_57")

files = sorted(cache_dir.glob("*.npz"))
print("num npz:", len(files))

bad = []
seg_counts = []
dims = set()

for f in files:
    d = np.load(f, allow_pickle=True)
    X = d["X"]
    ss = d["seg_start_s"]
    ee = d["seg_end_s"]

    dims.add(X.shape[1] if X.ndim == 2 and X.shape[0] >= 0 else None)
    seg_counts.append(int(X.shape[0]))

    ok = True
    if X.ndim != 2:
        ok = False
    if X.shape[0] != len(ss) or X.shape[0] != len(ee):
        ok = False
    if len(ss) > 1:
        if not (np.all(np.diff(ss) >= 0) and np.all(ee >= ss)):
            ok = False
    if X.size and np.isfinite(X).mean() < 0.5:
        ok = False

    if not ok:
        bad.append(f.name)

print("feature dims found:", dims)
print("segments: min/mean/max =", min(seg_counts, default=0),
      sum(seg_counts)/len(seg_counts) if seg_counts else 0,
      max(seg_counts, default=0))
print("bad files:", len(bad))
if bad[:20]:
    print("examples:", bad[:20])