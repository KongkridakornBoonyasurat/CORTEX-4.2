# reader.py — Concatenate dopamine traces across episodes and annotate "food" events
# Usage:
#   1) Set run_dir to your snake_run_* folder
#   2) python reader.py
# Notes:
#   - If files named eat_epXXX.npy exist, they are used for exact food markers.
#   - Otherwise we auto-detect dopamine spikes as a proxy.
#   - Annotated PNG is saved in the run_dir.

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

# ========= CONFIG =========
run_dir = Path(r"C:\Users\User\Desktop\Brain AI\cortex 4.2\cortex 4.2 v29\snake_run_20250928_015056")
file_glob = "dopamine_ep*.npy"   # per-episode dopamine traces
eat_glob  = "eat_ep*.npy"        # optional: per-episode boolean arrays

gap_steps = 0
smooth_k = 0
boundary_every = 5
spike_threshold = 0.80
min_separation = 3

# ========= HELPERS =========
pat_dop = re.compile(r"dopamine_ep(\d+)\.npy")
pat_eat = re.compile(r"eat_ep(\d+)\.npy")

def ep_idx(name: str, pat) -> int:
    m = pat.search(name)
    return int(m.group(1)) if m else -1

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1:
        return x
    k = int(k)
    w = np.ones(k) / k
    return np.convolve(x, w, mode="same")

def detect_peaks(sig: np.ndarray, thr: float, sep: int) -> np.ndarray:
    sig = np.asarray(sig, float)
    peaks = []
    last_p = -1e9
    for i in range(1, sig.size-1):
        if sig[i] > thr and sig[i] >= sig[i-1] and sig[i] >= sig[i+1]:
            if i - last_p >= sep:
                peaks.append(i)
                last_p = i
    return np.array(peaks, dtype=int)

# ========= LOAD =========
files = sorted(run_dir.glob(file_glob), key=lambda p: ep_idx(p.name, pat_dop))
if not files:
    raise SystemExit(f"No files matching {file_glob} in: {run_dir}")

eat_files = {ep_idx(p.name, pat_eat): p for p in run_dir.glob(eat_glob)}

series, ep_ids, eat_markers = [], [], []

for f in files:
    ep = ep_idx(f.name, pat_dop)
    arr = np.load(f).astype(float).squeeze()
    arr_sm = moving_average(arr, smooth_k)
    series.append(arr_sm)
    ep_ids.append(ep)

    if ep in eat_files:
        eat_vec = np.load(eat_files[ep]).astype(bool).squeeze()
        if eat_vec.size == arr.size:
            eat_markers.append(np.flatnonzero(eat_vec))
        else:
            eat_markers.append(detect_peaks(arr_sm, spike_threshold, min_separation))
    else:
        eat_markers.append(detect_peaks(arr_sm, spike_threshold, min_separation))

# ========= CONCAT =========
parts, offsets = [], [0]
for i, a in enumerate(series):
    if i > 0 and gap_steps > 0:
        parts.append(np.full(gap_steps, np.nan))
    parts.append(a)
    offsets.append(offsets[-1] + len(a) + (gap_steps if i > 0 else 0))

long_signal = np.concatenate(parts)
x = np.arange(len(long_signal))

global_eats = []
for i, peaks in enumerate(eat_markers):
    start = offsets[i]
    if gap_steps > 0 and i > 0:
        start += gap_steps
    global_eats.extend((start + peaks).tolist())
global_eats = np.array(global_eats, dtype=int)

# ========= PLOT =========
plt.figure(figsize=(14,6))
plt.plot(x, long_signal, lw=1, label="Dopamine")

valid = (global_eats >= 0) & (global_eats < len(long_signal))
if valid.any():
    plt.scatter(global_eats[valid], long_signal[global_eats[valid]],
                s=30, c="crimson", marker="v", alpha=0.9, label="Food events")

plt.title("Dopamine per step — episodes concatenated end-to-end")
plt.xlabel("Global step")
plt.ylabel("Dopamine")
plt.grid(True, alpha=0.25)

ymax = np.nanmax(long_signal) if np.isfinite(long_signal).any() else 1.0
for i, ep in enumerate(ep_ids):
    if i == 0: continue
    start = offsets[i]
    plt.axvline(start, color="k", lw=0.8, alpha=0.25)
    if boundary_every == 1 or (ep % boundary_every) == 0:
        plt.text(start, ymax, f"ep{ep:03d}", rotation=90,
                 va="bottom", ha="right", fontsize=8, alpha=0.7)

plt.legend(loc="upper right")
plt.tight_layout()

out_png = run_dir / "dopamine_concat_annotated.png"
plt.savefig(out_png, dpi=160)
print("Saved:", out_png)

plt.show()
