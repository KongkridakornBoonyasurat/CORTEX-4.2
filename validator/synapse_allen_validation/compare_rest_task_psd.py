import mne, numpy as np, matplotlib.pyplot as plt, argparse, os

def load_psd(path):
    # Allow giving the folder; auto-pick eeg_overall.fif inside it
    if os.path.isdir(path):
        path = os.path.join(path, "eeg_overall.fif")
    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    raw.filter(1., 40., verbose=False)
    nps = min(raw.n_times, 256)
    spec = raw.compute_psd(method="welch", fmin=1., fmax=40.,
                           n_per_seg=nps, n_overlap=nps//2, verbose=False)
    psd, f = spec.get_data(return_freqs=True)
    return f, psd.mean(axis=0)

def bandpow(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.mean(p[m])) if m.any() else 1e-12

ap = argparse.ArgumentParser("Rest vs Task PSD")
ap.add_argument("--rest", required=True, help="baseline eeg_overall.fif or its folder")
ap.add_argument("--task", required=True, help="task eeg_overall.fif or its folder")
ap.add_argument("--out",  default="Figure9_Rest_vs_Task.png")
args = ap.parse_args()

f_r, p_r = load_psd(args.rest)
f_t, p_t = load_psd(args.task)

Pa_r = bandpow(f_r, p_r, 8, 13)
Pa_t = bandpow(f_t, p_t, 8, 13)
Pb_t = bandpow(f_t, p_t, 13, 30)
Pg_t = bandpow(f_t, p_t, 30, 40)
alpha_drop_pct = ((Pa_r - Pa_t) / Pa_r) * 100.0

print("="*80)
print("Alpha-blocking check (Rest → Task)")
print(f"  Rest alpha  : {Pa_r:.2e}")
print(f"  Task alpha  : {Pa_t:.2e}")
print(f"  Suppression : {alpha_drop_pct:.1f}%")
print(f"  Task beta   : {Pb_t:.2e}")
print(f"  Task gamma  : {Pg_t:.2e}")
print("="*80)

plt.figure(figsize=(12,6), dpi=150)
plt.plot(f_r, p_r, label="Rest (baseline)", linewidth=2.5, color="tab:blue")
plt.plot(f_t, p_t, label="Task (Pac-Man)", linewidth=2.5, color="tab:orange")

plt.axvspan(8, 13,  color="green",  alpha=0.15, label="Alpha (8–13 Hz)")
plt.axvspan(13, 30, color="orange", alpha=0.10, label="Beta (13–30 Hz)")
plt.axvspan(30, 40, color="red",    alpha=0.10, label="Gamma (30–40 Hz)")

plt.xlim(1, 40)
plt.xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
plt.ylabel("PSD (μV²/Hz)",   fontsize=14, fontweight="bold")
plt.title("Figure 9: Rest vs Task PSD (Alpha-Blocking Test)", fontsize=16, fontweight="bold")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(args.out, dpi=300)
print(f"\nSaved: {args.out}")
