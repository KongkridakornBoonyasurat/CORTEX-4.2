import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt

print("="*80)
print("FINAL VALIDATION: Real Proxy Rest vs. Real Task")
print("="*80)

# ==== 1. INPUT: Path to your REAL Pac-Man data ====
# This path is correct, assuming your data is in this file.
PATH_TASK = r"C:\Users\User\Desktop\Brain AI\cortex 4.2\cortex 4.2 v42\pacman_run_20251013_103209\eeg_overall.fif"

# ==== 2. OUTPUT: Where to save the final plot ====
OUT_DIR = os.path.join(os.path.dirname(PATH_TASK), "final_validation_output")
os.makedirs(OUT_DIR, exist_ok=True)
FIG_PATH = os.path.join(OUT_DIR, "Figure9_FINAL_Alpha_Blocking.png")

# ==== 3. LOAD THE DATA ====
try:
    print(f"Loading task file: {PATH_TASK}")
    raw_task_full = mne.io.read_raw_fif(PATH_TASK, preload=True, verbose=False)
except Exception as e:
    print(f"--- ERROR ---")
    print(f"Could not load file: {e}")
    print("Please check the PATH_TASK variable is correct.")
    sys.exit(1)

# ==== 4. CREATE PROXY REST vs. TASK ====
print("Splitting file: First 1.0s (Rest) vs. Remainder (Task)")
T_REST = 2.0  # seconds
T_END = raw_task_full.times[-1]

if T_END <= T_REST:
    print(f"--- ERROR ---")
    print(f"File is too short ({T_END:.2f}s) to make a 1.0s proxy baseline.")
    sys.exit(1)

# ==== 5. PREPROCESS ONCE (full file), THEN SPLIT ====
print("Preprocessing full recording (reference + filter), then splitting...")
raw_proc = raw_task_full.copy()
raw_proc.set_eeg_reference('average', projection=False, verbose=False)
# Use IIR so the filter length is short (safe for short windows)
raw_proc.filter(1., 40., method='iir',
                iir_params=dict(order=4, ftype='butter'),
                verbose=False)

# Now split into proxy Rest and Task windows
TRANSIENT_TRIM = 0.25  # seconds
raw_rest = raw_proc.copy().crop(tmin=TRANSIENT_TRIM, tmax=T_REST)
raw_task = raw_proc.copy().crop(tmin=T_REST + TRANSIENT_TRIM, tmax=T_END)
print(f"  Rest (proxy): 0.0s – {T_REST}s")
print(f"  Task:         {T_REST}s – {T_END}s")

# ==== 6. COMPUTE PSDs (scales will now match) ====
print("Computing Power Spectral Densities (PSDs)...")

def mean_psd(rr):
    # Safe segment length: n_per_seg must be <= n_times
    nps = min(rr.n_times, 256)
    spec = rr.compute_psd(method='welch', fmin=1., fmax=40.,
                          n_per_seg=nps, n_overlap=nps//2, verbose=False)
    psds, freqs = spec.get_data(return_freqs=True)
    return freqs, psds.mean(axis=0)  # Average over channels

f_rest, psd_rest = mean_psd(raw_rest)
f_task, psd_task = mean_psd(raw_task)

# ==== 7. CALCULATE METRICS ====
def bandpow(f, p, lo, hi):
    m = (f >= lo) & (f < hi)
    if not m.any(): return 1e-12 # Avoid empty slice
    return float(np.mean(p[m])) # Use mean, not trapz, for noisy data

Pa_r = bandpow(f_rest, psd_rest, 8, 13)
Pa_t = bandpow(f_task, psd_task, 8, 13)
Pb_t = bandpow(f_task, psd_task, 13, 30)
Pg_t = bandpow(f_task, psd_task, 30, 40)

alpha_drop_pct = ((Pa_r - Pa_t) / Pa_r) * 100

print("\n" + "="*80)
print("FINAL VALIDATION RESULTS (Real Data vs. Real Data)")
print(f"  Resting Alpha (proxy): {Pa_r:.2e}")
print(f"  Task Alpha:            {Pa_t:.2e}")
print(f"  Alpha Suppression:     {alpha_drop_pct:.1f}%")
print(f"\n  Task Beta Power:       {Pb_t:.2e}")
print(f"  Task Gamma Power:      {Pg_t:.2e}")
print("="*80)

# ==== 8. PLOT THE FINAL FIGURE ====
print("Generating final plot...")
plt.figure(figsize=(12, 6), dpi=150)
plt.plot(f_rest, psd_rest, linewidth=2.5, label='Rest (proxy, first 1s)', color='blue')
plt.plot(f_task, psd_task, linewidth=2.5, label='Task (Pac-Man, remainder)', color='orange')

# Shade bands
plt.axvspan(8, 13, alpha=0.15, color='green', label='Alpha (8-13 Hz)')
plt.axvspan(13, 30, alpha=0.1, color='orange', label='Beta (13-30 Hz)')
plt.axvspan(30, 40, alpha=0.1, color='red', label='Gamma (30-40 Hz)')

plt.xlim(1, 40)
plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
plt.ylabel('PSD (μV²/Hz)', fontsize=14, fontweight='bold')
plt.title('Figure 9: EEG Validation - Proxy Rest vs. Task', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig(FIG_PATH, dpi=300)

print(f"\n SUCCESS! Final plot saved to:\n{FIG_PATH}")
print("="*80)
plt.show()