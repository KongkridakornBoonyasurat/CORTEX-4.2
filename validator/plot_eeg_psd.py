import mne
import matplotlib.pyplot as plt
import numpy as np

# Load your EEG file
print("Loading EEG data...")
raw = mne.io.read_raw_fif(
    r'C:\Users\User\Desktop\Brain AI\cortex 4.2\cortex 4.2 v42\pacman_run_20251013_103209\eeg_overall.fif',
    preload=True
)

print(f"Signal length: {raw.n_times} samples")
print(f"Duration: {raw.times[-1]:.2f} seconds")
print(f"Sampling rate: {raw.info['sfreq']} Hz")

# Filter the data
raw.filter(1, 40)

# Compute PSD - FIX: Use smaller n_fft that fits your data
print("Computing Power Spectral Density...")
spectrum = raw.compute_psd(method='welch', fmin=1, fmax=40, n_fft=256, n_per_seg=256)
psds, freqs = spectrum.get_data(return_freqs=True)

# Average across all channels
psd_mean = psds.mean(axis=0)

# Create professional plot
fig, ax = plt.subplots(figsize=(14, 7))

# Plot PSD
ax.plot(freqs, psd_mean, linewidth=3, color='#E63946', label='Task State (Pac-Man)', zorder=3)

# Mark frequency bands with shading
ax.axvspan(1, 4, alpha=0.1, color='purple', label='Delta (1-4 Hz)', zorder=0)
ax.axvspan(4, 8, alpha=0.1, color='blue', label='Theta (4-8 Hz)', zorder=0)
ax.axvspan(8, 13, alpha=0.15, color='green', label='Alpha (8-13 Hz)', zorder=0)
ax.axvspan(13, 30, alpha=0.1, color='orange', label='Beta (13-30 Hz)', zorder=0)
ax.axvspan(30, 40, alpha=0.1, color='red', label='Gamma (30-40 Hz)', zorder=0)

# Find peaks in Alpha band
alpha_idx = (freqs >= 8) & (freqs <= 13)
if alpha_idx.any():
    alpha_peak_freq = freqs[alpha_idx][np.argmax(psd_mean[alpha_idx])]
    alpha_peak_power = np.max(psd_mean[alpha_idx])
    ax.plot(alpha_peak_freq, alpha_peak_power, 'go', markersize=12, 
            markeredgewidth=2, markeredgecolor='white', zorder=4)
    ax.annotate(f'Alpha Peak\n{alpha_peak_freq:.1f} Hz', 
               xy=(alpha_peak_freq, alpha_peak_power),
               xytext=(alpha_peak_freq+3, alpha_peak_power*1.2),
               fontsize=11, fontweight='bold', color='green',
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Styling
ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
ax.set_ylabel('Power Spectral Density (μV²/Hz)', fontsize=14, fontweight='bold')
ax.set_title('Figure 9: Power Spectral Density - CORTEX 4.2 EEG During Pac-Man Task',
            fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right', frameon=True, shadow=True, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([1, 40])

# Add band power summary
delta_power = np.mean(psd_mean[(freqs >= 1) & (freqs <= 4)])
theta_power = np.mean(psd_mean[(freqs >= 4) & (freqs <= 8)])
alpha_power = np.mean(psd_mean[(freqs >= 8) & (freqs <= 13)])
beta_power = np.mean(psd_mean[(freqs >= 13) & (freqs <= 30)])
gamma_power = np.mean(psd_mean[(freqs >= 30) & (freqs <= 40)])

textstr = f'''Band Power Summary:
Delta (1-4 Hz):   {delta_power:.2e}
Theta (4-8 Hz):   {theta_power:.2e}
Alpha (8-13 Hz):  {alpha_power:.2e}
Beta (13-30 Hz):  {beta_power:.2e}
Gamma (30-40 Hz): {gamma_power:.2e}'''

props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, fontweight='bold', family='monospace')

plt.tight_layout()
plt.savefig('Figure9_PSD.png', dpi=300, bbox_inches='tight')
print("\n" + "="*80)
print("SUCCESS! Figure 9 saved as: Figure9_PSD.png")
print("="*80)
print(f"\nBand Power Analysis:")
print(f"  Delta:  {delta_power:.2e}")
print(f"  Theta:  {theta_power:.2e}")
print(f"  Alpha:  {alpha_power:.2e} ← Brain rhythm")
print(f"  Beta:   {beta_power:.2e}")
print(f"  Gamma:  {gamma_power:.2e}")
print("\n This validates emergent oscillatory dynamics!")
print("="*80)
plt.show()