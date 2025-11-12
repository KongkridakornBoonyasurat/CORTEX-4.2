import os
import argparse
import numpy as np

# headless plotting for PSD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne


def _load_region_activity(path):
    # ANCHOR:LOAD-REGION
    region = np.load(path)
    if region.ndim != 2:
        raise ValueError(f"Expected 2D array (T x R). Got shape {region.shape}")
    return region.astype(float)


def _load_leadfield(path, R_expected):
    # ANCHOR:LOAD-LEADFIELD
    L = np.load(path)
    if L.ndim != 2:
        raise ValueError(f"leadfield must be 2D (E x R). Got shape {L.shape}")
    if L.shape[1] != R_expected:
        raise ValueError(f"leadfield has R={L.shape[1]} but region has R={R_expected}")
    return L.astype(float)


def _make_channel_names(E):
    # ANCHOR:CHANNEL-NAMES
    std_1020 = [
        'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
        'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz','Oz'
    ]
    if E <= len(std_1020):
        return std_1020[:E]
    extra = [f'EEG{i+1}' for i in range(E - len(std_1020))]
    return std_1020 + extra


def _build_raw_from_leadfield(region, L, sfreq, out_prefix, try_montage=True):
    # ANCHOR:OFFLINE-PATH-LEADFIELD
    # region: (T, R), L: (E, R)
    # Map region activity -> EEG
    a_centered = region - 1.0                          # remove 1.0 baseline
    eeg = (L @ a_centered.T)                           # (E, T)
    eeg = eeg - eeg.mean(axis=0, keepdims=True)        # common average reference
    eeg_uv = np.clip(eeg, -150.0, 150.0)               # microvolts
    data_V = eeg_uv * 1e-6                             # µV -> V
    E = data_V.shape[0]
    ch_names = _make_channel_names(E)
    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types='eeg')

    if try_montage:
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            # Let MNE handle missing channels gracefully
            info.set_montage(montage, match_case=False, on_missing='ignore')
        except Exception as e:
            print(f"[warn] montage attach skipped: {e}")

    raw = mne.io.RawArray(data_V, info)
    raw_path = f"{out_prefix}_raw.fif"
    raw.save(raw_path, overwrite=True)

    # PSD figure (optional)
    try:
        fig = raw.plot_psd(show=False)
        fig.savefig(f"{out_prefix}_psd.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] PSD plot failed: {e}")

    print(f"[OK] Saved Raw -> {raw_path}")
    return raw_path


def _build_raw_direct(region, sfreq, out_prefix, try_montage=True):
    # ANCHOR:OFFLINE-PATH-DIRECT
    # Fallback if no leadfield is given: treat each region as an EEG channel
    # Region names in your order (best-effort labels)
    region_names = [
        'motor','sensory','thalamus','cerebellum',
        'parietal','pfc','limbic','hippocampus','insula','basal_ganglia'
    ]
    T, R = region.shape
    ch_names = [region_names[i] if i < len(region_names) else f"R{i}" for i in range(R)]

    # Center and scale to a plausible microvolt range
    a_centered = region - 1.0
    # Scale each region roughly to ~±50 µV and CAR
    eeg_uv = a_centered.T * 50.0                        # (R, T), µV
    eeg_uv = eeg_uv - eeg_uv.mean(axis=0, keepdims=True)
    data_V = eeg_uv * 1e-6

    info = mne.create_info(ch_names=ch_names, sfreq=float(sfreq), ch_types='eeg')

    if try_montage:
        # We don’t have anatomical mapping here; skip montage
        pass

    raw = mne.io.RawArray(data_V, info)
    raw_path = f"{out_prefix}_raw.fif"
    raw.save(raw_path, overwrite=True)

    try:
        fig = raw.plot_psd(show=False)
        fig.savefig(f"{out_prefix}_psd.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] PSD plot failed: {e}")

    print(f"[OK] Saved Raw (direct region channels) -> {raw_path}")
    return raw_path


def main():
    # ANCHOR:ARGPARSE
    ap = argparse.ArgumentParser("Bridge Pacman region activity -> MNE Raw (offline)")
    ap.add_argument("--region-npy", required=True, help="Path to overall_region_activity.npy (T x 10)")
    ap.add_argument("--leadfield-npy", default="", help="Optional path to leadfield.npy (E x 10). If given, use L @ region.")
    ap.add_argument("--sfreq", type=float, default=250.0, help="Sampling frequency (Hz)")
    ap.add_argument("--out-prefix", default="pacman", help="Output filename prefix")
    args = ap.parse_args()

    mne.set_log_level("WARNING")

    # Load region activity
    region = _load_region_activity(args.region_npy)     # (T, R)
    T, R = region.shape
    print(f"[INFO] region shape = {region.shape} (T x R)")

    # If leadfield provided, use it; else direct channels
    if args.leadfield_npy:
        L = _load_leadfield(args.leadfield_npy, R_expected=R)  # (E, R)
        print(f"[INFO] leadfield shape = {L.shape} (E x R)")
        _build_raw_from_leadfield(region, L, args.sfreq, args.out_prefix, try_montage=True)
    else:
        print("[INFO] No leadfield provided. Building Raw directly from region channels.")
        _build_raw_direct(region, args.sfreq, args.out_prefix, try_montage=False)


if __name__ == "__main__":
    # ANCHOR:MAIN-GUARD
    main()
