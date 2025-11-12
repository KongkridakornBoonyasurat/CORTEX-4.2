#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG → 3D brain regional activity (MNE-Python, fsaverage template).

What this does
--------------
1) Loads an EEG recording (EDF/BDF/FIF/BrainVision .vhdr) or generates synthetic EEG.
2) Attaches a standard montage (10-20 by default) and does light filtering.
3) Builds an EEG forward model on the FreeSurfer "fsaverage" template (downloaded automatically).
4) Estimates source activity with a minimum-norm inverse (MNE/dSPM/sLORETA).
5) Aggregates source activity per cortical region (Desikan-Killiany "aparc" by default).
6) Plots a 3D brain with per-region activity and optionally the full time-resolved source map.
7) Exports a CSV of region activity values.

Requirements
------------
pip install mne pyvistaqt nibabel scipy numpy
# On some systems you may also need: pip install PyQt5

Example usage
-------------
# Run with a real EEG file (EDF) and compute alpha-band power per region:
python mne_eeg_3d_regions.py --eeg your_recording.edf --band 8 12 --method dSPM

# Without a file, it will synthesize 60 s of EEG and still show the full pipeline:
python mne_eeg_3d_regions.py --duration 60

# Save static PNGs instead of opening interactive windows:
python mne_eeg_3d_regions.py --eeg your_recording.edf --save-figs

Notes
-----
- This uses the fsaverage template MRI, so it's "good enough" to demo without a subject-specific
  MRI and head<->sensor coregistration. For publication-quality results, use your own MRI, BEM,
  and a measured head<->sensor transform (.trans). See MNE tutorials for details.
"""

import argparse
import os
import sys
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_raw
from scipy.signal import welch
from pathlib import Path

def read_raw_auto(path, *, preload=True):
    path = str(path)
    lower = path.lower()
    if lower.endswith('.edf') or lower.endswith('.bdf') or lower.endswith('.gdf'):
        return mne.io.read_raw_edf(path, preload=preload, stim_channel=None)
    if lower.endswith('.fif') or lower.endswith('.fif.gz'):
        return mne.io.read_raw_fif(path, preload=preload)
    if lower.endswith('.vhdr'):
        return mne.io.read_raw_brainvision(path, preload=preload)
    raise ValueError(f"Unsupported EEG file format: {path}")

def ensure_fsaverage(subjects_dir=None):
    """Download fsaverage and return (subjects_dir, fs_dir, bem_fname, src_fname)."""
    fs_dir = fetch_fsaverage(verbose=True)  # returns .../subjects/fsaverage
    fs_dir = Path(fs_dir)
    subj_dir = fs_dir.parent
    os.environ['SUBJECTS_DIR'] = str(subj_dir)

    # BEM and src files distributed with fsaverage dataset
    bem_dir = fs_dir / 'bem'
    # pick a bem solution if available
    bem_candidates = sorted([p for p in bem_dir.glob('*bem-sol.fif')])
    if not bem_candidates:
        raise RuntimeError("No BEM solution found in fsaverage. Check your MNE/data install.")
    bem_fname = str(bem_candidates[0])

    # pick a source space (ico-5 is typically present)
    src_candidates = sorted([p for p in bem_dir.glob('*-ico-5-src.fif')])
    if not src_candidates:
        # fallback: any src
        src_candidates = sorted([p for p in bem_dir.glob('*-src.fif')])
    if not src_candidates:
        raise RuntimeError("No source space found in fsaverage. Check your MNE/data install.")
    src_fname = str(src_candidates[0])
    return str(subj_dir), str(fs_dir), bem_fname, src_fname

def make_evoked_from_raw(raw, epoch_len=2.0, reject_by_annotation=True):
    """Create simple fixed-length epochs and average to get an Evoked; also compute noise covariance."""
    events = mne.make_fixed_length_events(raw, duration=epoch_len)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=epoch_len, proj=True, baseline=None,
                        picks=picks, preload=True, reject_by_annotation=reject_by_annotation)
    # Estimate noise covariance across epochs (robust auto method)
    noise_cov = mne.compute_covariance(epochs, method='auto', rank='info')
    evoked = epochs.average()
    return evoked, noise_cov

def compute_inverse(evoked, noise_cov, src_fname, bem_fname, method='dSPM', snr=3.0):
    """Build forward/inverse on fsaverage and return (fwd, inv, stc_evoked)."""
    # Build forward model on fsaverage template
    fwd = mne.make_forward_solution(evoked.info, trans='fsaverage', src=src_fname, bem=bem_fname,
                                    eeg=True, meg=False, mindist=5.0, n_jobs=1, verbose=True)
    inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose='auto', depth=0.8, verbose=True)
    lambda2 = 1.0 / (snr ** 2)
    stc = apply_inverse(evoked, inv, lambda2=lambda2, method=method, pick_ori=None, verbose=True)
    return fwd, inv, stc

def band_power(x, fs, fmin, fmax):
    """Return Welch PSD mean power in [fmin, fmax]."""
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return 0.0
    return float(Pxx[mask].mean())

def label_aggregate(stc, src_fname, parc='aparc', mode='mean', band=None, info=None):
    """
    Aggregate vertex-wise source estimates to cortical labels.
    If `band` is a tuple (fmin, fmax) and `info` is provided, compute band power per label.
    Otherwise return time-domain RMS per label.
    """
    src = mne.read_source_spaces(src_fname, verbose=False)
    labels = mne.read_labels_from_annot('fsaverage', parc=parc, subjects_dir=os.environ['SUBJECTS_DIR'])
    label_ts = mne.extract_label_time_course(stc, labels, src, mode=mode, return_generator=False)
    # label_ts shape: (n_labels, n_times)
    if band is not None and info is not None:
        fs = info['sfreq']
        vals = np.array([band_power(label_ts[i, :], fs, band[0], band[1]) for i in range(label_ts.shape[0])])
        metric_name = f"band_power_{band[0]}-{band[1]}Hz"
    else:
        # RMS across time
        vals = np.sqrt((label_ts ** 2).mean(axis=1))
        metric_name = "rms_time"
    # Return labels and their values
    return labels, vals, metric_name

def labels_to_static_stc(labels, values, src_fname):
    """Map per-label scalar values back onto the cortical surface as a static STC (one time point)."""
    src = mne.read_source_spaces(src_fname, verbose=False)
    lh_n = src[0]['nuse']
    rh_n = src[1]['nuse']
    lh_data = np.zeros(lh_n, float)
    rh_data = np.zeros(rh_n, float)

    # For each label, assign its scalar to all vertices inside the label (restricted to src)
    for lab, val in zip(labels, values):
        hemi_idx = 0 if lab.hemi == 'lh' else 1
        vertno = src[hemi_idx]['vertno']
        # intersect label vertices with src vertices to find indices
        verts = np.intersect1d(lab.vertices, vertno, assume_unique=False)
        if verts.size == 0:
            continue
        # map to indices in src vertex numbering
        idx = np.searchsorted(vertno, verts)
        if lab.hemi == 'lh':
            lh_data[idx] = val
        else:
            rh_data[idx] = val

    data = np.vstack([lh_data[:, None], rh_data[:, None]])  # (n_vertices_total, 1)
    stc_static = mne.SourceEstimate(data=data,
                                    vertices=[src[0]['vertno'], src[1]['vertno']],
                                    tmin=0.0, tstep=1.0, subject='fsaverage')
    return stc_static

def plot_3d(stc_static, subjects_dir, surface='inflated', clim='auto', title='Regional activity (static)',
            view_sulc=True, add_parc='aparc', save_figs=False, out_prefix='brain_plot'):
    """Plot the static label-valued STC on a 3D brain (PyVista). Optionally save PNGs."""
    brain = stc_static.plot(subject='fsaverage', hemi='both', surface=surface, subjects_dir=subjects_dir,
                            time_viewer=False, colormap='hot', smoothing_steps=5, clim=clim)
    if view_sulc:
        brain.add_sulc()
    if add_parc is not None:
        brain.add_annotation(add_parc)
    brain.add_text(0.01, 0.95, title, 'title', font_size=14)
    if save_figs:
        # Save a couple of canonical views
        for view in ['lateral', 'medial', 'dorsal', 'ventral']:
            brain.show_view(view)
            brain.save_image(f"{out_prefix}_{view}.png")
        print(f"Saved figures with prefix: {out_prefix}_*.png")
        brain.close()

def maybe_make_synthetic_raw(duration=60.0, sfreq=256.0, montage='standard_1020'):
    """Create a simple synthetic EEG Raw with alpha at occipital and beta at frontal sensors."""
    ch_names = mne.channels.make_standard_montage(montage).ch_names
    # Keep a reasonable subset of EEG channels
    ch_names = [c for c in ch_names if c.upper() in {'FP1','FP2','F3','F4','F7','F8','FZ','C3','C4','CZ','P3','P4','PZ','O1','O2'}]
    n_ch = len(ch_names)
    n_samp = int(duration * sfreq)
    times = np.arange(n_samp) / sfreq

    rng = np.random.RandomState(42)
    data = 1e-6 * rng.randn(n_ch, n_samp)  # baseline noise, in Volts (MNE expects SI)

    # Add alpha 10 Hz stronger at O1/O2/Pz
    alpha = np.sin(2 * np.pi * 10.0 * times)
    for idx, name in enumerate(ch_names):
        if name.upper() in {'O1','O2','PZ'}:
            data[idx] += 4e-6 * alpha
        if name.upper() in {'FP1','FP2','FZ'}:
            # add a small 20 Hz beta at frontal
            data[idx] += 2e-6 * np.sin(2 * np.pi * 20.0 * times)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    raw.set_montage(mne.channels.make_standard_montage(montage))
    return raw

def main():
    parser = argparse.ArgumentParser(description="EEG → 3D brain regional activity with MNE (fsaverage).")
    parser.add_argument('--eeg', type=str, default=None,
                        help='Path to EEG file (.edf/.bdf/.fif/.vhdr). If omitted, synthetic EEG is generated.')
    parser.add_argument('--montage', type=str, default='standard_1020',
                        help='Sensor montage name (e.g., standard_1020, standard_1005, EEG1005).')
    parser.add_argument('--l_freq', type=float, default=1.0, help='High-pass cutoff (Hz).')
    parser.add_argument('--h_freq', type=float, default=40.0, help='Low-pass cutoff (Hz).')
    parser.add_argument('--epoch_len', type=float, default=2.0, help='Fixed-length epoch length (s).')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='If synthesizing EEG, how many seconds to simulate.')
    parser.add_argument('--parc', type=str, default='aparc',
                        help='Cortical parcellation (aparc, aparc.a2009s, HCPMMP1, etc. if available).')
    parser.add_argument('--method', type=str, default='dSPM', choices=['MNE', 'dSPM', 'sLORETA'],
                        help='Inverse method.')
    parser.add_argument('--snr', type=float, default=3.0, help='Assumed SNR for inverse (lambda2 = 1/SNR^2).')
    parser.add_argument('--band', type=float, nargs=2, default=None,
                        help='If set, compute band power per region, e.g., --band 8 12 for alpha.')
    parser.add_argument('--save-figs', action='store_true', help='Save static PNGs instead of interactive viewer.')
    parser.add_argument('--out-csv', type=str, default='region_activity.csv', help='CSV filename for region metrics.')
    args = parser.parse_args()

    # Load EEG
    if args.eeg is None:
        print("No EEG provided; generating synthetic EEG...")
        raw = maybe_make_synthetic_raw(duration=args.duration, montage=args.montage)
    else:
        raw = read_raw_auto(args.eeg, preload=True)
        # Attach montage if not already present
        if raw.get_montage() is None:
            raw.set_montage(mne.channels.make_standard_montage(args.montage))

    # Basic preprocessing
    raw.load_data()
    raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, fir_design='firwin', phase='zero', verbose=True)
    raw.set_eeg_reference('average', projection=True)  # average reference (as proj)

    # Prepare fsaverage anatomy, BEM, and source space
    subjects_dir, fs_dir, bem_fname, src_fname = ensure_fsaverage()

    # Make Evoked and noise covariance
    evoked, noise_cov = make_evoked_from_raw(raw, epoch_len=args.epoch_len)

    # Compute inverse solution
    fwd, inv, stc_evoked = compute_inverse(evoked, noise_cov, src_fname, bem_fname,
                                           method=args.method, snr=args.snr)

    # Aggregate per label
    labels, values, metric_name = label_aggregate(stc_evoked, src_fname, parc=args.parc,
                                                  mode='mean', band=tuple(args.band) if args.band else None,
                                                  info=evoked.info)

    # Save CSV
    import csv
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['label_name', metric_name])
        for lab, val in zip(labels, values):
            w.writerow([lab.name, f"{val:.6e}"])
    print(f"Wrote regional metrics to: {args.out_csv} [{metric_name}]")

    # Map label values back to cortex for static 3D map
    stc_static = labels_to_static_stc(labels, values, src_fname)

    # Plot static regional map
    plot_3d(stc_static, subjects_dir=subjects_dir,
            title=f"Regional activity: {metric_name} ({args.parc})",
            save_figs=args.save_figs, out_prefix=Path(args.out_csv).stem)

    # Optionally, also show the full time-resolved STC viewer (if not saving figs)
    if not args.save_figs:
        print("Opening interactive time viewer for evoked STC... (close window to finish)")
        # This displays the full vertex-wise time course; you can scrub time.
        _ = stc_evoked.plot(subject='fsaverage', hemi='both', subjects_dir=subjects_dir,
                            surface='inflated', time_viewer=True, colormap='hot', smoothing_steps=5)

    print("Done.")

if __name__ == '__main__':
    main()
