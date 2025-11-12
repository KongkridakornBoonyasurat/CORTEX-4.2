#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG → 3D brain regional activity (MNE-Python, fsaverage template).
Stable for Windows/Anaconda + VS Code; supports interactive/off-screen, safe montage,
sphere alignment preview, and inverse-safe EEG referencing.
"""

import argparse
import os
os.environ["MNE_3D_BACKEND"] = "pyvistaqt"
# Try native OpenGL first; if the window still misbehaves, switch to "angle"
os.environ.setdefault("QT_OPENGL", "desktop")

# Tame DPI scaling & geometry
os.environ["QT_OPENGL"] = "angle"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = "1"
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse
from scipy.signal import welch
from pathlib import Path
import pyvista as pv

pv.global_theme.full_screen = False
pv.global_theme.window_size = (1000, 700)   

# Backend defaults (conservative to reduce GPU/driver crashes on Windows)
try:
    mne.viz.set_3d_backend('pyvistaqt')
    mne.viz.set_3d_options(antialias=False, smooth_shading=False)
except Exception:
    pass


def open_interactive_safely(stc, subjects_dir):
    # interactive viewer
    brain = stc.plot(
        subject="fsaverage", hemi="both", subjects_dir=subjects_dir,
        surface="inflated", cortex="low_contrast",
        time_viewer=True, show_traces=True
    )

    # Grab the actual Qt window and clamp size/position
    try:
        # MNE → Brain → internal renderer → PyVista plotter
        plotter = brain._renderer.plotter
        # PyVistaQt exposes the Qt main window here:
        app_win = getattr(plotter, "app_window", None)

        if app_win is not None:
            # Ensure a sane minimum, then resize & move onto primary screen
            app_win.setMinimumSize(800, 600)
            app_win.resize(1100, 800)
            app_win.move(100, 100)  # top-left corner (x,y) on \\.\DISPLAY1
        else:
            # Fallback: use PyVista's API if app_window isn't present
            try:
                plotter.window_size = (1100, 800)
            except Exception:
                pass
    except Exception as e:
        print("[warn] Could not adjust window geometry:", e)

    print("[ok] Interactive viewer opened. Press 'h' for help/shortcuts.")
    return brain

# ──────────────── Montage handling ────────────────
def attach_montage_safely(raw, montage_name="standard_1020"):
    """
    Attach a montage robustly:
      - case-insensitive name match
      - rename to montage's canonical casing (FP1→Fp1, FZ→Fz, etc.)
      - force kept channels to EEG type
      - drop channels missing from the montage
    """
    mont = mne.channels.make_standard_montage(montage_name)
    mont_upper_to_canon = {ch.upper(): ch for ch in mont.ch_names}

    file_chs = raw.info["ch_names"]
    file_upper = [c.upper() for c in file_chs]
    common_upper = sorted(set(file_upper).intersection(mont_upper_to_canon.keys()))
    if not common_upper:
        raise RuntimeError(
            f"No channel names match the {montage_name} montage.\n"
            f"First 10 file channels: {file_chs[:10]}"
        )

    rename_map, keep_names = {}, []
    for cu in common_upper:
        file_name = file_chs[file_upper.index(cu)]
        canon_name = mont_upper_to_canon[cu]
        rename_map[file_name] = canon_name
        keep_names.append(file_name)

    if len(keep_names) < len(file_chs):
        missing = sorted(set(file_chs) - set(keep_names))
        print(f"[warn] Dropping channels without positions in {montage_name}: {missing}")
        raw.pick(keep_names)

    raw.rename_channels(rename_map)
    types_map = {name: "eeg" for name in raw.info["ch_names"]}
    raw.set_channel_types(types_map)
    raw.set_montage(mont, match_case=True, on_missing="ignore")
    return raw


# ──────────────── Inverse-safe EEG reference ────────────────
def ensure_inverse_safe_reference(raw):
    """
    If a custom EEG reference was already applied (info['custom_ref_applied']),
    add back a virtual reference channel so inverse modeling is allowed,
    then set average reference as a projector (OK for forward/inverse).
    """
    candidates = ['FCz','Cz','Fz','CPz','Pz','A1','A2','M1','M2','TP9','TP10','REF']
    if raw.info.get('custom_ref_applied', False):
        ref_name = next((c for c in candidates if c not in raw.ch_names), 'EEG REF')
        mne.add_reference_channels(raw, ref_channels=[ref_name])
        print(f"[info] Added virtual reference channel: {ref_name}")
    raw.set_eeg_reference('average', projection=True)
    return raw


# ──────────────── Core helpers ────────────────
def read_raw_auto(path, *, preload=True):
    path = str(path)
    low = path.lower()
    if low.endswith(('.edf', '.bdf', '.gdf')):
        return mne.io.read_raw_edf(path, preload=preload, stim_channel=None)
    if low.endswith(('.fif', '.fif.gz')):
        return mne.io.read_raw_fif(path, preload=preload)
    if low.endswith('.vhdr'):
        return mne.io.read_raw_brainvision(path, preload=preload)
    raise ValueError(f"Unsupported EEG format: {path}")


def ensure_fsaverage():
    fs_dir = Path(fetch_fsaverage(verbose=True))
    subj_dir = fs_dir.parent
    os.environ['SUBJECTS_DIR'] = str(subj_dir)
    bem_dir = fs_dir / 'bem'
    bem = sorted(bem_dir.glob('*bem-sol.fif'))[0]
    src = (sorted(bem_dir.glob('*-ico-5-src.fif')) or
           sorted(bem_dir.glob('*-src.fif')))[0]
    return str(subj_dir), str(fs_dir), str(bem), str(src)


def make_evoked_from_raw(raw, epoch_len=2.0):
    events = mne.make_fixed_length_events(raw, duration=epoch_len)
    picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=epoch_len,
                        picks=picks, preload=True,
                        baseline=None,               # avoid one-sample baseline error
                        reject_by_annotation=True)
    noise_cov = mne.compute_covariance(epochs, method='auto', rank='info')
    return epochs.average(), noise_cov


def compute_inverse(evoked, noise_cov, src_fname, bem_fname, method='dSPM', snr=3.0):
    fwd = mne.make_forward_solution(evoked.info, trans='fsaverage',
                                    src=src_fname, bem=bem_fname,
                                    eeg=True, meg=False)
    inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose='auto', depth=0.8)
    stc = apply_inverse(evoked, inv, lambda2=1.0/(snr**2), method=method)
    return fwd, inv, stc


def band_power(x, fs, fmin, fmax):
    f, Pxx = welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (f >= fmin) & (f <= fmax)
    return float(Pxx[mask].mean()) if np.any(mask) else 0.0


def label_aggregate(stc, src_fname, parc='aparc', band=None, info=None):
    src = mne.read_source_spaces(src_fname, verbose=False)
    labels = mne.read_labels_from_annot('fsaverage', parc=parc,
                                        subjects_dir=os.environ['SUBJECTS_DIR'])
    labels = [lab for lab in labels if 'unknown' not in lab.name.lower()]
    try:
        ts = mne.extract_label_time_course(stc, labels, src, mode='mean',
                                           return_generator=False, allow_empty=True)
    except TypeError:  # older MNE
        ts = mne.extract_label_time_course(stc, labels, src, mode='mean',
                                           return_generator=False)
    if band and info:
        fs = info['sfreq']
        vals = np.array([band_power(ts[i], fs, band[0], band[1]) for i in range(len(labels))])
        metric = f"band_power_{band[0]}-{band[1]}Hz"
    else:
        vals = np.sqrt((ts ** 2).mean(axis=1))
        metric = "rms_time"
    return labels, vals, metric

def safe_plot_stc(stc, subjects_dir):
    try:
        return stc.plot(
            subject="fsaverage", hemi="both", subjects_dir=subjects_dir,
            surface="inflated", cortex="low_contrast", time_viewer=False
        )
    except Exception as e:
        print(f"[warn] Interactive plot failed: {e}")
        import os, pyvista
        os.environ["PYVISTA_OFF_SCREEN"] = "true"
        pv.OFF_SCREEN = True
        brain = stc.plot(
            subject="fsaverage", hemi="both", subjects_dir=subjects_dir,
            surface="inflated", cortex="low_contrast",
            time_viewer=True,   # ✅ enables interactive slider
            show_traces=True    # optional EEG traces panel
        )
        for view in ("lateral", "medial", "dorsal"):
            brain.show_view(view)
            brain.save_image(f"brain_fallback_{view}.png")
        brain.close()
        return None
    
def labels_to_static_stc(labels, values, src_fname):
    src = mne.read_source_spaces(src_fname, verbose=False)
    lh_data = np.zeros(src[0]['nuse']); rh_data = np.zeros(src[1]['nuse'])
    for lab, val in zip(labels, values):
        hemi = 0 if lab.hemi == 'lh' else 1
        verts = np.intersect1d(lab.vertices, src[hemi]['vertno'])
        if verts.size == 0:
            continue
        idx = np.searchsorted(src[hemi]['vertno'], verts)
        (lh_data if hemi == 0 else rh_data)[idx] = val
    data = np.vstack([lh_data[:, None], rh_data[:, None]])
    return mne.SourceEstimate(data, [src[0]['vertno'], src[1]['vertno']], 0, 1, subject='fsaverage')


def plot_3d(stc_static, subjects_dir, save_figs=False, out_prefix='brain_plot', offscreen=False):
    if offscreen or save_figs:
        os.environ['PYVISTA_OFF_SCREEN'] = 'true'
    brain = stc_static.plot(subject="fsaverage", hemi="both", surface="inflated",
                        subjects_dir=subjects_dir, time_viewer=False, cortex="low_contrast")
    for view in ("lateral", "medial", "dorsal", "ventral"):
        try:
            brain.show_view(view)
            brain.save_image(f"brain_{view}.png")
        except Exception:
            pass
    brain.close()
    print("Saved off-screen PNGs.")


def maybe_make_synthetic_raw(duration=60, sfreq=256, montage='standard_1020'):
    chs = ['Fp1','Fp2','F3','F4','F7','F8','Fz','C3','C4','Cz',
           'P3','P4','Pz','O1','O2']
    n_samp = int(duration * sfreq); t = np.arange(n_samp) / sfreq
    rng = np.random.RandomState(42)
    data = 1e-6 * rng.randn(len(chs), n_samp)
    alpha = np.sin(2 * np.pi * 10 * t)
    for i, n in enumerate(chs):
        if n in {'O1','O2','Pz'}: data[i] += 4e-6 * alpha
        if n in {'Fp1','Fp2','Fz'}: data[i] += 2e-6 * np.sin(2 * np.pi * 20 * t)
    info = mne.create_info(chs, sfreq, 'eeg')
    raw = mne.io.RawArray(data, info)
    raw.set_montage(mne.channels.make_standard_montage(montage))
    return raw


# ──────────────── Main ────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eeg')
    p.add_argument('--parc', default='aparc')
    p.add_argument('--offscreen', action='store_true')
    p.add_argument('--save-figs', action='store_true')
    args = p.parse_args()

    
    # Load / create data
    raw = maybe_make_synthetic_raw() if args.eeg is None else read_raw_auto(args.eeg)

    # Safe montage attach (10–20 → fallback to 10–05)
    try:
        raw = attach_montage_safely(raw, montage_name='standard_1020')
    except Exception as e:
        print(f"[warn] standard_1020 failed: {e}; trying standard_1005")
        raw = attach_montage_safely(raw, montage_name='standard_1005')

    # Preprocess
    raw.load_data()
    raw.filter(1, 40)
    raw = ensure_inverse_safe_reference(raw)   # ← inverse-safe referencing here

    # Anatomy
    subjects_dir, _, bem, src = ensure_fsaverage()

    # Evoked + inverse
    evoked, cov = make_evoked_from_raw(raw)
    _, _, stc = compute_inverse(evoked, cov, src, bem)

    # Regions
    labels, vals, metric = label_aggregate(stc, src, parc=args.parc, info=evoked.info)
    with open('region_activity.csv', 'w', encoding='utf-8') as f:
        f.write('label_name,' + metric + '\n')
        for l, v in zip(labels, vals):
            f.write(f"{l.name},{v:.6e}\n")

    # Static 3D map
    brain = open_interactive_safely(stc, subjects_dir)
    # after your plotting call
    brain = open_interactive_safely(stc, subjects_dir)

    # Keep the Qt event loop running until the user closes the window
    from pyvistaqt import QtInteractor
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.exec_()   # <- this line keeps the viewer open
if __name__ == '__main__':
    main()
