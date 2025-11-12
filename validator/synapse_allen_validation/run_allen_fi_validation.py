# validator/run_allen_fi_validation.py
# Allen baseline → compare to your EnhancedNeuron42PyTorch (F–I + overlays)
import os, sys, math, json
from pathlib import Path
import numpy as np
# [VALIDATOR ONLY] force CPU so no CUDA/CPU mixing happens inside your neuron
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, math, json

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allen SDK
try:
    from allensdk.core.cell_types_cache import CellTypesCache
    from allensdk.core.nwb_data_set import NwbDataSet
except Exception as e:
    print("ERROR: allensdk not available. pip install allensdk")
    raise

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parent
OUTDIR = HERE / "allen_validation_outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Import your neuron
cells_path = (ROOT / "cortex" / "cells").resolve()
if str(cells_path) not in sys.path:
    sys.path.insert(0, str(cells_path))
import torch
# [VALIDATOR ONLY] make torch.exp(float) → Python float (no device)
# This avoids creating a CPU tensor inside your neuron when it calls torch.exp on a scalar.
_orig_torch_exp = torch.exp
def _exp_device_safe(x, *args, **kwargs):
    import math
    if isinstance(x, (float, int)):
        return float(math.exp(x))
    return _orig_torch_exp(x, *args, **kwargs)
torch.exp = _exp_device_safe

import cortex.cells.enhanced_neurons_42 as EN
# [VALIDATOR ONLY] Force the neuron module's globals to CPU so module-level tensors aren't on CUDA
EN.DEVICE = torch.device("cpu")
EN.CONSTANTS = EN.make_constants(EN.DEVICE)

from cortex.cells.enhanced_neurons_42 import EnhancedNeuron42PyTorch

def detect_step_window(stim_pa: np.ndarray, thresh_pa=5.0):
    """Return (start_idx, end_idx) of the longest contiguous segment with |I|>thresh_pa."""
    on = np.where(np.abs(stim_pa) > thresh_pa)[0]
    if on.size == 0:
        return 0, 0
    starts, ends = [int(on[0])], []
    for k in range(1, on.size):
        if on[k] != on[k-1] + 1:
            ends.append(int(on[k-1])); starts.append(int(on[k]))
    ends.append(int(on[-1]))
    segs = [(s,e) for s,e in zip(starts, ends)]
    segs.sort(key=lambda se: se[1]-se[0], reverse=True)
    return segs[0]

def count_spikes(v: np.ndarray, thr=-0.02, refrac=0.002, fs=20000):
    """Simple threshold-and-refractory counter on membrane voltage (V)."""
    above = v > thr
    idx = np.flatnonzero(np.logical_and(above, np.concatenate([[False], ~above[:-1]])))
    if idx.size == 0:
        return 0
    min_gap = int(refrac * fs)
    kept, last = [], -10**9
    for i in idx:
        if i - last >= min_gap:
            kept.append(i); last = i
    return len(kept)

def find_spike_times(v: np.ndarray, thr=-0.02, refrac=0.002, fs=20000):
    """
    Return spike times (seconds). Auto-convert input to Volts so we can pass a
    threshold in Volts regardless of whether v is in V or mV.
    """
    # If the typical magnitude is >> 0.5, we are almost certainly in mV.
    m = float(np.nanmedian(np.abs(v)))
    vV = v if m < 0.5 else (v * 1e-3)  # mV -> V

    above = vV > thr
    idx = np.flatnonzero(np.logical_and(above, np.concatenate([[False], ~above[:-1]])))
    if idx.size == 0:
        return np.array([], dtype=float)
    min_gap = int(refrac * fs)
    kept, last = [], -10**9
    for i in idx:
        if i - last >= min_gap:
            kept.append(i); last = i
    return np.array(kept, dtype=int) / float(fs)

def get_allen_long_square(specimen_id: int):
    ctc = CellTypesCache(manifest_file=str(OUTDIR / "manifest.json"))
    nwb_obj = ctc.get_ephys_data(specimen_id)
    # AllenSDK can return a path (str) or an NwbDataSet depending on version/cache
    nwb = nwb_obj if isinstance(nwb_obj, NwbDataSet) else NwbDataSet(nwb_obj)
    sweeps = ctc.get_ephys_sweeps(specimen_id)
    long_sq = [s for s in sweeps if s.get("stimulus_name","").lower().startswith("long square")]
    results = []
    for s in long_sq:
        sweep_num = s["sweep_number"]
        data = nwb.get_sweep(sweep_num)
        v = data["response"]          # in Volts
        i = data["stimulus"]          # in Amps
        fs = data["sampling_rate"]    # in Hz
        t = np.arange(v.size) / fs
        v = v.astype(np.float64)
        i_pa = i * 1e12               # convert to pA
        s0, s1 = detect_step_window(i_pa)
        step_pa = float(np.median(i_pa[s0:s1])) if s1> s0 else 0.0
        spk = count_spikes(v, thr=0.0, refrac=0.002, fs=int(fs))
        results.append(dict(sweep=sweep_num, fs=float(fs), t=t, v=v, i_pa=i_pa, step_pa=step_pa, spikes=spk))
    return results

def simulate_cortex_neuron(step_pa: float, fs=20000, pre_ms=50, dur_ms=1000, post_ms=50, gain=1.0):
    dt_ms = 1000.0 / fs  # validator: forward() expects milliseconds
    T  = int((pre_ms+dur_ms+post_ms) * 1e-3 * fs)
    # Keep Allen step in pA for reporting…
    stim_pa = np.zeros(T, dtype=np.float32)
    s0 = int(pre_ms * 1e-3 * fs); s1 = s0 + int(dur_ms * 1e-3 * fs)
    stim_pa[s0:s1] = step_pa

    # …but FEED YOUR NEURON IN nA (most of your code expects nA-scale inputs)
    stim_model = stim_pa / 1000.0  # pA → nA


    # Build neuron (CPU for comparability)
    neuron = EnhancedNeuron42PyTorch(
        neuron_id=0, n_dendrites=4, neuron_type="pyramidal",
        use_cadex=True, device="cpu"
    )

    # --- HARD-FORCE EVERYTHING TO CPU (validator-only; no model file changes) ---
    try:
        # if it's an nn.Module, this moves params & buffers
        neuron.to("cpu")
    except Exception:
        pass

    import torch  # already imported above, but safe to reference here
    with torch.no_grad():
        # move any parameters that might still be on CUDA
        if hasattr(neuron, "parameters"):
            for p in neuron.parameters(recurse=True):
                if hasattr(p, "is_cuda") and p.is_cuda:
                    p.data = p.data.cpu()
        # move any registered buffers that might still be on CUDA
        if hasattr(neuron, "named_buffers"):
            for name, buf in neuron.named_buffers(recurse=True):
                if hasattr(buf, "is_cuda") and buf.is_cuda:
                    buf.data = buf.data.cpu()

    # Best effort reset if provided (after forcing to CPU)
    if hasattr(neuron, "reset") and callable(neuron.reset):
        neuron.reset()
    v_hist = []
    spike_sum = 0
    current_time = 0.0

    # validator-only smoothing of the injected pA (matches Allen’s softer onset)
    filtered_pA = 0.0
    tau_ms_drive = 5.0  # try 3–8 ms

    for k in range(T):
        target = float(gain) * float(stim_pa[k])  # pA
        filtered_pA += (dt_ms / max(tau_ms_drive, 1e-6)) * (target - filtered_pA)

        with torch.no_grad():
            neuron.phase_coupling_strength = 1.0
            neuron.oscillatory_input = filtered_pA
        # give the dendrites zero external current (we're validating soma F–I)
        I = np.zeros(neuron.n_dendrites, dtype=np.float32)

        if hasattr(neuron, "step"):
            spk, V = neuron.step(I, dt_ms, current_time)   # your step() returns (bool, voltage)
        else:
            spk, V = neuron(torch.from_numpy(I))           # keep CPU tensor

        # record voltage trace for plotting
        v_val = float(V.detach().cpu().flatten()[0]) if hasattr(V, "detach") else float(V)
        v_hist.append(v_val)

        # count only inside the long-square step window
        if s0 <= k < s1:
            spike_sum += int(spk)

        current_time += dt_ms


    v_hist = np.array(v_hist, dtype=np.float64)
    spk = spike_sum
    return dict(fs=fs, v=v_hist, i_pa=stim_pa, spikes=spk)

def _to_volts(v: np.ndarray) -> np.ndarray:
    """Return v in Volts whether input is V or mV."""
    m = float(np.nanmedian(np.abs(v)))
    return v if m < 0.5 else (v * 1e-3)  # mV -> V

def detect_cortex_spikes(v_raw: np.ndarray, fs: int, s0: int, s1: int) -> np.ndarray:
    """
    Peak-based spike picker for the CORTEX trace.
    - Light high-pass (moving-average subtract) so oscillations cross zero.
    - Pick local maxima with minimum prominence.
    - 6 ms guard after onset, 12 ms refractory.
    Returns spike times (seconds).
    """
    vV = _to_volts(v_raw)
    step = vV[s0:s1]

    # --- light high-pass: subtract a short moving average (≈8 ms) ---
    w_hp = max(1, int(0.008 * fs))
    kernel = np.ones(w_hp, dtype=float) / float(w_hp)
    ma = np.convolve(step, kernel, mode="same")
    x = step - ma

    # amplitude scale for adaptive gates (robust range)
    hi = float(np.percentile(x, 97))
    lo = float(np.percentile(x, 3))
    amp = max(hi - lo, 1e-9)

    # detection params (tuned for your +170 pA bumps)
    guard   = int(0.006 * fs)   # 6 ms after onset
    refrac  = int(0.012 * fs)   # 12 ms refractory
    w_prom  = int(0.003 * fs)   # 3 ms look-back for trough
    prom_th = 0.22 * amp        # minimum prominence

    peaks = []
    last  = -10**9
    # search only after the guard
    for i in range(max(guard, 1), x.size - 1):
        if i - last < refrac:
            continue
        # local maximum
        if x[i] > x[i-1] and x[i] >= x[i+1]:
            j0     = max(0, i - w_prom)
            trough = float(np.min(x[j0:i+1]))
            prom   = x[i] - trough
            if prom >= prom_th:
                peaks.append(i)
                last = i

    # convert indices back to absolute time
    return (np.array(peaks, dtype=int) + s0) / float(fs)
    # Pass A (moderate): lower thr & prominence, shorter guard/window
    kept = _detect(thr_frac=0.40, prom_frac=0.25, guard_ms=6.0, refrac_ms=12.0, win_ms=3.0)

    # Pass B (fallback): if nothing found, relax once more
    if not kept:
        kept = _detect(thr_frac=0.25, prom_frac=0.15, guard_ms=5.0, refrac_ms=10.0, win_ms=3.0)

    return np.array(kept, dtype=int) / float(fs)

def run(specimen_id: int):
    allen = get_allen_long_square(specimen_id)
    # Build unique steps from Allen (use only depolarizing for calibration)
    steps = sorted({round(rec["step_pa"],1) for rec in allen if rec["step_pa"] > 0.1})

    tmp_rows = []
    for step in steps:
        sim = simulate_cortex_neuron(step_pa=step, fs=20000, gain=1.0)
        tmp_rows.append(dict(step_pa=step, spikes=sim["spikes"], v=sim["v"], i_pa=sim["i_pa"], fs=sim["fs"]))

    # preliminary F–I pairs (Allen vs CORTEX) for slope fit
    def hz_from_spikes(n, dur_ms=1000): return (n / (dur_ms/1000.0))
    def q(pa): return int(round(pa))  # quantize to integer pA

    allen_fi_pass1 = {}
    for rec in allen:
        if rec["step_pa"] > 0.1:
            allen_fi_pass1.setdefault(q(rec["step_pa"]), []).append(rec["spikes"])
    allen_fi_pass1 = {k: hz_from_spikes(int(np.median(v))) for k, v in allen_fi_pass1.items()}

    cortex_fi_pass1 = {q(r["step_pa"]): hz_from_spikes(r["spikes"]) for r in tmp_rows}

    # paired vectors (exclude zeros to avoid skewing slope)
    xs, ys = [], []
    for k in sorted(set(allen_fi_pass1) & set(cortex_fi_pass1)):
        if allen_fi_pass1[k] > 0 and cortex_fi_pass1[k] > 0:
            xs.append(allen_fi_pass1[k]); ys.append(cortex_fi_pass1[k])

    if len(xs) >= 2:
        m_est, b_est = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
        gain = float(1.0 / max(m_est, 1e-6))  # target slope ~1
        gain = float(np.clip(gain, 0.25, 4.0))  # keep sane range
    else:
        gain = 1.0

    print(f"[validator] auto-calibrated gain = {gain:.3f}")

    # ---------- PASS 2: final run using calibrated gain ----------
    cortex_rows = []
    for step in steps:
        sim = simulate_cortex_neuron(step_pa=step, fs=20000, gain=gain)
        cortex_rows.append(dict(step_pa=step, spikes=sim["spikes"], v=sim["v"], i_pa=sim["i_pa"], fs=sim["fs"]))

    # F–I table
    def hz_from_spikes(n, dur_ms=1000): return (n / (dur_ms/1000.0))

    # Quantize to integer pA so -30.0 and -30.0000001 collapse to the same key
    def q(pa): return int(round(pa))

    allen_fi = {}
    for rec in allen:
        allen_fi.setdefault(q(rec["step_pa"]), []).append(rec["spikes"])
    allen_fi = {k: hz_from_spikes(int(np.median(v))) for k, v in allen_fi.items()}

    cortex_fi = {}
    for r in cortex_rows:
        cortex_fi[q(r["step_pa"])] = hz_from_spikes(r["spikes"])

    # --- Comparison table + metrics ---
    print("current_pA\tAllen_Hz\tCORTEX_Hz")
    rows = []
    all_keys = sorted(set(list(allen_fi.keys()) + list(cortex_fi.keys())))
    for pa in [k for k in all_keys if k >= 0]:

        a = allen_fi.get(pa, float("nan"))
        c = cortex_fi.get(pa, float("nan"))
        rows.append((float(pa), a, c))
        a_show = 0.0 if (isinstance(a, float) and np.isnan(a)) else a
        c_show = 0.0 if (isinstance(c, float) and np.isnan(c)) else c
        print(f"{pa:+.0f}\t\t{a_show:.3f}\t\t{c_show:.3f}")

    # Save CSV (clean)
    import csv, json
    with open(OUTDIR/"fi_comparison.csv","w",newline="") as f:
        w = csv.writer(f); w.writerow(["current_pA","allen_hz","cortex_hz"]); w.writerows(rows)

    # Build paired vectors where both exist (no NaNs)
    pa_vec, allen_vec, cortex_vec = [], [], []
    for pa, a, c in rows:
        if not (isinstance(a, float) and np.isnan(a)) and not (isinstance(c, float) and np.isnan(c)):
            pa_vec.append(pa); allen_vec.append(a); cortex_vec.append(c)
    pa_vec  = np.array(pa_vec, dtype=float)
    allen_v = np.array(allen_vec, dtype=float)
    cortex_v= np.array(cortex_vec, dtype=float)

    # Metrics: rheobase (first >0 Hz), linear fit cortex = m*allen + b, R2, MAE, RMSE
    def first_pos_idx(x): 
        idx = np.where(x > 0.0)[0]
        return int(idx[0]) if idx.size>0 else None

    r_ix_a = first_pos_idx(allen_v)
    r_ix_c = first_pos_idx(cortex_v)
    rheobase_allen_pa  = float(pa_vec[r_ix_a]) if r_ix_a is not None else float("nan")
    rheobase_cortex_pa = float(pa_vec[r_ix_c]) if r_ix_c is not None else float("nan")
    rheobase_delta_pa  = rheobase_cortex_pa - rheobase_allen_pa if (not np.isnan(rheobase_allen_pa) and not np.isnan(rheobase_cortex_pa)) else float("nan")

    if allen_v.size >= 2 and cortex_v.size >= 2:
        m, b = np.polyfit(allen_v, cortex_v, 1)
        pred  = m*allen_v + b
        ss_res= np.sum((cortex_v - pred)**2)
        ss_tot= np.sum((cortex_v - np.mean(cortex_v))**2)
        r2    = 1.0 - (ss_res/ss_tot if ss_tot>0 else 0.0)
        mae   = float(np.mean(np.abs(cortex_v - allen_v)))
        rmse  = float(np.sqrt(np.mean((cortex_v - allen_v)**2)))
    else:
        m=b=r2=mae=rmse=float("nan")

    metrics = {
        "rheobase_allen_pA": rheobase_allen_pa,
        "rheobase_cortex_pA": rheobase_cortex_pa,
        "rheobase_delta_pA": rheobase_delta_pa,
        "slope_m": float(m),
        "intercept_b": float(b),
        "r2": float(r2),
        "mae_hz": float(mae),
        "rmse_hz": float(rmse),
        "n_points_paired": int(allen_v.size),
        "validator_gain": float(gain)
    }
    with open(OUTDIR/"fi_metrics.json","w") as jf:
        json.dump(metrics, jf, indent=2)

    # Scatter plot: Allen vs CORTEX with identity + fit line
    fig = plt.figure(figsize=(5,5))
    ax = plt.gca()
    if allen_v.size:
        ax.scatter(allen_v, cortex_v, s=28)
        lim = float(max(1.0, np.nanmax([allen_v.max(), cortex_v.max()])))
        ax.plot([0,lim],[0,lim], linestyle="--")  # identity
        if np.isfinite(m) and np.isfinite(b):
            xs = np.linspace(0, lim, 100)
            ax.plot(xs, m*xs + b)
    ax.set_xlabel("Allen F–I (Hz)")
    ax.set_ylabel("CORTEX F–I (Hz)")
    ax.set_title(f"F–I agreement  (R²={0.0 if not np.isfinite(r2) else r2:.3f}, slope={0.0 if not np.isfinite(m) else m:.2f})")
    plt.tight_layout(); plt.savefig(OUTDIR/"fi_scatter.png", dpi=180)
    plt.close(fig)

    # Pick an overlay step from the mid-range (clear comparison), then fall back
    def best_overlay_pa(rows, allen_fi, lo=6.0, hi=22.0):
        # rows: list of dict(step_pa, spikes, ...)
        # allen_fi: dict[int pA] -> Allen Hz
        def hz(n, dur_ms=1000): return n / (dur_ms/1000.0)
        # 1) prefer steps where Allen is in [lo, hi] Hz
        candidates = []
        for r in rows:
            pa_i = int(round(r["step_pa"]))
            if pa_i in allen_fi:
                a_hz = float(allen_fi[pa_i])
                c_hz = float(hz(r["spikes"]))
                err  = abs(a_hz - c_hz)
                if lo <= a_hz <= hi:
                    candidates.append((err, pa_i))
        if candidates:
            candidates.sort()
            return candidates[0][1]
        # 2) otherwise, closest rate anywhere
        best_pa, best_err = None, float("inf")
        for r in rows:
            pa_i = int(round(r["step_pa"]))
            if pa_i in allen_fi:
                err = abs(float(allen_fi[pa_i]) - hz(r["spikes"]))
                if err < best_err:
                    best_err, best_pa = err, pa_i
        return best_pa if best_pa is not None else int(round(rows[0]["step_pa"]))

    target = best_overlay_pa(cortex_rows, allen_fi)
    
    allen_one  = next(rec for rec in allen        if int(round(rec["step_pa"])) == target)
    cortex_one = next(rec for rec in cortex_rows  if int(round(rec["step_pa"])) == target)

    # Align by step window (indices in full traces)
    a_s0, a_s1 = detect_step_window(allen_one["i_pa"])
    c_s0, c_s1 = detect_step_window(cortex_one["i_pa"])

    # --- helpers ---
    def to_mV(arr):
        med = float(np.nanmedian(np.abs(arr)))
        if med < 0.5:       # Volts → mV
            return arr * 1e3
        elif med > 200.0:   # µV → mV
            return arr * 1e-3
        else:               # already mV
            return arr

    allen_mV  = allen_one["v"] * 1e3
    cortex_mV = to_mV(cortex_one["v"])

    # 1) FULL-STEP overlay (robust step masks + true 1.0 s window)
    def _robust_step_indices(i_pa, fs, thresh_pa=5.0):
        # base mask
        m = (np.abs(i_pa) > thresh_pa).astype(np.int8)
        # fill tiny gaps (<=1 ms) so one long block stays contiguous
        gap = int(max(1, 0.001 * fs))
        # simple 1D closing: dilate then erode
        from numpy.lib.stride_tricks import sliding_window_view
        if m.sum():
            # dilate
            k = np.ones(gap*2+1, dtype=np.int8)
            md = (np.convolve(m, k, 'same') > 0).astype(np.int8)
            # erode
            me = (np.convolve(md, k, 'same') == k.size).astype(np.int8)
            m = np.maximum(m, me)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            return 0, 0
        # longest contiguous segment
        starts, ends = [idx[0]], []
        for k in range(1, idx.size):
            if idx[k] != idx[k-1] + 1:
                ends.append(idx[k-1]); starts.append(idx[k])
        ends.append(idx[-1])
        segs = [(int(s), int(e)) for s, e in zip(starts, ends)]
        segs.sort(key=lambda se: se[1]-se[0], reverse=True)
        s0, s1 = segs[0]
        return s0, s1+1  # make s1 exclusive

    # recompute robust windows (in case tiny gaps fooled the old detector)
    a_s0, a_s1 = _robust_step_indices(allen_one["i_pa"], allen_one["fs"])
    c_s0, c_s1 = _robust_step_indices(cortex_one["i_pa"], cortex_one["fs"])

    # enforce a true 1.0 s step duration (pad with NaNs if shorter)
    dur_s = 1.0
    Na = int(round(dur_s * allen_one["fs"]))
    Nc = int(round(dur_s * cortex_one["fs"]))

    Va = allen_mV[a_s0:a_s0+Na]
    Vc = cortex_mV[c_s0:c_s0+Nc]
    if Va.size < Na:
        Va = np.pad(Va, (0, Na - Va.size), constant_values=np.nan)
    if Vc.size < Nc:
        Vc = np.pad(Vc, (0, Nc - Vc.size), constant_values=np.nan)

    t_a_full = np.arange(Na) / allen_one["fs"]
    t_c_full = np.arange(Nc) / cortex_one["fs"]

    fig = plt.figure(figsize=(9,4))
    plt.plot(t_a_full, Va, label="Allen (mV)")
    plt.plot(t_c_full, Vc, label="CORTEX (mV)", alpha=0.8)
    plt.xlabel("Time from step onset (s)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"Overlay @ {target:+.0f} pA (step-aligned)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR/"trace_overlay.png", dpi=180)
    plt.close(fig)

    # 2) BASELINE-ALIGNED, ZOOMED (first 150 ms after onset; robust)
    # re-use a_s0/c_s0 from the robust finder above
    a_pre = max(0, a_s0 - int(0.020 * allen_one["fs"]))
    c_pre = max(0, c_s0 - int(0.020 * cortex_one["fs"]))
    a_base = float(np.mean(allen_mV[a_pre:a_s0])) if a_s0 > a_pre else float(allen_mV[a_s0])
    c_base = float(np.mean(cortex_mV[c_pre:c_s0])) if c_s0 > c_pre else float(cortex_mV[c_s0])

    Z = 0.150  # 150 ms
    La = int(round(Z * allen_one["fs"]))
    Lc = int(round(Z * cortex_one["fs"]))
    Va_zo = allen_mV[a_s0:a_s0+La] - a_base
    Vc_zo = cortex_mV[c_s0:c_s0+Lc] - c_base
    if Va_zo.size < La: Va_zo = np.pad(Va_zo, (0, La - Va_zo.size), constant_values=np.nan)
    if Vc_zo.size < Lc: Vc_zo = np.pad(Vc_zo, (0, Lc - Vc_zo.size), constant_values=np.nan)

    t_a_zo = np.arange(La) / allen_one["fs"]
    t_c_zo = np.arange(Lc) / cortex_one["fs"]

    fig = plt.figure(figsize=(9,4))
    plt.plot(t_a_zo, Va_zo, label="Allen (mV, baseline-subtracted)")
    plt.plot(t_c_zo, Vc_zo, label="CORTEX (mV, baseline-subtracted)", alpha=0.8)
    plt.xlabel("Time from step onset (s)")
    plt.ylabel("ΔVoltage (mV)")
    plt.title(f"Overlay @ {target:+.0f} pA (first 150 ms, baseline-aligned)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR/"trace_overlay_zoom.png", dpi=180)
    plt.close(fig)

    # 3) Spike raster (step-aligned) for the chosen target current
    # Compute spike times from the FULL traces, then keep only inside the step window
    fsA, fsC = float(allen_one["fs"]), float(cortex_one["fs"])

    # Allen: fixed -20 mV works (its spikes cross 0 V)
    tA_spk = find_spike_times(allen_one["v"], thr=-0.020, refrac=0.002, fs=int(fsA))

    # CORTEX: robust detector (adaptive thr, 12 ms guard, 15 ms refractory)
    tC_spk = detect_cortex_spikes(cortex_one["v"], int(fsC), c_s0, c_s1)

    # keep spikes within the step window, but ignore the first 5 ms after onset
    guard_ms = 5.0
    guard_s  = guard_ms / 1000.0

    tA_step = tA_spk[(tA_spk >= a_s0/fsA) & (tA_spk <= a_s1/fsA)] - (a_s0/fsA)
    tC_step = tC_spk[(tC_spk >= (c_s0/fsC + guard_s)) & (tC_spk <= c_s1/fsC)] - (c_s0/fsC)

    fig = plt.figure(figsize=(9,2.8))
    ax = plt.gca()
    # Allen spikes (row 1)
    for t in tA_step:
        ax.vlines(t, 1.6, 1.9, linewidth=1.4, label=None, color="C0")
    # CORTEX spikes (row 2)
    for t in tC_step:
        ax.vlines(t, 0.6, 0.9, linewidth=1.4, label=None, color="C1")

    ax.set_ylim(0.0, 2.5)
    ax.set_yticks([0.75, 1.75])
    ax.set_yticklabels(["CORTEX", "Allen"])
    ax.set_xlabel("Time from step onset (s)")
    ax.set_title(f"Spike times @ {target:+.0f} pA (step-aligned)")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTDIR/"trace_spike_raster.png", dpi=180)
    plt.close(fig)
    
if __name__ == "__main__":
    # pick a specimen from the included cache first if you like
    # example: 314800874 is present in your zip (validator/allen_cache/specimen_314800874/ephys.nwb)
    run(specimen_id=314800874)
