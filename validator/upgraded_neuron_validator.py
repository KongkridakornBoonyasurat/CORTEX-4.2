# validator/upgraded_neuron_validator.py
# Allen baseline → compare to your EnhancedNeuron42PyTorch (F–I + overlays + rich metrics)

import os, sys, math, json, csv
from pathlib import Path
import numpy as np

# [VALIDATOR ONLY] force CPU so no CUDA/CPU mixing happens inside your neuron
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTDIR = HERE / "allen_validation_outputs"

def _ensure_outdir():
    OUTDIR.mkdir(parents=True, exist_ok=True)

_ensure_outdir()
# Import your neuron
cells_path = (ROOT / "cortex" / "cells").resolve()
if str(cells_path) not in sys.path:
    sys.path.insert(0, str(cells_path))

import torch
# [VALIDATOR ONLY] make torch.exp(float) → Python float (no device)
_orig_torch_exp = torch.exp
def _exp_device_safe(x, *args, **kwargs):
    import math as _math
    if isinstance(x, (float, int)):
        return float(_math.exp(x))
    return _orig_torch_exp(x, *args, **kwargs)
torch.exp = _exp_device_safe

import cortex.cells.enhanced_neurons_42 as EN
# [VALIDATOR ONLY] Force the neuron module's globals to CPU so module-level tensors aren't on CUDA
EN.DEVICE = torch.device("cpu")
EN.CONSTANTS = EN.make_constants(EN.DEVICE)
from cortex.cells.enhanced_neurons_42 import EnhancedNeuron42PyTorch


# ----------------------------- Utils -----------------------------

def _to_volts(v: np.ndarray) -> np.ndarray:
    """Return v in Volts whether input is V or mV."""
    m = float(np.nanmedian(np.abs(v)))
    return v if m < 0.5 else (v * 1e-3)  # mV -> V

def to_mV(arr: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(np.abs(arr)))
    if med < 0.5:       # Volts → mV
        return arr * 1e3
    elif med > 200.0:   # µV → mV
        return arr * 1e-3
    else:               # already mV
        return arr

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
    vV = _to_volts(v)
    above = vV > thr
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
    """Return spike times (seconds) in a threshold & refractory manner (threshold in Volts)."""
    vV = _to_volts(v)
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

    # light high-pass (≈8 ms MA)
    w_hp = max(1, int(0.008 * fs))
    kernel = np.ones(w_hp, dtype=float) / float(w_hp)
    ma = np.convolve(step, kernel, mode="same")
    x = step - ma

    hi = float(np.percentile(x, 97))
    lo = float(np.percentile(x, 3))
    amp = max(hi - lo, 1e-9)

    guard   = int(0.006 * fs)
    refrac  = int(0.012 * fs)
    w_prom  = int(0.003 * fs)
    prom_th = 0.22 * amp

    peaks, last = [], -10**9
    for i in range(max(guard,1), x.size-1):
        if i - last < refrac:
            continue
        if x[i] > x[i-1] and x[i] >= x[i+1]:
            j0     = max(0, i - w_prom)
            trough = float(np.min(x[j0:i+1]))
            prom   = x[i] - trough
            if prom >= prom_th:
                peaks.append(i); last = i

    return (np.array(peaks, dtype=int) + s0) / float(fs)

def match_spikes(t_ref: np.ndarray, t_test: np.ndarray, tol_ms=3.0):
    """Greedy bipartite matching with ±tol_ms tolerance; return precision/recall/F1 and pairs."""
    tol = tol_ms / 1000.0
    ref_used = np.zeros(t_ref.size, dtype=bool)
    test_used= np.zeros(t_test.size, dtype=bool)
    pairs = []
    for j, tt in enumerate(t_test):
        # find nearest unused reference spike within tolerance
        diffs = np.abs(t_ref - tt)
        if diffs.size == 0: continue
        k = int(np.argmin(diffs))
        if not ref_used[k] and diffs[k] <= tol:
            ref_used[k] = True; test_used[j] = True
            pairs.append((k,j))
    tp  = int(np.sum(test_used))
    fp  = int(np.sum(~test_used))
    fn  = int(np.sum(~ref_used))
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    f1   = (2*prec*rec)/max(prec+rec, 1e-9)
    return prec, rec, f1, pairs

def isi_metrics(t: np.ndarray):
    """Return mean ISI, CV, adaptation index (last/first), and first-spike latency."""
    if t.size < 2:
        return dict(mean_ms=float("nan"), cv=float("nan"),
                    adapt=float("nan"),
                    latency_ms=(float(t[0])*1000.0 if t.size==1 else float("nan")))
    isi = np.diff(t)  # seconds
    mean = float(np.mean(isi))*1000.0
    cv   = float(np.std(isi)/max(np.mean(isi),1e-9))
    adapt = float(isi[-1]/max(isi[0],1e-9))
    lat  = float(t[0]*1000.0)
    return dict(mean_ms=mean, cv=cv, adapt=adapt, latency_ms=lat)

def ap_features(v: np.ndarray, fs: int, t_spk: np.ndarray, win_ms_pre=2.0, win_ms_post=4.0):
    """
    Extract spike shape metrics around each spike time.
    Returns dict with per-spike arrays and averaged values.
    """
    v_mV = to_mV(v)
    dt = 1.0/fs
    w_pre  = int(win_ms_pre/1000.0/fs**-1)  # samples before peak
    w_post = int(win_ms_post/1000.0/fs**-1)

    peaks, troughs, halfwidth_ms, ahp_depth_mV, dvdtmax_mVms = [], [], [], [], []

    for t in t_spk:
        k = int(round(t*fs))
        i0 = max(0, k - w_pre)
        i1 = min(v_mV.size-1, k + w_post)
        seg = v_mV[i0:i1+1]
        if seg.size < 5:
            continue
        # local peak near center
        pk_off = int(np.argmax(seg))
        pk_idx = i0 + pk_off
        pk_val = float(v_mV[pk_idx])
        # preceding trough for prominence
        tr_idx = max(i0, pk_idx - int(0.002*fs))
        tr_val = float(np.min(v_mV[tr_idx:pk_idx+1]))

        # half-width at half-amp above trough
        half = tr_val + 0.5*(pk_val - tr_val)
        # walk left
        L = pk_idx
        while L > i0 and v_mV[L] > half: L -= 1
        # walk right
        R = pk_idx
        while R < i1 and v_mV[R] > half: R += 1
        hw = (R - L) * dt * 1000.0  # ms

        # AHP depth: min within 4 ms after peak relative to pre-step baseline proxy
        ahp_win = v_mV[pk_idx:min(v_mV.size-1, pk_idx+int(0.004*fs))+1]
        ahp_val = float(np.min(ahp_win))
        ahp_depth = float(pk_val - ahp_val)  # mV drop after spike

        # dV/dt max around upstroke (central ±1 ms)
        j0 = max(i0, pk_idx - int(0.001*fs))
        j1 = min(i1, pk_idx + int(0.001*fs))
        dvdt = np.diff(v_mV[j0:j1+1]) / dt / 1000.0  # mV/ms
        dvdtmax = float(np.max(dvdt)) if dvdt.size else float("nan")

        peaks.append(pk_val)
        troughs.append(tr_val)
        halfwidth_ms.append(hw)
        ahp_depth_mV.append(ahp_depth)
        dvdtmax_mVms.append(dvdtmax)

    def _avg(x):
        return float(np.nanmean(x)) if len(x) else float("nan")

    return dict(
        n=len(peaks),
        peak_mV_avg=_avg(peaks),
        trough_mV_avg=_avg(troughs),
        halfwidth_ms_avg=_avg(halfwidth_ms),
        ahp_depth_mV_avg=_avg(ahp_depth_mV),
        dvdtmax_mVms_avg=_avg(dvdtmax_mVms)
    )

def subthreshold_metrics(v: np.ndarray, i_pa: np.ndarray, fs: int, s0: int, s1: int):
    """
    Baseline (pre), steady-state delta during step, Rin estimate (MΩ),
    and tau (ms) via simple mono-exponential fit on first 50 ms after onset.
    """
    vV = _to_volts(v)
    pre = float(np.mean(vV[max(0, s0-int(0.020*fs)): s0]))
    step = vV[s0:s1]
    if step.size == 0:
        return dict(baseline_mV=float("nan"), dV_ss_mV=float("nan"),
                    Rin_MOhm=float("nan"), tau_ms=float("nan"))

    ss = float(np.mean(step[-int(0.100*fs):]))  # last 100 ms average
    dV = ss - pre  # Volts

    # current amplitude (A): robust median in-window
    I = float(np.median((i_pa[s0:s1] if s1> s0 else np.array([0.0])))) * 1e-12
    Rin = (dV / I) / 1e6 if abs(I) > 1e-15 else float("nan")  # MΩ

    # tau fit: v(t) = pre + dV * (1 - exp(-(t)/tau))
    L = int(min(0.050*fs, step.size))  # first 50 ms
    t = np.arange(L)/fs
    y = step[:L]
    # guard against numerical issues
    eps = 1e-9
    y_norm = (y - pre) / (dV + eps)
    y_norm = np.clip(1.0 - y_norm, eps, 1.0-eps)
    # linearize: ln(1 - (y-pre)/dV) = -t/tau
    z = np.log(y_norm)
    tau = - (np.sum(t*z)/max(np.sum(t*t), eps))  # least squares slope
    tau_ms = float(tau*1000.0) if np.isfinite(tau) and tau>0 else float("nan")

    return dict(baseline_mV=pre*1e3, dV_ss_mV=dV*1e3, Rin_MOhm=float(Rin), tau_ms=tau_ms)


# -------------------- Allen data and simulation --------------------

def get_allen_long_square(specimen_id: int):
    ctc = CellTypesCache(manifest_file=str(OUTDIR / "manifest.json"))
    nwb_obj = ctc.get_ephys_data(specimen_id)
    nwb = nwb_obj if isinstance(nwb_obj, NwbDataSet) else NwbDataSet(nwb_obj)
    sweeps = ctc.get_ephys_sweeps(specimen_id)
    long_sq = [s for s in sweeps if s.get("stimulus_name","").lower().startswith("long square")]
    results = []
    for s in long_sq:
        sweep_num = s["sweep_number"]
        data = nwb.get_sweep(sweep_num)
        v = data["response"]          # V
        i = data["stimulus"]          # A
        fs = data["sampling_rate"]    # Hz
        t = np.arange(v.size) / fs
        v = v.astype(np.float64)
        i_pa = i * 1e12
        s0, s1 = detect_step_window(i_pa)
        step_pa = float(np.median(i_pa[s0:s1])) if s1 > s0 else 0.0
        spk = count_spikes(v, thr=0.0, refrac=0.002, fs=int(fs))
        results.append(dict(sweep=s["sweep_number"], fs=float(fs), t=t, v=v, i_pa=i_pa,
                            step_pa=step_pa, spikes=spk))
    return results

def simulate_cortex_neuron(step_pa: float, fs=20000, pre_ms=50, dur_ms=1000, post_ms=50,
                           gain=1.0, bias_pa: float = 0.0):
    """
    Simulate the CORTEX neuron for a 1 s long-square step.
    gain: multiplicative current gain (already in pA space).
    bias_pa: constant DC pA subtracted from the stimulus to align rheobase with Allen.
    """
    dt_ms = 1000.0 / fs
    T  = int((pre_ms+dur_ms+post_ms) * 1e-3 * fs)
    stim_pa = np.zeros(T, dtype=np.float32)
    s0 = int(pre_ms * 1e-3 * fs); s1 = s0 + int(dur_ms * 1e-3 * fs)
    stim_pa[s0:s1] = step_pa

    # Build neuron (CPU for comparability)
    neuron = EnhancedNeuron42PyTorch(
        neuron_id=0, n_dendrites=4, neuron_type="pyramidal",
        use_cadex=True, device="cpu"
    )

    # --- HARD-FORCE EVERYTHING TO CPU (validator-only) ---
    try:
        neuron.to("cpu")
    except Exception:
        pass
    with torch.no_grad():
        if hasattr(neuron, "parameters"):
            for p in neuron.parameters(recurse=True):
                if hasattr(p, "is_cuda") and p.is_cuda:
                    p.data = p.data.cpu()
        if hasattr(neuron, "named_buffers"):
            for name, buf in neuron.named_buffers(recurse=True):
                if hasattr(buf, "is_cuda") and buf.is_cuda:
                    buf.data = buf.data.cpu()

    if hasattr(neuron, "reset") and callable(neuron.reset):
        neuron.reset()

    v_hist = []
    spike_sum = 0
    current_time_ms = 0.0   # milliseconds (for neuron.step)
    current_time_s  = 0.0   # seconds (for t_spk)

    # validator-only smoothing of injected pA (matches Allen softer onset)
    filtered_pA = 0.0
    tau_ms_drive = 5.0  # try 3–8 ms
    spk_times = []
    for k in range(T):
        # effective pA the model receives: scaled AND bias-shifted
        target = (float(gain) * float(stim_pa[k])) - float(bias_pa)

        filtered_pA += (dt_ms / max(tau_ms_drive, 1e-6)) * (target - filtered_pA)
        with torch.no_grad():
            neuron.phase_coupling_strength = 1.0
            neuron.oscillatory_input = filtered_pA
        I = np.zeros(neuron.n_dendrites, dtype=np.float32)

        if hasattr(neuron, "step"):
            spk, V = neuron.step(I, dt_ms, current_time_ms)   # ms to the model
        else:
            spk, V = neuron(torch.from_numpy(I))

        if spk:
            spk_times.append(current_time_s)  # record in seconds

        v_val = float(V.detach().cpu().flatten()[0]) if hasattr(V, "detach") else float(V)
        v_hist.append(v_val)

        if s0 <= k < s1:
            spike_sum += int(spk)

        # advance clocks
        current_time_ms += dt_ms
        current_time_s  += 1.0 / fs

    v_hist = np.array(v_hist, dtype=np.float64)
    return dict(fs=fs, v=v_hist, i_pa=stim_pa, spikes=spike_sum,
                t_spk=np.array(spk_times, dtype=float))
# ----------------------------- Main run -----------------------------

def run(specimen_id: int):
    _ensure_outdir()
    allen = get_allen_long_square(specimen_id)
    steps = sorted({round(rec["step_pa"],1) for rec in allen if rec["step_pa"] > 0.1})

    # PASS 1: estimate gain from mid-range steps
    tmp_rows = []
    for step in steps:
        sim = simulate_cortex_neuron(step_pa=step, fs=20000, gain=1.0)
        tmp_rows.append(dict(step_pa=step, spikes=sim["spikes"], v=sim["v"], i_pa=sim["i_pa"], fs=sim["fs"]))

    def hz_from_spikes(n, dur_ms=1000): return (n / (dur_ms/1000.0))
    def q(pa): return int(round(pa))

    allen_fi_pass1 = {}
    for rec in allen:
        if rec["step_pa"] > 0.1:
            allen_fi_pass1.setdefault(q(rec["step_pa"]), []).append(rec["spikes"])
    allen_fi_pass1 = {k: hz_from_spikes(int(np.median(v))) for k, v in allen_fi_pass1.items()}
    cortex_fi_pass1 = {q(r["step_pa"]): hz_from_spikes(r["spikes"]) for r in tmp_rows}

    xs, ys = [], []
    for k in sorted(set(allen_fi_pass1) & set(cortex_fi_pass1)):
        if allen_fi_pass1[k] > 0 and cortex_fi_pass1[k] > 0:
            xs.append(allen_fi_pass1[k]); ys.append(cortex_fi_pass1[k])

    if len(xs) >= 2:
        m_est, b_est = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
        gain = float(1.0 / max(m_est, 1e-6))
        gain = float(np.clip(gain, 0.25, 4.0))
    else:
        gain = 1.0

    print(f"[validator] auto-calibrated gain = {gain:.3f}")

    # Estimate rheobase offset (in pA) from PASS 1 and correct it with a DC bias.
    def _first_step_over_zero(fi_dict):
        ks = sorted(k for k, hz in fi_dict.items() if hz > 0.0)
        return float(ks[0]) if ks else float("nan")

    rheo_allen_pA_pass1  = _first_step_over_zero(allen_fi_pass1)
    rheo_cortex_pA_pass1 = _first_step_over_zero(cortex_fi_pass1)

    if np.isfinite(rheo_allen_pA_pass1) and np.isfinite(rheo_cortex_pA_pass1):
        # Only subtract current to align rheobase; never add
        bias_pa = float(min(max(rheo_cortex_pA_pass1 - rheo_allen_pA_pass1, 0.0), 60.0))
    else:
        bias_pa = 0.0

    print(f"[validator] rheobase bias (CORTEX-Allen) = {bias_pa:+.1f} pA  (positive → subtract from model)")

    # PASS 2: calibrated run with rheobase-alignment bias
    cortex_rows = []
    for step in steps:
        sim = simulate_cortex_neuron(step_pa=step, fs=20000, gain=gain, bias_pa=bias_pa)
        cortex_rows.append(dict(step_pa=step, spikes=sim["spikes"], v=sim["v"],
                                i_pa=sim["i_pa"], fs=sim["fs"], t_spk=sim["t_spk"]))

    # F–I tables
    allen_fi = {}
    for rec in allen:
        allen_fi.setdefault(q(rec["step_pa"]), []).append(rec["spikes"])
    allen_fi = {k: hz_from_spikes(int(np.median(v))) for k, v in allen_fi.items()}
    cortex_fi = {q(r["step_pa"]): hz_from_spikes(r["spikes"]) for r in cortex_rows}

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

    _ensure_outdir()
    OUTDIR.mkdir(parents=True, exist_ok=True)  # hard-ensure the directory exists here
    csv_path = OUTDIR / "fi_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["current_pA","allen_hz","cortex_hz"])
        w.writerows(rows)
    
    # Paired vectors for summary fit
    pa_vec, allen_vec, cortex_vec = [], [], []
    for pa, a, c in rows:
        if not (isinstance(a, float) and np.isnan(a)) and not (isinstance(c, float) and np.isnan(c)):
            pa_vec.append(pa); allen_vec.append(a); cortex_vec.append(c)
    pa_vec  = np.array(pa_vec, dtype=float)
    allen_v = np.array(allen_vec, dtype=float)
    cortex_v= np.array(cortex_vec, dtype=float)

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
    _ensure_outdir()
    with open(OUTDIR/"fi_metrics.json","w") as jf:
        json.dump(metrics, jf, indent=2)
    
    # ---------------- Plots (unchanged core visuals) ----------------
    # Choose overlay current
    def best_overlay_pa(rows, allen_fi, lo=6.0, hi=22.0):
        def hz(n, dur_ms=1000): return n / (dur_ms/1000.0)
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

    # Align by step window
    a_s0, a_s1 = detect_step_window(allen_one["i_pa"])
    c_s0, c_s1 = detect_step_window(cortex_one["i_pa"])

    allen_mV  = to_mV(allen_one["v"])
    cortex_mV = to_mV(cortex_one["v"])

    # Robust step windows (closing and enforce 1.0 s)
    def _robust_step_indices(i_pa, fs, thresh_pa=5.0):
        m = (np.abs(i_pa) > thresh_pa).astype(np.int8)
        gap = int(max(1, 0.001 * fs))
        if m.sum():
            k = np.ones(gap*2+1, dtype=np.int8)
            md = (np.convolve(m, k, 'same') > 0).astype(np.int8)
            me = (np.convolve(md, k, 'same') == k.size).astype(np.int8)
            m = np.maximum(m, me)
        idx = np.flatnonzero(m)
        if idx.size == 0: return 0, 0
        starts, ends = [idx[0]], []
        for k in range(1, idx.size):
            if idx[k] != idx[k-1] + 1:
                ends.append(idx[k-1]); starts.append(idx[k])
        ends.append(idx[-1])
        segs = [(int(s), int(e)) for s, e in zip(starts, ends)]
        segs.sort(key=lambda se: se[1]-se[0], reverse=True)
        s0, s1 = segs[0]
        return s0, s1+1

    a_s0, a_s1 = _robust_step_indices(allen_one["i_pa"], allen_one["fs"])
    c_s0, c_s1 = _robust_step_indices(cortex_one["i_pa"], cortex_one["fs"])

    dur_s = 1.0
    Na = int(round(dur_s * allen_one["fs"]))
    Nc = int(round(dur_s * cortex_one["fs"]))

    Va = allen_mV[a_s0:a_s0+Na]; Vc = cortex_mV[c_s0:c_s0+Nc]
    if Va.size < Na: Va = np.pad(Va, (0, Na - Va.size), constant_values=np.nan)
    if Vc.size < Nc: Vc = np.pad(Vc, (0, Nc - Vc.size), constant_values=np.nan)

    t_a_full = np.arange(Na) / allen_one["fs"]
    t_c_full = np.arange(Nc) / cortex_one["fs"]

    fig = plt.figure(figsize=(9,4))
    plt.plot(t_a_full, Va, label="Allen (mV)")
    plt.plot(t_c_full, Vc, label="CORTEX (mV)", alpha=0.8)
    plt.xlabel("Time from step onset (s)"); plt.ylabel("Voltage (mV)")
    plt.title(f"Overlay @ {target:+.0f} pA (step-aligned)")
    plt.legend(); plt.tight_layout()
    _ensure_outdir()
    plt.savefig(OUTDIR/"trace_overlay.png", dpi=180); plt.close(fig)
    # Zoomed baseline-aligned (first 150 ms)
    a_pre = max(0, a_s0 - int(0.020 * allen_one["fs"]))
    c_pre = max(0, c_s0 - int(0.020 * cortex_one["fs"]))
    a_base = float(np.mean(allen_mV[a_pre:a_s0])) if a_s0 > a_pre else float(allen_mV[a_s0])
    c_base = float(np.mean(cortex_mV[c_pre:c_s0])) if c_s0 > c_pre else float(cortex_mV[c_s0])

    Z = 0.150
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
    plt.xlabel("Time from step onset (s)"); plt.ylabel("ΔVoltage (mV)")
    plt.title(f"Overlay @ {target:+.0f} pA (first 150 ms, baseline-aligned)")
    plt.legend(); plt.tight_layout()
    _ensure_outdir()
    plt.savefig(OUTDIR/"trace_overlay_zoom.png", dpi=180); plt.close(fig)
    
    # Raster using robust detectors
    fsA, fsC = float(allen_one["fs"]), float(cortex_one["fs"])
    tA_spk = find_spike_times(allen_one["v"], thr=-0.020, refrac=0.002, fs=int(fsA))
    tC_spk = np.asarray(cortex_one["t_spk"], dtype=float)

    guard_ms = 5.0
    guard_s  = guard_ms / 1000.0
    tA_step = tA_spk[(tA_spk >= a_s0/fsA) & (tA_spk <= a_s1/fsA)] - (a_s0/fsA)
    tC_step = tC_spk[(tC_spk >= (c_s0/fsC + guard_s)) & (tC_spk <= c_s1/fsC)] - (c_s0/fsC)

    fig = plt.figure(figsize=(9,2.8)); ax = plt.gca()
    for t in tA_step: ax.vlines(t, 1.6, 1.9, linewidth=1.4, label=None, color="C0")
    for t in tC_step: ax.vlines(t, 0.6, 0.9, linewidth=1.4, label=None, color="C1")
    ax.set_ylim(0.0, 2.5); ax.set_yticks([0.75, 1.75]); ax.set_yticklabels(["CORTEX", "Allen"])
    ax.set_xlabel("Time from step onset (s)")
    ax.set_title(f"Spike times @ {target:+.0f} pA (step-aligned)")
    ax.grid(True, axis="x", alpha=0.25); plt.tight_layout()
    _ensure_outdir()
    plt.savefig(OUTDIR/"trace_spike_raster.png", dpi=180); plt.close(fig)

    # ---------------- Rich metrics per step ----------------
    per_step_rows = []
    per_step_json = {}

    # Build Allen dict keyed by step (int pA) with a representative sweep
    allen_by_step = {}
    for rec in allen:
        k = int(round(rec["step_pa"]))
        # prefer one with spikes for depolarizing; otherwise any
        if k not in allen_by_step or (rec["spikes"] > allen_by_step[k]["spikes"]):
            allen_by_step[k] = rec

    for row in cortex_rows:
        pa_i = int(round(row["step_pa"]))
        if pa_i not in allen_by_step:
            continue

        A = allen_by_step[pa_i]
        C = row

        # step windows
        a_s0, a_s1 = detect_step_window(A["i_pa"])
        c_s0, c_s1 = detect_step_window(C["i_pa"])
        fsA = int(A["fs"]); fsC = int(C["fs"])

        # spike times in window
        ta = find_spike_times(A["v"], thr=-0.020, refrac=0.002, fs=fsA)
        tc = np.asarray(C["t_spk"], dtype=float)
        ta = ta[(ta >= a_s0/fsA) & (ta <= a_s1/fsA)] - (a_s0/fsA)
        tc = tc[(tc >= c_s0/fsC) & (tc <= c_s1/fsC)] - (c_s0/fsC)
        # spike matching
        prec, rec, f1, pairs = match_spikes(ta, tc, tol_ms=3.0)

        # ISI/adaptation/latency
        isiA = isi_metrics(ta)
        isiC = isi_metrics(tc)

        # spike shape metrics (averaged per step)
        apA = ap_features(A["v"], fsA, ta)
        apC = ap_features(C["v"], fsC, tc)

        # subthreshold metrics
        stA = subthreshold_metrics(A["v"], A["i_pa"], fsA, a_s0, a_s1)
        stC = subthreshold_metrics(C["v"], C["i_pa"], fsC, c_s0, c_s1)

        # store row for CSV
        per_step_rows.append(dict(
            step_pA=pa_i,
            allen_hz=count_spikes(A["v"], thr=0.0, refrac=0.002, fs=fsA),  # over 1s step
            cortex_hz=(len(tc) / ((c_s1 - c_s0) / fsC)),
            precision=prec, recall=rec, f1=f1,
            isi_mean_ms_allen=isiA["mean_ms"], isi_cv_allen=isiA["cv"], isi_adapt_allen=isiA["adapt"], latency_ms_allen=isiA["latency_ms"],
            isi_mean_ms_cortex=isiC["mean_ms"], isi_cv_cortex=isiC["cv"], isi_adapt_cortex=isiC["adapt"], latency_ms_cortex=isiC["latency_ms"],
            ap_peak_mV_allen=apA["peak_mV_avg"], ap_halfwidth_ms_allen=apA["halfwidth_ms_avg"], ap_ahp_mV_allen=apA["ahp_depth_mV_avg"], dvdtmax_mVms_allen=apA["dvdtmax_mVms_avg"],
            ap_peak_mV_cortex=apC["peak_mV_avg"], ap_halfwidth_ms_cortex=apC["halfwidth_ms_avg"], ap_ahp_mV_cortex=apC["ahp_depth_mV_avg"], dvdtmax_mVms_cortex=apC["dvdtmax_mVms_avg"],
            baseline_mV_allen=stA["baseline_mV"], dV_ss_mV_allen=stA["dV_ss_mV"], Rin_MOhm_allen=stA["Rin_MOhm"], tau_ms_allen=stA["tau_ms"],
            baseline_mV_cortex=stC["baseline_mV"], dV_ss_mV_cortex=stC["dV_ss_mV"], Rin_MOhm_cortex=stC["Rin_MOhm"], tau_ms_cortex=stC["tau_ms"],
        ))

        # and dump a detailed JSON per step
        per_step_json[str(pa_i)] = dict(
            precision=prec, recall=rec, f1=f1,
            matched_pairs=pairs,
            isi_allen=isiA, isi_cortex=isiC,
            ap_allen=apA, ap_cortex=apC,
            subthreshold_allen=stA, subthreshold_cortex=stC
        )

    # Write consolidated CSV + per-step JSON
    _ensure_outdir()
    with open(OUTDIR/"per_step_metrics.csv","w",newline="") as f:
        cols = list(per_step_rows[0].keys()) if per_step_rows else []
        w = csv.DictWriter(f, fieldnames=cols)
        if cols: w.writeheader()
        for r in sorted(per_step_rows, key=lambda d: d["step_pA"]):
            w.writerow(r)
   
    _ensure_outdir()
    with open(OUTDIR/"per_step_metrics.json","w") as jf:
        json.dump(per_step_json, jf, indent=2)
    # Scatter: Allen vs CORTEX F–I with identity + fit
    fig = plt.figure(figsize=(5,5)); ax = plt.gca()
    if allen_v.size:
        ax.scatter(allen_v, cortex_v, s=28)
        lim = float(max(1.0, np.nanmax([allen_v.max(), cortex_v.max()])))
        ax.plot([0,lim],[0,lim], linestyle="--")
        if np.isfinite(m) and np.isfinite(b):
            xs = np.linspace(0, lim, 100); ax.plot(xs, m*xs + b)
    ax.set_xlabel("Allen F–I (Hz)"); ax.set_ylabel("CORTEX F–I (Hz)")
    ax.set_title(f"F–I agreement  (R²={0.0 if not np.isfinite(r2) else r2:.3f}, slope={0.0 if not np.isfinite(m) else m:.2f})")
    plt.tight_layout()
    _ensure_outdir()
    plt.savefig(OUTDIR/"fi_scatter.png", dpi=180); plt.close(fig)

if __name__ == "__main__":
    # example specimen in your cache: 314800874
    run(specimen_id=314800874)
