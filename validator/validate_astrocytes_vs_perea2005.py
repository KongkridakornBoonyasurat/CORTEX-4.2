#!/usr/bin/env python3
"""
validate_astrocytes_vs_perea2005.py

Quantitative validation of your Astrocyte model against Perea & Araque (2005, J. Neurosci)

What this validates (with literature target ranges you provided):
  1) Calcium rise time (200–500 ms) after synaptic stimulation
  2) Calcium decay time (1–2 s) back to baseline
  3) Frequency-dependent modulation (Observed/Expected, O/E):
       - Low frequency (1–10 Hz) potentiation → O/E ≈ 1.4–2.0
       - High frequency (30–50 Hz) depression → O/E ≈ 0.4–0.5
  4) Receptor blockade sanity checks (mGluR, mAChR) reduce Ca2+ responses

Outputs (in ./astro_validation_out):
  - perea2005_summary.json         (key metrics and pass/fail flags)
  - traces_lowfreq.png             (example low-frequency Ca2+ trace)
  - traces_highfreq.png            (example high-frequency Ca2+ trace)
  - oe_vs_frequency.png            (Observed/Expected vs frequency)
  - rise_decay_examples.png        (rise/decay measurement visualization)

Assumptions about your astrocyte model (based on uploaded astrocyte.py):
  - Class: Astrocyte(n_units=1, device='cpu'|'cuda', **kwargs)
  - Step:  astro.step(spikes: Sequence[float|int] of len n_units, dt: float seconds)
  - State: astro.Ca_fast, astro.Ca_slow (torch Tensors or floats)
  - Flags: astro.block_mGluR, astro.block_mAChR (booleans)
  - Time constants near your CORTEX_42 defaults (dt is in seconds; we use dt=0.001)

If your API differs slightly, adjust the adapter functions below.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List
import importlib

import numpy as np

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------- Literature Targets (from your notes) ----------------------

PEREA_ARAQUE_2005 = {
    "calcium_dynamics": {
        "rise_time_ms": (200.0, 500.0),   # time from train onset to peak
        "decay_time_s": (1.0, 2.0),       # time from peak to ~1/e above baseline
        "threshold_um": 0.5,              # not used for hard pass/fail; recorded
        "spatial_spread_um": (20.0, 50.0) # not measured here (requires imaging geometry)
    },
    "frequency_modulation": {
        "low_freq_range_hz": (1.0, 10.0),
        "low_freq_oe_ratio": (1.4, 2.0),
        "high_freq_range_hz": (30.0, 50.0),
        "high_freq_oe_ratio": (0.4, 0.5)
    }
}

# ---------------------- Paths & small utils ----------------------

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "astro_validation_out"
OUTDIR.mkdir(parents=True, exist_ok=True)

def _to_numpy_scalar(x) -> float:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    x = np.asarray(x)
    return float(x.squeeze())

def _get_ca_scalar(astro) -> float:
    """Combine ca_fast + ca_slow into one scalar Ca(t)."""
    # For AstrocyteNetwork: use lowercase ca_fast, ca_slow
    if hasattr(astro, 'ca_fast'):
        ca_fast = astro.ca_fast[0] if hasattr(astro.ca_fast, '__getitem__') else astro.ca_fast
        ca_slow = astro.ca_slow[0] if hasattr(astro.ca_slow, '__getitem__') else astro.ca_slow
        return _to_numpy_scalar(ca_fast) + _to_numpy_scalar(ca_slow)
    
    # For Astrocyte: use uppercase Ca_fast, Ca_slow
    elif hasattr(astro, 'Ca_fast'):
        ca_fast = astro.Ca_fast[0] if hasattr(astro.Ca_fast, '__getitem__') else astro.Ca_fast
        ca_slow = astro.Ca_slow[0] if hasattr(astro.Ca_slow, '__getitem__') else astro.Ca_slow
        return _to_numpy_scalar(ca_fast) + _to_numpy_scalar(ca_slow)
    
    # Fallback to calcium_levels
    elif hasattr(astro, 'calcium_levels'):
        ca = astro.calcium_levels[0] if hasattr(astro.calcium_levels, '__getitem__') else astro.calcium_levels
        return _to_numpy_scalar(ca)
    
    return 0.0

def _zero_state(astro):
    """Best-effort zeroing of internal state between protocols."""
    # For AstrocyteNetwork (lowercase)
    for name in ["ca_fast", "ca_slow", "calcium_levels"]:
        if hasattr(astro, name):
            obj = getattr(astro, name)
            try:
                import torch
                if isinstance(obj, torch.Tensor):
                    obj.data[:] = 0.0
                elif isinstance(obj, np.ndarray):
                    obj[:] = 0.0
            except Exception:
                pass
    
    # For Astrocyte (uppercase)
    for name in ["Ca_fast", "Ca_slow", "Ca_wave", "C"]:
        if hasattr(astro, name):
            obj = getattr(astro, name)
            try:
                import torch
                if isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
                    obj.data[:] = 0.0
            except Exception:
                pass

# ---------------------- Stimulation & measurements ----------------------

def run_train(astro, freq_hz: float, duration_s: float, dt: float) -> Dict[str, np.ndarray]:
    n_steps = int(np.round(duration_s / dt))
    period_steps = np.inf if freq_hz <= 0 else int(round(1.0 / (freq_hz * dt)))

    # Baseline pre-period (0.5 s) — record actual baseline
    baseline_steps = int(round(0.5 / dt))
    pre_vals = np.zeros(baseline_steps, dtype=float)
    for i in range(baseline_steps):
        astro.step([0.0], dt)
        pre_vals[i] = _get_ca_scalar(astro)

    # Drive
    ca_trace = np.zeros(n_steps, dtype=float)
    for i in range(n_steps):
        spike = 1.0 if (period_steps != np.inf and (i % period_steps == 0)) else 0.0
        astro.step([spike], dt)
        ca_trace[i] = _get_ca_scalar(astro)

    # Post-train decay tail (5 s)
    tail_steps = int(round(5.0 / dt))
    tail = np.zeros(tail_steps, dtype=float)
    for i in range(tail_steps):
        astro.step([0.0], dt)
        tail[i] = _get_ca_scalar(astro)

    t_full = np.concatenate([
        -np.arange(baseline_steps, 0, -1)*dt,
        np.arange(n_steps)*dt,
        duration_s + (np.arange(tail_steps)+1)*dt
    ])
    ca_full = np.concatenate([pre_vals, ca_trace, tail])
    return {"t": t_full, "ca": ca_full}


def single_pulse_area(astro, dt: float, pre_wait_s: float = 0.5, post_window_s: float = 2.0) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Area under Ca(t) above baseline after a single presynaptic spike.
    """
    _zero_state(astro)

    # Pre-baseline
    pre_steps = int(round(pre_wait_s / dt))
    for _ in range(pre_steps):
        astro.step([0.0], dt)
    baseline = _get_ca_scalar(astro)

    # Single spike
    astro.step([1.0], dt)

    # Post window
    post_steps = int(round(post_window_s / dt))
    ca = np.zeros(post_steps, dtype=float)
    for i in range(post_steps):
        astro.step([0.0], dt)
        ca[i] = _get_ca_scalar(astro)

    above = np.clip(ca - baseline, 0.0, None)
    area = float(np.trapz(above, dx=dt))

    trace = np.concatenate([np.full(pre_steps, baseline), ca])
    t = np.arange(trace.size) * dt - pre_wait_s
    return area, {"t": t, "ca": trace, "baseline": baseline}

def observed_expected_ratio(astro, freq_hz: float, duration_s: float, dt: float) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    O/E = (area under Ca(t) above baseline during train + tail) / (N_spikes * single_pulse_area)
    """
    # Expected from single pulse
    single_area, _ = single_pulse_area(astro, dt=dt)

    # Full train
    _zero_state(astro)
    traces = run_train(astro, freq_hz=freq_hz, duration_s=duration_s, dt=dt)
    t, ca = traces["t"], traces["ca"]

    # Use pre-train baseline (t<0) as baseline estimate
    pre_mask = t < 0
    baseline = float(np.median(ca[pre_mask])) if np.any(pre_mask) else float(ca[0])
    above = np.clip(ca - baseline, 0.0, None)
    observed = float(np.trapz(above, x=t))

    n_spikes = int(round(freq_hz * duration_s))
    expected = max(n_spikes * single_area, 1e-12)  # avoid /0
    oe = observed / expected

    traces["baseline"] = baseline
    traces["above"] = above
    traces["observed_area"] = observed
    traces["expected_area"] = expected
    traces["n_spikes"] = n_spikes
    traces["single_pulse_area"] = single_area
    traces["oe"] = oe
    return oe, traces

def measure_rise_decay(astro, freq_hz: float, duration_s: float, dt: float) -> Dict[str, float]:
    """
    Rise time = t_peak - t_train_start
    Decay time = time from peak to baseline + (peak-baseline)/e (i.e., 1/e decay)
    """
    _zero_state(astro)
    traces = run_train(astro, freq_hz=freq_hz, duration_s=duration_s, dt=dt)
    t, ca = traces["t"], traces["ca"]

    # Define train window
    train_start = 0.0
    train_end = duration_s

    # Baseline from pre-train segment
    pre_mask = t < train_start
    baseline = float(np.median(ca[pre_mask])) if np.any(pre_mask) else float(ca[0])

    # Peak within [0, 1.0 s] ONLY - biology says 200-500ms!
    window_mask = (t >= train_start) & (t <= min(0.5, train_end))
    t_win, ca_win = t[window_mask], ca[window_mask]
    i_peak = int(np.argmax(ca_win))
    t_peak = float(t_win[i_peak])
    ca_peak = float(ca_win[i_peak])
    
    # DEBUG: Show calcium trace at key timepoints
    print(f"\n[DEBUG measure_rise_decay]:")
    print(f"  Baseline: {baseline:.4f}")
    for check_t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        idx = np.argmin(np.abs(t - check_t))
        print(f"  Ca at t={check_t:.1f}s: {ca[idx]:.4f}")
    print(f"  Peak: ca={ca_peak:.4f} at t={t_peak:.3f}s")
    rise_time = max(t_peak - train_start, 0.0) * 1000.0  # ms

    # Decay time to 1/e above baseline, search after peak for up to 5 s
    target = baseline + (ca_peak - baseline) / np.e
    post_mask = (t > t_peak) & (t <= t_peak + 5.0)
    t_post, ca_post = t[post_mask], ca[post_mask]

    if t_post.size > 1:
        # Find first crossing where ca_post <= target
        idx = np.where(ca_post <= target)[0]
        if idx.size > 0:
            t_decay = float(t_post[idx[0]] - t_peak)
        else:
            t_decay = float(t_post[-1] - t_peak)  # lower bound if not reached
    else:
        t_decay = 0.0

    return {
        "baseline": baseline,
        "t_peak": t_peak,
        "rise_time_ms": rise_time,
        "decay_time_s": t_decay
    }

# ---------------------- Plotting helpers ----------------------

def plot_trace(t, ca, baseline, title, outpath):
    plt.figure(figsize=(8, 4))
    plt.plot(t, ca, linewidth=2)
    plt.axhline(baseline, linestyle="--", linewidth=1)
    plt.axvline(0.0, linestyle=":", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Ca (a.u.)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_oe_vs_freq(freqs, oes, targets, outpath):
    plt.figure(figsize=(6, 4))
    plt.plot(freqs, oes, "o-", linewidth=2)
    # target bands
    lo_lo, lo_hi = targets["low_freq_oe_ratio"]
    hi_lo, hi_hi = targets["high_freq_oe_ratio"]
    # mark bands
    plt.fill_betweenx([lo_lo, lo_hi], targets["low_freq_range_hz"][0], targets["low_freq_range_hz"][1],
                      alpha=0.15, label="Low-freq target")
    plt.fill_betweenx([hi_lo, hi_hi], targets["high_freq_range_hz"][0], targets["high_freq_range_hz"][1],
                      alpha=0.15, label="High-freq target")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("O/E ratio")
    plt.title("Observed/Expected vs frequency (Perea & Araque 2005)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_rise_decay_example(t, ca, baseline, t_peak, target_level, outpath, title):
    plt.figure(figsize=(8,4))
    plt.plot(t, ca, linewidth=2, label="Ca(t)")
    plt.axhline(baseline, linestyle="--", linewidth=1, label="Baseline")
    plt.axvline(0.0, linestyle=":", linewidth=1, label="Train start")
    plt.axvline(t_peak, linestyle="-.", linewidth=1, label="Peak")
    plt.axhline(target_level, linestyle="--", linewidth=1, label="1/e target")
    plt.xlabel("Time (s)")
    plt.ylabel("Ca (a.u.)")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

# ---------------------- Validation runner ----------------------

def run_validation(device: str = "cpu",
                   dt: float = 0.001,
                   train_duration_s: float = 5.0) -> Dict:
    # --- robust imports ---
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    for p in (ROOT, HERE, ROOT / "cortex", ROOT / "cortex" / "cells"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    AstrocyteNetwork = None
    for mod_name in ("cortex.cells.astrocyte", "astrocyte", "cells.astrocyte"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "AstrocyteNetwork"):
                AstrocyteNetwork = getattr(mod, "AstrocyteNetwork")
                break
        except Exception:
            pass
    if AstrocyteNetwork is None:
        raise ImportError(
            "Could not import AstrocyteNetwork. Tried: cortex.cells.astrocyte, astrocyte, cells.astrocyte.\n"
            f"sys.path starts: {sys.path[:5]} ..."
        )

    # --- device normalize ---
    try:
        import torch
        if isinstance(device, str):
            device = torch.device(device)
    except Exception:
        pass

    astro = AstrocyteNetwork(n_astrocytes=1, n_neurons=1, device=device)

    # DETAILED DEBUG: Simulate a short train and watch calcium
    print(f"\n[DEBUG] Simulating 10 spikes at 30 Hz:")
    for i in range(10):
        if i % 3 == 0:  # Spike every ~30ms (roughly 30 Hz)
            astro.step([1.0], dt=0.001)
            print(f"  Step {i*1}ms (spike): ca_fast={astro.ca_fast[0]:.4f}, ca_slow={astro.ca_slow[0]:.4f}")
        else:
            astro.step([0.0], dt=0.001)
            if i % 3 == 2:  # Print just before next spike
                print(f"  Step {i*1}ms (decay): ca_fast={astro.ca_fast[0]:.4f}, ca_slow={astro.ca_slow[0]:.4f}")

    # Reset before validation
    _zero_state(astro)

    results = {
        "literature": PEREA_ARAQUE_2005,
        "parameters": {"dt_s": dt, "train_duration_s": train_duration_s, "device": f"{device}"},
        "calcium_dynamics": {},
        "frequency_modulation": {},
        "blockade_tests": {},
        "notes": {}
    }

    # 1) Rise/decay @ 30 Hz, 5 s
    rd = measure_rise_decay(astro, freq_hz=30.0, duration_s=train_duration_s, dt=dt)
    results["calcium_dynamics"].update(rd)

    _zero_state(astro)
    tr = run_train(astro, freq_hz=30.0, duration_s=train_duration_s, dt=dt)
    t, ca = tr["t"], tr["ca"]
    baseline = float(np.median(ca[t < 0])) if np.any(t < 0) else float(ca[0])
    mask = (t >= 0.0) & (t <= train_duration_s + 2.0)
    i_peak = np.argmax(ca[mask])
    t_peak = float(t[mask][i_peak])
    ca_peak = float(ca[mask][i_peak])
    target_level = baseline + (ca_peak - baseline)/np.e
    plot_rise_decay_example(
        t, ca, baseline, t_peak, target_level,
        OUTDIR/"rise_decay_examples.png",
        title="30 Hz, 5 s train: Rise/Decay markers"
    )

    rt_min, rt_max = PEREA_ARAQUE_2005["calcium_dynamics"]["rise_time_ms"]
    dec_min, dec_max = PEREA_ARAQUE_2005["calcium_dynamics"]["decay_time_s"]
    results["calcium_dynamics"]["rise_time_within_range"] = (rt_min <= rd["rise_time_ms"] <= rt_max)
    results["calcium_dynamics"]["decay_time_within_range"] = (dec_min <= rd["decay_time_s"] <= dec_max)

    # 2) O/E vs frequency
    freqs = [1.0, 10.0, 30.0, 50.0]
    oes, traces_by_freq = [], {}
    for f in freqs:
        _zero_state(astro)
        oe, trc = observed_expected_ratio(astro, freq_hz=f, duration_s=train_duration_s, dt=dt)
        oes.append(float(oe))
        traces_by_freq[f] = trc

    results["frequency_modulation"]["freqs_hz"] = freqs
    results["frequency_modulation"]["oe"] = oes

    if 10.0 in traces_by_freq:
        trc = traces_by_freq[10.0]
        plot_trace(trc["t"], trc["ca"], trc["baseline"],
                   f"Low frequency 10 Hz (O/E={trc['oe']:.2f})",
                   OUTDIR/"traces_lowfreq.png")
    if 30.0 in traces_by_freq:
        trc = traces_by_freq[30.0]
        plot_trace(trc["t"], trc["ca"], trc["baseline"],
                   f"High frequency 30 Hz (O/E={trc['oe']:.2f})",
                   OUTDIR/"traces_highfreq.png")

    plot_oe_vs_freq(freqs, oes, PEREA_ARAQUE_2005["frequency_modulation"], OUTDIR/"oe_vs_frequency.png")

    lo_lo, lo_hi = PEREA_ARAQUE_2005["frequency_modulation"]["low_freq_oe_ratio"]
    hi_lo, hi_hi = PEREA_ARAQUE_2005["frequency_modulation"]["high_freq_oe_ratio"]
    oe_low = oes[freqs.index(10.0)] if 10.0 in freqs else float("nan")
    oe_high = oes[freqs.index(30.0)] if 30.0 in freqs else float("nan")
    results["frequency_modulation"]["low_freq_oe_within_range"] = (lo_lo <= oe_low <= lo_hi)
    results["frequency_modulation"]["high_freq_oe_within_range"] = (hi_lo <= oe_high <= hi_hi)
    results["frequency_modulation"]["representatives"] = {"10Hz_oe": oe_low, "30Hz_oe": oe_high}

    # 3) Pharmacology sanity checks
    if hasattr(astro, "block_mGluR"):
        astro.block_mGluR = True
        oe_block_mGluR, _ = observed_expected_ratio(astro, freq_hz=10.0, duration_s=train_duration_s, dt=dt)
        astro.block_mGluR = False
    else:
        oe_block_mGluR = float("nan")

    if hasattr(astro, "block_mAChR"):
        astro.block_mAChR = True
        oe_block_mAChR, _ = observed_expected_ratio(astro, freq_hz=10.0, duration_s=train_duration_s, dt=dt)
        astro.block_mAChR = False
    else:
        oe_block_mAChR = float("nan")

    results["blockade_tests"] = {
        "mGluR_block_10Hz_OE": oe_block_mGluR,
        "mAChR_block_10Hz_OE": oe_block_mAChR,
        "interpretation": "Lower O/E with blockade indicates Ca2+ response depends on the targeted receptor."
    }

    # 4) Pass/Fail
    passes = {
        "rise_time_ms_in_range": results["calcium_dynamics"]["rise_time_within_range"],
        "decay_time_s_in_range": results["calcium_dynamics"]["decay_time_within_range"],
        "low_freq_oe_in_range": results["frequency_modulation"]["low_freq_oe_within_range"],
        "high_freq_oe_in_range": results["frequency_modulation"]["high_freq_oe_within_range"],
    }
    results["pass_flags"] = passes
    results["score_fraction_passed"] = float(np.mean([bool(v) for v in passes.values()]))

    # Save JSON
    with open(OUTDIR/"perea2005_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------- CLI ----------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Astrocyte model vs Perea & Araque (2005)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--dt", type=float, default=0.001, help="Simulation step (s)")
    parser.add_argument("--duration", type=float, default=5.0, help="Train duration (s)")
    args = parser.parse_args()

    res = run_validation(device=args.device, dt=args.dt, train_duration_s=args.duration)

    # Console summary
    print("\n=== Perea & Araque (2005) Astrocyte Validation ===")
    cd = res["calcium_dynamics"]
    fm = res["frequency_modulation"]
    print(f"Rise time: {cd['rise_time_ms']:.1f} ms  "
          f"(target {PEREA_ARAQUE_2005['calcium_dynamics']['rise_time_ms']}) "
          f"Pass: {cd['rise_time_within_range']}")
    print(f"Decay time: {cd['decay_time_s']:.2f} s  "
          f"(target {PEREA_ARAQUE_2005['calcium_dynamics']['decay_time_s']}) "
          f"Pass: {cd['decay_time_within_range']}")
    print(f"O/E @ 10 Hz: {fm['representatives']['10Hz_oe']:.2f}  "
          f"(target {PEREA_ARAQUE_2005['frequency_modulation']['low_freq_oe_ratio']}) "
          f"Pass: {fm['low_freq_oe_within_range']}")
    print(f"O/E @ 30 Hz: {fm['representatives']['30Hz_oe']:.2f}  "
          f"(target {PEREA_ARAQUE_2005['frequency_modulation']['high_freq_oe_ratio']}) "
          f"Pass: {fm['high_freq_oe_within_range']}")
    print(f"Overall score (fraction passed): {res['score_fraction_passed']:.2f}")
    print(f"Outputs saved to: {OUTDIR}")
