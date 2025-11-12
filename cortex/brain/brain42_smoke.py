# scripts/brain42_smoke.py
# Minimal end-to-end wiring: Eye -> Sensory -> Neocortex -> Thalamus

import torch

# ---- imports from your repo layout ----
from ..sensory.biological_eye import retina_simulate_optimized, SensoryCortex42Enhanced
from ..regions.thalamus_42 import ThalamusSystem42PyTorch
from ..regions.unified_neocortex_42 import UnifiedNeocortex42PyTorch

def to_device(t):
    return t.cuda() if torch.cuda.is_available() else t.cpu()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SPIKE_PROBE = True
    print(f"[SMOKE] Device = {device}")

    # --- instantiate regions ---
    sensory = SensoryCortex42Enhanced(n_neurons=32, device=device, use_biological_eye=True)
    thal    = ThalamusSystem42PyTorch(n_neurons=16, n_sensory_channels=16, device=device)
    neo     = UnifiedNeocortex42PyTorch(device=device)

    # --- simple loop ---
    dt = 0.001
    for step_idx in range(50):
        # 1) make a small moving scene (84x84, grayscale)
        scene = retina_simulate_optimized(scene="pong", resolution=(84, 84), device=device)

        # 2) sensory path (uses optimized biological eye inside)
        s_out = sensory.process_visual_input(scene, dt=dt, current_time=step_idx*dt)
        
        if step_idx % 10 == 0:
            try:
                nspk = float(torch.sum((s_out['spikes'] > 0.0).float()))
                print(f"[dbg] sensory_spikes>0: {nspk:.0f}")
            except:
                pass        

        # s_out['features'] is [F, H, W] spatial tensor from eye feature extractor
        spatial = s_out['features']  # [F,H,W]
        # average over space -> feature vector [F]
        feat_vec = spatial.view(spatial.shape[0], -1).mean(dim=1)  # [F]

        # 3) feed neocortex (expects ~8 dims at level 0) -> pad/crop to 8
        if feat_vec.shape[0] < 8:
            pad = torch.zeros(8 - feat_vec.shape[0], device=device)
            neo_in = torch.cat([feat_vec, pad], dim=0)
        else:
            neo_in = feat_vec[:8]

        neo_out = neo(neo_in, dt=dt)  # uses internal modulators
        # --- safety checks: catch NaNs/Infs during the run ---
        def _assert_ok(t, name):
            if torch.isnan(t).any() or torch.isinf(t).any():
                raise RuntimeError(f"[SAFETY] {name} has NaN/Inf")
        # 4) thalamus sensory_input needs 16 dims -> pad/crop feat_vec to 16
        if feat_vec.shape[0] < 16:
            pad = torch.zeros(16 - feat_vec.shape[0], device=device)
            thal_sens = torch.cat([feat_vec, pad], dim=0)
        else:
            thal_sens = feat_vec[:16]
        # ---- probe pulses to elicit a visible thalamic response
        if step_idx in (5, 25, 45):
            thal_sens = thal_sens + 3.0
        # optional brief drive to elicit spikes at step 5
        if SPIKE_PROBE and step_idx == 5:
            thal_sens = thal_sens + 6.0  # transient DC bump

        # 5) cortical feedback for thalamus: use prefrontal belief (8 dims) -> tile/pad to 16
        pfc = neo_out['prefrontal_control']  # [8]
        if pfc.shape[0] < 16:
            pfc_fb = torch.cat([pfc, pfc], dim=0)[:16]  # simple repeat to 16
        else:
            pfc_fb = pfc[:16]

        thal_out = thal(
            sensory_input=thal_sens,
            cortical_feedback=pfc_fb,
            attention_level=1.0,
            arousal_level=1.0,
            context_input=None,
            dt=dt,
            step_idx=step_idx
        )
        _assert_ok(neo_out['prefrontal_control'], "neo.prefrontal_control")
        _assert_ok(neo_out['sensory_prediction'], "neo.sensory_prediction")
        _assert_ok(neo_out['top_belief'], "neo.top_belief")
        _assert_ok(thal_out['relay_output'], "thalamus.relay_output")
        # 6) brief summary each step
        err = float(neo_out['total_prediction_error'])
        rel = float(torch.mean(torch.abs(thal_out['relay_output'])))
        act = float(thal_out['neural_activity'])
        burst = thal_out['mode_state']['burst_active']
        tonic = thal_out['mode_state']['tonic_active']
        print(f"[{step_idx:02d}] neocortex_err={err:.4f}  thal_relay={rel:.3f}  thal_act={act:.3f}  burst={burst:.2%} tonic={tonic:.2%}")

    # --- final checks ---
    neo_diag = neo.get_diagnostics()
    thal_state = thal.get_region_state()
    print("\n[OK] Diagnostics:")
    print(f"  Neo avg pred error: {neo_diag['average_prediction_error']:.4f}")
    print(f"  Thal compliance: {thal_state['cortex_42_compliance']:.1%}, burst_fraction={thal_state['burst_fraction']:.1%}")

    # quick health summary
    levels = neo_diag['levels']
    print("  Neo levels:")
    print(f"    sensory  L2/3={levels['sensory']['L2_3_activity']:.3f}  L5={levels['sensory']['L5_belief']:.3f}  precision={levels['sensory']['precision']:.2f}")
    print(f"    parietal L2/3={levels['parietal']['L2_3_activity']:.3f} L5={levels['parietal']['L5_belief']:.3f} precision={levels['parietal']['precision']:.2f}")
    print(f"    pfc      L2/3={levels['prefrontal']['L2_3_activity']:.3f} L5={levels['prefrontal']['L5_belief']:.3f} precision={levels['prefrontal']['precision']:.2f}")

    print(f"  Neo avg pred error: {neo_diag['average_prediction_error']:.4f}")
    print(f"  Thal compliance: {thal_state['cortex_42_compliance']:.1%}, burst_fraction={thal_state['burst_fraction']:.1%}")

if __name__ == "__main__":
    main()
