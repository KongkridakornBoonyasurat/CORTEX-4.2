# cortex/integration/thalamo_cortical_loop_42.py
import torch
import numpy as np
from collections import deque

# --- anchors: these imports must resolve exactly as written ---
from cortex.sensory.biological_eye import BiologicalEye42Optimized
from cortex.regions.thalamus_42 import ThalamusSystem42PyTorch
from cortex.regions.unified_neocortex_42 import UnifiedNeocortex42PyTorch

class ThalamoCorticalLoop42:
    """
    Minimal orchestrator that wires: Eye -> Thalamus -> Neocortex -> (feedback) -> Thalamus
    and exposes an EEG-like proxy signal each step.
    """

    def __init__(self,
                 resolution=(84, 84),
                 thal_neurons=32,
                 thal_channels=8,
                 device=None,
                 history_len=5000):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        # --- build regions (original modules, unmodified) ---
        self.eye = BiologicalEye42Optimized(resolution=resolution, device=self.device)
        self.thalamus = ThalamusSystem42PyTorch(n_neurons=thal_neurons,
                                                n_sensory_channels=thal_channels,
                                                device=self.device)
        self.thalamus.sensory_relay.test_enable_baseline = False
        self.neocortex = UnifiedNeocortex42PyTorch(device=self.device)

        # feedback buffer from cortex->thalamus (size must match thalamic n_neurons)
        self._thal_neurons = thal_neurons
        self.cortical_feedback = torch.zeros(thal_neurons, device=self.device)

        # EEG trace history
        self.eeg_history = deque(maxlen=history_len)
        self.step_idx = 0

        # (optional) soften artificial baseline drive in thalamic sensory relay
        # You can flip this off once the loop is running stably:
        self.thalamus.sensory_relay.test_enable_baseline = False

    def _build_thalamic_sensory_vector(self, eye_out, target_dim=8):
        """
        Build an 8-D (default) sensory vector for the thalamus from the eye output.
        Uses global features + two simple stats for a stable mapping.
        """
        gf = eye_out.get('global_features', None)
        if gf is None:
            # fallback: zeros
            base = torch.zeros(6, device=self.device)
        else:
            # ensure 6-D (motion, activity, contrast, rg, by, color_contrast) per your eye code
            v = gf
            if v.dim() > 1:
                v = v.view(-1)
            if v.shape[0] < 6:
                pad = torch.zeros(6 - v.shape[0], device=self.device)
                v = torch.cat([v, pad], dim=0)
            else:
                v = v[:6]
            base = v.to(self.device)

        ganglion = eye_out.get('ganglion_output', None)
        if ganglion is not None:
            g_mean = torch.mean(ganglion).to(self.device)
            g_std = torch.std(ganglion).to(self.device)
        else:
            g_mean = torch.tensor(0.0, device=self.device)
            g_std = torch.tensor(0.0, device=self.device)

        vec8 = torch.stack([base[0], base[1], base[2], base[3], base[4], base[5], g_mean, g_std], dim=0)

        if target_dim == 8:
            return vec8
        elif target_dim < 8:
            return vec8[:target_dim]
        else:
            # pad with zeros if thalamus expects more channels
            pad = torch.zeros(target_dim - 8, device=self.device)
            return torch.cat([vec8, pad], dim=0)

    def _map_relay_to_neocortex_input(self, relay_vec):
        """
        Map thalamic relay output to neocortex level-0 input.
        Level-0 neuron count is defined in UnifiedNeocortex42 (default 8).
        """
        lvl0_n = self.neocortex.hierarchy.levels[0].n_neurons
        if relay_vec.shape[0] >= lvl0_n:
            return relay_vec[:lvl0_n]
        pad = torch.zeros(lvl0_n - relay_vec.shape[0], device=self.device)
        return torch.cat([relay_vec, pad], dim=0)

    def _get_cortical_feedback(self):
        """
        Use level-0 L6 (prediction) as cortical feedback to thalamus.
        Pad/truncate to thalamic neuron count.
        """
        l6 = self.neocortex.hierarchy.levels[0].L6.activity
        if l6.shape[0] >= self._thal_neurons:
            return l6[:self._thal_neurons]
        pad = torch.zeros(self._thal_neurons - l6.shape[0], device=self.device)
        return torch.cat([l6, pad], dim=0)

    def _compute_eeg_proxy(self, neo_out):
        """
        EEG-like proxy: cortical dipole (L2/3 - L5 mean) + small TC alpha component.
        """
        lvl0 = neo_out['hierarchical_state']['level_outputs'][0]
        l23_mean = torch.mean(lvl0['L2_3'])
        l5_mean  = torch.mean(lvl0['L5'])
        cortical_dipole = (l23_mean - l5_mean)

        # thalamic alpha proxy (emergent TC activity)
        tc_alpha = getattr(self.thalamus, 'alpha_tc_activity', None)
        if tc_alpha is not None and isinstance(tc_alpha, torch.Tensor) and tc_alpha.numel() > 0:
            alpha_term = torch.mean(tc_alpha)
        else:
            alpha_term = torch.tensor(0.0, device=self.device)

        eeg = 0.85 * cortical_dipole + 0.15 * alpha_term
        return float(eeg.detach().cpu().item())

    @torch.no_grad()
    def step(self, raw_visual_input, dt=0.001, t=0.0,
             attention=0.7, arousal=0.8):
        """
        One simulation step:
        Eye -> Thalamus -> Neocortex -> feedback -> EEG proxy
        """
        # 1) eye
        eye_out = self.eye(raw_visual_input, dt=dt, current_time=t)
        thal_x = self._build_thalamic_sensory_vector(eye_out, target_dim=self.thalamus.n_sensory_channels)

        # 2) thalamus (uses previous cortical_feedback)
        th_out = self.thalamus(sensory_input=thal_x,
                               cortical_feedback=self.cortical_feedback,
                               attention_level=attention,
                               arousal_level=arousal,
                               dt=dt,
                               step_idx=self.step_idx)

        # 3) neocortex
        relay = th_out['relay_output']
        if isinstance(relay, torch.Tensor):
            relay_vec = relay.to(self.device)
        else:
            relay_vec = torch.tensor(relay, device=self.device, dtype=torch.float32)

        neo_in = self._map_relay_to_neocortex_input(relay_vec)
        neo_out = self.neocortex(neo_in, dt=dt)

        # 4) update feedback for next step
        self.cortical_feedback = self._get_cortical_feedback()

        # 5) EEG-like proxy
        eeg_val = self._compute_eeg_proxy(neo_out)
        self.eeg_history.append(eeg_val)

        self.step_idx += 1

        return {
            'eeg_proxy': eeg_val,
            'eeg_trace': np.array(self.eeg_history, dtype=np.float32),
            'eye': eye_out,
            'thalamus': th_out,
            'neocortex': neo_out
        }

    def get_eeg_trace(self):
        return np.array(self.eeg_history, dtype=np.float32)
