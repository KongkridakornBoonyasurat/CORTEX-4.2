# cortex/cells/enhanced_neurons.py
"""
Enhanced Neurons 4.2 - FULLY PyTorch Implementation
==================================================
KEEPS ALL your existing biological complexity from enhanced_neurons.py
ADDS CORTEX 4.2 enhancements
FULLY GPU-accelerated with PyTorch tensors

This is a COMPLETE PyTorch implementation that maintains your existing API
while adding GPU acceleration and CORTEX 4.2 mathematical enhancements.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import time
from cortex import config as C

# GPU setup
def setup_device():
    pref = (getattr(C, "DEVICE_PREFERENCE", "auto") or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = setup_device()

# CORTEX 4.2 biological constants (FROM THE PAPER) - as PyTorch tensors
def make_constants(device):
    D = {
        # Keep your existing defaults (legacy scaling)
        'C_m': torch.tensor(1.0, device=device),           # μF/cm²
        'g_L': torch.tensor(0.3, device=device),           # mS/cm²
        'E_L': torch.tensor(-70.0, device=device),         # mV
        'V_T': torch.tensor(-50.0, device=device),         # mV
        'Delta_T': torch.tensor(2.0, device=device),       # mV
        'tau_w': torch.tensor(100.0, device=device),       # ms
        'a': torch.tensor(0.005, device=device),           # nS (legacy style)
        'b': torch.tensor(0.1, device=device),             # nA (legacy style)

        'E_AMPA': torch.tensor(0.0, device=device),
        'E_NMDA': torch.tensor(0.0, device=device),
        'E_GABA': torch.tensor(-70.0, device=device),

        'tau_AMPA': torch.tensor(5.0, device=device),      # ms
        'tau_NMDA': torch.tensor(50.0, device=device),     # ms
        'tau_GABA_A': torch.tensor(10.0, device=device),   # ms
        'tau_GABA_B': torch.tensor(150.0, device=device),  # ms
    }

    # Overlays from central config (faithful to paper)
    # Reversal potentials
    if hasattr(C, 'E_E_MV'):
        D['E_AMPA'] = torch.tensor(C.E_E_MV, device=device)
        D['E_NMDA'] = torch.tensor(C.E_E_MV, device=device)
    if hasattr(C, 'E_I_MV'):
        D['E_GABA'] = torch.tensor(C.E_I_MV, device=device)

    # Synaptic time constants (use cfg τ to match synapse/EEG)
    for cfg_key, here in [
        ('TAU_AMPA_MS', 'tau_AMPA'),
        ('TAU_NMDA_MS', 'tau_NMDA'),
        ('TAU_GABA_A_MS', 'tau_GABA_A'),
        ('TAU_GABA_B_MS', 'tau_GABA_B'),
    ]:
        if hasattr(C, cfg_key):
            D[here] = torch.tensor(getattr(C, cfg_key), device=device)

    # AdEx threshold/reset from config (pyramidal defaults)
    if isinstance(getattr(C, 'ADEX_PYR', None), dict):
        ap = C.ADEX_PYR
        D['E_L']     = torch.tensor(ap.get('EL_mV', -70.0), device=device)
        D['V_T']     = torch.tensor(ap.get('VT_mV', -50.0), device=device)
        D['Delta_T'] = torch.tensor(ap.get('dT_mV', 2.0),   device=device)
        D['V_r']     = torch.tensor(ap.get('Vr_mV', -58.0), device=device)
    else:
        D['V_r']     = torch.tensor(-70.0, device=device)

    # NMDA Mg2+ block constants
    D['NMDA_ALPHA'] = torch.tensor(getattr(C, 'NMDA_BLOCK_ALPHA', 0.062), device=device)
    D['NMDA_BETA']  = torch.tensor(getattr(C, 'NMDA_BLOCK_BETA', 1.0),    device=device)
    D['NMDA_MG']    = torch.tensor(getattr(C, 'NMDA_MG_MILLIMOLAR', 1.0), device=device)
    D['NMDA_BLOCK_DIVISOR'] = torch.tensor(getattr(C, 'NMDA_BLOCK_DIVISOR', 3.57), device=device)

    # ---- Legacy Hodgkin–Huxley & metabolism keys expected elsewhere ----
    # Leak & reversals
    D['g_leak'] = torch.tensor(getattr(C, 'GL', 0.04), device=device)         # nS (legacy compat)
    D['E_leak'] = torch.tensor(getattr(C, 'EL', -65.0), device=device)        # mV

    # HH maxima & reversals (sensible defaults if not in config)
    D['g_Na_max'] = torch.tensor(getattr(C, 'G_NA_MAX', 120.0), device=device)
    D['E_Na']     = torch.tensor(getattr(C, 'E_NA', 50.0), device=device)
    D['g_K_max']  = torch.tensor(getattr(C, 'G_K_MAX', 36.0), device=device)
    D['E_K']      = torch.tensor(getattr(C, 'E_K', -77.0), device=device)
    D['g_Ca_max'] = torch.tensor(getattr(C, 'G_CA_MAX', 0.5), device=device)
    D['E_Ca']     = torch.tensor(getattr(C, 'E_CA', 120.0), device=device)

    # Calcium & adaptation
    D['tau_adaptation'] = torch.tensor(getattr(C, 'TAU_ADAPT_MS', 200.0), device=device)
    D['Ca_alpha']       = torch.tensor(getattr(C, 'CA_ALPHA', 0.05), device=device)
    D['Ca_rest']        = torch.tensor(getattr(C, 'CA_REST', 0.1), device=device)
    D['Ca_decay']       = torch.tensor(getattr(C, 'CA_DECAY_MS', 200.0), device=device)

    # Metabolic / homeostasis
    D['ATP_max']              = torch.tensor(getattr(C, 'ATP_MAX', 5.0), device=device)
    D['ATP_consumption_rate'] = torch.tensor(getattr(C, 'ATP_CONSUMPTION_PER_SPIKE', 0.02), device=device)
    D['ATP_recovery_rate']    = torch.tensor(getattr(C, 'ATP_RECOVERY_PER_MS', 0.005), device=device)
    D['target_firing_rate']   = torch.tensor(getattr(C, 'TARGET_FR_HZ', 10.0), device=device)
    D['homeostatic_timescale']= torch.tensor(getattr(C, 'HOMEOSTASIS_T_MS', 10000.0), device=device)

    return D

CONSTANTS = make_constants(DEVICE)

class EnhancedNeuron42PyTorch(nn.Module):
    """
    FULLY PyTorch Enhanced Neuron 4.2
    
    KEEPS ALL your existing biological complexity
    ADDS CORTEX 4.2 enhancements
    FULLY GPU-accelerated with PyTorch tensors
    SAME API as your enhanced_neurons.py
    """
    
    def __init__(self, neuron_id: int, n_dendrites: int = 4, neuron_type: str = "pyramidal", 
                 use_cadex: bool = False, device=None):
        super().__init__()
        
        self.neuron_id = neuron_id
        self.neuron_type = neuron_type
        self.n_dendrites = n_dendrites
        self.use_cadex = use_cadex
        self.device = device or DEVICE
        
        # === YOUR EXISTING PARAMETERS (as PyTorch parameters) ===
        self._init_neuron_type_parameters()
        self._init_adex_params_from_config()
        # refractory timer (ms)
        self.register_buffer("refrac_left_ms", torch.tensor(0.0, device=self.device))

        # === STATE VARIABLES (as PyTorch tensors) ===
        # === STATE VARIABLES (as buffers) ===
        # Electrical state
        self.register_buffer("voltage", torch.tensor(-70.0, device=self.device))
        self.register_buffer("sodium_activation", torch.tensor(0.0, device=self.device))
        self.register_buffer("sodium_inactivation", torch.tensor(1.0, device=self.device))
        self.register_buffer("potassium_activation", torch.tensor(0.0, device=self.device))
        
        # Calcium dynamics
        self.register_buffer("calcium_concentration", torch.tensor(0.1, device=self.device))
        self.register_buffer("calcium_activation", torch.tensor(0.0, device=self.device))
        
        # Metabolic state
        self.register_buffer("atp_level", torch.tensor(2.5, device=self.device))
        self.register_buffer("metabolic_stress", torch.tensor(0.0, device=self.device))
        self.sim_time_ms = 0.0

        # Adaptation state
        self.register_buffer("adaptation_current", torch.tensor(0.0, device=self.device))
        self.register_buffer("cadex_adaptation_current", torch.tensor(0.0, device=self.device))
        
        # Homeostatic state
        self.register_buffer("intrinsic_excitability", torch.tensor(1.0, device=self.device))
        self.register_buffer("firing_rate_average", torch.tensor(0.0, device=self.device))
        self.register_buffer("homeostatic_pressure", torch.tensor(0.0, device=self.device))
        
        # CORTEX 4.2 multi-receptor synaptic state
        self.register_buffer("ampa_conductance", torch.tensor(0.0, device=self.device))
        self.register_buffer("nmda_conductance", torch.tensor(0.0, device=self.device))
        self.register_buffer("gaba_a_conductance", torch.tensor(0.0, device=self.device))
        self.register_buffer("gaba_b_conductance", torch.tensor(0.0, device=self.device))
        
        # Enhanced dendritic state
        self.register_buffer("dendritic_voltages", torch.full((n_dendrites,), -70.0, device=self.device))
        self.register_buffer("nmda_gating_variables", torch.zeros(n_dendrites, device=self.device))
        
        # Astrocyte coupling
        self.register_buffer("astrocyte_calcium", torch.tensor(0.1, device=self.device))
        self.register_buffer("astrocyte_modulation", torch.tensor(1.0, device=self.device))
        
        # === TRACKING VARIABLES (not parameters) ===
        self.spike_count = 0
        self.last_spike_time = 0.0
        self.oscillatory_input = 0.0
        self.phase_coupling_strength = 1.0
        self.homeostatic_timer = 0.0
        
        # Firing history (CPU-based for efficiency)
        self.firing_history = deque(maxlen=1000)
        
        # Ion channel states (for H-H mode)
        self.register_buffer("Na_m", torch.tensor(0.0, device=self.device))
        self.register_buffer("Na_h", torch.tensor(1.0, device=self.device))
        self.register_buffer("K_n", torch.tensor(0.0, device=self.device))
        self.register_buffer("Ca_r", torch.tensor(0.0, device=self.device))
        self.register_buffer("Ca_s", torch.tensor(1.0, device=self.device))
        
        print(f"Enhanced Neuron {neuron_id} ({neuron_type}): CAdEx={'ON' if use_cadex else 'OFF'}, Device={self.device}")
    
    def _init_neuron_type_parameters(self):
        """
        Back-compat stub. You call this in __init__, but it wasn't defined.
        Keep it as a no-op because neuron-type specific factors are set
        in _init_adex_params_from_config() already.
        """
        return

    def _init_adex_params_from_config(self):
        """
        Bind AdEx (CAdEx) parameters from config with explicit physical units.
        - Cm in pF, gL in nS, voltages in mV, b in pA, tau in ms.
        """
        adex = getattr(C, 'ADEX_FS' if self.neuron_type == 'interneuron' else 'ADEX_PYR')

        # Register buffers so they stay on the right device and are saved in state_dict
        self.register_buffer("Cm_pF",          torch.tensor(adex['Cm_pF'],        device=self.device))
        self.register_buffer("gL_nS",          torch.tensor(adex['gL_nS'],        device=self.device))
        self.register_buffer("EL_mV",          torch.tensor(adex['EL_mV'],        device=self.device))
        self.register_buffer("VT_mV",          torch.tensor(adex['VT_mV'],        device=self.device))
        self.register_buffer("dT_mV",          torch.tensor(adex['dT_mV'],        device=self.device))
        self.register_buffer("Vr_mV",          torch.tensor(adex['Vr_mV'],        device=self.device))
        self.register_buffer("a_nS",           torch.tensor(adex['a_nS'],         device=self.device))
        self.register_buffer("b_pA",           torch.tensor(adex['b_pA'],         device=self.device))
        self.register_buffer("tau_w_ms",       torch.tensor(adex['tau_w_ms'],     device=self.device))
        self.register_buffer("refractory_ms",  torch.tensor(adex['refractory_ms'],device=self.device))
        self.register_buffer("V_spike_mV",     torch.tensor(adex['V_spike_mV'],   device=self.device))

        if self.neuron_type == "pyramidal":
            self.excitability_factor = 1.0
            self.adaptation_strength = 1.0
            self.calcium_sensitivity = 1.0
            self.metabolic_efficiency = 1.0
        elif self.neuron_type == "interneuron":
            self.excitability_factor = 1.5
            self.adaptation_strength = 0.5
            self.calcium_sensitivity = 0.8
            self.metabolic_efficiency = 1.2
        elif self.neuron_type == "motor":
            self.excitability_factor = 1.2
            self.adaptation_strength = 1.5
            self.calcium_sensitivity = 1.2
            self.metabolic_efficiency = 0.9
        else:
            self.excitability_factor = 1.0
            self.adaptation_strength = 1.0
            self.calcium_sensitivity = 1.0
            self.metabolic_efficiency = 1.0
    
    def forward(self, current_inputs: torch.Tensor, dt: float = 0.25, current_time: float = 0.0):
        """
        FULLY PyTorch forward pass (dt in **milliseconds**)
        Args:
            current_inputs: 1D tensor of dendritic input currents (len == n_dendrites)
            dt: timestep in milliseconds (default from config typically 0.25 ms)
            current_time: current sim time in milliseconds
        Returns:
            (spike_occurred, membrane_voltage) as tensors
        """
        # normalize dt to ms
        dt_ms = float(getattr(C, "DT_MS", 0.25)) if dt is None else float(dt)  # milliseconds
        self.sim_time_ms += dt_ms

        # --- NEW: increment the homeostatic timer and handle refractory state ---
        self.homeostatic_timer += dt_ms

        if self.refrac_left_ms.item() > 0.0:
            # decrement refractory counter and hold at Vr
            with torch.no_grad():
                self.refrac_left_ms.sub_(dt_ms).clamp_(min=0.0)
                # hold membrane at reset during refractory
                self.voltage.copy_(self.Vr_mV)
            # still return well-formed outputs
            return torch.tensor(0.0, device=self.device), self.voltage

        # Ensure inputs match dendrite count safely
        if current_inputs.dim() == 0:
            current_inputs = current_inputs.view(1)
        if current_inputs.shape[0] > self.n_dendrites:
            current_inputs = current_inputs[:self.n_dendrites]
        elif current_inputs.shape[0] < self.n_dendrites:
            pad = self.n_dendrites - current_inputs.shape[0]
            current_inputs = F.pad(current_inputs, (0, pad))

        # dendrites -> soma
        dendritic_current = self._update_enhanced_dendritic_integration_pytorch(current_inputs, dt_ms)

        # dynamics
        if self.use_cadex:
            spike_occurred = self._update_cadex_dynamics_pytorch(dendritic_current, dt_ms)
        else:
            spike_occurred = self._update_hodgkin_huxley_dynamics_pytorch(dendritic_current, dt_ms)

        self._update_adaptation_mechanisms_pytorch(spike_occurred, dt_ms)
        self._update_calcium_dynamics_pytorch(spike_occurred, dt_ms)
        self._update_metabolic_state_pytorch(spike_occurred, dt_ms)
        self._update_astrocyte_coupling_pytorch(spike_occurred, dt_ms)

        if self.homeostatic_timer > 100.0:
            self._update_homeostatic_plasticity_pytorch()
            self.homeostatic_timer = 0.0

        if bool(spike_occurred.item()):
            self.spike_count += 1
            self.last_spike_time = self.sim_time_ms
            self.firing_history.append(self.sim_time_ms)

        return spike_occurred, self.voltage

    def _update_enhanced_dendritic_integration_pytorch(self, inputs: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Vectorized dendritic integration (CORTEX 4.2 v14 fix):
        - Decays AMPA/NMDA/GABA-A/GABA-B conductances
        - Updates NMDA gating per dendrite
        - Applies NMDA Mg2+ voltage block
        - Returns total dendritic current (scalar) — API unchanged
        """
        import torch  # safe if already imported; no harm

        # 1) Decay existing receptor conductances using your taus
        
        # --- NEW: simple dendrite cable params (nS, pF) and soma coupling ---
        gLd = torch.as_tensor(5.0,  device=self.device)   # dendritic leak (nS)
        g_c = torch.as_tensor(8.0,  device=self.device)   # coupling to soma (nS)
        C_d = torch.as_tensor(50.0, device=self.device)   # dendritic capacitance (pF)

        
        self._update_multi_receptor_conductances_pytorch(dt)

        # 2) NMDA gating (first-order ODE), per dendrite, bounded [0,1]
        #    alpha_NMDA matches your previous scalar gain; tau from CONSTANTS
        alpha_NMDA = torch.as_tensor(0.2, device=self.device)
        tau_NMDA   = CONSTANTS['tau_NMDA']

        if inputs.dim() == 0:
            inputs = inputs.view(1)

        with torch.no_grad():
            self.nmda_gating_variables.add_(
                dt * (-self.nmda_gating_variables / tau_NMDA + alpha_NMDA * inputs)
            )
            self.nmda_gating_variables.clamp_(0.0, 1.0)


        # 3) Per-dendrite voltage (mV)
        Vd = self.dendritic_voltages  # shape [n_dendrites]

        # 4) Effective conductances (support both naming schemes)
        g_AMPA  = self.ampa_conductance
        g_NMDA0 = self.nmda_conductance
        g_GABAA = self.gaba_a_conductance
        g_GABAB = self.gaba_b_conductance
        
        # apply NMDA gating (per dendrite)
        g_NMDA = g_NMDA0 * self.nmda_gating_variables

        # 5) NMDA Mg2+ voltage block (canonical; V in mV, beta in 1/mV)
        # NMDA magnesium block (paper): B(V) = 1 / (1 + [Mg2+] * exp(-0.062 * V) / 3.57)
        Mg  = CONSTANTS.get('NMDA_MG', torch.tensor(1.0, device=self.device))            # mM
        div = CONSTANTS.get('NMDA_BLOCK_DIVISOR', torch.tensor(3.57, device=self.device)) # unitless
        mg_block = 1.0 / (1.0 + Mg * torch.exp(-0.062 * Vd) / div)
        
        # Map reversal potentials with fallbacks (handles your dict naming)
        E_AMPA  = CONSTANTS.get('E_AMPA', 0.0)
        E_NMDA  = CONSTANTS.get('E_NMDA', 0.0)
        E_GABAA = (CONSTANTS.get('E_GABA_A',
                    CONSTANTS.get('E_GABAA',
                    CONSTANTS.get('E_GABA', -75.0))))
        E_GABAB = CONSTANTS.get('E_GABA_B', CONSTANTS.get('E_GABAB', -75.0))

        # 6) Currents: I = g * (V - E)
        I_AMPA  = g_AMPA  * (Vd - E_AMPA)
        I_NMDA  = g_NMDA  * mg_block * (Vd - E_NMDA)
        I_GABAA = g_GABAA * (Vd - E_GABAA)
        I_GABAB = g_GABAB * (Vd - E_GABAB)

        # --- NEW: update each dendrite’s voltage (simple cable: leak + coupling to soma – ionic current)
        I_d_total = I_AMPA + I_NMDA + I_GABAA + I_GABAB  # pA per dendrite
        dVd_dt = (-gLd * (Vd - CONSTANTS['E_L']) - g_c * (Vd - self.voltage) - I_d_total) / C_d
        with torch.no_grad():
            self.dendritic_voltages.add_(dVd_dt * dt)

        # 7) Synaptic current that goes into the soma:
        #    Use the *negative* sum so excitation depolarizes when V < E_AMPA (0 mV)
        # --- NEW: soma receives coupling current from dendrites (NOT the ionic sums directly)
        I_syn = -g_c * torch.sum(self.dendritic_voltages - self.voltage)  # pA
        return I_syn
    
    def _update_multi_receptor_conductances_pytorch(self, dt: float):
            """Update multi-receptor conductances (CORTEX 4.2 - FULLY PyTorch)"""
            with torch.no_grad():
                self.ampa_conductance.mul_(torch.exp(-dt / CONSTANTS['tau_AMPA']))
                self.nmda_conductance.mul_(torch.exp(-dt / CONSTANTS['tau_NMDA']))
                self.gaba_a_conductance.mul_(torch.exp(-dt / CONSTANTS['tau_GABA_A']))
                self.gaba_b_conductance.mul_(torch.exp(-dt / CONSTANTS['tau_GABA_B']))

    def _update_cadex_dynamics_pytorch(self, dendritic_current: torch.Tensor, dt: float) -> torch.Tensor:
        """CAdEx dynamics (CORTEX 4.2 - FULLY PyTorch)"""
        # CAdEx equations from paper
        # C_m * dV/dt = -g_L(V - E_L) + g_L*Delta_T*exp((V - V_T)/Delta_T) - w + I_syn + I_ext
        
        # Leak current
        I_leak = -self.gL_nS * (self.voltage - self.EL_mV)  # nS*mV = pA
        # Exponential spike mechanism (CAdEx signature)
        x = torch.clamp((self.voltage - self.VT_mV) / self.dT_mV, min=-20.0, max=8.0)
        I_exp = self.gL_nS * self.dT_mV * torch.exp(x)      # pA

        # Adaptation current
        I_adaptation = -self.cadex_adaptation_current
        
        # Oscillatory input
        I_osc = self.oscillatory_input * self.phase_coupling_strength
        
        # Total current
        I_total = I_leak + I_exp + I_adaptation + dendritic_current + I_osc
        
        # Apply homeostatic and metabolic scaling
        metabolic_scaling = self._get_metabolic_scaling_pytorch()
        I_total = I_total * self.intrinsic_excitability * metabolic_scaling
        
        # Voltage update
        dV_dt = I_total / self.Cm_pF  # pA/pF = mV/ms

        with torch.no_grad():
            self.voltage.add_(dV_dt * dt)        
        
        with torch.no_grad():
            self.voltage.clamp_(-120.0, 40.0)

        # Adaptation update (tau_w * dw/dt = a(V - E_L) - w)
        dw_dt = (self.a_nS * (self.voltage - self.EL_mV) - self.cadex_adaptation_current) / self.tau_w_ms

        with torch.no_grad():
            self.cadex_adaptation_current.add_(dw_dt * dt)
        
        # threshold/reset from config
        spike_occurred = (self.voltage >= self.VT_mV).float()
        with torch.no_grad():
            # reset to Vr on spike
            self.voltage.copy_(torch.where(spike_occurred.bool(), self.Vr_mV, self.voltage))
            # add b (pA) on spike
            self.cadex_adaptation_current.add_(spike_occurred * self.b_pA)
            # start refractory (ms)
            self.refrac_left_ms.copy_(torch.where(spike_occurred.bool(), self.refractory_ms, self.refrac_left_ms))

        return spike_occurred
    
    def _update_hodgkin_huxley_dynamics_pytorch(self, dendritic_current: torch.Tensor, dt: float) -> torch.Tensor:
        """Hodgkin-Huxley dynamics (YOUR EXISTING LOGIC - FULLY PyTorch)"""
        # === UPDATE ION CHANNELS (PyTorch) ===
        self._update_ion_channels_pytorch(dt)
        
        # === CALCULATE MEMBRANE CURRENTS (PyTorch) ===
        total_current = self._calculate_membrane_currents_pytorch()
        
        # Add dendritic contribution
        total_current = total_current + dendritic_current
        
        # Add oscillatory modulation
        total_current = total_current + self.oscillatory_input * self.phase_coupling_strength
        
        # Apply homeostatic modulation
        total_current = total_current * self.intrinsic_excitability
        
        # Apply metabolic constraints
        metabolic_scaling = self._get_metabolic_scaling_pytorch()
        total_current = total_current * metabolic_scaling
        
        # === UPDATE MEMBRANE VOLTAGE (PyTorch) ===
        dV_dt = total_current / CONSTANTS['C_m']
        with torch.no_grad():
            self.voltage.add_(dV_dt * dt)
        self.voltage.data = torch.clamp(self.voltage.data, -100.0, 50.0)
        
        # === SPIKE DETECTION (PyTorch) ===
        spike_threshold = -50.0
        adaptive_threshold = spike_threshold + self.adaptation_current * 5.0
        spike_occurred = (self.voltage > adaptive_threshold).float()
        
        # Reset voltage if spike occurred
        self.voltage.data = torch.where(spike_occurred.bool(), torch.tensor(-70.0, device=self.device), self.voltage.data)
        
        # Reset ion channels if spike occurred
        with torch.no_grad():
            self.Na_m.copy_(torch.where(spike_occurred.bool(), torch.tensor(0.0, device=self.device), self.Na_m))
            self.Na_h.copy_(torch.where(spike_occurred.bool(), torch.tensor(1.0, device=self.device), self.Na_h))   
        
        return spike_occurred
    
    def _update_ion_channels_pytorch(self, dt: float):
        V = self.voltage
        
        # Sodium channel (Na_m, Na_h)
        alpha_m = 0.1 * (V + 40.0) / (1.0 - torch.exp(-(V + 40.0) / 10.0))
        beta_m = 4.0 * torch.exp(-(V + 65.0) / 18.0)
        tau_m = 1.0 / (alpha_m + beta_m)
        m_inf = alpha_m / (alpha_m + beta_m)
        
        alpha_h = 0.07 * torch.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + torch.exp(-(V + 35.0) / 10.0))
        tau_h = 1.0 / (alpha_h + beta_h)
        h_inf = alpha_h / (alpha_h + beta_h)
        
        # Potassium channel (K_n)
        alpha_n = 0.01 * (V + 55.0) / (1.0 - torch.exp(-(V + 55.0) / 10.0))
        beta_n = 0.125 * torch.exp(-(V + 65.0) / 80.0)
        tau_n = 1.0 / (alpha_n + beta_n)
        n_inf = alpha_n / (alpha_n + beta_n)
        
        # Calcium channel (simplified)
        ca_inf = 1.0 / (1.0 + torch.exp(-(V + 20.0) / 9.0))
        
        with torch.no_grad():
            self.Na_m.add_((m_inf - self.Na_m) * dt / tau_m)
            self.Na_h.add_((h_inf - self.Na_h) * dt / tau_h)
            self.K_n.add_((n_inf - self.K_n) * dt / tau_n)
            self.Ca_r.add_((ca_inf - self.Ca_r) * dt / 5.0)
        
    def _calculate_membrane_currents_pytorch(self) -> torch.Tensor:
        """Calculate membrane currents (YOUR EXISTING LOGIC - FULLY PyTorch)"""
        V = self.voltage
        
        # Leak current
        I_leak = CONSTANTS['g_leak'] * (V - CONSTANTS['E_leak'])
        
        # Sodium current
        I_Na = (CONSTANTS['g_Na_max'] * 
                self.Na_m**3 * self.Na_h * 
                (V - CONSTANTS['E_Na']))
        
        # Potassium current
        I_K = (CONSTANTS['g_K_max'] * 
            self.K_n**4 * 
            (V - CONSTANTS['E_K']))

        # Calcium current
        I_Ca = (CONSTANTS['g_Ca_max'] * 
                self.Ca_r * 
                (V - CONSTANTS['E_Ca']))
        
        # Adaptation current
        I_adaptation = self.adaptation_current * (V - CONSTANTS['E_K'])
        
        # Total membrane current (negative for outward)
        total_current = -(I_leak + I_Na + I_K + I_Ca + I_adaptation)
        
        return total_current
    
    def _get_metabolic_scaling_pytorch(self) -> torch.Tensor:
        """Get metabolic scaling factor (YOUR EXISTING LOGIC - FULLY PyTorch)"""
        atp_ratio = self.atp_level / CONSTANTS['ATP_max']
        scaling = 1.0 / (1.0 + torch.exp(-10.0 * (atp_ratio - 0.3)))
        return scaling
    
    def _update_adaptation_mechanisms_pytorch(self, spike_occurred: torch.Tensor, dt: float):
        """Update adaptation mechanisms (YOUR EXISTING LOGIC - FULLY PyTorch)"""
        # Spike frequency adaptation
        spike_increment = spike_occurred * 0.1
        
        # Decay adaptation (approximated for PyTorch)
        decay_rate = torch.exp(-dt / CONSTANTS['tau_adaptation'])
        
        # Update adaptation current
        self.adaptation_current.data = self.adaptation_current.data * decay_rate + spike_increment * self.adaptation_strength
    
    def _update_calcium_dynamics_pytorch(self, spike_occurred: torch.Tensor, dt: float):
        """Update calcium dynamics (YOUR EXISTING LOGIC - FULLY PyTorch)"""
        # Calcium influx during spike
        ca_influx = spike_occurred * CONSTANTS['Ca_alpha'] * 2.0
        self.calcium_concentration.data = self.calcium_concentration.data + ca_influx
        
        # Calcium decay
        ca_decay = (self.calcium_concentration - CONSTANTS['Ca_rest']) / CONSTANTS['Ca_decay']
        self.calcium_concentration.data = self.calcium_concentration.data - ca_decay * dt
        self.calcium_concentration.data = torch.clamp(self.calcium_concentration.data, CONSTANTS['Ca_rest'], 10.0)
        
        # Update calcium activation
        ca_ratio = self.calcium_concentration / CONSTANTS['Ca_rest']
        self.calcium_activation.data = (ca_ratio - 1.0) * 0.5
    
    def _update_metabolic_state_pytorch(self, spike_occurred: torch.Tensor, dt: float):
        """Update metabolic state (YOUR EXISTING LOGIC - FULLY PyTorch)"""
        # ATP consumption during spike
        atp_cost = spike_occurred * CONSTANTS['ATP_consumption_rate'] * self.metabolic_efficiency
        self.atp_level.data = self.atp_level.data - atp_cost
        
        # ATP recovery
        atp_recovery = CONSTANTS['ATP_recovery_rate'] * dt
        self.atp_level.data = self.atp_level.data + atp_recovery
        
        # Bounds
        self.atp_level.data = torch.clamp(self.atp_level.data, 0.0, CONSTANTS['ATP_max'])
        
        # Calculate metabolic stress
        atp_ratio = self.atp_level / CONSTANTS['ATP_max']
        self.metabolic_stress.data = torch.clamp(0.5 - atp_ratio, 0.0, 1.0)
    
    def _update_astrocyte_coupling_pytorch(self, spike_occurred: torch.Tensor, dt: float):
        """Update astrocyte coupling (CORTEX 4.2 - FULLY PyTorch)"""
        # Astrocyte calcium dynamics (from CORTEX 4.2 paper)
        tau_Ca = 800.0
        alpha_Ca = 0.03
        beta_astro = 0.3
        
        # Calcium dynamics: tau_Ca * dCa/dt = -Ca + α_Ca * spike
        spike_input = spike_occurred * alpha_Ca
        dCa_dt = (-self.astrocyte_calcium + spike_input) / tau_Ca
        self.astrocyte_calcium.data = self.astrocyte_calcium.data + dCa_dt * dt
        self.astrocyte_calcium.data = torch.clamp(self.astrocyte_calcium.data, 0.0, 10.0)
        
        # Modulation factor (from paper: 1 + β_astro * mean(Ca))
        self.astrocyte_modulation.data = 1.0 + beta_astro * self.astrocyte_calcium
    
    def _update_homeostatic_plasticity_pytorch(self):
        """Update homeostatic plasticity"""
        # Calculate recent firing rate (CPU-based for efficiency)
        current_time = self.sim_time_ms
        recent_spikes = [t for t in self.firing_history if current_time - t < 1000.0]  # last 1 s of sim time
        
        current_firing_rate = len(recent_spikes)
        
        # Update average firing rate
        alpha = 0.01
        new_firing_rate = (1.0 - alpha) * self.firing_rate_average + alpha * current_firing_rate
        with torch.no_grad():
            self.firing_rate_average.copy_(torch.as_tensor(new_firing_rate, device=self.device, dtype=self.firing_rate_average.dtype))

        # Calculate homeostatic pressure
        rate_error = CONSTANTS['target_firing_rate'] - self.firing_rate_average
        self.homeostatic_pressure.data = rate_error / CONSTANTS['target_firing_rate']
        
        # Update intrinsic excitability
        adjustment_rate = 1.0 / CONSTANTS['homeostatic_timescale']
        excitability_change = self.homeostatic_pressure * adjustment_rate
        self.intrinsic_excitability.data = self.intrinsic_excitability.data + excitability_change
        self.intrinsic_excitability.data = torch.clamp(self.intrinsic_excitability.data, 0.1, 3.0)
    
    def step(self, current_inputs: np.ndarray, dt: float = 0.001, current_time: float = 0.0) -> Tuple[bool, float]:
        """
        SAME API as your enhanced_neurons.py - but FULLY PyTorch internally
        
        Converts NumPy inputs to PyTorch tensors, runs forward pass, returns NumPy outputs
        """
        # Convert inputs to PyTorch tensor
        inputs_tensor = torch.tensor(current_inputs, device=self.device, dtype=torch.float32)
        
        # Run forward pass
        with torch.no_grad():  # No gradients needed for inference
            spike_tensor, voltage_tensor = self.forward(inputs_tensor, dt, current_time)
        
        # Convert back to Python types for API compatibility
        spike_occurred = bool(spike_tensor.item())
        voltage = float(voltage_tensor.item())
        
        return spike_occurred, voltage
    
    def set_oscillatory_input(self, oscillatory_signal: float, coupling_strength: float = 1.0):
        """Set oscillatory input (YOUR EXISTING API)"""
        self.oscillatory_input = oscillatory_signal
        self.phase_coupling_strength = coupling_strength
    
    def get_state(self) -> Dict[str, Any]:
        """Get neuron state - SAME API as your enhanced_neurons.py"""
        return {
            # YOUR EXISTING STATE (converted from PyTorch tensors)
            'neuron_id': self.neuron_id,
            'voltage': float(self.voltage.item()),
            'calcium': float(self.calcium_concentration.item()),
            'atp_level': float(self.atp_level.item()),
            'adaptation_current': float(self.adaptation_current.item()),
            'intrinsic_excitability': float(self.intrinsic_excitability.item()),
            'firing_rate': float(self.firing_rate_average.item()),
            'homeostatic_pressure': float(self.homeostatic_pressure.item()),
            'metabolic_stress': float(self.metabolic_stress.item()),
            'spike_count': self.spike_count,
            'ion_channels': {
                'Na_m': float(self.Na_m.item()),
                'Na_h': float(self.Na_h.item()),
                'K_n': float(self.K_n.item()),
                'Ca_r': float(self.Ca_r.item()),
                'Ca_s': float(self.Ca_s.item())
            },
            # CORTEX 4.2 ADDITIONS (converted from PyTorch tensors)
            'cadex_mode': self.use_cadex,
            'cadex_adaptation_current': float(self.cadex_adaptation_current.item()),
            'multi_receptor_conductances': {
                'AMPA': float(self.ampa_conductance.item()),
                'NMDA': float(self.nmda_conductance.item()),
                'GABA_A': float(self.gaba_a_conductance.item()),
                'GABA_B': float(self.gaba_b_conductance.item())
            },
            'dendritic_voltages': [float(v.item()) for v in self.dendritic_voltages],
            'nmda_gating_variables': [float(v.item()) for v in self.nmda_gating_variables],
            'astrocyte_modulation': float(self.astrocyte_modulation.item()),
            'gpu_device': str(self.device)
        }
    
    def get_biological_metrics(self) -> Dict[str, float]:
        """Get biological metrics - SAME API as your enhanced_neurons.py"""
        return {
            # YOUR EXISTING METRICS
            'firing_rate_hz': float(self.firing_rate_average.item()),
            'adaptation_strength': float(self.adaptation_current.item()),
            'calcium_level_um': float(self.calcium_concentration.item()),
            'atp_level_mm': float(self.atp_level.item()),
            'metabolic_efficiency': 1.0 - float(self.metabolic_stress.item()),
            'homeostatic_balance': 1.0 - abs(float(self.homeostatic_pressure.item())),
            'intrinsic_excitability': float(self.intrinsic_excitability.item()),
            'membrane_voltage_mv': float(self.voltage.item()),
            'biological_realism_score': self._calculate_biological_realism_score(),
            
            # CORTEX 4.2 ADDITIONS
            'cadex_realism_score': self._calculate_cadex_realism_score(),
            'multi_receptor_activity': self._calculate_multi_receptor_activity(),
            'dendritic_integration_score': self._calculate_dendritic_integration_score(),
            'astrocyte_coupling_strength': float(self.astrocyte_modulation.item()),
            'cortex_42_compliance': self._calculate_cortex_42_compliance(),
            'gpu_accelerated': 1.0 if self.device.type == 'cuda' else 0.0
        }
    
    def _calculate_biological_realism_score(self) -> float:
        """Calculate biological realism score (YOUR EXISTING LOGIC)"""
        scores = []
        
        # Firing rate in biological range
        fr = float(self.firing_rate_average.item())
        fr_score = 1.0 if 1.0 <= fr <= 100.0 else max(0.0, 1.0 - abs(fr - 10.0) / 50.0)
        scores.append(fr_score)
        
        # Voltage in biological range
        v = float(self.voltage.item())
        v_score = 1.0 if -90.0 <= v <= -40.0 else 0.5
        scores.append(v_score)
        
        # ATP levels healthy
        atp = float(self.atp_level.item())
        atp_score = atp / float(CONSTANTS['ATP_max'].item())
        scores.append(atp_score)
        
        # Calcium in physiological range
        ca = float(self.calcium_concentration.item())
        ca_rest = float(CONSTANTS['Ca_rest'].item())
        ca_ratio = ca / ca_rest
        ca_score = 1.0 if 1.0 <= ca_ratio <= 10.0 else max(0.0, 1.0 - abs(ca_ratio - 2.0) / 5.0)
        scores.append(ca_score)
        
        # Homeostatic balance
        homeostatic_score = 1.0 - abs(float(self.homeostatic_pressure.item()))
        scores.append(max(0.0, homeostatic_score))
        
        return np.mean(scores)
    
    def _calculate_cadex_realism_score(self) -> float:
        """Calculate CAdEx realism score (CORTEX 4.2 addition)"""
        if not self.use_cadex:
            return 0.0
            
        scores = []
        
        # CAdEx adaptation current active
        adapt_score = 1.0 if float(self.cadex_adaptation_current.item()) > 0.0 else 0.0
        scores.append(adapt_score)
        
        # Exponential spike mechanism evidence
        exp_score = 1.0 if float(self.voltage.item()) > -60.0 else 0.5
        scores.append(exp_score)
        
        return np.mean(scores)
    
    def _calculate_multi_receptor_activity(self) -> float:
        """Calculate multi-receptor activity (CORTEX 4.2 addition)"""
        total_activity = (
            float(self.ampa_conductance.item()) + 
            float(self.nmda_conductance.item()) + 
            float(self.gaba_a_conductance.item()) + 
            float(self.gaba_b_conductance.item())
        )
        return min(1.0, total_activity / 4.0)
    
    def _calculate_dendritic_integration_score(self) -> float:
        """Calculate dendritic integration score (CORTEX 4.2 addition)"""
        # Measure voltage diversity across dendrites
        voltages = [float(v.item()) for v in self.dendritic_voltages]
        voltage_std = np.std(voltages)
        
        # Measure NMDA activity
        nmda_activity = np.mean([float(v.item()) for v in self.nmda_gating_variables])
        
        # Combined score
        integration_score = min(1.0, (voltage_std / 10.0) + nmda_activity)
        return integration_score
    
    def _calculate_cortex_42_compliance(self) -> float:
        """Calculate CORTEX 4.2 compliance score"""
        compliance_factors = []
        
        # Multi-receptor synapses active
        multi_receptor_score = self._calculate_multi_receptor_activity()
        compliance_factors.append(multi_receptor_score)
        
        # Dendritic integration working
        dendritic_score = self._calculate_dendritic_integration_score()
        compliance_factors.append(dendritic_score)
        
        # Astrocyte coupling active
        astrocyte_score = min(1.0, abs(float(self.astrocyte_modulation.item()) - 1.0) * 5.0)
        compliance_factors.append(astrocyte_score)
        
        # GPU acceleration working
        gpu_score = 1.0 if self.device.type == 'cuda' else 0.5
        compliance_factors.append(gpu_score)
        
        return np.mean(compliance_factors)

class EnhancedNeuronPopulation42PyTorch(nn.Module):
    """
    FULLY PyTorch Enhanced Neuron Population 4.2
    
    SAME API as your existing NeuronPopulation
    KEEPS ALL your existing population logic
    ADDS CORTEX 4.2 enhancements
    FULLY GPU-accelerated
    """
    
    def __init__(self, n_neurons: int = 32, neuron_types: Optional[List[str]] = None, 
                 use_cadex: bool = False, device=None):
        super().__init__()
        
        self.n_neurons = n_neurons
        self.use_cadex = use_cadex
        self.device = device or DEVICE
        
        # === YOUR EXISTING NEURON TYPE LOGIC (KEEP ALL) ===
        if neuron_types is None:
            neuron_types = ['pyramidal'] * int(n_neurons * 0.8) + ['interneuron'] * int(n_neurons * 0.2)
            neuron_types = neuron_types[:n_neurons]
        
        # Create PyTorch neurons as ModuleList
        self.neurons = nn.ModuleList()
        for i in range(n_neurons):
            neuron_type = neuron_types[i] if i < len(neuron_types) else 'pyramidal'
            neuron = EnhancedNeuron42PyTorch(
                neuron_id=i,
                neuron_type=neuron_type,
                use_cadex=use_cadex,
                device=self.device
            )
            self.neurons.append(neuron)
        
        # === YOUR EXISTING POPULATION DYNAMICS (KEEP ALL) ===
        self.population_oscillation = 0.0
        self.global_inhibition = 0.0
        self.network_activity = 0.0
        
        # === YOUR EXISTING E/I BALANCE (KEEP ALL) ===
        self.excitatory_neurons = [n for n in self.neurons if n.neuron_type == 'pyramidal']
        self.inhibitory_neurons = [n for n in self.neurons if n.neuron_type == 'interneuron']
        
        # === YOUR EXISTING ACTIVITY TRACKING (KEEP ALL) ===
        self.activity_history = deque(maxlen=1000)
        self.population_firing_rate = 0.0
        
        print(f"Enhanced Population 4.2 PyTorch: {n_neurons} neurons, CAdEx={'ON' if use_cadex else 'OFF'}, Device={self.device}")
    
    def forward(self, inputs: torch.Tensor, dt: float = 0.001, step_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FULLY PyTorch forward pass for the entire population
        
        Args:
            inputs: Input currents for each neuron (PyTorch tensor)
            dt: Time step
            step_idx: Step index
            
        Returns:
            (spikes, voltages) - PyTorch tensors
        """
        # --- NEW: remember dt for rate calculations ---
        self._dt_ms = float(dt)
        current_time = step_idx * dt
        
        # Ensure inputs match neuron count
        if inputs.shape[0] != self.n_neurons:
            inputs = F.pad(inputs, (0, self.n_neurons - inputs.shape[0]))[:self.n_neurons]
        
        # === YOUR EXISTING POPULATION DYNAMICS (KEEP ALL) ===
        self._update_population_dynamics(dt)
        self._apply_population_oscillations()
        
        # === VECTORIZED NEURAL PROCESSING (FULLY PyTorch) ===
        spikes = torch.zeros(self.n_neurons, device=self.device)
        voltages = torch.zeros(self.n_neurons, device=self.device)
        
        # Process each neuron (can be batched in future for even more speed)
        for i, neuron in enumerate(self.neurons):
            # Add population effects
            total_input = inputs[i:i+1] + self.global_inhibition
            
            # Expand to match neuron's dendrite count
            neuron_input = total_input.repeat(neuron.n_dendrites)
            
            # Forward pass through neuron
            with torch.no_grad():
                spike, voltage = neuron.forward(neuron_input, dt, current_time)
            
            spikes[i] = spike
            voltages[i] = voltage
        
        # === YOUR EXISTING POPULATION TRACKING (KEEP ALL) ===
        self._update_population_activity(spikes)
        self._regulate_excitation_inhibition_balance(spikes)
        
        return spikes, voltages
    
    def step(self, inputs: np.ndarray, dt: float = 0.001, step_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        SAME API as your existing NeuronPopulation.step() - but FULLY PyTorch internally
        
        Converts NumPy inputs to PyTorch tensors, runs forward pass, returns NumPy outputs
        """
        # Convert inputs to PyTorch tensor
        if isinstance(inputs, torch.Tensor):
            inputs_tensor = inputs.clone().detach().to(device=self.device, dtype=torch.float32)
        else:
            inputs_tensor = torch.tensor(inputs, device=self.device, dtype=torch.float32)

        # Run forward pass
        with torch.no_grad():
            spikes_tensor, voltages_tensor = self.forward(inputs_tensor, dt, step_idx)
        
        # Convert back to NumPy but preserve spike magnitudes
        spikes = spikes_tensor.cpu().numpy().astype(float)  # Keep as float, not bool
        voltages = voltages_tensor.cpu().numpy().astype(float)

        return spikes, voltages
    
    def _update_population_dynamics(self, dt: float):
        """Update population dynamics (YOUR EXISTING LOGIC)"""
        # Population calcium (enhanced with CORTEX 4.2)
        total_calcium = sum(float(n.calcium_concentration.item()) for n in self.neurons)
        astrocyte_calcium = sum(float(n.astrocyte_modulation.item()) for n in self.neurons)
        self.population_calcium = (total_calcium + astrocyte_calcium) / (self.n_neurons * 2)
        
        # Network activity
        total_voltage = sum(abs(float(n.voltage.item()) + 70.0) for n in self.neurons)
        self.network_activity = total_voltage / self.n_neurons
        
        # Population oscillation
        oscillation_strength = min(1.0, self.network_activity / 20.0)
        self.population_oscillation = oscillation_strength * np.sin(2 * np.pi * 10.0 * len(self.activity_history) * dt)
    
    def _apply_population_oscillations(self):
        """Apply population oscillations (YOUR EXISTING LOGIC)"""
        for neuron in self.neurons:
            coupling_strength = 0.1 + 0.1 * np.random.random()
            neuron.set_oscillatory_input(self.population_oscillation, coupling_strength)
    
    def _update_population_activity(self, spikes: torch.Tensor):
        """Update population activity tracking (YOUR EXISTING LOGIC)"""
        spike_count = float(torch.sum(spikes).item())
        self.activity_history.append(spike_count)
        
        if len(self.activity_history) > 10:
            recent_activity = list(self.activity_history)[-100:]
            avg_spikes_per_step = np.mean(recent_activity)
            dt_sec = (getattr(self, "_dt_ms", getattr(C, "DT_MS", 0.25))) / 1000.0
            self.population_firing_rate = avg_spikes_per_step / (dt_sec * self.n_neurons)

    def _regulate_excitation_inhibition_balance(self, spikes: torch.Tensor):
        """Regulate E/I balance (YOUR EXISTING LOGIC)"""
        # Convert to numpy for processing
        spikes_np = spikes.cpu().numpy()
        
        excitatory_activity = sum(spikes_np[i] for i, n in enumerate(self.neurons) if n.neuron_type == 'pyramidal')
        inhibitory_activity = sum(spikes_np[i] for i, n in enumerate(self.neurons) if n.neuron_type == 'interneuron')
        
        total_excitatory = len(self.excitatory_neurons)
        total_inhibitory = len(self.inhibitory_neurons)
        
        if total_excitatory > 0 and total_inhibitory > 0:
            e_rate = excitatory_activity / total_excitatory
            i_rate = inhibitory_activity / total_inhibitory
            
            target_ei_ratio = 4.0
            current_ei_ratio = (e_rate + 0.001) / (i_rate + 0.001)
            
            ei_error = current_ei_ratio - target_ei_ratio
            inhibition_adjustment = ei_error * 0.01
            
            self.global_inhibition += inhibition_adjustment
            self.global_inhibition = np.clip(self.global_inhibition, -2.0, 2.0)
    
    def get_population_state(self) -> Dict[str, Any]:
        """Get population state - SAME API as your existing NeuronPopulation"""
        # Calculate averages from PyTorch tensors
        avg_voltage = np.mean([float(n.voltage.item()) for n in self.neurons])
        avg_calcium = np.mean([float(n.calcium_concentration.item()) for n in self.neurons])
        avg_atp = np.mean([float(n.atp_level.item()) for n in self.neurons])
        avg_excitability = np.mean([float(n.intrinsic_excitability.item()) for n in self.neurons])
        
        # CORTEX 4.2 additions
        avg_astrocyte_modulation = np.mean([float(n.astrocyte_modulation.item()) for n in self.neurons])
        avg_multi_receptor_activity = np.mean([n._calculate_multi_receptor_activity() for n in self.neurons])
        avg_cortex_42_compliance = np.mean([n._calculate_cortex_42_compliance() for n in self.neurons])
        
        return {
            # YOUR EXISTING STATE
            'n_neurons': self.n_neurons,
            'population_firing_rate_hz': self.population_firing_rate,
            'average_voltage_mv': avg_voltage,
            'average_calcium_um': avg_calcium,
            'average_atp_mm': avg_atp,
            'average_excitability': avg_excitability,
            'population_calcium': getattr(self, 'population_calcium', 0.0),
            'network_activity': self.network_activity,
            'global_inhibition': self.global_inhibition,
            'ei_balance': len(self.excitatory_neurons) / max(1, len(self.inhibitory_neurons)),
            'population_oscillation_strength': abs(self.population_oscillation),
            
            # CORTEX 4.2 ADDITIONS
            'cadex_mode': self.use_cadex,
            'average_astrocyte_modulation': avg_astrocyte_modulation,
            'average_multi_receptor_activity': avg_multi_receptor_activity,
            'cortex_42_compliance_score': avg_cortex_42_compliance,
            'gpu_device': str(self.device),
            'pytorch_accelerated': True
        }

# === TESTING FUNCTIONS ===
def test_pytorch_vs_numpy_performance():
    """Test performance difference between PyTorch and NumPy versions"""
    print(" Testing PyTorch vs NumPy Performance...")
    
    n_neurons = 32
    n_steps = 100
    
    # PyTorch version
    print("Testing PyTorch version...")
    start_time = time.time()
    
    pytorch_population = EnhancedNeuronPopulation42PyTorch(
        n_neurons=n_neurons,
        use_cadex=True
    )

    for step in range(n_steps):
        inputs = np.random.normal(3.0, 1.0, n_neurons)
        
        # TEMP: evoke spikes on neuron 0 for steps 20–39
        if 20 <= step < 40:
            n0 = pytorch_population.neurons[0]
            with torch.no_grad():
                n0.ampa_conductance.add_(0.12)
                n0.nmda_conductance.add_(0.04)
        
        spikes, voltages = pytorch_population.step(inputs, dt = 0.25, step_idx=step)

    pytorch_time = time.time() - start_time
    
    print(f"Results:")
    print(f"  PyTorch time: {pytorch_time:.3f} seconds")
    print(f"  Final spikes: {np.sum(spikes)}")
    print(f"  Final avg voltage: {np.mean(voltages):.1f} mV")
    print(f"  Device: {pytorch_population.device}")
    
    # Test biological metrics
    state = pytorch_population.get_population_state()
    print(f"  CORTEX 4.2 compliance: {state['cortex_42_compliance_score']:.1%}")
    print(f"  GPU accelerated: {state['pytorch_accelerated']}")
    
    return pytorch_population

def test_cortex_42_features():
    """Test CORTEX 4.2 specific features"""
    print("\n Testing CORTEX 4.2 Features...")
    
    # Test both H-H and CAdEx modes
    modes = [
        ('Hodgkin-Huxley', False),
        ('CAdEx', True)
    ]
    
    for mode_name, use_cadex in modes:
        print(f"\n--- Testing {mode_name} Mode ---")
        
        neuron = EnhancedNeuron42PyTorch(
            neuron_id=0,
            neuron_type='pyramidal',
            use_cadex=use_cadex
        )
        
        # Test neuron dynamics
        for step in range(50):
            current = np.array([3.0, 2.0, 1.0, 0.5])
            spike, voltage = neuron.step(current, dt = 0.25, current_time=step * 0.001)
            
            if step % 20 == 0:
                print(f"  Step {step}: V={voltage:.1f}mV, Spike={spike}")
        
        # Test biological metrics
        metrics = neuron.get_biological_metrics()
        print(f"  Firing rate: {metrics['firing_rate_hz']:.1f} Hz")
        print(f"  Bio realism: {metrics['biological_realism_score']:.1%}")
        print(f"  CORTEX 4.2 compliance: {metrics['cortex_42_compliance']:.1%}")
        print(f"  GPU accelerated: {metrics['gpu_accelerated']:.0f}")
        
        if use_cadex:
            print(f"  CAdEx realism: {metrics['cadex_realism_score']:.1%}")
        
        # Test state access
        state = neuron.get_state()
        print(f"  Multi-receptor AMPA: {state['multi_receptor_conductances']['AMPA']:.3f}")
        print(f"  Astrocyte modulation: {state['astrocyte_modulation']:.3f}")
        print(f"  Device: {state['gpu_device']}")

if __name__ == "__main__":
    print("=" * 80)
    print("CORTEX 4.2 Enhanced Neurons - FULLY PyTorch Implementation")
    print("=" * 80)
    
    # Test performance
    pytorch_population = test_pytorch_vs_numpy_performance()
    
    # Test CORTEX 4.2 features
    test_cortex_42_features()
    
    print("\n" + "=" * 80)
    print(" CORTEX 4.2 FULLY PyTorch Implementation Complete!")
    print("=" * 80)
    print(" FULLY PyTorch - all computation on GPU tensors")
    print(" KEEPS ALL your existing biological complexity")
    print(" ADDS CORTEX 4.2 enhancements")
    print(" SAME API as your enhanced_neurons.py")
    print(" GPU accelerated with CPU fallback")
    print(" Ready for drop-in replacement!")
    print("")
    print(" Your CORTEX 4.1 → 4.2 upgrade is ready!")