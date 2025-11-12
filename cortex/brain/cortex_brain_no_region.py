#cortex.brain.cortex_no_region.py
# CORTEX 4.2 — Neurons+Synapses+Astrocytes, no regions. Fast training + RWS LFP proxy.
# Search anchors: [#CBNR]

from typing import Dict, Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# [#CBNR] Optional real modules
# =============================
try:
    from cortex.cells.enhanced_neurons_42 import EnhancedNeuronPopulation42PyTorch  # type: ignore
except Exception:
    EnhancedNeuronPopulation42PyTorch = None

try:
    from cortex.cells.enhanced_synapses_42 import EnhancedSynapticSystem42PyTorch  # type: ignore
except Exception:
    EnhancedSynapticSystem42PyTorch = None

try:
    from cortex.cells.astrocyte import AstrocyteNetwork42  # type: ignore
except Exception:
    AstrocyteNetwork42 = None

# ==================================================
# [#CBNR] Lightweight fallbacks (keep it self-contained)
# ==================================================
class _FastLIF(nn.Module):
    def __init__(self, n: int, dt: float = 1.0, tau_mem: float = 20.0, v_th: float = 1.0, device: str = "cpu"):
        super().__init__()
        self.n = n
        self.dt = dt
        self.register_buffer("v", torch.zeros(1, n, device=device))
        self.register_buffer("tau_mem", torch.tensor(tau_mem, device=device))
        self.register_buffer("v_th", torch.tensor(v_th, device=device))

    def reset_state(self, batch: int = 1):
        self.v = torch.zeros(batch, self.n, device=self.v.device)

    def forward_step(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_current: [B, n]
        alpha = torch.exp(-self.dt / self.tau_mem)
        self.v = alpha * self.v + (1.0 - alpha) * input_current
        spikes = (self.v >= self.v_th).to(input_current.dtype)
        self.v = self.v * (1.0 - spikes)  # reset-on-spike
        return self.v, spikes


class _NullAstro(nn.Module):
    """Pass-through astrocyte 'network': returns per-neuron gain of 1.0 and keeps a tiny EMA of spikes."""
    def __init__(self, n_neurons: int, device: torch.device):
        super().__init__()
        self.n = n_neurons
        self.register_buffer("ema_spike", torch.zeros(1, n_neurons, device=device))
        self.momentum = 0.98

    def reset_state(self, batch: int = 1):
        self.ema_spike = torch.zeros(batch, self.n, device=self.ema_spike.device)

    def step(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: [B, n] → return astro_gain [B, n]
        self.ema_spike = self.momentum * self.ema_spike + (1 - self.momentum) * spikes
        return torch.ones_like(spikes)  # neutral gain


class _NullSynapses(nn.Module):
    """
    Cheap synapse approximation providing AMPA/GABA 'currents' to enable the RWS LFP proxy.
    We map pre activity to a signed current and split into AMPA (relu(+)) and GABA (relu(-)).
    """
    def __init__(self, n_neurons: int, device: torch.device):
        super().__init__()
        self.lin = nn.Linear(n_neurons, n_neurons, bias=False).to(device)

    def reset_state(self, batch: int = 1):
        pass

    def forward_step(self, pre_activity: torch.Tensor, astro_gain: torch.Tensor) -> Dict[str, torch.Tensor]:
        # pre_activity: [B, n] (use spikes or voltage)
        y = self.lin(pre_activity) * astro_gain  # astro modulation of syn drive
        i_ampa = F.relu(y)
        i_gaba = F.relu(-y)
        return {"i_ampa": i_ampa, "i_gaba": i_gaba}


# =========================================================
# [#CBNR] Main neurons-only brain with RWS LFP + RL heads
# =========================================================
class CortexBrainNoRegion42(nn.Module):
    """
    Neurons-only brain (no regions) with:
      - Input projection → Neuron population → Synapses (AMPA/GABA) + Astrocytes → Policy/Value heads
      - RWS LFP proxy: RWS(t) = zscore( sum(AMPA(t-6ms)) - 1.65 * sum(GABA(t)) )
    Returned dict keeps a 'region_activity' stub {} for compatibility with callers that expect it.
    """

    def __init__(
        self,
        n_inputs: int,
        n_actions: int,
        n_neurons: int = 512,
        dt: float = 1.0,                       # milliseconds per step (your sim dt)
        use_amp: bool = True,
        use_torch_compile: bool = False,
        device: Optional[str] = None,
        spikes_into_policy: bool = True,       # if False, use voltage into heads
    ):
        super().__init__()
        # ------------- device -------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.n_inputs = n_inputs
        self.n_actions = n_actions
        self.n_neurons = n_neurons
        self.dt = float(dt)
        self.use_amp = bool(use_amp)
        self.spikes_into_policy = bool(spikes_into_policy)

        # ------------- I/O -------------
        self.input_proj = nn.Linear(n_inputs, n_neurons, bias=True)

        # ------------- Neurons -------------
        if EnhancedNeuronPopulation42PyTorch is not None:
            self.neurons = EnhancedNeuronPopulation42PyTorch(
                n_neurons=n_neurons,
                dt=self.dt,
                device=self.device,
            )
            self._enhanced = True
        else:
            self.neurons = _FastLIF(n_neurons, dt=self.dt, device=str(self.device))
            self._enhanced = False

        # ------------- Astrocytes -------------
        if AstrocyteNetwork42 is not None:
            try:
                self.astro = AstrocyteNetwork42(n_neurons=n_neurons, device=self.device)
            except Exception:
                self.astro = _NullAstro(n_neurons, self.device)
        else:
            self.astro = _NullAstro(n_neurons, self.device)

        # ------------- Synapses -------------
        if EnhancedSynapticSystem42PyTorch is not None:
            try:
                self.syn = EnhancedSynapticSystem42PyTorch(n_neurons=n_neurons, device=self.device)
                self._syn_enhanced = True
            except Exception:
                self.syn = _NullSynapses(n_neurons, self.device)
                self._syn_enhanced = False
        else:
            self.syn = _NullSynapses(n_neurons, self.device)
            self._syn_enhanced = False

        # ------------- RL heads -------------
        self.action_head = nn.Linear(n_neurons, n_actions, bias=True)
        self.value_head  = nn.Linear(n_neurons, 1, bias=True)
        self.register_buffer("policy_tau", torch.tensor(1.0, device=self.device))  # temperature if you want soft sampling

        # ------------- RWS LFP buffers -------------
        self.rws_weight_gaba = 1.65
        delay_ms = 6.0
        self.delay_steps = max(1, int(round(delay_ms / max(1e-6, self.dt))))  # at least 1
        self._ampa_fifo: List[torch.Tensor] = []  # stores [B] AMPA sums
        # Rolling z-score (EMA) for RWS
        self.register_buffer("lfp_mean", torch.zeros(1, device=self.device))
        self.register_buffer("lfp_var", torch.ones(1, device=self.device))
        self._lfp_momentum = 0.99

        # ------------- move + optional compile -------------
        self.to(self.device)
        if use_torch_compile and hasattr(torch, "compile"):
            try:
                self.input_proj = torch.compile(self.input_proj)  # type: ignore
                self.action_head = torch.compile(self.action_head)  # type: ignore
                self.value_head  = torch.compile(self.value_head)   # type: ignore
                self.neurons = torch.compile(self.neurons)           # type: ignore
                self.syn = torch.compile(self.syn)                   # type: ignore
                self.astro = torch.compile(self.astro)               # type: ignore
            except Exception:
                pass

    # --------------------------
    # [#CBNR] Public API methods
    # --------------------------
    def reset_state(self, batch_size: int = 1):
        if hasattr(self.neurons, "reset_state"):
            self.neurons.reset_state(batch_size)
        if hasattr(self.astro, "reset_state"):
            self.astro.reset_state(batch_size)
        if hasattr(self.syn, "reset_state"):
            self.syn.reset_state(batch_size)
        self._ampa_fifo = [torch.zeros(batch_size, device=self.device) for _ in range(self.delay_steps)]
        self.lfp_mean.zero_()
        self.lfp_var.fill_(1.0)

    def forward(self, sensory: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        sensory: [B, n_inputs] float32
        returns dict with:
          - action_logits: [B, n_actions]
          - value: [B, 1]
          - lfp_proxy: [B]   (RWS z-scored)
          - spike_rate: [B]  (mean spikes)
          - region_activity: {}  (stub for compatibility)
        """
        x = sensory.to(self.device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            I = self.input_proj(x)                       # [B, n_neurons]
            v, spikes = self._neuron_step(I)            # [B, n], [B, n]

            # Astrocyte modulation (per-neuron gain)
            astro_gain = self._astro_step(spikes)       # [B, n]

            # Synaptic currents; must provide AMPA/GABA tensors
            syn_out = self._syn_step(pre_activity=spikes if self.spikes_into_policy else v,
                                     astro_gain=astro_gain)
            i_ampa = syn_out["i_ampa"]                  # [B, n]
            i_gaba = syn_out["i_gaba"]                  # [B, n]

            # RWS LFP proxy
            lfp_proxy = self._rws_proxy(i_ampa, i_gaba) # [B]

            # Readout for RL
            read = spikes if self.spikes_into_policy else v
            action_logits = self.action_head(read)      # [B, A]
            value        = self.value_head(read)        # [B, 1]

        spike_rate = spikes.mean(dim=1)                 # [B]
        return {
            "action_logits": action_logits,
            "value": value,
            "lfp_proxy": lfp_proxy,
            "spike_rate": spike_rate,
            "region_activity": {},   # keep API compatibility
        }

    @torch.no_grad()
    def step(self, sensory: torch.Tensor, epsilon: float = 0.0, sample_softmax: bool = False) -> Dict[str, torch.Tensor]:
        out = self.forward(sensory)
        logits = out["action_logits"]

        if sample_softmax:
            probs = F.softmax(logits / self.policy_tau.clamp_min(1e-3), dim=-1)
            actions = torch.distributions.Categorical(probs=probs).sample()
        elif epsilon > 0.0:
            B = logits.shape[0]
            greedy = torch.argmax(logits, dim=1)
            rand = torch.randint(0, logits.shape[1], (B,), device=logits.device)
            mask = (torch.rand(B, device=logits.device) < epsilon)
            actions = torch.where(mask, rand, greedy)
        else:
            actions = torch.argmax(logits, dim=1)

        out["actions"] = actions
        return out

    # --------------------------
    # [#CBNR] Internals
    # --------------------------
    def _neuron_step(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if EnhancedNeuronPopulation42PyTorch is not None and self._enhanced:
            v, spikes = self.neurons.forward_step(input_current=I)
        else:
            v, spikes = self.neurons.forward_step(I)
        return v, spikes

    def _astro_step(self, spikes: torch.Tensor) -> torch.Tensor:
        if AstrocyteNetwork42 is not None and isinstance(self.astro, nn.Module) and not isinstance(self.astro, _NullAstro):
            try:
                # Expect astro.step to accept spikes and return per-neuron gain [B, n]
                gain = self.astro.step(spikes)
                return gain
            except Exception:
                pass
        return self.astro.step(spikes)  # _NullAstro path returns ones

    def _syn_step(self, pre_activity: torch.Tensor, astro_gain: torch.Tensor) -> Dict[str, torch.Tensor]:
        if EnhancedSynapticSystem42PyTorch is not None and getattr(self, "_syn_enhanced", False):
            try:
                # Expect dict with 'i_ampa', 'i_gaba' tensors [B, n]
                syn_out = self.syn.forward_step(pre_activity=pre_activity, astro_gain=astro_gain)
                if ("i_ampa" in syn_out) and ("i_gaba" in syn_out):
                    return {"i_ampa": syn_out["i_ampa"], "i_gaba": syn_out["i_gaba"]}
            except Exception:
                pass
        # Fallback: signed current split into AMPA/GABA
        return self.syn.forward_step(pre_activity, astro_gain)

    def _rws_proxy(self, i_ampa: torch.Tensor, i_gaba: torch.Tensor) -> torch.Tensor:
        """
        RWS(t) = zscore( sum(AMPA(t-6ms)) - 1.65 * sum(GABA(t)) )
        Uses an EMA-based rolling mean/var for z-scoring (no large buffers).
        """
        # population sums per sample
        ampa_sum = i_ampa.sum(dim=1)         # [B]
        gaba_sum = i_gaba.sum(dim=1)         # [B]

        # AMPA delay FIFO
        if len(self._ampa_fifo) < self.delay_steps:
            # build if reset_state wasn't called
            self._ampa_fifo = [torch.zeros_like(ampa_sum) for _ in range(self.delay_steps)]
        self._ampa_fifo.append(ampa_sum)
        ampa_delayed = self._ampa_fifo.pop(0)  # [B]

        raw = ampa_delayed - self.rws_weight_gaba * gaba_sum  # [B]

        # EMA z-score
        # update mean/var with momentum, broadcasting over batch
        m = self._lfp_momentum
        batch_mean = raw.mean()
        batch_var = raw.var(unbiased=False) + 1e-6
        self.lfp_mean = m * self.lfp_mean + (1 - m) * batch_mean
        self.lfp_var  = m * self.lfp_var  + (1 - m) * batch_var

        std = torch.sqrt(self.lfp_var.clamp_min(1e-6))
        z = (raw - self.lfp_mean) / std
        return z  # [B]
