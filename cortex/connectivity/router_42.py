# cortex/connectivity/router_42.py
"""
DelayRouter42 — distance-based axonal delay router for CORTEX 4.2
-----------------------------------------------------------------
- Routes spikes/rates from region A -> region B with biologically
  plausible delays (ms) and weights.
- Ring-buffer per destination region: O(1) push/pull, tiny memory.
- Works on CPU or CUDA; stays in torch tensors end-to-end.

API
---
router = DelayRouter42(regions_meta, edges, dt_ms, conduction_mm_per_ms=0.5, device=None)

router.push(src_name, pre_spikes_tensor)  # shape [n_src], float or 0/1
drive = router.pull(dst_name)             # shape [n_dst], float (+E, -I)
router.tick()                             # advance time index by one step
router.reset()                            # zero buffers, reset time index
router.to(device)                         # move buffers/weights across devices

Edge spec
---------
Each edge dict can have:
{
  'src': 'V1', 'dst': 'M1',
  'W': torch.Tensor [n_dst, n_src] | numpy array | None (identity-ish),
  'distance_mm': 12.0,               # optional if delay_ms provided
  'delay_ms': 24.0,                  # optional; overrides distance
  'weight': 1.0,                     # scalar gain applied after W @ spikes
  'receptor': 'E' | 'I'              # E => +, I => -
}

Notes
-----
- If W is None, uses a sparse identity from min(n_src, n_dst) for quick tests.
- pre_spikes are treated as *instantaneous drive* this step (0/1 or rates).
- You can chain multiple edges into one dst; drives are summed in the buffer.
"""

from typing import Dict, List, Optional, Any
import math
import torch
import numpy as np


def _as_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


class DelayRouter42:
    def __init__(
        self,
        regions_meta: Dict[str, int],
        edges: List[Dict[str, Any]],
        dt_ms: float,
        conduction_mm_per_ms: float = 0.3,  # If paper says 0.3 mm/ms
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        regions_meta: { 'RegionName': n_neurons, ... }
        edges: list of edge dicts (see module docstring)
        dt_ms: simulation timestep in milliseconds
        conduction_mm_per_ms: default axonal speed (mm/ms) if only distance is given
        """
        self.dt_ms = float(dt_ms)
        self.conduction = float(conduction_mm_per_ms)
        self.regions = dict(regions_meta)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype

        # Compile edges
        compiled = []
        max_delay_steps = 0

        for e in edges:
            src = e["src"]
            dst = e["dst"]
            n_src = self.regions[src]
            n_dst = self.regions[dst]

            # Weight matrix
            W = e.get("W", None)
            if W is None:
                # Default: identity-ish map (min(n_src, n_dst))
                k = min(n_src, n_dst)
                W = torch.zeros((n_dst, n_src), dtype=self.dtype)
                idx = torch.arange(k, dtype=torch.long)
                W[idx, idx] = 1.0
            W = _as_tensor(W, self.device, self.dtype)

            # Basic shape safety: pad/trim if user passed mismatched W
            if W.shape != (n_dst, n_src):
                W = self._reshape_W(W, n_dst, n_src)

            # Delay in ms
            if "delay_ms" in e and e["delay_ms"] is not None:
                delay_ms = float(e["delay_ms"])
            else:
                dist = float(e.get("distance_mm", 0.0))
                delay_ms = dist / max(self.conduction, 1e-6)

            delay_steps = max(0, int(round(delay_ms / self.dt_ms)))
            max_delay_steps = max(max_delay_steps, delay_steps)

            # Sign for receptor
            receptor = e.get("receptor", "E")
            sign = +1.0 if str(receptor).upper().startswith("E") else -1.0

            gain = float(e.get("weight", 1.0))

            compiled.append({
                "src": src,
                "dst": dst,
                "W": W,
                "delay_steps": delay_steps,
                "sign": sign,
                "gain": gain,
            })

        self.edges = compiled

        # Ring buffers per destination region
        L = max(1, max_delay_steps + 1)
        self.buffer_len = L
        self.t_idx = 0

        self.buffers: Dict[str, torch.Tensor] = {}
        for dst, n in self.regions.items():
            self.buffers[dst] = torch.zeros((L, n), device=self.device, dtype=self.dtype)

        # Index edges by source/dest for fast loops
        self.by_src: Dict[str, List[Dict[str, Any]]] = {}
        self.by_dst: Dict[str, List[Dict[str, Any]]] = {}
        for ee in self.edges:
            self.by_src.setdefault(ee["src"], []).append(ee)
            self.by_dst.setdefault(ee["dst"], []).append(ee)

    # ---------- public API ----------

    def push(self, src_name: str, pre_spikes: torch.Tensor) -> None:
        """
        Queue drive from 'src_name' into destination buffers at the appropriate future slots.
        pre_spikes: tensor shape [n_src], float (rates or 0/1 spikes)
        """
        if src_name not in self.by_src:
            return  # no outgoing edges

        pre_spikes = _as_tensor(pre_spikes, self.device, self.dtype).view(-1)
        n_src = self.regions[src_name]
        if pre_spikes.numel() != n_src:
            # pad/trim to expected length
            pre_spikes = self._reshape_vec(pre_spikes, n_src)

        for ee in self.by_src[src_name]:
            dst = ee["dst"]
            W = ee["W"]                          # [n_dst, n_src]
            delay = ee["delay_steps"]
            sign = ee["sign"]
            gain = ee["gain"]

            # Compute drive for this edge
            drive = (W @ pre_spikes) * (gain * sign)   # [n_dst]

            # Place into (current + delay) slot for destination buffer
            write_idx = (self.t_idx + delay) % self.buffer_len
            self.buffers[dst][write_idx].add_(drive)

    def pull(self, dst_name: str) -> torch.Tensor:
        """
        Read and clear current time-slot drive for 'dst_name'.
        Returns tensor shape [n_dst].
        """
        buf = self.buffers[dst_name]
        out = buf[self.t_idx].clone()
        buf[self.t_idx].zero_()
        return out

    def tick(self) -> None:
        """Advance router by one time step."""
        self.t_idx = (self.t_idx + 1) % self.buffer_len

    def reset(self) -> None:
        """Zero all buffers and reset time index."""
        for k in self.buffers:
            self.buffers[k].zero_()
        self.t_idx = 0

    def to(self, device: torch.device) -> "DelayRouter42":
        """Move buffers and weights to a new device."""
        self.device = device
        for dst in self.buffers:
            self.buffers[dst] = self.buffers[dst].to(device=device, dtype=self.dtype)
        for ee in self.edges:
            ee["W"] = ee["W"].to(device=device, dtype=self.dtype)
        return self

    # ---------- helpers ----------

    def _reshape_W(self, W: torch.Tensor, n_dst: int, n_src: int) -> torch.Tensor:
        """Pad/trim a user-provided W to [n_dst, n_src] safely."""
        out = torch.zeros((n_dst, n_src), device=self.device, dtype=self.dtype)
        r = min(n_dst, W.shape[0])
        c = min(n_src, W.shape[1])
        out[:r, :c] = W[:r, :c]
        return out

    def _reshape_vec(self, v: torch.Tensor, n: int) -> torch.Tensor:
        """Pad/trim a vector to length n."""
        out = torch.zeros((n,), device=self.device, dtype=self.dtype)
        m = min(n, v.numel())
        out[:m] = v.view(-1)[:m]
        return out


# --------- minimal smoke test ----------
if __name__ == "__main__":
    # Two regions, 8 neurons each
    regions = {"R1": 8, "R2": 8}
    dt = 0.25  # ms
    # Edge: R1 -> R2, 10 mm distance, ~20 ms delay at 0.5 mm/ms => 80 steps
    W = torch.zeros(8, 8); W[3, 2] = 1.0  # pre #2 strongly drives post #3
    edges = [{
        "src": "R1", "dst": "R2",
        "W": W, "distance_mm": 10.0, "weight": 0.9, "receptor": "E"
    }]

    router = DelayRouter42(regions, edges, dt_ms=dt)
    # Stim: pre #2 fires at t=0
    pre = torch.zeros(8); pre[2] = 1.0
    router.push("R1", pre)

    # Walk time and check when R2 receives it
    hit = None
    for t in range(200):
        drive = router.pull("R2")
        if drive.abs().sum().item() > 0 and hit is None:
            hit = t
        router.tick()

    print("First nonzero at step:", hit, " => ms:", hit * dt)
