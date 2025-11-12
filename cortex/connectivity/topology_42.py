# cortex/connectivity/topology_42.py
"""
Topology42 — region sizes, coordinates, and edges for CORTEX 4.2 routing
------------------------------------------------------------------------
- Defines small, biologically sensible region set (names + neuron counts)
- Optional 3D coordinates in millimeters (coarse, left-hemisphere-ish)
- Compiles an `edges` list suitable for DelayRouter42 (distance->delay)

Exports
-------
make_default_topology() -> (regions_meta, edges)
make_small_topology()   -> (regions_meta, edges)     # even smaller counts
build_edges(regions_meta, coords, spec_list, *, rng=None) -> edges
weight_matrix(kind, n_dst, n_src, density=0.15, gain=0.8, seed=None)

Notes
-----
- If an edge has no `delay_ms`, we compute it from Euclidean distance (mm)
  using the router's conduction velocity.
- If `W` is omitted, we generate a weight matrix by `weight_matrix(...)`.
- Router will apply sign from `receptor` ('E' -> +, 'I' -> -), so `W` can be
  nonnegative magnitudes.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import math
import numpy as np


# -------------------------- coordinates (mm) -------------------------- #
# Coarse centroids (you can replace with your lab’s coordinates later)
DEFAULT_COORDS_MM: Dict[str, Tuple[float, float, float]] = {
    "V1":       (30.0, -85.0, 10.0),
    "V2":       (35.0, -75.0, 12.0),
    "S1":       (40.0, -25.0, 50.0),
    "M1":       (35.0, -15.0, 55.0),
    "PFC":      (30.0,  45.0, 30.0),
    "THAL":     (10.0, -10.0,  5.0),
    "HIPPO":    (15.0, -30.0,  0.0),
    "AMY":      (20.0, -10.0, -10.0),
    "STR":      (15.0,   0.0,  5.0),
    "CBL":      ( 5.0, -60.0, -30.0),  # cerebellum
}

# -------------------------- region sizes ----------------------------- #
# “Small brain”: keep GPU-friendly. Tweak as needed.
DEFAULT_REGION_SIZES: Dict[str, int] = {
    "V1": 64, "V2": 48, "S1": 48, "M1": 48,
    "PFC": 48, "THAL": 32, "HIPPO": 48, "AMY": 24, "STR": 32, "CBL": 32
}

SMALL_REGION_SIZES: Dict[str, int] = {
    "V1": 24, "V2": 16, "S1": 16, "M1": 16,
    "PFC": 16, "THAL": 12, "HIPPO": 16, "AMY": 8, "STR": 12, "CBL": 12
}

# -------------------------- utilities -------------------------------- #
def euclidean_mm(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    ax, ay, az = a; bx, by, bz = b
    return float(math.sqrt((ax-bx)**2 + (ay-by)**2 + (az-bz)**2))

def weight_matrix(
    kind: str,
    n_dst: int,
    n_src: int,
    density: float = 0.15,
    gain: float = 0.8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    kind: 'identity' | 'dense' | 'sparse'
    Returns nonnegative magnitudes; router applies E/I sign.
    """
    rng = np.random.default_rng(seed)
    if kind == "identity":
        k = min(n_dst, n_src)
        W = np.zeros((n_dst, n_src), dtype=np.float32)
        idx = np.arange(k)
        W[idx, idx] = gain
        return W
    if kind in ("dense", "sparse"):
        W = rng.normal(loc=gain, scale=0.15*gain, size=(n_dst, n_src)).astype(np.float32)
        W = np.clip(W, 0.0, None)
        if kind == "sparse":
            mask = (rng.random((n_dst, n_src)) < density).astype(np.float32)
            W *= mask
        # fan-in scaling to keep postsyn current stable-ish
        fan_in = np.maximum(np.sum(W > 0, axis=1, keepdims=True), 1)
        W = W / np.sqrt(fan_in)
        return W
    raise ValueError(f"Unknown weight kind: {kind}")

def build_edges(
    regions_meta: Dict[str, int],
    coords_mm: Dict[str, Tuple[float,float,float]],
    spec_list: List[Dict[str, Any]],
    *,
    rng: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    """
    spec_list items support:
      {
        'src': 'V1', 'dst': 'V2',
        'receptor': 'E'|'I',
        'weight': 1.0,                  # scalar post-gain
        'W': np.ndarray | None,         # optional; if None -> built from 'kind'
        'kind': 'identity'|'dense'|'sparse',
        'density': 0.15, 'gain': 0.8,   # used if W is None
        'delay_ms': float | None,       # if None -> computed from coordinates
        'distance_mm': float | None     # override distance if provided
      }
    """
    edges: List[Dict[str, Any]] = []
    for spec in spec_list:
        src = spec["src"]; dst = spec["dst"]
        n_src = regions_meta[src]; n_dst = regions_meta[dst]
        receptor = spec.get("receptor", "E")
        post_gain = float(spec.get("weight", 1.0))

        # Weight matrix
        W = spec.get("W", None)
        if W is None:
            kind = spec.get("kind", "sparse")
            density = float(spec.get("density", 0.15))
            gain = float(spec.get("gain", 0.8))
            W = weight_matrix(kind, n_dst, n_src, density=density, gain=gain, seed=rng)

        # Distance / delay
        if spec.get("delay_ms", None) is not None:
            delay_ms = float(spec["delay_ms"])
            dist_mm = float(spec.get("distance_mm", 0.0))
        else:
            if spec.get("distance_mm", None) is not None:
                dist_mm = float(spec["distance_mm"])
            else:
                if src not in coords_mm or dst not in coords_mm:
                    # Fallback small distance if coords missing
                    dist_mm = 5.0
                else:
                    dist_mm = euclidean_mm(coords_mm[src], coords_mm[dst])
            delay_ms = None  # router will compute from distance via conduction

        edge = {
            "src": src, "dst": dst,
            "W": W,
            "distance_mm": dist_mm,
            "delay_ms": delay_ms,
            "weight": post_gain,
            "receptor": receptor,
        }
        edges.append(edge)
    return edges

# -------------------------- canned topologies ------------------------ #
def make_default_topology() -> Tuple[Dict[str,int], List[Dict[str,Any]]]:
    """
    Medium-small brain with sensible pathways.
    """
    regions = dict(DEFAULT_REGION_SIZES)
    coords = dict(DEFAULT_COORDS_MM)

    # Visual chain; thalamocortical; somatomotor; PFC loops; limbic/basal ganglia
    specs = [
        # Vision
        {"src":"V1","dst":"V2","receptor":"E","weight":1.0,"kind":"sparse","density":0.20,"gain":0.9},
        # Thalamo-cortical sensory
        {"src":"THAL","dst":"S1","receptor":"E","weight":1.0,"kind":"sparse","density":0.25,"gain":0.9},
        {"src":"THAL","dst":"V1","receptor":"E","weight":1.0,"kind":"sparse","density":0.25,"gain":0.9},
        # Cortico-cortical forward / feedback
        {"src":"S1","dst":"M1","receptor":"E","weight":0.9,"kind":"sparse","density":0.20,"gain":0.8},
        {"src":"V2","dst":"PFC","receptor":"E","weight":0.7,"kind":"sparse","density":0.15,"gain":0.7},
        {"src":"PFC","dst":"M1","receptor":"E","weight":0.8,"kind":"sparse","density":0.20,"gain":0.8},
        {"src":"PFC","dst":"S1","receptor":"I","weight":0.6,"kind":"sparse","density":0.15,"gain":0.6},  # top-down inhibition
        # Hippocampus → PFC memory channel
        {"src":"HIPPO","dst":"PFC","receptor":"E","weight":0.8,"kind":"sparse","density":0.20,"gain":0.8},
        # Amygdala ↔ PFC affect
        {"src":"AMY","dst":"PFC","receptor":"E","weight":0.7,"kind":"sparse","density":0.15,"gain":0.7},
        {"src":"PFC","dst":"AMY","receptor":"I","weight":0.6,"kind":"sparse","density":0.15,"gain":0.6},
        # Basal ganglia loop (very simplified)
        {"src":"PFC","dst":"STR","receptor":"E","weight":0.8,"kind":"sparse","density":0.20,"gain":0.8},
        {"src":"STR","dst":"THAL","receptor":"I","weight":0.7,"kind":"sparse","density":0.20,"gain":0.8},
        {"src":"THAL","dst":"PFC","receptor":"E","weight":0.8,"kind":"sparse","density":0.25,"gain":0.9},
        # Cerebellum → motor (simplified feedback)
        {"src":"CBL","dst":"M1","receptor":"E","weight":0.7,"kind":"sparse","density":0.20,"gain":0.7},
        # Intrinsic PFC (short feedback)
        {"src":"PFC","dst":"PFC","receptor":"E","weight":0.6,"kind":"dense","gain":0.6},
    ]
    edges = build_edges(regions, coords, specs)
    return regions, edges

def make_small_topology() -> Tuple[Dict[str,int], List[Dict[str,Any]]]:
    """
    Tiny counts for laptops; same pathways.
    """
    regions = dict(SMALL_REGION_SIZES)
    # reuse coords; neuron counts change only
    coords = dict(DEFAULT_COORDS_MM)

    # keep the same pathway spec but use 'identity' W for a few to be cheap
    specs = [
        {"src":"V1","dst":"V2","receptor":"E","weight":1.0,"kind":"identity","gain":0.9},
        {"src":"THAL","dst":"S1","receptor":"E","weight":1.0,"kind":"sparse","density":0.25,"gain":0.9},
        {"src":"THAL","dst":"V1","receptor":"E","weight":1.0,"kind":"sparse","density":0.25,"gain":0.9},
        {"src":"S1","dst":"M1","receptor":"E","weight":0.9,"kind":"identity","gain":0.8},
        {"src":"V2","dst":"PFC","receptor":"E","weight":0.7,"kind":"sparse","density":0.15,"gain":0.7},
        {"src":"PFC","dst":"M1","receptor":"E","weight":0.8,"kind":"sparse","density":0.20,"gain":0.8},
        {"src":"PFC","dst":"S1","receptor":"I","weight":0.6,"kind":"sparse","density":0.15,"gain":0.6},
        {"src":"HIPPO","dst":"PFC","receptor":"E","weight":0.8,"kind":"sparse","density":0.20,"gain":0.8},
        {"src":"AMY","dst":"PFC","receptor":"E","weight":0.7,"kind":"sparse","density":0.15,"gain":0.7},
        {"src":"PFC","dst":"AMY","receptor":"I","weight":0.6,"kind":"sparse","density":0.15,"gain":0.6},
        {"src":"PFC","dst":"STR","receptor":"E","weight":0.8,"kind":"sparse","density":0.20,"gain":0.8},
        {"src":"STR","dst":"THAL","receptor":"I","weight":0.7,"kind":"sparse","density":0.20,"gain":0.8},
        {"src":"THAL","dst":"PFC","receptor":"E","weight":0.8,"kind":"sparse","density":0.25,"gain":0.9},
        {"src":"CBL","dst":"M1","receptor":"E","weight":0.7,"kind":"identity","gain":0.7},
        {"src":"PFC","dst":"PFC","receptor":"E","weight":0.6,"kind":"dense","gain":0.6},
    ]
    edges = build_edges(regions, coords, specs)
    return regions, edges
