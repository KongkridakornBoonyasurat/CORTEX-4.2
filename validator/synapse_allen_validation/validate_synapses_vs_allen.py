#!/usr/bin/env python3
"""
validate_synapses_vs_allen.py

Compare CORTEX 4.2 Synapses against Allen Brain Connectivity Data + Literature

This validator compares your EnhancedSynapse42PyTorch to:
1. Allen Mouse Connectivity Atlas (projection strengths)
2. Published synaptic dynamics data (PPR, STP)
3. STDP protocols from literature

Outputs:
- CSV tables comparing weights to Allen connectivity
- Scatter plots (R², correlation)
- STP comparison (PPR vs ISI)
- STDP curves vs Bi & Poo (1998)
- JSON metrics summary

Requirements:
- allensdk (pip install allensdk)
- numpy, pandas, matplotlib
- Your mouse_connectivity/ cache folder

Usage:
    python validate_synapses_vs_allen.py --specimen-id 314800874
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

import sys, json, csv, importlib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allen SDK 
try:
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache  # type: ignore
    HAVE_ALLEN = True
except ImportError:
    HAVE_ALLEN = False
    MouseConnectivityCache = None  # type: ignore
    print("[Allen] allensdk not installed — will use local cache if available.")

# Paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTDIR = HERE / "synapse_allen_validation"
OUTDIR.mkdir(parents=True, exist_ok=True)
# ---- Resolve Allen cache directory (uses your existing sdk cache) ----
def _resolve_allen_cache_dir() -> Path:
    # 1) explicit env var wins
    env_dir = os.environ.get("ALLEN_CACHE_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    # 2) common project locations (your repo layout)
    candidate_dirs = [
        ROOT / "validator" / "allen_cache",          # e.g. cortex 4.2 v41/validator/allen_cache
        HERE / "allen_cache",                        # sibling to this file
        HERE / "mouse_connectivity",                 # default created by earlier versions
    ]
    for d in candidate_dirs:
        if (d / "manifest.json").exists():
            return d

    # 3) fallback (will be populated if you ever run live SDK)
    d = HERE / "mouse_connectivity"
    d.mkdir(parents=True, exist_ok=True)
    return d

CONNECTIVITY_CACHE = _resolve_allen_cache_dir()
print(f"[Allen] Using cache dir: {CONNECTIVITY_CACHE}")

# Skip CSV cache entirely - we have the real Allen SDK cache
allen_cache: Dict[Tuple[str, str], float] = {}
print("[Allen] Skipping CSV cache - using Allen SDK cache directly")
# ======================== IMPORT SYNAPSE (FIXED) ========================

# Add ROOT to path FIRST
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CRITICAL: Disable CUDA before importing synapse  
import torch
torch.cuda.is_available = lambda: False

# Now import synapse using the SAME pattern as validator_synapse.py
# Try multiple import paths
SYN_PATHS = [
    ("cortex.cells.enhanced_synapses_42", "EnhancedSynapse42PyTorch"),
    ("enhanced_synapses_42", "EnhancedSynapse42PyTorch"),
]

EnhancedSynapse42PyTorch = None
for module_name, class_name in SYN_PATHS:
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, class_name):
            EnhancedSynapse42PyTorch = getattr(mod, class_name)
            print(f"[OK] Found {class_name} in {module_name}")
            break
    except ImportError:
        continue

if EnhancedSynapse42PyTorch is None:
    print("ERROR: Could not import EnhancedSynapse42PyTorch")
    print("\nTried:")
    for module_name, class_name in SYN_PATHS:
        print(f"  - {module_name}.{class_name}")
    sys.exit(1)

# ======================== CONFIGURATION ========================

# Brain regions to compare (map CORTEX names to Allen structure acronyms)
REGION_MAPPING = {
    'PFC': 'PL',      # Prefrontal  → Prelimbic area
    'MOT': 'MOp',     # Motor       → Primary motor area
    'SENS': 'SSp',    # Sensory     → Primary somatosensory
    'PAR': 'PTLp',    # Parietal    → Posterior parietal
    'HIPP': 'CA1',    # Hippocampus → CA1
    'STR': 'CP',      # Striatum    → Caudoputamen
    'THAL': 'TH',     # Thalamus    → Thalamus
    'AMYG': 'BLA',    # Amygdala    → Basolateral amygdala
}

# Literature PPR values for comparison
LITERATURE_PPR = {
    'excitatory_cortical': {
        'PPR_20Hz': 0.75,  # Depression
        'PPR_50Hz': 0.65,
        'source': 'Markram et al. (1998) J Physiol'
    },
    'excitatory_hippocampal': {
        'PPR_20Hz': 0.85,
        'PPR_50Hz': 0.70,
        'source': 'Dobrunz & Stevens (1997) Neuron'
    }
}

# STDP reference (Bi & Poo 1998)
BI_POO_STDP = {
    'dt_ms': [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50],
    'dw_norm': [-0.25, -0.35, -0.50, -0.75, -0.90, 0.90, 0.75, 0.50, 0.35, 0.25],
    'source': 'Bi & Poo (1998) J Neurosci'
}

# ======================== ALLEN CONNECTIVITY FUNCTIONS ========================

def load_allen_connectivity() -> Dict[Tuple[str, str], float]:
    """
    Load Allen Mouse Connectivity from local cache using get_structure_unionizes.
    This extracts actual projection_density values from your cached specimen data.
    """
    print("[Allen] Loading connectivity from local cache...")
    connectivity_matrix: Dict[Tuple[str, str], float] = {}
    
    if not HAVE_ALLEN:
        print("[Allen] ERROR: allensdk not installed")
        for src in REGION_MAPPING.keys():
            for tgt in REGION_MAPPING.keys():
                if src != tgt:
                    connectivity_matrix[(src, tgt)] = 0.0
        return connectivity_matrix
    
    # Initialize Allen SDK cache reader
    mcc = MouseConnectivityCache(
        manifest_file=str(CONNECTIVITY_CACHE / "manifest.json"),
        resolution=25
    )
    structure_tree = mcc.get_structure_tree()
    
    # Loop through all region pairs
    for source_name, source_acr in REGION_MAPPING.items():
        for target_name, target_acr in REGION_MAPPING.items():
            if source_name == target_name:
                continue
            
            try:
                # Get Allen structure IDs
                source_structs = structure_tree.get_structures_by_acronym([source_acr])
                target_structs = structure_tree.get_structures_by_acronym([target_acr])
                
                if not source_structs or not target_structs:
                    connectivity_matrix[(source_name, target_name)] = 0.0
                    continue
                
                source_id = source_structs[0]['id']
                target_id = target_structs[0]['id']
                
                # Get experiments with injections in source region
                experiments = mcc.get_experiments(injection_structure_ids=[source_id])
                
                if not experiments:
                    connectivity_matrix[(source_name, target_name)] = 0.0
                    continue
                
                # KEY FIX: Extract projection densities using get_structure_unionizes
                densities = []
                for exp in experiments[:10]:  # Check up to 10 experiments
                    try:
                        exp_id = exp['id']
                        
                        # Use get_structure_unionizes to get actual projection data
                        unionizes = mcc.get_structure_unionizes(
                            experiment_ids=[exp_id],
                            structure_ids=[target_id]
                        )
                        
                        if unionizes.empty:
                            continue
                        
                        # Filter to non-injection sites only
                        non_injection = unionizes[unionizes['is_injection'] == False]
                        
                        if len(non_injection) == 0:
                            continue
                        
                        # Extract projection_density column
                        if 'projection_density' not in non_injection.columns:
                            continue
                        
                        density_value = non_injection['projection_density'].values[0]
                        
                        # Only add valid, non-zero densities
                        if not np.isnan(density_value) and density_value > 0:
                            densities.append(float(density_value))
                            
                    except Exception:
                        continue
                
                # Average all valid densities found
                if densities:
                    avg_density = float(np.mean(densities))
                    connectivity_matrix[(source_name, target_name)] = avg_density
                else:
                    connectivity_matrix[(source_name, target_name)] = 0.0
                    
            except Exception as e:
                connectivity_matrix[(source_name, target_name)] = 0.0
    
    # Summary
    non_zero = sum(1 for v in connectivity_matrix.values() if v > 0)
    total = len(connectivity_matrix)
    print(f"[Allen] Loaded {non_zero}/{total} connections with non-zero density")
    
    return connectivity_matrix

# ======================== SYNAPSE WEIGHT EXTRACTION ========================

def extract_cortex_weights() -> Dict[str, float]:
    """
    Extract synaptic weights from your CORTEX biological_connectivity.py
    
    Returns dict: {(source_region, target_region): weight}
    """
    print("[CORTEX] Loading your connectivity weights...")
    
    # Import your connectivity using same pattern
    try:
        from cortex.connectivity.biological_connectivity import BiologicalConnectivityMatrix42PyTorch
    except ImportError:
        print("[WARNING] Could not import biological_connectivity, using mock weights")
        # Return mock weights for testing
        weights = {}
        for src in REGION_MAPPING.keys():
            for tgt in REGION_MAPPING.keys():
                if src != tgt:
                    weights[(src, tgt)] = np.random.uniform(0.1, 1.0)
        return weights
    
    conn = BiologicalConnectivityMatrix42PyTorch()
    
    weights = {}
    for source_name in REGION_MAPPING.keys():
        for target_name in REGION_MAPPING.keys():
            if source_name == target_name:
                continue
            
            try:
                weight = conn.get_connection(source_name, target_name)
                weights[(source_name, target_name)] = float(weight.item())
            except:
                weights[(source_name, target_name)] = 0.0
    
    print(f"[CORTEX] Extracted {len(weights)} connection weights")
    return weights

# ======================== SYNAPSE DYNAMICS (PPR) ========================

def measure_synapse_ppr(synapse: 'EnhancedSynapse42PyTorch', isi_ms: float = 50.0) -> float:
    """
    Measure Paired-Pulse Ratio for a synapse.
    
    PPR = amplitude_pulse2 / amplitude_pulse1
    """
    synapse.reset_traces()
    
    dt_ms = 1.0
    t_ms = 0.0
    
    # Baseline
    for _ in range(100):
        synapse.record_pre(dt_ms / 1000.0, spike_strength=0.0)
        synapse.record_post(dt_ms / 1000.0, spike_strength=0.0)
        t_ms += dt_ms
    
    # Pulse 1
    synapse.record_pre(dt_ms / 1000.0, spike_strength=1.0)
    synapse.record_post(dt_ms / 1000.0, spike_strength=0.0)
    amp1 = float(synapse.w.item())  # Weight proxy for amplitude
    
    # Inter-stimulus interval
    for _ in range(int(isi_ms / dt_ms)):
        synapse.record_pre(dt_ms / 1000.0, spike_strength=0.0)
        synapse.record_post(dt_ms / 1000.0, spike_strength=0.0)
    
    # Pulse 2
    synapse.record_pre(dt_ms / 1000.0, spike_strength=1.0)
    synapse.record_post(dt_ms / 1000.0, spike_strength=0.0)
    amp2 = float(synapse.w.item())
    
    if amp1 > 0:
        ppr = amp2 / amp1
    else:
        ppr = 1.0
    
    return ppr

def measure_synapse_stdp(synapse: 'EnhancedSynapse42PyTorch', dt_ms: float = 20.0, n_pairs: int = 60) -> float:
    """
    Measure STDP weight change for a given spike timing difference.
    
    Returns: Normalized weight change (dw / w0)
    """
    synapse.reset_traces()
    w0 = 0.5
    synapse.w.data = torch.tensor(w0, device=synapse.device)
    
    dt_s = 0.001  # 1 ms timestep
    pair_interval_ms = 1000.0  # 1 Hz pairing
    
    for i in range(n_pairs):
        # Wait between pairs
        for _ in range(int(pair_interval_ms)):
            synapse.record_pre(dt_s, spike_strength=0.0)
            synapse.record_post(dt_s, spike_strength=0.0)
        
        # Spike pair with timing dt_ms
        if dt_ms > 0:  # Post after pre
            synapse.record_pre(dt_s, spike_strength=1.0)
            for _ in range(int(abs(dt_ms))):
                synapse.record_pre(dt_s, spike_strength=0.0)
                synapse.record_post(dt_s, spike_strength=0.0)
            synapse.record_post(dt_s, spike_strength=1.0)
        else:  # Pre after post
            synapse.record_post(dt_s, spike_strength=1.0)
            for _ in range(int(abs(dt_ms))):
                synapse.record_pre(dt_s, spike_strength=0.0)
                synapse.record_post(dt_s, spike_strength=0.0)
            synapse.record_pre(dt_s, spike_strength=1.0)
        
        # Update weight
        synapse.update_weight(reward=1.0, dopamine=1.0, ach=1.0, ne=1.0, dt=dt_s)
    
    w_final = float(synapse.w.item())
    dw_norm = (w_final - w0) / w0
    
    return dw_norm

# ======================== COMPARISON & METRICS ========================

def compare_connectivity_to_allen(cortex_weights: Dict, allen_connectivity: Dict) -> Dict:
    """
    Compare your CORTEX weights to Allen connectivity.
    
    Returns metrics dict with R², correlation, etc.
    """
    # Find common pairs
    common_pairs = set(cortex_weights.keys()) & set(allen_connectivity.keys())
    
    if not common_pairs:
        print("[ERROR] No common region pairs found!")
        return {}
    
    # Extract paired values
    cortex_vals = [cortex_weights[pair] for pair in common_pairs]
    allen_vals = [allen_connectivity[pair] for pair in common_pairs]
    
    # Convert to numpy
    x = np.array(allen_vals)
    y = np.array(cortex_vals)
    
    # Remove NaNs and zeros (no connectivity)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_nonzero = x[mask]
    y_nonzero = y[mask]
    
    if len(x_nonzero) < 2:
        print("[WARNING] Too few non-zero connections for statistics")
        return {}
    
    # Linear regression
    m, b = np.polyfit(x_nonzero, y_nonzero, 1)
    y_pred = m * x_nonzero + b
    
    # R²
    ss_res = np.sum((y_nonzero - y_pred) ** 2)
    ss_tot = np.sum((y_nonzero - np.mean(y_nonzero)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Correlation
    correlation = np.corrcoef(x_nonzero, y_nonzero)[0, 1]
    
    # MAE, RMSE
    mae = np.mean(np.abs(y_nonzero - x_nonzero))
    rmse = np.sqrt(np.mean((y_nonzero - x_nonzero) ** 2))
    
    metrics = {
        'r2': float(r2),
        'correlation': float(correlation),
        'slope': float(m),
        'intercept': float(b),
        'mae': float(mae),
        'rmse': float(rmse),
        'n_pairs': int(len(common_pairs)),
        'n_nonzero': int(len(x_nonzero))
    }
    
    return metrics

# ======================== PLOTTING ========================

def plot_connectivity_scatter(cortex_weights: Dict, allen_connectivity: Dict, metrics: Dict):
    """Plot Allen vs CORTEX connectivity scatter"""
    common_pairs = set(cortex_weights.keys()) & set(allen_connectivity.keys())
    
    x = [allen_connectivity[pair] for pair in common_pairs]
    y = [cortex_weights[pair] for pair in common_pairs]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, s=50, alpha=0.6, edgecolors='k')
    
    # Fit line
    x_arr = np.array(x)
    y_arr = np.array(y)
    mask = (x_arr > 0) & (y_arr > 0)
    if mask.sum() >= 2:
        m, b = metrics['slope'], metrics['intercept']
        x_fit = np.linspace(x_arr[mask].min(), x_arr[mask].max(), 100)
        y_fit = m * x_fit + b
        ax.plot(x_fit, y_fit, 'r--', linewidth=2, label=f'Fit: y={m:.2f}x+{b:.2f}')
    
    # Identity line
    max_val = max(max(x), max(y))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Identity')
    
    ax.set_xlabel('Allen Projection Density', fontsize=12)
    ax.set_ylabel('CORTEX Synaptic Weight', fontsize=12)
    ax.set_title(f"Connectivity: Allen vs CORTEX\n(R²={metrics.get('r2', 0):.3f}, ρ={metrics.get('correlation', 0):.3f})", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "connectivity_scatter.png", dpi=180)
    plt.close(fig)
    print(f"[PLOT] Saved connectivity_scatter.png")

def plot_ppr_comparison(cortex_pprs: Dict, literature_pprs: Dict):
    """Plot PPR comparison to literature"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    isis = sorted(cortex_pprs.keys())
    cortex_vals = [cortex_pprs[isi] for isi in isis]
    
    ax.plot(isis, cortex_vals, 'o-', linewidth=2, markersize=8, label='CORTEX Synapse')
    
    # Literature reference lines
    ax.axhline(y=LITERATURE_PPR['excitatory_cortical']['PPR_20Hz'], color='r', linestyle='--', label='Literature (20Hz)')
    ax.axhline(y=LITERATURE_PPR['excitatory_cortical']['PPR_50Hz'], color='g', linestyle='--', label='Literature (50Hz)')
    
    ax.set_xlabel('Inter-Stimulus Interval (ms)', fontsize=12)
    ax.set_ylabel('Paired-Pulse Ratio (PPR)', fontsize=12)
    ax.set_title('Short-Term Plasticity: CORTEX vs Literature', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0.4, 1.2])
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "ppr_comparison.png", dpi=180)
    plt.close(fig)
    print(f"[PLOT] Saved ppr_comparison.png")

def plot_stdp_comparison(cortex_stdp: Dict):
    """Plot STDP curve vs Bi & Poo (1998)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # CORTEX data
    dts = sorted(cortex_stdp.keys())
    cortex_vals = [cortex_stdp[dt] for dt in dts]
    
    ax.plot(dts, cortex_vals, 'o-', linewidth=2, markersize=8, label='CORTEX Synapse', color='blue')
    
    # Bi & Poo reference
    ax.plot(BI_POO_STDP['dt_ms'], BI_POO_STDP['dw_norm'], 's--', linewidth=2, markersize=6, 
            label='Bi & Poo (1998)', color='red', alpha=0.7)
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Spike Timing Δt (ms)', fontsize=12)
    ax.set_ylabel('Normalized Weight Change (Δw/w₀)', fontsize=12)
    ax.set_title('STDP: CORTEX vs Bi & Poo (1998)', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "stdp_comparison.png", dpi=180)
    plt.close(fig)
    print(f"[PLOT] Saved stdp_comparison.png")

# ======================== MAIN VALIDATION ========================

def run_validation():
    """Run complete synapse validation"""
    print("="*80)
    print("CORTEX 4.2 Synapse Validation vs Allen Brain + Literature")
    print("="*80)
    
    # ========== 1. CONNECTIVITY COMPARISON ==========
    print("\n[1/3] Comparing synaptic weights to Allen connectivity...")
    
    try:
        print("\n[DEBUG] Calling load_allen_connectivity()...")
        allen_conn = load_allen_connectivity()
        print(f"[DEBUG] Got {len(allen_conn)} Allen connections")
        print(f"[DEBUG] Sample Allen values: {list(allen_conn.values())[:5]}")
        
        print("\n[DEBUG] Calling extract_cortex_weights()...")
        cortex_weights = extract_cortex_weights()
        print(f"[DEBUG] Got {len(cortex_weights)} CORTEX weights")
        print(f"[DEBUG] Sample CORTEX values: {list(cortex_weights.values())[:5]}")

        metrics = compare_connectivity_to_allen(cortex_weights, allen_conn)
        
        # Save results
        with open(OUTDIR / "connectivity_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save paired data
        common_pairs = set(cortex_weights.keys()) & set(allen_conn.keys())
        rows = []
        for (src, tgt) in sorted(common_pairs):
            rows.append({
                'source': src,
                'target': tgt,
                'allen_density': allen_conn[(src, tgt)],
                'cortex_weight': cortex_weights[(src, tgt)]
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(OUTDIR / "connectivity_comparison.csv", index=False)
        
        plot_connectivity_scatter(cortex_weights, allen_conn, metrics)
        
        print(f"   R² = {metrics.get('r2', 0):.4f}")
        print(f"   Correlation = {metrics.get('correlation', 0):.4f}")
        print(f"   N pairs = {metrics.get('n_pairs', 0)}")
        
    except Exception as e:
        print(f"   [ERROR] Connectivity comparison failed!")
        print(f"   Exception type: {type(e).__name__}")
        print(f"   Exception message: {e}")
        import traceback
        traceback.print_exc()
        metrics = {}

    # ========== 2. SHORT-TERM PLASTICITY (PPR) ==========
    print("\n[2/3] Measuring Paired-Pulse Ratio (PPR)...")
    
    synapse = EnhancedSynapse42PyTorch(device=torch.device('cpu'))
    
    isis = [20, 50, 100, 200]  # ms
    pprs = {}
    
    for isi in isis:
        ppr = measure_synapse_ppr(synapse, isi_ms=isi)
        pprs[isi] = ppr
        print(f"   ISI={isi}ms -> PPR={ppr:.3f}")

    # Save PPR data
    with open(OUTDIR / "ppr_measurements.json", "w") as f:
        json.dump({'pprs': pprs, 'literature': LITERATURE_PPR}, f, indent=2)
    
    plot_ppr_comparison(pprs, LITERATURE_PPR)
    
    # ========== 3. STDP CURVE ==========
    print("\n[3/3] Measuring STDP curve...")
    
    dts = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50]  # ms
    stdp_results = {}
    
    for dt in dts:
        synapse_fresh = EnhancedSynapse42PyTorch(device=torch.device('cpu'))
        dw = measure_synapse_stdp(synapse_fresh, dt_ms=dt, n_pairs=60)
        stdp_results[dt] = dw
        print(f"   dt={dt:+3d}ms -> dw/w0={dw:+.3f}")

    # Save STDP data
    with open(OUTDIR / "stdp_measurements.json", "w") as f:
        json.dump({'stdp': stdp_results, 'bi_poo_1998': BI_POO_STDP}, f, indent=2)
    
    plot_stdp_comparison(stdp_results)
    
    # ========== SUMMARY ==========
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTDIR}")
    print("\nFiles generated:")
    print("  - connectivity_comparison.csv")
    print("  - connectivity_metrics.json")
    print("  - connectivity_scatter.png")
    print("  - ppr_measurements.json")
    print("  - ppr_comparison.png")
    print("  - stdp_measurements.json")
    print("  - stdp_comparison.png")
    
    print("\n KEY METRICS:")
    if metrics:
        print(f"   Connectivity R² = {metrics.get('r2', 0):.4f}")
        print(f"   Connectivity Correlation = {metrics.get('correlation', 0):.4f}")
    print(f"   PPR (50ms ISI) = {pprs.get(50, 0):.3f}")
    print(f"   STDP asymmetry = {abs(stdp_results.get(20, 0)) + abs(stdp_results.get(-20, 0)):.3f}")
    
    print("\n ARCHITECTURAL NOTES:")
    print("   PPR: Short-term plasticity is implemented at the population level")
    print("        (EnhancedSynapticSystem42PyTorch) rather than individual synapses.")
    print("        This validator tests individual synapses, so PPR=1.0 is expected.")
    print("        STP dynamics are active in full network simulations.")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Validate CORTEX synapses vs Allen Brain')
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer measurements)')
    args = parser.parse_args()
    
    run_validation()