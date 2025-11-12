# cortex/config.py
"""
Fixed Central Configuration for CORTEX 4.1 → 4.3 small-bio v0
Units: time=ms, voltage=mV, conductance=nS, capacitance=pF, current=pA
This file is the ONLY source of truth for parameters and units.
"""

NMDA_BLOCK_DIVISOR = 3.57

# ─────────────────────────────────────────────────────────────────────────────
# Version & reproducibility
CONFIG_VERSION     = "4.3-smallbio-v0"
SEED               = 1337

# ─────────────────────────────────────────────────────────────────────────────
# Simulation control (units: ms, mV, nS, pF, pA)
DT_MS           = 0.25     # ms per time step (was 1.0). Finer dt for AdEx + delays
DT              = DT_MS    # backward-compatible alias
SIM_DURATION_MS = 50.0     # per episode duration (ms)
SIM_DURATION    = SIM_DURATION_MS  # alias
SIM_STEPS       = int(SIM_DURATION_MS / DT_MS)
N_EPISODES      = 5
SNAPSHOT_EVERY  = 10

# Logging
LOG_DIR         = "logs"
LOG_LEVEL       = "INFO"
USE_TENSORBOARD = False
VERBOSE         = False     # gate prints in modules

# ─────────────────────────────────────────────────────────────────────────────
# Backend & device
BACKEND            = "torch"     # unified
DEVICE_PREFERENCE  = "auto"      # "cpu" | "cuda" | "auto"

# ─────────────────────────────────────────────────────────────────────────────
# Feature flags (keep legacy pieces off while we build biology-first path)
ENABLE_ADEX            = True
ENABLE_TWO_COMP_PYR    = False
ENABLE_ASTRO           = False
ENABLE_OSC             = False
ENABLE_LEGACY_PREDICT  = False

# ─────────────────────────────────────────────────────────────────────────────
# Predictive‐Coding (legacy; kept for compat but disabled by flag)
PRED_LAYERS     = [32, 16]
PRED_LR_FF      = 0.005     # Reduced for stability
PRED_LR_FB      = 0.005     # Reduced for stability

# ─────────────────────────────────────────────────────────────────────────────
# Readout & Self‐Model (legacy; kept for compat but disabled by flag)
READOUT_HIDDEN  = 32        # Reduced from 64
READOUT_LR      = 0.001     # Conservative learning rate
SELF_MODEL_LR   = 0.001     # Conservative learning rate

# ─────────────────────────────────────────────────────────────────────────────
# Memory Buffer & Replay Config (legacy scaffolding)
MEMORY_SIZE     = 500       # Reduced to prevent memory issues
REPLAY_PROB     = 0.05      # Less frequent replay
TRACE_DECAY     = 0.98      # Faster decay for stability

# Memory & Replay (dual‐phase)
WAKE_INTERVAL      = 200.0     # ms between wake‐phase replays (increased)
SLEEP_INTERVAL     = 5000.0    # ms between sleep‐phase replays (reduced)
EPOCHS_WAKE        = 3         # Fewer events per wake replay
EPOCHS_SLEEP       = 20        # Fewer events per sleep replay
REPLAY_BOOST_SLEEP = 2.0       # Reduced ACh multiplier

# ─────────────────────────────────────────────────────────────────────────────
# Astrocyte–Neuron Coupling (deferred in v0; values kept for later)
N_ASTRO            = 8         # Reduced from 10
TAU_C              = 800.0     # ms, Ca²⁺ decay (faster)
ALPHA_C            = 0.03      # Reduced Ca bump per spike
BETA_ASTRO         = 0.3       # Reduced modulation strength

# ─────────────────────────────────────────────────────────────────────────────
# Oscillator Rhythms (deferred in v0; values kept for later)
OSC_FREQ           = 8.0       # Hz (unchanged - good frequency)
OSC_AMP            = 0.15      # Reduced amplitude
# ---- EEG/oscillator helpers (used by oscillator.py & modulators.py)
OSC_PHASE_NOISE_STD = 0.02      # rad/sqrt(s), small phase noise for realism
OSC_BANDS_HZ = {                # central band freqs for quick mixes
    'delta': 2.5, 'theta': 6.0, 'alpha': 10.0, 'beta': 20.0, 'gamma': 40.0
}
OSC_DEFAULT_WEIGHTS = {         # default regional mixture (unitless)
    'delta': 0.00, 'theta': 1.00, 'alpha': 0.30, 'beta': 0.20, 'gamma': 0.10
}

# ─────────────────────────────────────────────────────────────────────────────
# Neuromodulators & Plasticity (legacy + shared gate)
TAU_D              = 150.0     # Faster dopamine decay
TAU_ACH            = 200.0     # Faster ACh decay
TAU_NE             = 250.0     # Faster NE decay
D_BASE             = 0.15      # Higher baseline dopamine
BETA_ACH           = 0.25      # Reduced ACh effect
BETA_NE            = 0.15      # Reduced NE effect
TAU_PLUS           = 30.0      # Reduced STDP time constants
TAU_MINUS          = 35.0      # Slightly asymmetric for balanced LTP/LTD
A_PLUS             = 0.02      # Reduced LTP strength
A_MINUS            = 0.015     # Reduced LTD strength
MAX_CHANGE         = 0.1       # Smaller max weight change per step

# Modulator gate (used by STDP)
MOD_BASELINE = {'DA': 0.0, 'ACh': 0.0, 'NE': 0.0}
MOD_GAINS    = {'DA': 1.0, 'ACh': 1.0, 'NE': 1.0}
M_MIN, M_MAX = 0.5, 1.5   # clamp for STDP gate multiplier

# STDP unified bounds (tie to learning bounds below)
MIN_WEIGHT         = 0.05      # Minimum synaptic weight
MAX_WEIGHT         = 0.8       # Maximum synaptic weight
STDP_W_MIN         = MIN_WEIGHT
STDP_W_MAX         = MAX_WEIGHT

# ─────────────────────────────────────────────────────────────────────────────
# Neuron biophysics & dendrites (legacy block kept for compat; NOT used by AdEx)
CM                 = 1.0       # Membrane capacitance (legacy units)
GL                 = 0.04      # Reduced leak conductance for stability
EL                 = -65.0     # Resting potential (unchanged)
V_THRESH           = -40.0     # Higher threshold (less excitable)
V_RESET            = -65.0     # Reset to resting (unchanged)
NOISE_SCALE        = 2.0       # Much reduced noise (was 8.0)

# Dendritic compartments - legacy placeholders
N_DENDRITES        = 4
G_NMDA             = 0.3
TAU_NMDA           = 80.0
ALPHA_NMDA         = 0.15
G_LEAK_DEND        = 0.02
G_COUPLE           = [0.08, 0.08, 0.08, 0.08]
E_NMDA             = 0.0
E_LEAK             = -65.0

# ─────────────────────────────────────────────────────────────────────────────
# Small-bio AdEx path (USED by the new cells/synapses)
# Neuron classes with explicit units in keys
ADEX_PYR = dict(    
    Cm_pF=200.0, gL_nS=10.0, EL_mV=-70.0, VT_mV=-52.0, dT_mV=3.0,
    Vr_mV=-58.0, a_nS=2.0, b_pA=60.0, tau_w_ms=200.0,
    refractory_ms=2.0, V_spike_mV=20.0
)
ADEX_FS = dict(
    Cm_pF=100.0, gL_nS=14.0, EL_mV=-68.0, VT_mV=-52.0, dT_mV=2.0,
    Vr_mV=-55.0, a_nS=0.0,  b_pA=0.0,  tau_w_ms=5.0,
    refractory_ms=1.0, V_spike_mV=20.0
)

# Synaptic reversal potentials for AdEx path
E_E_MV = 0.0      # excitatory (AMPA/NMDA)
E_I_MV = -70.0    # inhibitory (GABA-A/B)

# Synaptic kinetics & NMDA block (explicit names to avoid legacy collisions)
TAU_AMPA_MS   = 3.0
TAU_NMDA_MS   = 80.0
TAU_GABA_A_MS = 8.0
TAU_GABA_B_MS = 150.0
NMDA_BLOCK_ALPHA     = 0.062   # 1/mV
NMDA_BLOCK_BETA      = 1.0
NMDA_MG_MILLIMOLAR   = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Sensor settings
SENSOR_CHANNELS      = {"food": True, "pain": True}
SENSOR_JITTER_STD    = 0.5       # Reduced jitter
SENSOR_DROPOUT_RATE  = 0.0       # No dropout for stability

# ─────────────────────────────────────────────────────────────────────────────
# Symbol & Workspace - REASONABLE SIZES
K_SYMBOLS          = 12        # Reduced from 16
K_BROADCAST        = 4         # Keep at 4 (good size)

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap exploration - GENTLE PARAMETERS
BOOTSTRAP_RATE     = 0.03      # Slightly more exploration
BOOTSTRAP_STRENGTH = 10.0      # Reduced strength (was 15.0)

# ─────────────────────────────────────────────────────────────────────────────
# Environment
WORLD_SIZE         = 5         # Keep simple world

# ─────────────────────────────────────────────────────────────────────────────
# Additional stability parameters
MAX_FIRING_RATE        = 0.8       # Maximum allowed firing rate
MIN_FIRING_RATE        = 0.01      # Minimum firing rate for homeostasis
HOMEOSTATIC_STRENGTH   = 0.001     # Homeostatic regulation strength
WEIGHT_DECAY           = 0.0001    # Small weight decay for stability

# Voltage bounds for all neurons
V_MIN              = -80.0     # mV
V_MAX              = 40.0      # mV

# Learning bounds (global)
LEARNING_RATE_MAX  = 0.05      # Maximum learning rate

# ─────────────────────────────────────────────────────────────────────────────
# Delays (geometry-based; ranges for sampling if positions missing)
INTRA_DELAY_MS_RANGE    = (1.0, 3.0)
INTER_DELAY_MS_RANGE    = (3.0, 10.0)
AXON_VELOCITY_MM_PER_MS = 0.5   # if deriving d from distance; adjust later

# ─────────────────────────────────────────────────────────────────────────────
# EEG-lite
EEG_SAMPLING_RATE_HZ = 512.0
EEG_USE_CAR          = True
EEG_BANDPASS_HZ      = (0.5, 45.0)
EEG_MAINS_HZ         = 50.0      # set 60.0 if needed

# ─────────────────────────────────────────────────────────────────────────────
# Validation targets (for headless tests/scripts)
VALIDATION = dict(
    isi_cv_range=(0.5, 1.2),
    fano_min=1.0,
    v_bounds_mV=(V_MIN, V_MAX),
    eeg_band_hz=EEG_BANDPASS_HZ,
    channel_corr_max=0.98
)

# ─────────────────────────────────────────────────────────────────────────────
# Small-brain region neuron counts (from the plan; used later)
REGION_COUNTS = {
    'PFC':      {'E': 24, 'I': 6},
    'HPC_CA3':  {'E': 16, 'I': 4},
    'HPC_CA1':  {'E': 16, 'I': 4},
    'AMY_LA':   {'E': 12, 'I': 3},
    'AMY_CeA':  {'E': 0,  'I': 8},
    'BG':       {'D1': 12, 'D2': 12, 'I': 6, 'GPiSNr': 2},  # minimal; STN optional later
    'THAL':     {'TC': 6,  'RTN': 4},
    'S1':       {'E': 12, 'I': 3},
    'M1':       {'E': 12, 'I': 3},
    'INS':      {'E': 8,  'I': 2},
    'PAR':      {'E': 10, 'I': 3},
    'CB':       {'Granule': 40, 'Purkinje': 1, 'DN': 1},
}

if __name__ == "__main__":
    print("CORTEX config loaded:", CONFIG_VERSION)
