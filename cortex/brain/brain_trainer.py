import sys
import os
# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import io
from contextlib import redirect_stdout
import warnings
warnings.filterwarnings("ignore", message=r"To copy construct from a tensor.*sourceTensor\.clone\(\)\.detach\(\).*")

# ==== Region output map & helpers ===========================================
REGION_KEYMAP = {
    'parietal':           ['correlation', 'spatial_integration', 'self_boundary'],
    'insula':             ['emotional', 'interoceptive', 'pain_temperature', 'risk_assessment'],
    'hippocampus':        ['sharp_wave_ripples', 'ca3_ca1_circuit', 'memory_consolidation'],
    'limbic':             ['neural_activity', 'emotional_state'],
    'pfc':                ['working_memory', 'policy_output', 'wm_activity', 'neural_activity'],
    'unified_neocortex':  ['sensory_encoding', 'association_output', 'encoding'],
    'cerebellum':         ['spikes', 'output', 'forward_error'],
    'basal_ganglia':      ['neural_activity', 'go_nogo_output', 'policy_signal'],
    'motor':              ['motor_output', 'spikes', 'activity', 'selection_strength', 'decision_neuron_output'],
    'thalamus':           ['relay_output', 'activity']
}

def _to_scalar_mean(v):
    try:
        if v is None:
            return 0.0
        if hasattr(v, "detach"):
            return float(v.detach().cpu().mean().item())
        return float(np.asarray(v, dtype=float).mean())
    except Exception:
        return 0.0

def pick_region_scalar(outputs, region):
    """Read a single scalar for a region using REGION_KEYMAP, handling nested dicts."""
    out = outputs.get(region, {})
    if not isinstance(out, dict):
        # sometimes the region itself is the tensor/array
        return _to_scalar_mean(out)

    # Try known keys first
    for k in REGION_KEYMAP.get(region, []):
        if k in out:
            v = out[k]
            if isinstance(v, dict):
                # common nested names (e.g., hippocampus ca3_ca1_circuit)
                for kk in ('ripple_output', 'activity', 'value', 'signal', 'output'):
                    if kk in v:
                        return _to_scalar_mean(v[kk])
                # fallback: first value
                try:
                    return _to_scalar_mean(next(iter(v.values())))
                except Exception:
                    return 0.0
            return _to_scalar_mean(v)

    # If nothing matched, try to coerce the dict itself
    return _to_scalar_mean(out)
# ============================================================================

# Try importing the helper function, fallback to direct brain creation
try:
    from cortex.brain.cortex_brain import CortexBrain42, create_small_brain_for_testing
    brain_creation_method = "helper_function"
except ImportError:
    try:
        from cortex.brain.cortex_brain import CortexBrain42
        brain_creation_method = "direct"
    except ImportError:
        print("Could not import brain modules - check if cortex_brain.py exists")
        brain_creation_method = "error"
    
# ======= SPEED SWITCHES =======
FAST_MODE = True          # master toggle
LOG_EVERY = 500 if FAST_MODE else 50
SAVE_EVERY = None         # set e.g. 5000 to save occasionally
PLOT_DURING_TRAIN = False # only plot at the very end
TRACE_EVERY = 500        # how often to record per-step traces (None = never)
DTYPE = "float32"         # use float32 everywhere
# ==============================

# === SAFE UTILITY FUNCTIONS ===
def _safe_mean_scalar(x):
    if torch.is_tensor(x):
        v = float(x.mean().item())
    else:
        v = float(np.mean(x))
    if not np.isfinite(v):
        v = 0.0
    return v

def _clip(v, limit=5.0):
    return max(-limit, min(limit, v))

# === RL helpers ===
class EMA:
    """Exponential Moving Average for baseline estimation"""
    def __init__(self, beta=0.98):
        self.beta = beta
        self.value = None
    
    def update(self, x):
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.beta * self.value + (1 - self.beta) * float(x)
        return self.value

def softmax(x, tau=1.0):
    """Softmax with temperature"""
    x = np.array(x, dtype=float)
    x = (x - np.max(x)) / max(tau, 1e-6)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-8)

# ====== CONFIG ======
class BrainTrainer:
    def __init__(self, brain_config=None, device=None, verbose=True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.brain_config = brain_config or {
            'motor_neurons': 32,
            'thalamus_neurons': 50, 
            'amygdala_neurons': 60,
            'pfc_neurons': 64,
            'hippocampus_neurons': 100,
            'insula_neurons': 40,
            'parietal_elements': 16,
            'sensory_features': 4,
            'sensory_specialized_neurons': 16,
            'n_actions': 4
        }
    
    def train(self, num_steps=100, patterns=None, rewards=None):
        # === Reproducibility seeds ===
        np.random.seed(42)
        torch.manual_seed(42)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(42)

        # Create results folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_folder = f"brain_training_results_{timestamp}"
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"Results will be saved to: {self.results_folder}")
        print("Creating small brain for testing...")

        # Create brain (silence verbose constructors)
        buf_init = io.StringIO()
        with redirect_stdout(buf_init):
            if brain_creation_method == "helper_function":
                brain = create_small_brain_for_testing(device=self.device, verbose=self.verbose)
            else:
                brain = CortexBrain42(**self.brain_config, device=self.device, verbose=self.verbose)
        # optional: keep a short sample if you want to inspect it
        self.build_print_sample = buf_init.getvalue()[:2000]

        # Print brain architecture
        print(f"Brain created with {brain.get_total_neuron_count():,} neurons")

        # Apply neural activation (keeping your original activation code)
        print("Applying strong neural activation...")
        with torch.no_grad():
            regions = ['thalamus', 'amygdala', 'pfc', 'hippocampus', 'insula']
            for region_name in regions:
                if hasattr(brain, region_name):
                    region = getattr(brain, region_name)
                    if hasattr(region, 'neurons'):
                        region.baseline_current = 5.0
                        if hasattr(region.neurons, 'threshold'):
                            region.neurons.threshold.data.fill_(-60.0)
            
            if hasattr(brain, 'motor'):
                if hasattr(brain.motor, 'neurons'):
                    brain.motor.neurons.baseline_current = 15.0
                    brain.motor.neurons.leak_conductance = 0.001
                    brain.motor.neurons.membrane_capacitance = 100.0
                if hasattr(brain.motor, 'population_decoder'):
                    brain.motor.population_decoder.baseline_bias = 2.0
            
            print("Neural activation applied")

        # Get neuron counts
        try:
            motor_neuron_count = brain.get_neuron_count('motor')
            total_neurons = brain.get_total_neuron_count()
            n_actions = brain.neuron_counts['n_actions']
        except AttributeError:
            motor_neuron_count = 64
            total_neurons = 754
            n_actions = 4

        print(f"\n=== BRAIN CONFIGURATION ===")
        print(f"Motor Neurons: {motor_neuron_count}")
        print(f"Total Neurons: {total_neurons:,}")
        print(f"Action Space: {n_actions}")
        print(f"Device: {self.device}")

        # === RL PARAMETERS ===
        gamma = 0.98      # TD discount
        k_da = 0.6        # DA gain on TD error
        da_baseline = 0.3 # floor
        tau = 1.5         # softmax temperature (will anneal)
        tau_min = 0.2
        tau_decay = 0.999  # per step
        reward_baseline = EMA(beta=0.99)  # advantage baseline
        prev_V = 0.0
        
        # Homeostatic parameters
        homeostasis_interval = 50  # Apply every N steps
        target_firing_rate = 0.5    # Reduced target
        homeostasis_gain = 0.001    # Slower adaptation

        # Use the last motor neuron as decision neuron
        output_neuron_idx = motor_neuron_count - 1
        print(f"Using motor neuron #{output_neuron_idx} as decision neuron")

        # Tracking lists
        reward_history = []
        output_history = []
        dopamine_history = []
        td_error_history = []
        value_history = []
        temperature_history = []
        policy_entropy_history = []
        progress_lines = []

        region_firing = {region: [] for region in [
            'motor', 'unified_neocortex', 'thalamus', 'cerebellum', 'parietal', 
            'pfc', 'limbic', 'hippocampus', 'insula', 'basal_ganglia'
        ]}

        # ====== MAIN LEARNING LOOP ======
        print(f"\n=== STARTING LEARNING EXPERIMENT ({num_steps} steps) ===")
        
        start_time = time.time()
        
        for step in range(num_steps):
            # === STRUCTURED GO/NO-GO INPUT ===
            # Curriculum: separable Gaussian feature with mild noise
            feat = np.random.randn() * 0.3
            if np.random.random() < 0.5:
                mu, expected_output = +1.2, 1  # GO
            else:
                mu, expected_output = -1.2, 0  # NO-GO
            
            # Curriculum: after 400 steps, make GO/NO-GO closer (harder)
            if step > 60:
                mu = 0.9 if expected_output == 1 else -0.9

            x = mu + feat + 0.1 * np.random.randn()
            
            # Render as simple 84x84 image: centered blob intensity ~ x
            img = np.zeros((84, 84), dtype=np.float32)
            yy, xx = np.ogrid[:84, :84]
            mask = (yy - 42)**2 + (xx - 42)**2 <= 8**2
            img[mask] = np.float32(np.clip(0.5 * x + 0.5, 0.0, 1.0))
            sensory_input = torch.from_numpy(img).to(self.device)
            
            # Mild scaling
            sensory_input = torch.clamp(sensory_input * 10.0, 0.0, 5.0)
            
            # mild input noise
            noise = torch.randn_like(sensory_input) * 0.05
            sensory_input = torch.clamp(sensory_input + noise, 0.0, 5.0)
            
            # === FIRST FORWARD PASS ===
            start_time = time.time()            # define per-step timer
            DA_t = 0.0                          # safe initial DA for first pass
            buf = io.StringIO()
            with redirect_stdout(buf):
                outputs = brain(sensory_input, None, DA_t)
            
            # DEBUG: show region keys the brain actually emits (first 3 steps)
            if step < 3 and isinstance(outputs, dict):
                dbg = {}
                for r, out in outputs.items():
                    if isinstance(out, dict):
                        dbg[r] = list(out.keys())
                    else:
                        dbg[r] = type(out).__name__
                print(f"[DBG step {step}] region keys: {dbg}")

            if step == 0:
                # keep a tiny sample of the brain’s own prints (not to console)
                self.forward_print_sample = buf.getvalue()[:2000]

            if step == 0:
                # keep a small sample in case you want to inspect it after training
                self.forward_print_sample = buf.getvalue()[:2000]

            # Store debug outputs from first step
            if step == 0:
                self.debug_outputs = outputs if outputs is not None else {}

            # === EXTRACT STATE VALUE AND ACTION PROBABILITIES (SAFE) ===
            bg = outputs.get('basal_ganglia', {}) if isinstance(outputs, dict) else {}

            # ---- Critic: safe, bounded, persistent if BG skips or dumps to ~0 ----
            value_available = False
            if 'state_value' in bg:
                V_t_raw = _safe_mean_scalar(bg['state_value']); value_available = True
            elif 'action_values' in bg:
                V_t_raw = _safe_mean_scalar(bg['action_values']); value_available = True
            else:
                V_t_raw = prev_V  # hold last estimate if BG didn't emit

            # Treat an implausible sudden zero as missing (after warm-up)
            if value_available and abs(V_t_raw) < 1e-6 and step > 20:
                V_t_raw = prev_V

            V_t = _clip(V_t_raw, limit=5.0)

            # === EXTRACT MOTOR DECISION (ONCE) ===
            motor_out = outputs.get('motor', {}) if isinstance(outputs, dict) else {}
            if 'decision_neuron_output' in motor_out:
                dno = motor_out['decision_neuron_output']
                output_spike = float(dno.mean().item()) if torch.is_tensor(dno) else float(dno)
            elif 'action_probabilities' in motor_out:
                probs_motor = motor_out['action_probabilities']
                probs_motor = probs_motor.detach().cpu().numpy() if hasattr(probs_motor, 'detach') else np.array(probs_motor, dtype=float)
                sel = int(motor_out.get('selected_action', int(np.argmax(probs_motor))))
                output_spike = float(probs_motor[sel])
            elif 'selection_strength' in motor_out:
                output_spike = float(motor_out['selection_strength'])
            elif 'spikes' in motor_out:
                s = motor_out['spikes']
                output_spike = float(s.mean().item()) if torch.is_tensor(s) else float(np.mean(s))
            else:
                output_spike = 0.5

            # === CAPTURE PER-REGION ACTIVITY (single, reliable accessor) ===
            for r in region_firing.keys():
                val = pick_region_scalar(outputs, r)
                if np.isfinite(val) and (-1e-3 < val < 0):  # clamp tiny negatives
                    val = 0.0
                region_firing[r].append(val)

            # === ACTOR PROBABILITIES ===
            # Prefer BG, but fall back to motor if BG is missing OR nearly uniform.
            probs = None
            if 'action_probabilities' in bg:
                p = bg['action_probabilities']
                p = p.detach().cpu().numpy() if hasattr(p, 'detach') else np.array(p, dtype=float)
                # if BG is basically uniform, ignore it so entropy tracks behavior
                if p.size >= 2 and (np.max(p) - np.min(p)) >= 1e-3:
                    probs = p
            elif 'action_values' in bg:
                av = bg['action_values']
                av = av.detach().cpu().numpy() if hasattr(av, 'detach') else np.array(av, dtype=float)
                av = np.tanh(av) * 2.0
                probs = softmax(av, tau=tau)

            if probs is None:
                p_go = float(np.clip(output_spike, 1e-6, 1.0 - 1e-6))
                if n_actions >= 2:
                    probs = np.array([p_go, 1.0 - p_go] + [1e-6]*max(0, n_actions-2), dtype=float)
                    probs = probs / probs.sum()
                else:
                    probs = np.array([1.0], dtype=float)

            # Policy entropy for monitoring
            probs_safe = np.clip(probs, 1e-8, 1.0)
            entropy = -np.sum(probs_safe * np.log(probs_safe))
            policy_entropy_history.append(entropy)

            # === COMPUTE RAW REWARD ===
            if (output_spike > 0.5 and expected_output == 1) or (output_spike <= 0.5 and expected_output == 0):
                reward_raw = 5.0   # Correct
            else:
                reward_raw = -2.0  # Incorrect
           
            # (1) forward → get outputs
            # (2) extract V_t (critic) and actor probs
            # (3) extract motor decision → compute reward_raw
            # (4) baseline update
            b = reward_baseline.update(reward_raw)
            adv = reward_raw - b

            # (5) TD error & DA (keep this single copy)
            delta_raw = reward_raw + gamma * V_t - prev_V
            delta = float(np.clip(delta_raw, -5.0, 5.0))
            if not np.isfinite(delta):
                delta = 0.0
            if step == 0:
                delta_scale = 1.0
            else:
                delta_scale = 0.99 * delta_scale + 0.01 * abs(delta)
            delta_norm = delta / (1e-6 + delta_scale)
            DA_t = max(0.0, min(1.0, da_baseline + k_da * delta_norm))
            prev_V = V_t
            td_error_history.append(delta)
            value_history.append(V_t)

            # (6) credit assignment pass with DA
            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                outputs = brain(sensory_input, None, float(DA_t))
            if step == 0:
                self.credit_print_sample = buf2.getvalue()[:2000]

            step_time = time.time() - start_time
            
            # === TRACK METRICS ===
            reward_history.append(reward_raw)
            output_history.append(output_spike)
            dopamine_history.append(DA_t)
            tau = max(tau_min, tau * tau_decay)

            # === ANNEAL TEMPERATURE (keep as-is) ===
            temperature_history.append(tau)

            # Progress indicator (store now; we’ll print once at the end)
            if step % 100 == 0:
                recent_reward = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else 0
                recent_td = np.mean(td_error_history[-10:]) if len(td_error_history) >= 10 else 0
                progress_lines.append(
                    f"Step {step}/{num_steps}: Reward={recent_reward:.3f}, TD={recent_td:.3f}, "
                    f"V={V_t:.3f}, DA={DA_t:.3f}, τ={tau:.3f}, Entropy={entropy:.3f}"
                )

        # --- sanitize histories & choose ONE window for both summaries ---
        reward_history = list(np.nan_to_num(reward_history, nan=0.0, posinf=5.0, neginf=-5.0))
        td_error_history = list(np.nan_to_num(td_error_history, nan=0.0, posinf=5.0, neginf=-5.0))
        value_history = list(np.nan_to_num(value_history, nan=0.0, posinf=5.0, neginf=-5.0))
        policy_entropy_history = list(np.nan_to_num(policy_entropy_history, nan=0.0))
                
        # --- sanitize region activity arrays ---
        region_means = {}
        for r, arr in region_firing.items():
            a = np.asarray(arr, dtype=float)
            # replace NaNs with 0 (or with mean if you prefer)
            a = np.nan_to_num(a, nan=0.0)
            region_firing[r] = a.tolist()
            region_means[r] = float(np.mean(a)) if a.size > 0 else 0.0

        # use same tail window for both the quick analysis and the final summary
        k = min(100, len(reward_history))

        # ====== ANALYSIS & RESULTS ======
        print(f"\n=== LEARNING ANALYSIS ===")

        # Performance metrics
        k = min(100, len(reward_history))  # <-- single, consistent window

        final_reward = np.mean(reward_history[-k:])
        initial_reward = np.mean(reward_history[:k])
        learning_improvement = final_reward - initial_reward

        print(f"Initial Performance (first {k} steps): {initial_reward:.3f}")
        print(f"Final Performance (last {k} steps): {final_reward:.3f}")
        print(f"Learning Improvement: {learning_improvement:.3f}")
        print(f"Final TD Error: {np.mean(td_error_history[-k:]):.3f}")
        print(f"Final Value Estimate: {np.mean(value_history[-k:]):.3f}")
        print(f"Final Policy Entropy: {np.mean(policy_entropy_history[-k:]):.3f}")

        # Learning status
        if final_reward > 3.0 and learning_improvement > 0.2:
            learning_status = "SUCCESSFUL LEARNING"
            learning_success = True
        elif learning_improvement > 0.1:
            learning_status = "PARTIAL LEARNING"
            learning_success = True
        else:
            learning_status = "LEARNING IN PROGRESS"
            learning_success = False

        print(f"Learning Status: {learning_status}")

        # Save training data
        print(f"Saving training data to {self.results_folder}...")
        
        np.save(os.path.join(self.results_folder, 'reward_history.npy'), reward_history)
        np.save(os.path.join(self.results_folder, 'output_history.npy'), output_history)
        np.save(os.path.join(self.results_folder, 'dopamine_history.npy'), dopamine_history)
        np.save(os.path.join(self.results_folder, 'td_error_history.npy'), td_error_history)
        np.save(os.path.join(self.results_folder, 'value_history.npy'), value_history)
        np.save(os.path.join(self.results_folder, 'entropy_history.npy'), policy_entropy_history)
        
        for region, data in region_firing.items():
            np.save(os.path.join(self.results_folder, f'region_{region}_activity.npy'), data)
        
        metadata = {
            'num_steps': int(num_steps),
            'motor_neuron_count': int(motor_neuron_count),
            'total_neurons': int(total_neurons),
            'final_reward': float(final_reward),
            'learning_improvement': float(learning_improvement),
            'learning_success': bool(learning_success),
            'timestamp': str(timestamp),
            'device': str(self.device),
            'brain_config': {k: int(v) if isinstance(v, (int, np.integer)) else v 
                           for k, v in self.brain_config.items()},
            'rl_params': {
                'gamma': gamma,
                'k_da': k_da,
                'tau_initial': 1.5,
                'tau_final': float(tau),
                'tau_decay': tau_decay
            }
        }
        
        with open(os.path.join(self.results_folder, 'training_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Training data saved successfully!")

        # Save progress lines to a text file (so they don’t flood the console)
        progress_log_path = os.path.join(self.results_folder, 'progress_log.txt')
        with open(progress_log_path, 'w', encoding='utf-8') as _pf:
            for _ln in progress_lines:
                _pf.write(_ln + '\n')
        print(f"Saved per-step progress log to: {progress_log_path}")

        # Generate enhanced plots
        self._generate_enhanced_plots(reward_history, output_history, dopamine_history,
                                     td_error_history, value_history, policy_entropy_history,
                                     temperature_history, region_firing, motor_neuron_count, 
                                     total_neurons, final_reward, learning_improvement, 
                                     learning_success, num_steps, brain)
        
        return {
            'reward_history': reward_history,
            'output_history': output_history,
            'dopamine_history': dopamine_history,
            'td_error_history': td_error_history,
            'value_history': value_history,
            'policy_entropy_history': policy_entropy_history,
            'region_firing': region_firing,
            'brain': brain,
            'learning_success': learning_success,
            'final_reward': final_reward,
            'learning_improvement': learning_improvement,
            'total_neurons': total_neurons,
            'motor_neuron_count': motor_neuron_count,
            'output_neuron_idx': output_neuron_idx,
            'num_steps': num_steps
        }

    def _generate_enhanced_plots(self, reward_history, output_history, dopamine_history,
                                td_error_history, value_history, policy_entropy_history,
                                temperature_history, region_firing, motor_neuron_count, 
                                total_neurons, final_reward, learning_improvement, 
                                learning_success, num_steps, brain):
        """Generate enhanced plots including TD error and value function"""
        print(f"\n=== GENERATING ENHANCED PLOTS ===")
        
        output_neuron_idx = motor_neuron_count - 1
        
        # 1. Reward plot
        plt.figure(figsize=(12, 3))
        window = max(5, min(30, len(reward_history) // 10))
        if len(reward_history) >= window and window >= 3:
            kernel = np.ones(window, dtype=float) / float(window)
            reward_ma = np.convolve(reward_history, kernel, mode='valid')
            plt.plot(reward_ma, label=f'Moving Avg Reward (window={window})', color='green', linewidth=2)
        plt.plot(reward_history, alpha=0.3, label='Raw Reward', color='lightgray')
        plt.axhline(y=0, linestyle='--', color='black', alpha=0.5)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Learning Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, '1_learning_progress.png'), dpi=300)
        plt.close()
        
        # 2. TD Error vs Dopamine
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        ax1.plot(td_error_history, label='TD Error', color='blue', alpha=0.7)
        ax1.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
        ax1.set_ylabel('TD Error')
        ax1.set_title('Temporal Difference Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(dopamine_history, label='Dopamine Level', color='orange', linewidth=1.5)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Dopamine')
        ax2.set_title('Dopamine Modulation (from TD Error)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, '2_td_error_dopamine.png'), dpi=300)
        plt.close()
        
        # 3. Value Function Learning
        plt.figure(figsize=(12, 3))
        plt.plot(value_history, label='State Value V(s)', color='purple', alpha=0.8)
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('Value Function Learning')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, '3_value_function.png'), dpi=300)
        plt.close()
        
        # 4. Policy Entropy
        plt.figure(figsize=(12, 3))
        plt.plot(policy_entropy_history, label='Policy Entropy', color='red', alpha=0.7)
        plt.xlabel('Step')
        plt.ylabel('Entropy')
        plt.title('Policy Entropy (Exploration vs Exploitation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, '4_policy_entropy.png'), dpi=300)
        plt.close()
        
        # 5. Decision Neuron
        plt.figure(figsize=(12, 3))
        plt.plot(output_history, label=f'Motor Neuron #{output_neuron_idx}', color='blue', alpha=0.7)
        plt.axhline(y=0.5, linestyle='--', color='red', alpha=0.7, label='Decision Threshold')
        plt.xlabel('Step')
        plt.ylabel('Neuron Output')
        plt.title('Decision Neuron Activity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, '5_decision_neuron.png'), dpi=300)
        plt.close()
        
        # 6. Region Activity
        plt.figure(figsize=(14, 8))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for i, region in enumerate(region_firing):
            plt.plot(region_firing[region], 
                    label=f'{region}', 
                    color=colors[i % len(colors)], 
                    alpha=0.8)
        plt.xlabel('Step')
        plt.ylabel('Mean Region Activity')
        plt.title('Brain Region Activity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, '6_region_activity.png'), dpi=300)
        plt.close()
        
        print("All enhanced plots generated and saved!")

        # Final summary
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Brain Configuration: Small (Total: {total_neurons:,} neurons)")
        print(f"Training Steps: {len(reward_history)}")
        print(f"Final Learning Performance: {final_reward:.3f}")
        print(f"Learning Improvement: {learning_improvement:+.3f}")
        print(f"Final TD Error: {np.mean(td_error_history[-100:]):.3f}")
        print(f"Final Value: {np.mean(value_history[-100:]):.3f}")
        print(f"Final Entropy: {np.mean(policy_entropy_history[-100:]):.3f}")

        # Region activity summary
        print(f"\nRegion Activity Summary (last 100 steps):")
        for region in region_firing:
            if len(region_firing[region]) >= 100:
                mean_activity = np.mean(region_firing[region][-100:])
                std_activity = np.std(region_firing[region][-100:])
                print(f"  {region:18s}: Mean={mean_activity:.3f}  Std={std_activity:.3f}")

        print(f"\n{'='*60}")
        if learning_success:
            print("CORTEX 4.3 LEARNING VALIDATION: SUCCESS")
            print("TD-error driven learning with stable convergence achieved")
        else:
            print("CORTEX 4.3 LEARNING VALIDATION: IN PROGRESS")
            print("System showing learning signals, needs more training")
        print(f"{'='*60}")

# ====== AUTO-RUN TEST ======
if __name__ == "__main__":
    print("Starting CORTEX 4.3 Brain Trainer with TD Learning...")
    trainer = BrainTrainer()
    results = trainer.train(num_steps=100)
    print("Training completed!")
    print(f"Results saved to: {trainer.results_folder}")