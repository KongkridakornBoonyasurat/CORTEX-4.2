"""
BME Data Collector for CORTEX 4.2
Comprehensive neural data logging for biomedical engineering thesis validation
"""
import os
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from collections import deque
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import json

class BMEDataCollector:
    """
    Comprehensive data collector for BME thesis validation.
    Captures neural dynamics, synaptic plasticity, network interactions,
    neuromodulation, and computational metrics.
    """
    
    def __init__(self, output_dir: str, sampling_rate_hz: float = 1000.0):
        self.output_dir = output_dir
        self.dt = 1.0 / sampling_rate_hz  # seconds
        os.makedirs(output_dir, exist_ok=True)
        
        # Buffers for continuous data (per episode)
        self.reset_episode_buffers()
        
        # Statistics accumulators
        self.episode_stats = []
        
    def reset_episode_buffers(self):
        """Reset all data buffers for new episode"""
        # Neural dynamics
        self.voltages_buffer = []  # List of dicts {region: voltage_array}
        self.spikes_buffer = []     # List of dicts {region: spike_array}
        self.conductances_buffer = [] # AMPA, NMDA, GABA per region
        
        # Synaptic plasticity
        self.weights_buffer = []    # Weight snapshots
        self.stdp_events = []       # LTP/LTD events
        
        # Network measures
        self.coherence_buffer = []
        self.sync_index_buffer = []
        
        # Neuromodulators
        self.neuromod_buffer = []   # DA, ACh, NE, serotonin
        self.receptor_buffer = []    # Receptor activation states
        
        # Oscillations
        self.oscillation_buffer = []
        self.phase_coupling_buffer = []
        
        # Learning metrics
        self.td_error_buffer = []
        self.value_buffer = []
        self.entropy_buffer = []
        self.gradient_buffer = []
        
        # Astrocyte data
        self.astrocyte_ca_buffer = []
        self.gliotransmitter_buffer = []
        
        # Behavioral
        self.reaction_time_buffer = []
        self.decision_confidence_buffer = []
        
    def collect_step_data(self, brain_output: Dict, step_time: float):
        """
        Collect all available data from one simulation step.
        
        Args:
            brain_output: Full output dictionary from brain.forward()
            step_time: Computation time for this step (seconds)
        """
        
        # === 1. NEURAL DYNAMICS ===
        voltages_dict = {}
        spikes_dict = {}
        conductances_dict = {}
        
        for region_name in ['motor', 'thalamus', 'basal_ganglia', 'cerebellum',
                        'unified_neocortex', 'pfc', 'hippocampus', 'limbic', 
                        'insula', 'parietal']:
            region_out = brain_output.get(region_name, {})
            if isinstance(region_out, dict):
                # Try multiple spike keys
                spike_data = None
                for key in ['spikes', 'spike_train', 'neural_activity', 'activity']:
                    if key in region_out:
                        spike_data = region_out[key]
                        break
                
                # Also check nested neural_dynamics
                if spike_data is None and 'neural_dynamics' in region_out:
                    nd = region_out['neural_dynamics']
                    if isinstance(nd, dict) and 'spikes' in nd:
                        spike_data = nd['spikes']
                
                if spike_data is not None:
                    s = self._to_numpy(spike_data)
                    if s.size > 0:
                        spikes_dict[region_name] = s
                    else:
                        # Fallback: use neural_activity as proxy
                        if 'neural_activity' in region_out:
                            activity = self._to_numpy(region_out['neural_activity'])
                            # Convert scalar to pseudo-spike array
                            if activity.ndim == 0:
                                spikes_dict[region_name] = np.array([activity])
                            else:
                                spikes_dict[region_name] = activity
                # Conductances
                if 'synaptic_currents' in region_out:
                    curr = region_out['synaptic_currents']
                    if isinstance(curr, dict):
                        conductances_dict[region_name] = {
                            'ampa': self._to_numpy(curr.get('ampa', 0)),
                            'nmda': self._to_numpy(curr.get('nmda', 0)),
                            'gaba': self._to_numpy(curr.get('gaba', 0))
                        }
        
        self.voltages_buffer.append(voltages_dict)
        self.spikes_buffer.append(spikes_dict)
        self.conductances_buffer.append(conductances_dict)
        
        # === 2. SYNAPTIC PLASTICITY ===
        if 'synaptic_weights' in brain_output:
            weights = self._extract_weight_snapshot(brain_output['synaptic_weights'])
            self.weights_buffer.append(weights)
        
        # === 3. NETWORK MEASURES ===
        if len(self.spikes_buffer) > 10:  # Need history
            coherence = self._compute_pairwise_coherence(spikes_dict)
            self.coherence_buffer.append(coherence)
            
            sync_idx = self._compute_synchronization_index(spikes_dict)
            self.sync_index_buffer.append(sync_idx)
        
        # === 4. NEUROMODULATORS ===
        neuromod_state = {}
        if 'neuromodulators' in brain_output:
            mods = brain_output['neuromodulators']
            if isinstance(mods, dict):
                neuromod_state = {
                    'dopamine': float(mods.get('dopamine', 0)),
                    'acetylcholine': float(mods.get('acetylcholine', 0)),
                    'norepinephrine': float(mods.get('norepinephrine', 0)),
                    'system_coherence': float(mods.get('system_coherence', 0))
                }
        self.neuromod_buffer.append(neuromod_state)
        
        # Receptor states
        receptor_state = self._extract_receptor_states(brain_output)
        self.receptor_buffer.append(receptor_state)
        
        # === 5. OSCILLATIONS - EMERGENT FROM SPIKES ===
        # Don't use brain_output['oscillations'] (that's synthetic)
        # Extract from actual spike trains every 10 steps
        if len(self.spikes_buffer) >= 20 and len(self.spikes_buffer) % 10 == 0:
            osc_state = self._extract_emergent_oscillations(
                self.spikes_buffer[-min(50, len(self.spikes_buffer)):],  # Use what we have
                dt_ms=1.0
            )
        else:
            # Placeholder until enough data
            osc_state = {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0}

        self.oscillation_buffer.append(osc_state)
        
        # Phase coupling (theta-gamma)
        if len(self.oscillation_buffer) > 20:
            coupling = self._compute_phase_amplitude_coupling()
            self.phase_coupling_buffer.append(coupling)
        
        # === 6. LEARNING METRICS ===
        if 'basal_ganglia' in brain_output:
            bg = brain_output['basal_ganglia']
            if isinstance(bg, dict):
                self.td_error_buffer.append(float(bg.get('td_error', 0)))
                self.value_buffer.append(float(bg.get('state_value', 0)))
                self.entropy_buffer.append(float(bg.get('policy_entropy', 0)))
        
        # === 7. ASTROCYTE DATA ===
        astro_data = self._extract_astrocyte_data(brain_output)
        if astro_data:
            self.astrocyte_ca_buffer.append(astro_data['calcium'])
            self.gliotransmitter_buffer.append(astro_data['gliotransmitter'])
        
        # === 8. BEHAVIORAL METRICS ===
        self.reaction_time_buffer.append(step_time)
        
        if 'motor' in brain_output:
            motor = brain_output['motor']
            if isinstance(motor, dict) and 'selection_strength' in motor:
                confidence = float(motor['selection_strength'])
                self.decision_confidence_buffer.append(confidence)
    
    def save_episode_data(self, ep_idx: int, ep_dir: str):
        """Save all collected data for this episode"""
        os.makedirs(ep_dir, exist_ok=True)
        
        # === NEURAL DYNAMICS ===
        self._save_neural_dynamics(ep_idx, ep_dir)
        
        # === SYNAPTIC PLASTICITY ===
        self._save_plasticity_data(ep_idx, ep_dir)
        
        # === NETWORK ANALYSIS ===
        self._save_network_analysis(ep_idx, ep_dir)
        
        # === NEUROMODULATION ===
        self._save_neuromodulator_data(ep_idx, ep_dir)
        
        # === OSCILLATIONS ===
        self._save_oscillation_analysis(ep_idx, ep_dir)
        
        # === LEARNING ===
        self._save_learning_metrics(ep_idx, ep_dir)
        
        # === ASTROCYTES ===
        self._save_astrocyte_data(ep_idx, ep_dir)
        
        # === BEHAVIORAL ===
        self._save_behavioral_metrics(ep_idx, ep_dir)
        
        # === SUMMARY STATS ===
        self._save_episode_summary(ep_idx, ep_dir)
        
        print(f"[BME] Comprehensive data saved for episode {ep_idx}")
    
    def _save_neural_dynamics(self, ep_idx: int, ep_dir: str):
        """Save voltage traces, spike rasters, conductances"""
        
        # Spike raster plot
        if self.spikes_buffer:
            try:
                fig, axes = plt.subplots(len(self.spikes_buffer[0]), 1, 
                                        figsize=(12, 8), sharex=True)
                if not isinstance(axes, np.ndarray):
                    axes = [axes]
                
                for idx, (region_name, ax) in enumerate(zip(self.spikes_buffer[0].keys(), axes)):
                    spike_train = []
                    for t, spikes_dict in enumerate(self.spikes_buffer):
                        s = spikes_dict.get(region_name, np.array([]))
                        if s.size > 0:
                            spike_train.append(s)
                    
                    if spike_train:
                        spike_array = np.array(spike_train)
                        if spike_array.ndim == 2 and spike_array.shape[1] > 0:
                            # Raster plot
                            for neuron_idx in range(min(50, spike_array.shape[1])):  # Max 50 neurons
                                spike_times = np.where(spike_array[:, neuron_idx] > 0)[0]
                                ax.scatter(spike_times, [neuron_idx] * len(spike_times), 
                                         s=1, c='k', alpha=0.5)
                    
                    ax.set_ylabel(f'{region_name}\nNeuron #')
                    ax.set_ylim(-1, 50)
                
                axes[-1].set_xlabel('Time step')
                fig.suptitle(f'Spike Rasters ep{ep_idx:03d}')
                fig.tight_layout()
                fig.savefig(os.path.join(ep_dir, f'spike_rasters_ep{ep_idx:03d}.png'), dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[BME] Spike raster plot failed: {e}")
        
        # Save raw data
        np.savez_compressed(
            os.path.join(ep_dir, f'neural_dynamics_ep{ep_idx:03d}.npz'),
            voltages=self.voltages_buffer,
            spikes=self.spikes_buffer,
            conductances=self.conductances_buffer
        )
    
    def _save_plasticity_data(self, ep_idx: int, ep_dir: str):
        """Save synaptic weight evolution"""
        if not self.weights_buffer:
            return
        
        # Weight statistics over time
        try:
            means = [w['mean'] for w in self.weights_buffer if 'mean' in w]
            stds = [w['std'] for w in self.weights_buffer if 'std' in w]
            
            if means:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                
                ax1.plot(means, linewidth=1, color='blue', alpha=0.8)
                ax1.set_ylabel('Mean Weight')
                ax1.set_title(f'Synaptic Weight Evolution ep{ep_idx:03d}')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(stds, linewidth=1, color='red', alpha=0.8)
                ax2.set_ylabel('Weight Std Dev')
                ax2.set_xlabel('Sample')
                ax2.grid(True, alpha=0.3)
                
                fig.tight_layout()
                fig.savefig(os.path.join(ep_dir, f'weight_evolution_ep{ep_idx:03d}.png'), dpi=150)
                plt.close(fig)
        except Exception as e:
            print(f"[BME] Weight plot failed: {e}")
        
        # Save raw weight data
        np.save(os.path.join(ep_dir, f'weights_ep{ep_idx:03d}.npy'), 
               self.weights_buffer)
    
    def _save_network_analysis(self, ep_idx: int, ep_dir: str):
        """Save coherence matrices, synchronization indices"""
        if not self.coherence_buffer:
            return
        
        # Average coherence matrix
        try:
            avg_coherence = np.mean(self.coherence_buffer, axis=0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(avg_coherence, cmap='RdYlBu_r', vmin=-1, vmax=1)
            ax.set_title(f'Average Cross-Regional Coherence ep{ep_idx:03d}')
            ax.set_xlabel('Region')
            ax.set_ylabel('Region')
            plt.colorbar(im, ax=ax, label='Coherence')
            fig.tight_layout()
            fig.savefig(os.path.join(ep_dir, f'coherence_matrix_ep{ep_idx:03d}.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[BME] Coherence plot failed: {e}")
        
        # Synchronization index over time
        if self.sync_index_buffer:
            try:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(self.sync_index_buffer, linewidth=1, color='purple', alpha=0.8)
                ax.set_ylabel('Sync Index')
                ax.set_xlabel('Time step')
                ax.set_title(f'Network Synchronization ep{ep_idx:03d}')
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(ep_dir, f'synchronization_ep{ep_idx:03d}.png'), dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[BME] Sync plot failed: {e}")
        
        np.savez_compressed(
            os.path.join(ep_dir, f'network_analysis_ep{ep_idx:03d}.npz'),
            coherence=self.coherence_buffer,
            synchronization=self.sync_index_buffer
        )
    
    def _save_neuromodulator_data(self, ep_idx: int, ep_dir: str):
        """Save neuromodulator traces and receptor states"""
        if not self.neuromod_buffer:
            return
        
        # Neuromodulator traces
        try:
            da = [m.get('dopamine', 0) for m in self.neuromod_buffer]
            ach = [m.get('acetylcholine', 0) for m in self.neuromod_buffer]
            ne = [m.get('norepinephrine', 0) for m in self.neuromod_buffer]
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            
            axes[0].plot(da, linewidth=1, color='orange', alpha=0.8)
            axes[0].set_ylabel('Dopamine')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(ach, linewidth=1, color='green', alpha=0.8)
            axes[1].set_ylabel('Acetylcholine')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(ne, linewidth=1, color='purple', alpha=0.8)
            axes[2].set_ylabel('Norepinephrine')
            axes[2].set_xlabel('Time step')
            axes[2].grid(True, alpha=0.3)
            
            fig.suptitle(f'Neuromodulator Dynamics ep{ep_idx:03d}')
            fig.tight_layout()
            fig.savefig(os.path.join(ep_dir, f'neuromodulators_ep{ep_idx:03d}.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[BME] Neuromodulator plot failed: {e}")
        
        np.savez_compressed(
            os.path.join(ep_dir, f'neuromodulation_ep{ep_idx:03d}.npz'),
            neuromodulators=self.neuromod_buffer,
            receptors=self.receptor_buffer
        )
    
    def _save_oscillation_analysis(self, ep_idx: int, ep_dir: str):
        """Save oscillation power spectra and phase coupling"""
        if not self.oscillation_buffer:
            return
        
        # Oscillation power over time
        try:
            theta = [o.get('theta', 0) for o in self.oscillation_buffer]
            alpha = [o.get('alpha', 0) for o in self.oscillation_buffer]
            beta = [o.get('beta', 0) for o in self.oscillation_buffer]
            gamma = [o.get('gamma', 0) for o in self.oscillation_buffer]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(theta, label='Theta (4-8 Hz)', linewidth=1, alpha=0.8)
            ax.plot(alpha, label='Alpha (8-13 Hz)', linewidth=1, alpha=0.8)
            ax.plot(beta, label='Beta (13-30 Hz)', linewidth=1, alpha=0.8)
            ax.plot(gamma, label='Gamma (30-100 Hz)', linewidth=1, alpha=0.8)
            ax.set_ylabel('Power')
            ax.set_xlabel('Time step')
            ax.set_title(f'Oscillatory Power ep{ep_idx:03d}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(ep_dir, f'oscillations_ep{ep_idx:03d}.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[BME] Oscillation plot failed: {e}")
        
        np.savez_compressed(
            os.path.join(ep_dir, f'oscillations_ep{ep_idx:03d}.npz'),
            oscillations=self.oscillation_buffer,
            phase_coupling=self.phase_coupling_buffer
        )
    
    def _save_learning_metrics(self, ep_idx: int, ep_dir: str):
        """Save TD-error, value estimates, policy entropy"""
        if not self.td_error_buffer:
            return
        
        try:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            
            axes[0].plot(self.td_error_buffer, linewidth=1, color='red', alpha=0.8)
            axes[0].set_ylabel('TD Error')
            axes[0].axhline(0, linestyle='--', color='gray', alpha=0.5)
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(self.value_buffer, linewidth=1, color='blue', alpha=0.8)
            axes[1].set_ylabel('Value Estimate')
            axes[1].grid(True, alpha=0.3)
            
            axes[2].plot(self.entropy_buffer, linewidth=1, color='purple', alpha=0.8)
            axes[2].set_ylabel('Policy Entropy')
            axes[2].set_xlabel('Time step')
            axes[2].grid(True, alpha=0.3)
            
            fig.suptitle(f'Learning Dynamics ep{ep_idx:03d}')
            fig.tight_layout()
            fig.savefig(os.path.join(ep_dir, f'learning_metrics_ep{ep_idx:03d}.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[BME] Learning plot failed: {e}")
        
        np.savez_compressed(
            os.path.join(ep_dir, f'learning_ep{ep_idx:03d}.npz'),
            td_error=self.td_error_buffer,
            value=self.value_buffer,
            entropy=self.entropy_buffer,
            gradients=self.gradient_buffer
        )
    
    def _save_astrocyte_data(self, ep_idx: int, ep_dir: str):
        """Save astrocyte calcium and gliotransmitter data"""
        if not self.astrocyte_ca_buffer:
            return
        
        np.savez_compressed(
            os.path.join(ep_dir, f'astrocyte_ep{ep_idx:03d}.npz'),
            calcium=self.astrocyte_ca_buffer,
            gliotransmitter=self.gliotransmitter_buffer
        )
    
    def _save_behavioral_metrics(self, ep_idx: int, ep_dir: str):
        """Save reaction times and decision confidence"""
        if not self.reaction_time_buffer:
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            
            ax1.plot(self.reaction_time_buffer, linewidth=1, color='blue', alpha=0.8)
            ax1.set_ylabel('Reaction Time (s)')
            ax1.grid(True, alpha=0.3)
            
            if self.decision_confidence_buffer:
                ax2.plot(self.decision_confidence_buffer, linewidth=1, color='green', alpha=0.8)
            ax2.set_ylabel('Decision Confidence')
            ax2.set_xlabel('Time step')
            ax2.grid(True, alpha=0.3)
            
            fig.suptitle(f'Behavioral Metrics ep{ep_idx:03d}')
            fig.tight_layout()
            fig.savefig(os.path.join(ep_dir, f'behavioral_ep{ep_idx:03d}.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[BME] Behavioral plot failed: {e}")
        
        np.savez_compressed(
            os.path.join(ep_dir, f'behavioral_ep{ep_idx:03d}.npz'),
            reaction_time=self.reaction_time_buffer,
            decision_confidence=self.decision_confidence_buffer
        )
    
    def _save_episode_summary(self, ep_idx: int, ep_dir: str):
        """Save statistical summary of episode"""
        
        summary = {
            'episode': ep_idx,
            'neural_dynamics': {
                'mean_firing_rate': self._compute_mean_firing_rate(),
                'spike_count': self._compute_total_spikes(),
                'voltage_stats': self._compute_voltage_statistics()
            },
            'network': {
                'mean_coherence': float(np.mean(self.coherence_buffer)) if self.coherence_buffer else 0,
                'mean_synchronization': float(np.mean(self.sync_index_buffer)) if self.sync_index_buffer else 0
            },
            'learning': {
                'mean_td_error': float(np.mean(self.td_error_buffer)) if self.td_error_buffer else 0,
                'final_value': float(self.value_buffer[-1]) if self.value_buffer else 0,
                'mean_entropy': float(np.mean(self.entropy_buffer)) if self.entropy_buffer else 0
            },
            'neuromodulation': {
                'mean_dopamine': float(np.mean([m.get('dopamine', 0) for m in self.neuromod_buffer])) if self.neuromod_buffer else 0,
                'mean_acetylcholine': float(np.mean([m.get('acetylcholine', 0) for m in self.neuromod_buffer])) if self.neuromod_buffer else 0,
                'mean_norepinephrine': float(np.mean([m.get('norepinephrine', 0) for m in self.neuromod_buffer])) if self.neuromod_buffer else 0
            },
            'behavioral': {
                'mean_reaction_time': float(np.mean(self.reaction_time_buffer)) if self.reaction_time_buffer else 0,
                'mean_confidence': float(np.mean(self.decision_confidence_buffer)) if self.decision_confidence_buffer else 0
            }
        }
        
        with open(os.path.join(ep_dir, f'summary_ep{ep_idx:03d}.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.episode_stats.append(summary)
    
    # === HELPER METHODS ===
    
    def _to_numpy(self, x):
        """Convert to numpy array safely"""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)
    
    def _extract_weight_snapshot(self, weights_dict):
        """Extract weight statistics"""
        all_weights = []
        for k, v in weights_dict.items():
            w = self._to_numpy(v)
            if w.size > 0:
                all_weights.append(w.flatten())
        
        if all_weights:
            all_w = np.concatenate(all_weights)
            return {
                'mean': float(np.mean(all_w)),
                'std': float(np.std(all_w)),
                'min': float(np.min(all_w)),
                'max': float(np.max(all_w)),
                'sparsity': float(np.mean(np.abs(all_w) < 1e-6))
            }
        return {}
    
    def _compute_pairwise_coherence(self, spikes_dict):
        """Compute cross-correlation between regions"""
        regions = list(spikes_dict.keys())
        n = len(regions)
        coherence_matrix = np.zeros((n, n))
        
        for i, r1 in enumerate(regions):
            for j, r2 in enumerate(regions):
                s1 = spikes_dict.get(r1, np.array([]))
                s2 = spikes_dict.get(r2, np.array([]))
                
                if s1.size > 0 and s2.size > 0:
                    # Mean firing rate correlation
                    f1 = np.mean(s1) if s1.size > 0 else 0
                    f2 = np.mean(s2) if s2.size > 0 else 0
                    coherence_matrix[i, j] = f1 * f2  # Simplified
        
        return coherence_matrix
    
    def _compute_synchronization_index(self, spikes_dict):
        """Compute network synchronization index"""
        all_rates = []
        for region, spikes in spikes_dict.items():
            if spikes.size > 0:
                all_rates.append(np.mean(spikes))
        
        if len(all_rates) > 1:
            return float(np.std(all_rates))  # Low std = high sync
        return 0.0
    
    def _compute_phase_amplitude_coupling(self):
        """Compute theta-gamma phase-amplitude coupling"""
        if len(self.oscillation_buffer) < 20:
            return 0.0
        
        theta = np.array([o.get('theta', 0) for o in self.oscillation_buffer[-20:]])
        gamma = np.array([o.get('gamma', 0) for o in self.oscillation_buffer[-20:]])
        
        if theta.size > 0 and gamma.size > 0:
            corr, _ = pearsonr(theta, gamma)
            return float(corr) if np.isfinite(corr) else 0.0
        return 0.0
    
    def _extract_receptor_states(self, brain_output):
        """Extract receptor activation states"""
        # Placeholder - implement based on your brain's receptor model
        return {}
    
    def _extract_astrocyte_data(self, brain_output):
        """Extract astrocyte calcium and gliotransmitter data"""
        # Placeholder - implement based on your astrocyte model
        return None
    
    def _compute_mean_firing_rate(self):
        """Compute mean firing rate across all regions"""
        all_rates = []
        for spikes_dict in self.spikes_buffer:
            for region, spikes in spikes_dict.items():
                if spikes.size > 0:
                    all_rates.append(np.mean(spikes))
        return float(np.mean(all_rates)) if all_rates else 0.0
    
    def _compute_total_spikes(self):
        """Count total spikes"""
        total = 0
        for spikes_dict in self.spikes_buffer:
            for region, spikes in spikes_dict.items():
                if spikes.size > 0:
                    total += int(np.sum(spikes > 0))
        return total
    
    def _compute_voltage_statistics(self):
        """Compute voltage statistics"""
        all_voltages = []
        for volt_dict in self.voltages_buffer:
            for region, volts in volt_dict.items():
                if volts.size > 0:
                    all_voltages.append(volts.flatten())
        
        if all_voltages:
            all_v = np.concatenate(all_voltages)
            return {
                'mean': float(np.mean(all_v)),
                'std': float(np.std(all_v)),
                'min': float(np.min(all_v)),
                'max': float(np.max(all_v))
            }
        return {}
    
    def _extract_emergent_oscillations(self, spike_buffer: list, dt_ms: float = 1.0):
            """
            Extract oscillatory power from actual spike trains using spectral analysis.
            This is how real neuroscience measures oscillations from recordings.
            """
            from scipy import signal as scipy_signal
            
            # Aggregate population firing rate across all regions
            firing_rates = []
            for spike_dict in spike_buffer:
                total_rate = 0
                count = 0
                for region, spikes in spike_dict.items():
                    try:
                        spike_array = np.asarray(spikes, dtype=np.float32)
                        # Handle scalar, 0-d, or empty arrays
                        if spike_array.ndim == 0:
                            # Scalar value
                            total_rate += float(spike_array)
                            count += 1
                        elif spike_array.size > 0:
                            # Array with values
                            total_rate += float(np.mean(spike_array))
                            count += 1
                    except (TypeError, ValueError):
                        # Skip if conversion fails
                        continue
                
                avg_rate = total_rate / max(1, count)
                firing_rates.append(avg_rate)
            if len(firing_rates) < 10:  # Lower threshold from 30 to 10
                return {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0}
            # Compute power spectral density using Welch's method
            fs = 1000.0 / dt_ms  # Sampling rate (Hz)
            signal_array = np.array(firing_rates, dtype=np.float32)
            
            try:
                freqs, psd = scipy_signal.welch(
                    signal_array, 
                    fs=fs, 
                    nperseg=min(len(signal_array), 32),
                    noverlap=min(len(signal_array)//2, 16)
                )
                
                # Extract band power
                def band_power(f_low, f_high):
                    idx = np.logical_and(freqs >= f_low, freqs <= f_high)
                    return float(np.sum(psd[idx])) if np.any(idx) else 0.0
                
                return {
                    'delta': band_power(1, 4),
                    'theta': band_power(4, 8),
                    'alpha': band_power(8, 13),
                    'beta': band_power(13, 30),
                    'gamma': band_power(30, 100)
                }
            except Exception as e:
                # Fallback if spectral analysis fails
                return {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'gamma': 0}
    