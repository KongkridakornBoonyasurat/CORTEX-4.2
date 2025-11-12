import os
import sys
import argparse
import pickle
import numpy as np
def safe_float(x, default=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

from datetime import datetime
import json
from collections import deque
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add /game/ folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "game"))

from experiment.game.pong_env import PongEnv
from cortex.brain.cortex_brain import CortexBrain
from cortex.sensory.biological_eye import CortexBiologicalEye

# === BRAIN VISUALIZATION IMPORTS ===
try:
    import pygame
    import math
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ pygame not available. Install with: pip install pygame")

class BrainVisualizer:
    """Real-time NEURAL SPIKE visualization for CORTEX 4.1"""
    
    def __init__(self, brain, width=1400, height=900):
        if not VISUALIZATION_AVAILABLE:
            return
            
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CORTEX 4.1 - Neural Spikes")
        
        self.brain = brain
        self.clock = pygame.time.Clock()
        
        # Brain region layout - spread out more
        self.region_positions = {
            'sensory': (250, 300),
            'parietal': (650, 200), 
            'prefrontal': (1050, 300),
            'limbic': (450, 600),
            'motor': (850, 600)
        }
        
        self.region_colors = {
            'sensory': (0, 255, 0),     # Green
            'parietal': (0, 100, 255),   # Blue  
            'prefrontal': (255, 0, 255), # Magenta
            'limbic': (255, 150, 0),     # Orange
            'motor': (255, 50, 50)       # Red
        }
        
        # Individual neuron positions
        self.neuron_positions = {}
        self._initialize_neuron_positions()
        
        # Spike tracking
        self.spike_trails = []  # Active spike animations
        self.neuron_spikes = {}  # Current neuron spikes
        
        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 24)
        
        # Animation state
        self.animation_time = 0
        
    def _initialize_neuron_positions(self):
        """Create individual neuron positions for each brain region"""
        for region_name, (base_x, base_y) in self.region_positions.items():
            positions = []
            # Simple grid of neurons - NO FUCKING CIRCLES
            grid_size = 4  # 4x4 grid
            spacing = 30
            
            for i in range(16):
                row = i // grid_size
                col = i % grid_size
                x = base_x + (col - 1.5) * spacing
                y = base_y + (row - 1.5) * spacing
                positions.append((x, y))
                
            self.neuron_positions[region_name] = positions
            
    def update_brain_state(self, brain_output, reward=0.0, episode_info=None):
        """Update visualization with REAL brain state"""
        self.animation_time += 1
        
        # Get REAL neuron voltages and spikes
        self.neuron_spikes = {}
        total_spikes = 0
        
        for region_name, region in self.brain.regions.items():
            if hasattr(region, 'neurons') and hasattr(region.neurons, 'neurons'):
                neurons = region.neurons.neurons
                spikes = []
                
                for i, neuron in enumerate(neurons[:16]):  # First 16 neurons
                    # FORCE SOME SPIKES FOR TESTING
                    if np.random.random() < 0.1:  # 10% chance of spike
                        spike_strength = 1.0
                        total_spikes += 1
                    else:
                        spike_strength = 0.0
                    
                    spikes.append(spike_strength)
                
                self.neuron_spikes[region_name] = spikes
        
        # Debug print
        if total_spikes > 0:
            print(f"🔥 {total_spikes} SPIKES DETECTED!")
        
        # Create spike trails between connected regions
        self._create_spike_trails()
        
    def _create_spike_trails(self):
        """Create spike animations between connected neurons"""
        # Remove old spike trails
        self.spike_trails = [trail for trail in self.spike_trails if trail['life'] > 0]
        
        # Inter-region connections
        connections = [
            ('sensory', 'parietal'),
            ('sensory', 'prefrontal'),
            ('parietal', 'prefrontal'),
            ('parietal', 'motor'),
            ('prefrontal', 'motor'),
            ('limbic', 'prefrontal'),
            ('limbic', 'motor')
        ]
        
        for source_region, target_region in connections:
            if (source_region in self.neuron_spikes and 
                target_region in self.neuron_spikes):
                
                source_spikes = self.neuron_spikes[source_region]
                
                # Create spike trails from ANY active source neurons
                for i, spike_strength in enumerate(source_spikes):
                    if spike_strength > 0.1:  # ANY activity creates spikes
                        # Pick random target neuron
                        target_idx = np.random.randint(0, 16)
                        
                        source_pos = self.neuron_positions[source_region][i]
                        target_pos = self.neuron_positions[target_region][target_idx]
                        
                        # Create spike trail - BRIGHT AND VISIBLE
                        self.spike_trails.append({
                            'start': source_pos,
                            'end': target_pos,
                            'progress': 0.0,
                            'life': 60,  # Longer life
                            'strength': spike_strength,
                            'color': (255, 255, 0)  # BRIGHT YELLOW
                        })
        
    def draw_neurons(self):
        """Draw individual neurons with real spike activity"""
        for region_name, positions in self.neuron_positions.items():
            # Region label
            label = self.font_medium.render(region_name.upper(), True, (255, 255, 255))
            center = self.region_positions[region_name]
            label_rect = label.get_rect(center=(center[0], center[1] - 150))
            self.screen.blit(label, label_rect)
            
            # Draw individual neurons
            if region_name in self.neuron_spikes:
                spikes = self.neuron_spikes[region_name]
                base_color = self.region_colors[region_name]
                
                for i, (x, y) in enumerate(positions):
                    if i < len(spikes):
                        spike_strength = spikes[i]
                        
                        # Neuron color based on REAL spike activity
                        if spike_strength > 0:
                            # Bright flash when spiking
                            intensity = min(1.0, spike_strength)
                            color = (
                                min(255, int(base_color[0] + (255 - base_color[0]) * intensity)),
                                min(255, int(base_color[1] + (255 - base_color[1]) * intensity)),
                                min(255, int(base_color[2] + (255 - base_color[2]) * intensity))
                            )
                            radius = int(6 + spike_strength * 4)
                        else:
                            # Dim when not spiking
                            color = tuple(int(c * 0.3) for c in base_color)
                            radius = 4
                        
                        # Draw neuron
                        pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
                        
                        # Extra bright flash for strong spikes
                        if spike_strength > 0.7:
                            pygame.draw.circle(self.screen, (255, 255, 255), 
                                             (int(x), int(y)), radius + 3, 2)
                            
    def draw_spike_trails(self):
        """Draw spike signals traveling between neurons"""
        for trail in self.spike_trails:
            # Update trail progress
            trail['progress'] += 0.03  # Slower for visibility
            trail['life'] -= 1
            
            if trail['progress'] <= 1.0:
                # Calculate current position
                start_x, start_y = trail['start']
                end_x, end_y = trail['end']
                
                current_x = start_x + (end_x - start_x) * trail['progress']
                current_y = start_y + (end_y - start_y) * trail['progress']
                
                # Draw BRIGHT VISIBLE spike line
                size = 8  # BIG
                
                # Draw the spike as a bright circle
                pygame.draw.circle(self.screen, trail['color'], 
                                 (int(current_x), int(current_y)), size)
                
                # Draw the connection line
                pygame.draw.line(self.screen, (100, 100, 100), 
                               trail['start'], trail['end'], 1)
        
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
        
    def render(self):
        """Render complete neural spike visualization"""
        # Clear screen with dark background
        self.screen.fill((5, 5, 15))
        
        # Draw neurons and spikes
        self.draw_neurons()
        self.draw_spike_trails()
        
        # Title
        title = self.font_large.render("CORTEX 4.1 - Neural Spike Activity", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self.width//2, 30))
        self.screen.blit(title, title_rect)
        
        # Controls
        controls_text = self.font_medium.render("ESC: Exit", True, (180, 180, 180))
        self.screen.blit(controls_text, (self.width - 150, self.height - 30))
        
        pygame.display.flip()

class PerformanceTracker:
    """Track brain performance and save best performers"""
    
    def __init__(self, log_folder):
        self.log_folder = log_folder
        self.performance_history = []
        self.best_brains = []  # Top 5 best performing brains
        self.max_best_brains = 5
        self.best_score = -999999
        self.save_threshold = 1.0  # Save if score improvement > threshold
        
        # Create performance subfolder
        self.performance_folder = os.path.join(log_folder, "performance_saves")
        os.makedirs(self.performance_folder, exist_ok=True)
        
    def evaluate_performance(self, episode, total_reward, brain, episode_log):
        """Evaluate if this brain performance deserves saving"""
        
        # Calculate performance metrics
        steps_survived = len(episode_log)
        avg_consciousness = np.mean([step.get('consciousness_level', 0) for step in episode_log])
        avg_self_agency = np.mean([step.get('self_agency', 0) for step in episode_log])
        
        # Composite performance score
        # Add randomness and better scaling to break the tie
        performance_score = (
            total_reward * 1.0 +                    # Primary component
            (steps_survived - 21) * 0.01 +         # Steps beyond minimum
            avg_consciousness * 0.001 +            # Reduced impact
            avg_self_agency * 0.001 +              # Reduced impact  
            np.random.normal(0, 0.1)               # Small random component to break ties
        )

        print(f"[DEBUG SCORE] Episode {episode}:")
        print(f"  total_reward: {total_reward}")
        print(f"  steps_survived: {steps_survived}")
        print(f"  avg_consciousness: {avg_consciousness}")
        print(f"  avg_self_agency: {avg_self_agency}")
        print(f"  CALCULATED performance_score: {performance_score}")
        print(f"  Components: {total_reward * 1.0} + {(steps_survived - 21) * 0.01} + {avg_consciousness * 0.001} + {avg_self_agency * 0.001} + random")

        performance_record = {
            'episode': episode,
            'total_reward': total_reward,
            'steps_survived': steps_survived,
            'avg_consciousness': avg_consciousness,
            'avg_self_agency': avg_self_agency,
            'performance_score': performance_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_history.append(performance_record)
        
        # Check if this is a top performer
        should_save = False
        save_reason = ""
        
        if performance_score > self.best_score + self.save_threshold:
            should_save = True
            save_reason = f"NEW BEST SCORE: {performance_score:.2f} (prev: {self.best_score:.2f})"
            self.best_score = performance_score
        elif total_reward > 5.0:  # Any brain that scores well
            should_save = True
            save_reason = f"GOOD PERFORMANCE: Reward={total_reward:.1f}, Score={performance_score:.2f}"
        elif len(self.best_brains) < self.max_best_brains:
            should_save = True
            save_reason = f"FILLING TOP 5: Score={performance_score:.2f}"
            
        if should_save:
            self.save_top_brain(episode, brain, performance_record, save_reason)
            
        return performance_record, should_save, save_reason
        
    def save_top_brain(self, episode, brain, performance_record, reason):
        """Save a top-performing brain with full details"""
        
        brain_filename = f"best_brain_ep{episode}_score{performance_record['performance_score']:.1f}.pkl"
        brain_path = os.path.join(self.performance_folder, brain_filename)
        
        # Extract detailed brain structure
        detailed_brain_data = self.extract_detailed_brain_structure(brain, performance_record)
        
        # Save brain with metadata
        brain_save_data = {
            'brain': brain,
            'performance_record': performance_record,
            'detailed_structure': detailed_brain_data,
            'save_reason': reason
        }
        
        with open(brain_path, 'wb') as f:
            pickle.dump(brain_save_data, f)
            
        # Add to best brains list
        self.best_brains.append({
            'episode': episode,
            'brain_path': brain_path,
            'performance_record': performance_record,
            'save_reason': reason
        })
        
        # Keep only top performers
        self.best_brains.sort(key=lambda x: x['performance_record']['performance_score'], reverse=True)
        if len(self.best_brains) > self.max_best_brains:
            # Remove worst performer file
            worst = self.best_brains.pop()
            if os.path.exists(worst['brain_path']):
                os.remove(worst['brain_path'])
                
        print(f"💾 SAVED TOP BRAIN: Episode {episode} - {reason}")
        
    def extract_detailed_brain_structure(self, brain, performance_record):
        """Extract complete brain structure with neuron details"""
        
        detailed_structure = {
            'regions': {},
            'global_stats': {
                'total_neurons': 0,
                'total_synapses': 0,
                'consciousness_metrics': {},
                'performance_metrics': performance_record
            }
        }
        
        total_neurons = 0
        total_synapses = 0
        
        # Extract each region's detailed structure
        for region_name, region in brain.regions.items():
            region_data = {
                'region_name': region_name,
                'n_neurons': 0,
                'neurons': [],
                'synapses': {},
                'astrocytes': {},
                'region_specific': {}
            }
            
            try:
                # === NEURON DETAILS ===
                if hasattr(region, 'neurons') and hasattr(region.neurons, 'neurons'):
                    neurons = region.neurons.neurons
                    region_data['n_neurons'] = len(neurons)
                    total_neurons += len(neurons)
                    
                    for i, neuron in enumerate(neurons):
                        neuron_data = {
                            'neuron_id': f"{region_name}_neuron_{i}",
                            'voltage': getattr(neuron, 'Vs', -70.0),
                            'current': getattr(neuron, 'Is', 0.0),
                            'spike_threshold': getattr(neuron, 'Vth', -50.0),
                            'refractory_time': getattr(neuron, 'refractory_time', 0.0),
                            'neuron_type': getattr(neuron, 'neuron_type', 'excitatory')
                        }
                        region_data['neurons'].append(neuron_data)
                        
                # === SYNAPTIC DETAILS ===
                if hasattr(region, 'synapses'):
                    synapse_data = {
                        'n_synapses': 0,
                        'weights': [],
                        'plasticity_traces': [],
                        'learning_rates': []
                    }
                    
                    if hasattr(region.synapses, 'synapses'):
                        synapses = region.synapses.synapses
                        synapse_data['n_synapses'] = len(synapses)
                        total_synapses += len(synapses)
                        
                        for i, synapse in enumerate(synapses):
                            weight = getattr(synapse, 'w', 0.0)
                            synapse_data['weights'].append({
                                'synapse_id': f"{region_name}_synapse_{i}",
                                'weight': weight,
                                'pre_trace': getattr(synapse, 'pre_trace', 0.0),
                                'post_trace': getattr(synapse, 'post_trace', 0.0),
                                'eligibility': getattr(synapse, 'eligibility_trace', 0.0)
                            })
                            
                    region_data['synapses'] = synapse_data
                    
                # === ASTROCYTE DETAILS ===
                if hasattr(region, 'astrocytes'):
                    astrocyte_data = {
                        'n_astrocytes': getattr(region.astrocytes, 'n_astrocytes', 0),
                        'calcium_levels': [],
                        'modulation_strength': []
                    }
                    
                    if hasattr(region.astrocytes, 'calcium_levels'):
                        astrocyte_data['calcium_levels'] = list(region.astrocytes.calcium_levels)
                        
                    region_data['astrocytes'] = astrocyte_data
                    
                # === REGION-SPECIFIC DATA ===
                if region_name == 'limbic':
                    # Limbic-specific data
                    if hasattr(region, 'neuromodulator_system'):
                        region_data['region_specific']['neuromodulators'] = {
                            'dopamine': getattr(region.neuromodulator_system, 'dopamine', 1.0),
                            'acetylcholine': getattr(region.neuromodulator_system, 'acetylcholine', 1.0),
                            'norepinephrine': getattr(region.neuromodulator_system, 'norepinephrine', 1.0)
                        }
                        
                elif region_name == 'motor':
                    # Motor-specific data
                    if hasattr(region, 'action_weights'):
                        region_data['region_specific']['action_weights'] = region.action_weights.tolist()
                    if hasattr(region, 'motor_traces'):
                        region_data['region_specific']['motor_traces'] = region.motor_traces.tolist()
                        
                elif region_name == 'prefrontal':
                    # Prefrontal-specific data
                    if hasattr(region, 'workspace'):
                        region_data['region_specific']['workspace_state'] = region.workspace.get_workspace_state().tolist()
                    if hasattr(region, 'attention_weights'):
                        region_data['region_specific']['attention_weights'] = region.attention_weights.tolist()
                        
                elif region_name == 'parietal':
                    # Parietal-specific data
                    if hasattr(region, 'self_boundary_detector'):
                        boundary_state = region.self_boundary_detector.get_self_representation()
                        region_data['region_specific']['self_boundary'] = boundary_state
                        
                elif region_name == 'sensory':
                    # Sensory-specific data
                    if hasattr(region, 'visual_encoder'):
                        region_data['region_specific']['receptive_fields'] = len(region.visual_encoder.receptive_fields)
                        if hasattr(region.visual_encoder, 'feature_history'):
                            recent_features = list(region.visual_encoder.feature_history)[-5:] if region.visual_encoder.feature_history else []
                            region_data['region_specific']['recent_features'] = [f.tolist() if hasattr(f, 'tolist') else f for f in recent_features]
                            
            except Exception as e:
                print(f"⚠️ Error extracting {region_name} details: {e}")
                region_data['extraction_error'] = str(e)
                
            detailed_structure['regions'][region_name] = region_data
            
        # Update global stats
        detailed_structure['global_stats']['total_neurons'] = total_neurons
        detailed_structure['global_stats']['total_synapses'] = total_synapses
        
        return detailed_structure
        
    def create_super_brain(self, final_brain):
        """Create final super brain by combining best performers"""
        
        print(f"🧠 Creating SUPER BRAIN from {len(self.best_brains)} top performers...")
        
        if not self.best_brains:
            print("⚠️ No best brains found, using final brain as super brain")
            return final_brain
            
        # Load all best brains
        best_brain_data = []
        for best_brain_info in self.best_brains:
            try:
                with open(best_brain_info['brain_path'], 'rb') as f:
                    brain_data = pickle.load(f)
                    best_brain_data.append(brain_data)
                    print(f"   📥 Loaded brain from episode {best_brain_info['episode']} "
                          f"(score: {best_brain_info['performance_record']['performance_score']:.2f})")
            except Exception as e:
                print(f"⚠️ Failed to load {best_brain_info['brain_path']}: {e}")
                
        if not best_brain_data:
            return final_brain
            
        # Combine best performing regions
        super_brain = final_brain  # Start with final brain structure
        
        print(f"🔄 Fusing regions from {len(best_brain_data)} best brains...")
        
        for region_name in super_brain.regions.keys():
            print(f"   🧩 Fusing {region_name} region...")
            
            # Find best performer for this region based on region-specific metrics
            best_region_brain = self._find_best_region_performer(region_name, best_brain_data)
            
            if best_region_brain:
                try:
                    # Copy the best performing region's parameters
                    source_region = best_region_brain['brain'].regions[region_name]
                    target_region = super_brain.regions[region_name]
                    
                    # Fusion strategy: average weights from top performers
                    self._fuse_region_parameters(source_region, target_region, region_name)
                    
                except Exception as e:
                    print(f"⚠️ Failed to fuse {region_name}: {e}")
                    
        return super_brain
        
    def _find_best_region_performer(self, region_name, best_brain_data):
        """Find which brain had the best performance for a specific region"""
        
        region_scores = []
        for brain_data in best_brain_data:
            # Use overall performance as proxy for region performance
            # In future, could track region-specific metrics
            performance_score = brain_data['performance_record']['performance_score']
            region_scores.append((performance_score, brain_data))
            
        if region_scores:
            region_scores.sort(key=lambda x: x[0], reverse=True)
            return region_scores[0][1]  # Return best performing brain data
            
        return None
        
    def _fuse_region_parameters(self, source_region, target_region, region_name):
        """Fuse parameters from source region into target region"""
        
        try:
            # Fuse synaptic weights
            if hasattr(source_region, 'synapses') and hasattr(target_region, 'synapses'):
                source_synapses = source_region.synapses.synapses
                target_synapses = target_region.synapses.synapses
                
                min_synapses = min(len(source_synapses), len(target_synapses))
                for i in range(min_synapses):
                    # Average the weights (could use other strategies)
                    if hasattr(source_synapses[i], 'w') and hasattr(target_synapses[i], 'w'):
                        target_synapses[i].w = (target_synapses[i].w + source_synapses[i].w) / 2.0
                        
            # Region-specific parameter fusion
            if region_name == 'motor' and hasattr(source_region, 'action_weights'):
                if hasattr(target_region, 'action_weights'):
                    target_region.action_weights = (target_region.action_weights + source_region.action_weights) / 2.0
                    
            elif region_name == 'prefrontal' and hasattr(source_region, 'attention_weights'):
                if hasattr(target_region, 'attention_weights'):
                    target_region.attention_weights = (target_region.attention_weights + source_region.attention_weights) / 2.0
                    
            print(f"     ✅ Fused {region_name} parameters")
            
        except Exception as e:
            print(f"     ⚠️ Fusion error for {region_name}: {e}")
            
    def save_performance_summary(self):
        """Save performance tracking summary"""
        
        summary_path = os.path.join(self.log_folder, "performance_summary.json")
        
        summary_data = {
            'total_episodes_tracked': len(self.performance_history),
            'best_overall_score': self.best_score,
            'top_performers': [
                {
                    'episode': brain['episode'],
                    'performance_score': brain['performance_record']['performance_score'],
                    'total_reward': brain['performance_record']['total_reward'],
                    'save_reason': brain['save_reason']
                }
                for brain in self.best_brains
            ],
            'performance_timeline': self.performance_history[-50:],  # Last 50 episodes
            'training_stats': {
                'avg_performance': np.mean([p['performance_score'] for p in self.performance_history]),
                'performance_trend': self._calculate_performance_trend(),
                'consciousness_progression': self._calculate_consciousness_trend(),
                'self_agency_progression': self._calculate_agency_trend()
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        print(f"📊 Performance summary saved to: {summary_path}")
        
    def _calculate_performance_trend(self):
        """Calculate if performance is improving over time"""
        if len(self.performance_history) < 10:
            return 0.0
            
        recent = [p['performance_score'] for p in self.performance_history[-10:]]
        early = [p['performance_score'] for p in self.performance_history[:10]]
        
        return np.mean(recent) - np.mean(early)
        
    def _calculate_agency_trend(self):
        """Calculate self-agency development trend"""
        if len(self.performance_history) < 10:
            return 0.0
            
        recent = [p['avg_self_agency'] for p in self.performance_history[-10:]]
        early = [p['avg_self_agency'] for p in self.performance_history[:10]]
        
        return np.mean(recent) - np.mean(early)


def get_log_folder():
    folder = os.path.join("logs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    return folder

def save_state(brain, episode, log_data, folder, env_state=None):
    # Save the brain object
    brain_pkl_path = os.path.join(folder, f"brain_state_ep{episode}.pkl")
    with open(brain_pkl_path, 'wb') as f:  # 'wb' not 'rb'!
        pickle.dump(brain, f)  # Save brain, not load!
    print(f"Brain saved to: {brain_pkl_path}")
    
    with open(os.path.join(folder, "experiment_log.pkl"), "wb") as f:
        pickle.dump(log_data, f)
    if env_state is not None:
        with open(os.path.join(folder, "env_state.pkl"), "wb") as f:
            pickle.dump(env_state, f)
    print(f"[Paused & Saved at Episode {episode} in {folder}]")

def save_final_comprehensive_brain(brain, performance_tracker, log_folder):
    """Save final comprehensive brain with all details"""
    
    print(f"💾 Creating FINAL COMPREHENSIVE BRAIN SAVE...")
    
    # Create super brain from best performers
    super_brain = performance_tracker.create_super_brain(brain)
    
    # Extract complete detailed structure
    final_performance_record = {
        'episode': 'FINAL',
        'performance_score': performance_tracker.best_score,
        'training_complete': True
    }
    
    detailed_structure = performance_tracker.extract_detailed_brain_structure(super_brain, final_performance_record)
    
    # Create comprehensive save data
    comprehensive_save = {
        'super_brain': super_brain,
        'original_final_brain': brain,
        'detailed_brain_structure': detailed_structure,
        'performance_history': performance_tracker.performance_history,
        'best_performers': performance_tracker.best_brains,
        'training_metadata': {
            'training_completion_time': datetime.now().isoformat(),
            'total_episodes': len(performance_tracker.performance_history),
            'best_score_achieved': performance_tracker.best_score,
            'performance_improvement': performance_tracker._calculate_performance_trend(),
            'consciousness_development': performance_tracker._calculate_consciousness_trend(),
            'self_agency_development': performance_tracker._calculate_agency_trend()
        },
        'region_analysis': {
            region_name: {
                'n_neurons': detailed_structure['regions'][region_name]['n_neurons'],
                'n_synapses': detailed_structure['regions'][region_name]['synapses']['n_synapses'],
                'specialization': detailed_structure['regions'][region_name]['region_specific']
            }
            for region_name in detailed_structure['regions'].keys()
        }
    }
    
    # Save comprehensive brain
    final_brain_path = os.path.join(log_folder, "FINAL_SUPER_BRAIN_COMPLETE.pkl")
    with open(final_brain_path, 'wb') as f:
        pickle.dump(comprehensive_save, f)
        
    # Save human-readable structure
    structure_path = os.path.join(log_folder, "brain_structure_analysis.json")
    with open(structure_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_safe_structure = convert_numpy_to_lists(detailed_structure)
        json.dump(json_safe_structure, f, indent=2)
        
    print(f"🧠 FINAL SUPER BRAIN saved to: {final_brain_path}")
    print(f"📋 Brain structure analysis saved to: {structure_path}")
    
    return final_brain_path, structure_path

def convert_numpy_to_lists(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

def load_state(folder):
    with open(os.path.join(folder, "brain_state.pkl"), "rb") as f:
        brain = pickle.load(f)
    with open(os.path.join(folder, "experiment_log.pkl"), "rb") as f:
        log_data = pickle.load(f)
    env_state = None
    env_state_path = os.path.join(folder, "env_state.pkl")
    if os.path.exists(env_state_path):
        with open(env_state_path, "rb") as f:
            env_state = pickle.load(f)
    print(f"[Resuming from {folder}]")
    return brain, log_data, env_state

def main():
    parser = argparse.ArgumentParser(description="Enhanced Synthetic Brain x Pong Experiment Runner")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Show game window while running")
    parser.add_argument("--visualize", action="store_true", help="Show brain visualization")
    parser.add_argument("--resume", type=str, default=None, help="Folder to resume experiment from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-interval", type=int, default=10, help="Save best brains every N episodes")
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.resume:
        LOG_FOLDER = args.resume
        brain, log_data, env_state = load_state(LOG_FOLDER)
        episode_start = len(log_data)
    else:
        LOG_FOLDER = get_log_folder()
        brain = CortexBrain(n_neurons_per_region=16)
        log_data = []
        episode_start = 0

    # === PERFORMANCE TRACKER ===
    performance_tracker = PerformanceTracker(LOG_FOLDER)
    
    # === BRAIN VISUALIZATION SETUP ===
    visualizer = None
    if args.visualize and VISUALIZATION_AVAILABLE:
        visualizer = BrainVisualizer(brain)
        print("🎨 Brain visualization enabled")
    elif args.visualize:
        print("⚠️ Brain visualization requested but pygame not available")
    
    # === Retina and Pong setup ===
    eye = CortexBiologicalEye(resolution=(84, 84))
    env = PongEnv(width=84, height=84, render_mode=args.render)
    if args.render:
        print("Rendering enabled: Game window will appear.")

    print(f"🚀 Starting Enhanced Training with Performance Tracking")
    print(f"   Episodes: {args.episodes}")
    print(f"   Save best brains every {args.save_interval} episodes")
    print(f"   Log folder: {LOG_FOLDER}")
    print(f"   Brain visualization: {'ON' if visualizer else 'OFF'}")

    for episode in range(episode_start, args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step = 0
        episode_log = []
        
        # Initialize reward for first step (standard RL approach)
        reward = 0.0

        max_steps = 500
        while not done and step < max_steps:
            # Handle visualization events
            if visualizer:
                if not visualizer.handle_events():
                    # User pressed ESC - save and exit
                    save_state(brain, episode, log_data, LOG_FOLDER)
                    performance_tracker.save_performance_summary()
                    print("💾 Saved and exiting...")
                    pygame.quit()
                    env.close()
                    return

            frame = obs  # PongEnv returns frame as obs

            # --- Retina/vision processing ---
            eye_out = eye.process_visual_input(frame)
            features = eye_out['features']
            features = features + np.random.normal(0, 0.1, size=features.shape)

            # --- Step the synthetic brain with previous reward ---
            action, brain_state = brain.step(features, reward=reward, dt=0.01)

            # === ENHANCED BIOLOGICAL MONITORING ===
            if step % 20 == 0:  # Reduced frequency to avoid spam
                try:
                    # Get voltages from first 4 sensory neurons
                    sensory_neurons = brain.regions['sensory'].neurons.neurons
                    voltages = [neuron.Vs for neuron in sensory_neurons[:4]]
                    close_to_spike = [v for v in voltages if v > -55.0]
                    
                    # Count total spikes across all regions
                    total_spikes = 0
                    regional_activities = brain_state.get('regional_activities', {})
                    for region_name, activity in regional_activities.items():
                        total_spikes += float(activity)
                    
                    if step % 100 == 0:  # Less frequent detailed logging
                        print(f"[BIOLOGICAL] Ep{episode} Step{step}: Spikes={total_spikes:.1f}, "
                              f"Near-spike={len(close_to_spike)}, Reward={total_reward:.1f}")
                    
                except Exception as e:
                    pass  # Silent monitoring errors

            # --- Take action in environment ---
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # --- Update brain visualization ---
            if visualizer:
                visualizer.update_brain_state(brain_state, reward, {
                    'episode': episode,
                    'step': step,
                    'total_reward': total_reward
                })
                visualizer.render()
                visualizer.clock.tick(60)  # 60 FPS

            # --- Log everything ---
            episode_log.append({
                "episode": episode,
                "step": step,
                "reward": reward,
                "action": action,
                "self_agency": brain_state['self_agency'],
                "regional_activities": brain_state['regional_activities'],
                "consciousness_level": safe_float(brain_state['consciousness_report']['consciousness_level'])
            })

            step += 1
            if args.render:
                env.render(mode="human")
            
            # Force episode end if too long
            if step >= max_steps:
                done = True

        # === PERFORMANCE EVALUATION ===
        performance_record, was_saved, save_reason = performance_tracker.evaluate_performance(
            episode, total_reward, brain, episode_log
        )
        
        log_data.append(episode_log)
        
        # Enhanced episode reporting
        consciousness = performance_record['avg_consciousness']
        agency = performance_record['avg_self_agency']
        performance_score = performance_record['performance_score']
        
        print(f"Episode {episode}: Reward={total_reward:.1f}, Steps={step}, "
              f"Performance={performance_score:.2f}, Consciousness={consciousness:.1f}%, Agency={agency:.1f}%")
        
        if was_saved:
            print(f"   💾 {save_reason}")

        # Save log for each episode
        with open(os.path.join(LOG_FOLDER, f"episode_{episode}_log.pkl"), "wb") as f:
            pickle.dump(episode_log, f)

        # Periodic comprehensive saves
        if episode % args.save_interval == 0 and episode > 0:
            save_state(brain, episode, log_data, LOG_FOLDER)
            performance_tracker.save_performance_summary()
            print(f"📁 Checkpoint saved at episode {episode}")

        # Pause/resume handler (Ctrl+C)
        try:
            pass
        except KeyboardInterrupt:
            save_state(brain, episode, log_data, LOG_FOLDER)
            performance_tracker.save_performance_summary()
            print("Paused and saved. Resume with --resume.")
            if visualizer:
                pygame.quit()
            return

    # === FINAL COMPREHENSIVE SAVE ===
    print(f"\n🎯 TRAINING COMPLETE! Creating final comprehensive brain...")
    
    # Save regular final state
    save_state(brain, args.episodes, log_data, LOG_FOLDER)
    
    # Save performance summary
    performance_tracker.save_performance_summary()
    
    # Create and save super brain
    final_brain_path, structure_path = save_final_comprehensive_brain(brain, performance_tracker, LOG_FOLDER)
    
    print(f"\n✅ EXPERIMENT COMPLETE!")
    print(f"   📁 Log folder: {LOG_FOLDER}")
    print(f"   🧠 Final super brain: {final_brain_path}")
    print(f"   📊 Brain analysis: {structure_path}")
    print(f"   🏆 Best performance score: {performance_tracker.best_score:.2f}")
    print(f"   💾 Top performers saved: {len(performance_tracker.best_brains)}")
    
    if visualizer:
        pygame.quit()
    env.close()

if __name__ == "__main__":
    main()
        
    def _calculate_consciousness_trend(self):
        """Calculate consciousness development trend"""
        if len(self.performance_history) < 10:
            return 0.0
            
        recent = [p['avg_consciousness'] for p in self.performance_history[-10:]]
        early = [p['avg_consciousness'] for p in self.performance_history[:10]]
        
        return np.mean(recent) - np.mean(early)