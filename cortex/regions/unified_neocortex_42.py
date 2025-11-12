# cortex/regions/unified_neocortex_42.py
"""
CORTEX 4.2 Unified Neocortex - Hierarchical Predictive Processing
================================================================
FULLY PyTorch GPU-accelerated implementation following CORTEX 4.2 specifications

Implements the canonical cortical microcircuit with:
- Hierarchical predictive coding (Rao & Ballard, 1999)
- Precision-weighted belief updates (Feldman & Friston, 2010)
- 6-layer cortical architecture (Douglas & Martin, 2004)
- Integration with CORTEX 4.2 components

Key Features:
- Bottom-up prediction errors
- Top-down predictions
- Precision-weighted learning
- Canonical microcircuit mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from typing import List
import torch.nn.functional as F

# Import CORTEX 4.2 components
from cortex.cells.enhanced_neurons_42 import EnhancedNeuronPopulation42PyTorch
from cortex.cells.enhanced_synapses_42 import EnhancedSynapticSystem42PyTorch
from cortex.cells.astrocyte import AstrocyteNetwork
from cortex.modulation.modulators import ModulatorSystem42
from cortex.modulation.oscillator import Oscillator

# GPU setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CORTEX 4.2 Neocortex constants
CORTEX_42_NEOCORTEX_CONSTANTS = {
    # Hierarchical levels
    'n_levels': 3,  # 0=sensory, 1=parietal, 2=prefrontal
    'neurons_per_level': [8, 16, 8],  # Matching your existing modules
    
    # Time constants
    'tau_mu': 10.0,  # Neural time constant (ms)
    'tau_epsilon': 5.0,  # Error time constant (ms)
    
    # Learning rates
    'alpha_pe': 0.001,  # Prediction error learning rate
    'alpha_td': 0.0001,  # Top-down learning rate
    
    # Precision parameters
    'base_precision': [1.0, 0.5, 0.25],  # Level 0, 1, 2
    'beta_ach': 0.5,  # ACh effect on precision
    'beta_ne': 0.3,  # NE effect on precision
    
    # Layer connectivity
    'layer_2_3_weight': 1.0,  # Error computation
    'layer_4_weight': 1.0,  # Input reception
    'layer_5_weight': 1.0,  # Belief maintenance
    'layer_6_weight': 1.0,  # Prediction generation
}

class CorticalLayer(nn.Module):
    """
    Single cortical layer implementation
    
    Maps to specific laminar functions:
    - Layer 2/3: Error computation
    - Layer 4: Input reception
    - Layer 5: Belief representation
    - Layer 6: Prediction generation
    """
    
    def __init__(self, n_neurons: int, layer_type: str, device=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.layer_type = layer_type
        self.device = device or DEVICE

        # Neural state - initialize with small random values for L5
        if layer_type == 'L5':
            self.activity = torch.randn(n_neurons, device=self.device) * 0.1
        else:
            self.activity = torch.zeros(n_neurons, device=self.device)
        
        # Layer-specific parameters
        if layer_type == 'L2_3':  # Error neurons
            self.threshold = nn.Parameter(torch.tensor(0.1, device=self.device))
        elif layer_type == 'L4':  # Input neurons
            self.gain = nn.Parameter(torch.ones(1, device=self.device))
        elif layer_type == 'L5':  # Belief neurons
            self.tau = nn.Parameter(torch.tensor(10.0, device=self.device))        
        elif layer_type == 'L6':  # Prediction neurons
            self.prediction_weight = nn.Parameter(torch.randn(n_neurons, n_neurons, device=self.device) * 0.1)
    
    def forward(self, input_signal: torch.Tensor, dt: float = 0.001) -> torch.Tensor:
        """Process input based on layer type"""
        
        if self.layer_type == 'L2_3':
            # Error computation with threshold
            self.activity = torch.relu(input_signal - self.threshold)
            
        elif self.layer_type == 'L4':
            # Direct relay with gain
            self.activity = input_signal * self.gain
            
        elif self.layer_type == 'L5':
            # Leaky integration with momentum for beliefs
            if not hasattr(self, 'momentum'):
                self.momentum = torch.zeros_like(self.activity)
            
            # Calculate momentum update
            self.momentum = 0.9 * self.momentum + (dt / self.tau) * (input_signal - self.activity)
            self.activity = self.activity + self.momentum         

        elif self.layer_type == 'L6':
            # Generate predictions
            self.activity = torch.matmul(self.prediction_weight, input_signal)
            
        return self.activity

class CanonicalMicrocircuit(nn.Module):
    """
    Canonical cortical microcircuit
    
    Implements the 6-layer structure found across neocortex
    with predictive coding dynamics
    """
    
    def __init__(self, n_neurons: int, level: int, device=None):
        super().__init__()
        self.n_neurons = n_neurons
        self.level = level
        self.device = device or DEVICE
        
        # Create layers
        self.L2_3 = CorticalLayer(n_neurons, 'L2_3', self.device)  # Error
        self.L4 = CorticalLayer(n_neurons, 'L4', self.device)      # Input
        self.L5 = CorticalLayer(n_neurons, 'L5', self.device)      # Belief
        self.L6 = CorticalLayer(n_neurons, 'L6', self.device)      # Prediction
        
        # Inter-layer connections
        self.L4_to_L2_3 = nn.Linear(n_neurons, n_neurons, device=self.device)
        self.L5_to_L2_3 = nn.Linear(n_neurons, n_neurons, device=self.device)
        self.L5_to_L6 = nn.Linear(n_neurons, n_neurons, device=self.device)
        
        # Initialize with biological constraints
        with torch.no_grad():
            self.L4_to_L2_3.weight.data = torch.eye(n_neurons, device=self.device)
            self.L5_to_L2_3.weight.data = -torch.eye(n_neurons, device=self.device)
        
        print(f"CanonicalMicrocircuit Level {level}: {n_neurons} neurons")
    
    def forward(self, bottom_up: torch.Tensor, top_down: torch.Tensor, 
                precision: float = 1.0, dt: float = 0.001) -> Dict[str, torch.Tensor]:
        """
        Process one timestep of cortical dynamics
        
        Args:
            bottom_up: Input from lower level or sensory input
            top_down: Prediction from higher level
            precision: Confidence in this level
            dt: Time step
            
        Returns:
            Dictionary with layer activities
        """
        
        # Layer 4: Receive bottom-up input
        L4_activity = self.L4(bottom_up, dt)

        # Layer 6: Generate prediction based on belief + top-down
        L6_input = self.L5.activity + 0.6 * top_down
        L6_activity = self.L6(L6_input, dt)

        # Layer 2/3: Compute prediction error
        # Ensure sizes match for linear layers
        if L4_activity.shape[0] > self.n_neurons:
            L4_input = L4_activity[:self.n_neurons]
        else:
            L4_input = F.pad(L4_activity, (0, self.n_neurons - L4_activity.shape[0]))

        if L6_activity.shape[0] > self.n_neurons:
            L6_input_resized = L6_activity[:self.n_neurons]
        else:
            L6_input_resized = F.pad(L6_activity, (0, self.n_neurons - L6_activity.shape[0]))

        error_input = self.L4_to_L2_3(L4_input) + self.L5_to_L2_3(L6_input_resized)
        
        raw_error = self.L2_3(error_input * precision, dt)

        # === Noise-robust adaptation ===
        noise_estimate = torch.std(raw_error)
        adaptive_precision = precision / (1.0 + 0.5 * noise_estimate)
        L2_3_activity = raw_error * adaptive_precision


        # Layer 5: Update belief based on weighted error
        # Add damping factor to prevent explosion
        damping = 0.95  # Increased damping
        belief_update = 0.3 * precision * L2_3_activity - 0.03 * (self.L5.activity - top_down)
        belief_update = torch.clamp(belief_update, -0.2, 0.2)  # Tighter bounds

        # More conservative integration
        bias = 0.0001  # Reduced bias
        L5_input = damping * self.L5.activity + belief_update * dt * 0.2 + bias  # Added extra scaling
        L5_activity = self.L5(L5_input, dt)

        return {
            'L2_3': L2_3_activity,  # Error to send up
            'L4': L4_activity,      # Input received
            'L5': L5_activity,      # Current belief
            'L6': L6_activity,      # Prediction to send down
            'error': L2_3_activity,
            'belief': L5_activity,
            'prediction': L6_activity
        }

class HierarchicalPredictiveCoding(nn.Module):
    """
    Hierarchical predictive coding across multiple levels
    
    Implements the mathematics from the CORTEX 4.2 paper:
    - Belief updates: μ(t) = (1-α)μ(t-1) + α[g(μ_up) + ε]
    - Error computation: ε = x - x̂
    - Precision weighting: Π = Π_base * [1 + β_ACh*ACh + β_NE*NE]
    """
    
    def __init__(self, n_levels: int = 3, neurons_per_level: List[int] = None, device=None):
        super().__init__()
        self.n_levels = n_levels
        self.device = device or DEVICE
        
        
        if neurons_per_level is None:
            neurons_per_level = [8, 16, 8]  # Match sensory output: 8 -> 16 -> 8
        # Create hierarchical levels
        self.levels = nn.ModuleList([
            CanonicalMicrocircuit(neurons_per_level[i], i, self.device)
            for i in range(n_levels)
        ])
        
        # Inter-level connections (generative model)
        self.bottom_up_weights = nn.ModuleList([
            nn.Linear(neurons_per_level[i], neurons_per_level[i+1], device=self.device)
            for i in range(n_levels-1)
        ])
        
        self.top_down_weights = nn.ModuleList([
            nn.Linear(neurons_per_level[i+1], neurons_per_level[i], device=self.device)
            for i in range(n_levels-1)
        ])
        
        # Precision parameters
        self.base_precision = torch.tensor(
            CORTEX_42_NEOCORTEX_CONSTANTS['base_precision'], 
            device=self.device
        )
        
        # Learning parameters
        self.alpha_pe = CORTEX_42_NEOCORTEX_CONSTANTS['alpha_pe']
        self.alpha_td = CORTEX_42_NEOCORTEX_CONSTANTS['alpha_td']
        
        # Initialize inter-level connections properly
        with torch.no_grad():
            for i in range(n_levels-1):
                # Initialize with small random values, not default random
                nn.init.xavier_uniform_(self.bottom_up_weights[i].weight, gain=0.1)
                nn.init.xavier_uniform_(self.top_down_weights[i].weight, gain=0.1)
        
        print(f"HierarchicalPredictiveCoding: {n_levels} levels")

    def compute_precision(self, level: int, ach: float, ne: float) -> float:
        """
        Compute precision-weighted confidence
            
        Implements: Π(t) = Π_base * [1 + β_ACh*ACh + β_NE*NE]
        """
        base = self.base_precision[level]
        beta_ach = CORTEX_42_NEOCORTEX_CONSTANTS['beta_ach']
        beta_ne = CORTEX_42_NEOCORTEX_CONSTANTS['beta_ne']
            
        precision = base * (1.0 + beta_ach * ach + beta_ne * ne)
        return float(precision)
    
    def forward(self, sensory_input: torch.Tensor, 
                neuromodulators: Dict[str, float],
                    dt: float = 0.001) -> Dict[str, Any]:
            """
            Full hierarchical processing
            
            Args:
                sensory_input: Bottom-up sensory signal
                neuromodulators: {'ACh': float, 'NE': float}
                dt: Time step
                
            Returns:
                Hierarchical state and predictions
            """
            
            # Extract neuromodulator levels
            ach = neuromodulators.get('ACh', 1.0)
            ne = neuromodulators.get('NE', 1.0)
            
            # Initialize outputs
            level_outputs = []
            errors = []
            beliefs = []
            predictions = []
            
            # Bottom-up pass
            bottom_up_input = sensory_input
            
            for level in range(self.n_levels):
                # Get precision for this level
                precision = self.compute_precision(level, ach, ne)
                
                # Get top-down prediction
                if level < self.n_levels - 1:
                    # Get prediction from level above
                    with torch.no_grad():
                        top_down = self.levels[level + 1].L5.activity
                        top_down = self.top_down_weights[level](top_down)
                else:
                    # Highest level has no top-down
                    top_down = torch.zeros_like(self.levels[level].L5.activity)
                
                # Process in microcircuit
                output = self.levels[level](bottom_up_input, top_down, precision, dt)
                level_outputs.append(output)
                
                # Store for analysis
                errors.append(output['error'])
                beliefs.append(output['belief'])
                predictions.append(output['prediction'])
                
                # Debug print for level 0
                #if level == 0:  # Debug sensory level
                #    print(f"Debug - Level 0: bottom_up norm={torch.norm(bottom_up_input):.3f}, "
                #          f"belief norm={torch.norm(output['belief']):.3f}, "
                #          f"error norm={torch.norm(output['error']):.3f}")

                # Prepare bottom-up for next level
                if level < self.n_levels - 1:
                    bottom_up_input = self.bottom_up_weights[level](output['error'])
            
            # Top-down pass (update generative weights)
            self._update_generative_model(errors, beliefs, dt)
            
            return {
                'level_outputs': level_outputs,
                'errors': errors,
                'beliefs': beliefs,
                'predictions': predictions,
                'top_belief': beliefs[-1],
                'sensory_prediction': predictions[0]
            }
        
    def _update_generative_model(self, errors: List[torch.Tensor], 
                                beliefs: List[torch.Tensor], dt: float):
        """Update inter-level connections based on prediction errors"""
        
        with torch.no_grad():
            # Update bottom-up weights (error propagation)
            for i in range(self.n_levels - 1):
                # Check actual weight dimensions
                weight_shape = self.bottom_up_weights[i].weight.shape
                # weight_shape is [output_features, input_features]
                
                # Normalize errors to prevent explosion
                eps = 1e-8
                error_i = errors[i].flatten()
                error_i = error_i / (error_i.norm(p=2) + eps)
                error_i_plus_1 = errors[i+1].flatten()
                error_i_plus_1 = error_i_plus_1 / (error_i_plus_1.norm(p=2) + eps)
                # Check weight matrix expected shape
                expected_out, expected_in = self.bottom_up_weights[i].weight.shape
                # Reshape errors to match weight expectations
                if error_i_plus_1.shape[0] != expected_out:
                    error_i_plus_1 = F.pad(error_i_plus_1, (0, expected_out - error_i_plus_1.shape[0]))[:expected_out]
                if error_i.shape[0] != expected_in:
                    error_i = F.pad(error_i, (0, expected_in - error_i.shape[0]))[:expected_in]
                
                error_i = error_i.view(-1); error_i_plus_1 = error_i_plus_1.view(-1)
                error_grad = torch.outer(error_i_plus_1, error_i)
                
                # === Confidence-based learning ===
                confidence = 1.0 / (1.0 + torch.mean(error_i**2))
                update = self.alpha_pe * error_grad * dt * confidence

                # Clip to prevent explosion
                update = torch.clamp(update, -0.001, 0.001)
                self.bottom_up_weights[i].weight.data += update

                # Keep weights bounded
                self.bottom_up_weights[i].weight.data = torch.clamp(
                    self.bottom_up_weights[i].weight.data, -2.0, 2.0
                )
            
            # Update top-down weights (generative model)
            for i in range(self.n_levels - 1):
                # Check actual weight dimensions
                weight_shape = self.top_down_weights[i].weight.shape
                # weight_shape is [output_features, input_features]
                
                # Normalize beliefs
                belief_i = F.normalize(beliefs[i], dim=0)
                belief_i_plus_1 = F.normalize(beliefs[i+1], dim=0)
                
                # Create gradient with CORRECT dimensions
                # We need [output_features, input_features] = [64, 32] for level 1->0
                # beliefs[i] should map to output_features (64)
                # beliefs[i+1] should map to input_features (32)
                pred_grad = torch.matmul(
                    belief_i.unsqueeze(1),        # [64, 1]
                    belief_i_plus_1.unsqueeze(0)  # [1, 32]
                )  # Result: [64, 32] - matches weight shape
                
                # Apply learning with gradient clipping
                update = self.alpha_td * pred_grad * dt
                update = torch.clamp(update, -0.001, 0.001)
                
                self.top_down_weights[i].weight.data += update
                
                # Keep weights bounded
                self.top_down_weights[i].weight.data = torch.clamp(
                    self.top_down_weights[i].weight.data, -1.0, 1.0
                )

class UnifiedNeocortex42PyTorch(nn.Module):
    """
    CORTEX 4.2 Unified Neocortex - Complete Implementation
    
    Unifies all cortical processing under predictive coding:
    - Sensory areas predict input
    - Motor areas predict consequences
    - Prefrontal areas predict abstract states
    
    All use the same canonical microcircuit
    """
    
    def __init__(self, device=None):
        super().__init__()
        self.device = device or DEVICE
        self.region_name = "unified_neocortex_42"
        
        # Hierarchical predictive coding system
        self.hierarchy = HierarchicalPredictiveCoding(device=self.device)
        
        # Integration with CORTEX 4.2 modulators
        self.modulators = ModulatorSystem42()
        
        # Map existing regions to hierarchy levels
        self.region_mapping = {
            'sensory': 0,
            'parietal': 1,
            'prefrontal': 2
        }
        
        # Activity history
        self.belief_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=1000)
        
        print(f"UnifiedNeocortex42PyTorch initialized")
    
    def forward(self, sensory_input: torch.Tensor, 
                motor_feedback: Optional[torch.Tensor] = None,
                reward_signal: float = 0.0,
                dt: float = 0.001) -> Dict[str, torch.Tensor]:
        """
        Unified cortical processing
        
        Maps existing CORTEX 4.2 regions onto hierarchy
        """
        
        # Get current neuromodulator levels
        neuromodulators = {
            'ACh': float(self.modulators.acetylcholine.level.item()),
            'NE': float(self.modulators.norepinephrine.level.item()),
            'DA': float(self.modulators.dopamine.level.item())
        }
        
        # Process through hierarchy
        hierarchical_output = self.hierarchy(sensory_input, neuromodulators, dt)
        
        # Extract outputs for each conceptual region
        sensory_belief = hierarchical_output['beliefs'][0]
        parietal_belief = hierarchical_output['beliefs'][1]
        prefrontal_belief = hierarchical_output['beliefs'][2]
        
        # Store history
        self.belief_history.append({
            'sensory': sensory_belief.detach().cpu().numpy(),
            'parietal': parietal_belief.detach().cpu().numpy(),
            'prefrontal': prefrontal_belief.detach().cpu().numpy()
        })
        
        total_error = sum(torch.mean(e**2) for e in hierarchical_output['errors'])
        self.error_history.append(float(total_error))

        # Update modulators based on prediction error
        surprise = float(total_error)
        self.modulators.norepinephrine.step(pulse=(surprise > 0.5))
        self.modulators.acetylcholine.step(pulse=(surprise > 0.3))
        self.modulators.dopamine.step(pulse=(reward_signal > 0.0))

        return {
            'sensory_encoding': sensory_belief,
            'parietal_integration': parietal_belief,
            'prefrontal_control': prefrontal_belief,
            'top_belief': hierarchical_output.get('top_belief', prefrontal_belief),
            'sensory_prediction': hierarchical_output['sensory_prediction'],
            'total_prediction_error': total_error,
            'hierarchical_state': hierarchical_output
        }
    
    def get_layer_activity(self, region: str, layer: str) -> torch.Tensor:
        """Get activity from specific layer of specific region"""
        level = self.region_mapping.get(region, 0)
        layer_activity = getattr(self.hierarchy.levels[level], layer).activity
        return layer_activity
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Comprehensive diagnostics"""
        recent_errors = list(self.error_history)[-100:] if self.error_history else [0]
        
        diagnostics = {
            'average_prediction_error': np.mean(recent_errors),
            'error_trend': np.polyfit(range(len(recent_errors)), recent_errors, 1)[0] if len(recent_errors) > 1 else 0,
            'levels': {}
        }
        
        # Per-level diagnostics
        for i in range(self.hierarchy.n_levels):
            level_name = ['sensory', 'parietal', 'prefrontal'][i]
            diagnostics['levels'][level_name] = {
                'L2_3_activity': float(torch.mean(self.hierarchy.levels[i].L2_3.activity)),
                'L5_belief': float(torch.mean(self.hierarchy.levels[i].L5.activity)),
                'precision': float(self.hierarchy.base_precision[i])
            }
        
        return diagnostics

# Testing functions
def test_unified_neocortex():
    """Test unified neocortex functionality"""
    print("\nTesting UnifiedNeocortex42PyTorch...")
    
    neocortex = UnifiedNeocortex42PyTorch()
    
    # Test with different inputs
    test_scenarios = [
        {"name": "Clear sensory input", "noise": 0.1},
        {"name": "Noisy sensory input", "noise": 0.5},
        {"name": "Ambiguous input", "noise": 1.0}
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario['name']}")
        
        # Generate a stable target pattern
        true_pattern = torch.randn(64, device=neocortex.device)

        # Run multiple steps to allow adaptation
        num_steps = 200  # you can increase to see more learning
        base_dt = 0.001  # Define base time step

        for step in range(num_steps):
            outputs = None  # ensure defined even if we early-continue
            # add noise each step to simulate realistic input
            noise = torch.randn_like(true_pattern) * scenario['noise']
            sensory_input = true_pattern + noise

            # Adaptive learning rate - reduce as error decreases
            if step > 50:
                current_dt = base_dt * 0.5  # Slower learning for fine-tuning
            else:
                current_dt = base_dt

            # forward pass through the neocortex
            output = neocortex(sensory_input, dt=current_dt)

            # print progress periodically
            if step % 10 == 0:
                err = output['total_prediction_error']
                print(f"  Step {step}/{num_steps}: Prediction error = {err:.6f}")

        # after training steps, test on clean input
        final_output = neocortex(true_pattern, dt=0.001)
        final_err = final_output['total_prediction_error']
        print(f"  Final clean test error: {final_err:.6f}")
        print(f"  Final encoding norm: {torch.norm(final_output['sensory_encoding']):.3f}")
        # Handle size mismatch for cosine similarity
        pred = final_output['sensory_prediction']
        if pred.shape[0] != true_pattern.shape[0]:
            if pred.shape[0] < true_pattern.shape[0]:
                pred = F.pad(pred, (0, true_pattern.shape[0] - pred.shape[0]))
            else:
                pred = pred[:true_pattern.shape[0]]
        print(f"  Prediction quality: {torch.cosine_similarity(pred, true_pattern, dim=0):.3f}")
    
    # Test layer-specific access
    print("\nTesting layer access:")
    for region in ['sensory', 'parietal', 'prefrontal']:
        L5_activity = neocortex.get_layer_activity(region, 'L5')
        print(f"  {region} L5 mean activity: {torch.mean(L5_activity):.3f}")
    
    # Get diagnostics
    diagnostics = neocortex.get_diagnostics()
    print(f"\nDiagnostics:")
    print(f"  Average prediction error: {diagnostics['average_prediction_error']:.4f}")
    for level, stats in diagnostics['levels'].items():
        print(f"  {level}: L2/3={stats['L2_3_activity']:.3f}, L5={stats['L5_belief']:.3f}")

if __name__ == "__main__":
    print("=" * 80)
    print("CORTEX 4.2 Unified Neocortex - Hierarchical Predictive Processing")
    print("=" * 80)
    
    test_unified_neocortex()
    
    print("\n" + "=" * 80)
    print("CORTEX 4.2 Unified Neocortex Implementation Complete!")
    print("=" * 80)
    print("Implemented Features:")
    print("  • Canonical 6-layer microcircuit")
    print("  • Hierarchical predictive coding")
    print("  • Precision-weighted updates")
    print("  • Bottom-up errors & top-down predictions")
    print("  • Integration with CORTEX 4.2 modulators")
    print("  • Unified processing across all cortical areas")