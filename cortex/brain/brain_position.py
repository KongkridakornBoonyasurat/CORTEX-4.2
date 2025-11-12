# cortex/brain/brain_position.py
"""
Brain Position Module for CORTEX 4.2
====================================
Provides spatial coordinates and connectivity for biologically realistic brain geometry.
Based on Harvard-Oxford atlas data, integrates seamlessly with existing CORTEX modules.

Usage:
    from cortex.brain_position import get_brain_coordinator
    
    coordinator = get_brain_coordinator(device='cuda')
    delay = coordinator.get_conduction_delay('PFC', 'M1')
"""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
import math

class BrainPositionCoordinator:
    """
    Manages 3D spatial coordinates and connectivity for CORTEX 4.2 brain regions.
    Integrates Harvard-Oxford atlas data with existing neural architecture.
    """
    
    def __init__(self, device='cpu', scale_factor=0.1):
        self.device = device
        self.scale_factor = scale_factor
        
        # Brain coordinates from Harvard-Oxford atlas (scaled for simulation)
        self.region_coordinates = {
            'PFC': torch.tensor([0.261, 5.255, 0.787], device=device),
            'M1': torch.tensor([-0.010, -1.139, 4.983], device=device),
            'S1': torch.tensor([-0.015, -2.656, 0.268], device=device),
            'INS': torch.tensor([0.076, 0.187, -0.005], device=device),
            'PAR': torch.tensor([0.016, -7.475, -0.174], device=device),
            'THAL': torch.tensor([-0.982, -1.921, 0.622], device=device),
            'HPC': torch.tensor([-2.484, -2.218, -1.440], device=device),
            'AMY_LA': torch.tensor([-2.257, -0.487, -1.807], device=device),
            'AMY_CeA': torch.tensor([-2.257, -0.287, -1.907], device=device),
            'BG_Caudate': torch.tensor([-1.269, 0.924, 0.969], device=device),
            'BG_Putamen': torch.tensor([-2.486, 0.057, 0.037], device=device),
            'HPC_CA3': torch.tensor([-2.284, -2.218, -1.340], device=device),
            'HPC_CA1': torch.tensor([-2.684, -2.218, -1.540], device=device),
            'CB': torch.tensor([0.056, -3.104, -3.417], device=device),
        }
        
        # Region radii (approximate size of each brain region)
        self.region_radii = {
            'PFC': 1.5, 'M1': 1.2, 'S1': 1.2, 'INS': 0.8, 'PAR': 1.3,
            'THAL': 0.8, 'HPC': 1.0, 'AMY_LA': 0.6, 'AMY_CeA': 0.5,
            'BG_Caudate': 0.9, 'BG_Putamen': 0.9, 'HPC_CA3': 0.8,
            'HPC_CA1': 0.8, 'CB': 1.5
        }
        
        # Conduction velocity for calculating delays (mm/ms)
        self.conduction_velocity = 0.06  # 60 m/s = 0.06 mm/ms
        
        # Pre-computed distance matrix for efficiency
        self.distance_matrix = self._compute_distance_matrix()
        self.delay_matrix = self._compute_delay_matrix()
        
    def _compute_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """Compute distances between all region pairs"""
        distances = {}
        regions = list(self.region_coordinates.keys())
        
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i <= j:
                    coord1 = self.region_coordinates[region1]
                    coord2 = self.region_coordinates[region2]
                    dist = torch.norm(coord1 - coord2).item()
                    distances[(region1, region2)] = dist
                    if i != j:  # Symmetric
                        distances[(region2, region1)] = dist
        
        return distances
    
    def _compute_delay_matrix(self) -> Dict[Tuple[str, str], float]:
        """Compute conduction delays between all region pairs"""
        delays = {}
        for (region1, region2), distance in self.distance_matrix.items():
            # Delay = distance / velocity, with minimum delay of 0.1ms
            delay = max(distance / self.conduction_velocity, 0.1)
            delays[(region1, region2)] = delay
        
        return delays
    
    def get_region_center(self, region_name: str) -> torch.Tensor:
        """Get the center coordinates of a brain region"""
        if region_name not in self.region_coordinates:
            raise ValueError(f"Unknown region: {region_name}")
        return self.region_coordinates[region_name].clone()
    
    def get_distance(self, region1: str, region2: str) -> float:
        """Get distance between two brain regions"""
        return self.distance_matrix.get((region1, region2), 0.0)
    
    def get_conduction_delay(self, region1: str, region2: str) -> float:
        """Get conduction delay between two brain regions (in ms)"""
        return self.delay_matrix.get((region1, region2), 0.1)
    
    def generate_neuron_positions(self, region_name: str, num_neurons: int, 
                                layer_structure: Optional[Dict] = None) -> torch.Tensor:
        """
        Generate 3D positions for neurons within a brain region.
        
        Args:
            region_name: Name of the brain region
            num_neurons: Number of neurons to position
            layer_structure: Optional dict specifying cortical layers
            
        Returns:
            Tensor of shape (num_neurons, 3) with x,y,z coordinates
        """
        center = self.get_region_center(region_name)
        radius = self.region_radii.get(region_name, 1.0)
        
        if layer_structure is None:
            # Simple spherical distribution
            positions = self._generate_spherical_positions(center, radius, num_neurons)
        else:
            # Layered cortical structure (for cortical regions)
            positions = self._generate_layered_positions(center, radius, num_neurons, layer_structure)
            
        return positions
    
    def _generate_spherical_positions(self, center: torch.Tensor, radius: float, 
                                    num_neurons: int) -> torch.Tensor:
        """Generate neurons in a spherical distribution around center"""
        # Generate random points in sphere using normal distribution
        positions = torch.randn(num_neurons, 3, device=self.device)
        
        # Normalize to unit sphere
        norms = torch.norm(positions, dim=1, keepdim=True)
        positions = positions / norms
        
        # Scale by random radius (cube root for uniform volume distribution)
        radii = radius * torch.pow(torch.rand(num_neurons, 1, device=self.device), 1/3)
        positions = positions * radii
        
        # Translate to region center
        positions = positions + center.unsqueeze(0)
        
        return positions
    
    def _generate_layered_positions(self, center: torch.Tensor, radius: float,
                                  num_neurons: int, layer_structure: Dict) -> torch.Tensor:
        """Generate neurons in cortical layers (for EEG-compatible positioning)"""
        positions = []
        neurons_assigned = 0
        
        # Default cortical layers if not specified
        if 'layers' not in layer_structure:
            layers = {
                'L1': 0.05,   # 5% in layer 1
                'L2_3': 0.30, # 30% in layers 2/3  
                'L4': 0.20,   # 20% in layer 4
                'L5': 0.30,   # 30% in layer 5 (main EEG contributors)
                'L6': 0.15    # 15% in layer 6
            }
        else:
            layers = layer_structure['layers']
        
        layer_depths = [0.1, 0.3, 0.5, 0.7, 0.9]  # Relative depths from surface
        
        for i, (layer_name, proportion) in enumerate(layers.items()):
            layer_neurons = int(num_neurons * proportion)
            if neurons_assigned + layer_neurons > num_neurons:
                layer_neurons = num_neurons - neurons_assigned
            
            if layer_neurons > 0:
                # Generate positions in this layer
                layer_center = center.clone()
                layer_center[2] += (layer_depths[i] - 0.5) * radius  # Adjust depth
                
                layer_positions = self._generate_spherical_positions(
                    layer_center, radius * 0.8, layer_neurons
                )
                positions.append(layer_positions)
                neurons_assigned += layer_neurons
        
        return torch.cat(positions, dim=0) if positions else torch.zeros(0, 3, device=self.device)
    
    def calculate_connection_probability(self, region1: str, region2: str, 
                                       base_probability: float = 0.1) -> float:
        """Calculate connection probability based on distance"""
        distance = self.get_distance(region1, region2)
        
        # Exponential decay with distance
        prob = base_probability * math.exp(-distance / 5.0)
        return min(prob, 1.0)
    
    def get_region_mapping(self) -> Dict[str, str]:
        """Map region names to CORTEX 4.2 components"""
        return {
            'PFC': 'prefrontal_cortex',
            'M1': 'motor_cortex', 
            'S1': 'sensory_cortex',
            'INS': 'insula',
            'PAR': 'parietal_cortex',
            'THAL': 'thalamus',
            'HPC': 'hippocampus',
            'HPC_CA3': 'hippocampus',
            'HPC_CA1': 'hippocampus', 
            'AMY_LA': 'amygdala',
            'AMY_CeA': 'amygdala',
            'BG_Caudate': 'basal_ganglia',
            'BG_Putamen': 'basal_ganglia',
            'CB': 'cerebellum'
        }
    
    def export_coordinates_for_eeg(self) -> Dict:
        """Export coordinates in format suitable for EEG calculation"""
        eeg_data = {
            'region_centers': {},
            'electrode_positions': self._generate_eeg_electrode_positions(),
            'scale_factor': self.scale_factor
        }
        
        for region, coords in self.region_coordinates.items():
            eeg_data['region_centers'][region] = coords.cpu().numpy().tolist()
            
        return eeg_data
    
    def _generate_eeg_electrode_positions(self) -> torch.Tensor:
        """Generate standard EEG electrode positions (10-20 system)"""
        # Simplified 10-20 electrode positions (scaled to match brain coordinates)
        electrode_positions = torch.tensor([
            [0.0, 8.0, 6.0],    # Fz
            [-6.0, 6.0, 5.0],   # F3
            [6.0, 6.0, 5.0],    # F4
            [0.0, 0.0, 8.0],    # Cz
            [-6.0, 0.0, 6.0],   # C3
            [6.0, 0.0, 6.0],    # C4
            [0.0, -6.0, 6.0],   # Pz
            [-6.0, -6.0, 5.0],  # P3
            [6.0, -6.0, 5.0],   # P4
            [0.0, -8.0, 4.0],   # Oz
        ], device=self.device) * self.scale_factor
        
        return electrode_positions
    
    def get_summary(self) -> Dict:
        """Get summary of spatial brain organization"""
        regions = list(self.region_coordinates.keys())
        avg_distance = np.mean(list(self.distance_matrix.values()))
        
        return {
            'num_regions': len(regions),
            'region_names': regions,
            'average_distance': avg_distance,
            'distance_range': (min(self.distance_matrix.values()), 
                             max(self.distance_matrix.values())),
            'delay_range_ms': (min(self.delay_matrix.values()),
                             max(self.delay_matrix.values())),
            'scale_factor': self.scale_factor,
            'coordinate_bounds': self._get_coordinate_bounds()
        }
    
    def _get_coordinate_bounds(self) -> Dict:
        """Get the spatial bounds of all brain regions"""
        all_coords = torch.stack(list(self.region_coordinates.values()))
        return {
            'x_range': (all_coords[:, 0].min().item(), all_coords[:, 0].max().item()),
            'y_range': (all_coords[:, 1].min().item(), all_coords[:, 1].max().item()),
            'z_range': (all_coords[:, 2].min().item(), all_coords[:, 2].max().item())
        }


# Global coordinator instance for easy import
_global_coordinator = None

def get_brain_coordinator(device='cpu', scale_factor=0.1, reset=False):
    """
    Get the global brain position coordinator instance.
    
    Args:
        device: PyTorch device ('cpu', 'cuda', etc.)
        scale_factor: Scaling factor for coordinates (default 0.1)
        reset: Force creation of new coordinator
        
    Returns:
        BrainPositionCoordinator instance
    """
    global _global_coordinator
    
    if _global_coordinator is None or reset:
        _global_coordinator = BrainPositionCoordinator(device=device, scale_factor=scale_factor)
    
    return _global_coordinator

def get_region_delay(source_region: str, target_region: str) -> float:
    """Quick function to get conduction delay between regions"""
    coordinator = get_brain_coordinator()
    return coordinator.get_conduction_delay(source_region, target_region)

def get_region_distance(source_region: str, target_region: str) -> float:
    """Quick function to get distance between regions"""
    coordinator = get_brain_coordinator()
    return coordinator.get_distance(source_region, target_region)

def get_region_center(region_name: str) -> torch.Tensor:
    """Quick function to get region center coordinates"""
    coordinator = get_brain_coordinator()
    return coordinator.get_region_center(region_name)

def generate_region_neurons(region_name: str, num_neurons: int, 
                          layer_structure: Optional[Dict] = None) -> torch.Tensor:
    """Quick function to generate neuron positions in a region"""
    coordinator = get_brain_coordinator()
    return coordinator.generate_neuron_positions(region_name, num_neurons, layer_structure)

# Test and demonstration
if __name__ == "__main__":
    print("=== Brain Position Module Test ===")
    
    # Test the module interface
    delay = get_region_delay('PFC', 'M1')
    distance = get_region_distance('PFC', 'M1')
    center = get_region_center('PFC')
    
    print(f"PFC to M1 delay: {delay:.2f} ms")
    print(f"PFC to M1 distance: {distance:.2f} mm")
    print(f"PFC center: {center}")
    
    # Test neuron generation
    positions = generate_region_neurons('PFC', 32)  # 32 neurons like your config
    print(f"Generated {len(positions)} PFC neurons")
    
    # Test coordinator access
    coordinator = get_brain_coordinator()
    summary = coordinator.get_summary()
    print(f"Brain regions available: {len(summary['region_names'])}")
    print(f"Available regions: {summary['region_names']}")
    
    print("Module ready for integration with CORTEX 4.2!")