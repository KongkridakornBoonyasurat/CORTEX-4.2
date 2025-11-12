import nibabel as nib
import numpy as np
from nilearn import datasets

# Load the atlas you just downloaded
atlas_cortical = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
atlas_subcortical = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-1mm')

# Load the actual brain images
cort_img = nib.load(atlas_cortical.filename)
subcort_img = nib.load(atlas_subcortical.filename)

# Get the 3D data arrays
cort_data = cort_img.get_fdata()
subcort_data = subcort_img.get_fdata()

print(f"Cortical atlas shape: {cort_data.shape}")
print(f"Subcortical atlas shape: {subcort_data.shape}")

def get_region_center(data, region_index):
    """Find the center coordinates of a brain region"""
    # Create a mask where this region exists
    region_mask = (data == region_index)
    
    # Find all voxel coordinates where region exists
    coords = np.where(region_mask)
    
    if len(coords[0]) == 0:
        return None  # Region not found
    
    # Calculate center-of-mass (average position)
    center_x = np.mean(coords[0])
    center_y = np.mean(coords[1]) 
    center_z = np.mean(coords[2])
    
    return (center_x, center_y, center_z)

def voxel_to_mm(voxel_coords, img):
    """Convert voxel coordinates to real-world mm coordinates"""
    # Get the transformation matrix from the image
    affine = img.affine
    
    # Convert voxel to mm using affine transformation
    voxel_coords_homogeneous = np.array([voxel_coords[0], voxel_coords[1], voxel_coords[2], 1])
    mm_coords = affine.dot(voxel_coords_homogeneous)[:3]
    
    return mm_coords

# Define your COMPLETE 12 region mapping for CORTEX 4.2
your_regions = {
    # Main cortical regions
    'PFC': {'atlas': 'cortical', 'index': 1, 'name': 'Frontal Pole'},
    'M1': {'atlas': 'cortical', 'index': 7, 'name': 'Precentral Gyrus'},
    'S1': {'atlas': 'cortical', 'index': 10, 'name': 'Postcentral Gyrus'},
    'INS': {'atlas': 'cortical', 'index': 2, 'name': 'Insular Cortex'},
    'PAR': {'atlas': 'cortical', 'index': 23, 'name': 'Superior Parietal Lobule'},
    
    # Subcortical regions
    'THAL': {'atlas': 'subcortical', 'index': 4, 'name': 'Left Thalamus'},
    'HPC': {'atlas': 'subcortical', 'index': 9, 'name': 'Left Hippocampus'},
    'AMY_LA': {'atlas': 'subcortical', 'index': 10, 'name': 'Left Amygdala'},
    'AMY_CeA': {'atlas': 'subcortical', 'index': 10, 'name': 'Left Amygdala', 'offset': [0, 2, -1]},
    
    # Basal Ganglia
    'BG_Caudate': {'atlas': 'subcortical', 'index': 5, 'name': 'Left Caudate'},
    'BG_Putamen': {'atlas': 'subcortical', 'index': 6, 'name': 'Left Putamen'},
    
    # Hippocampus subregions (offset from main hippocampus)
    'HPC_CA3': {'atlas': 'subcortical', 'index': 9, 'name': 'Left Hippocampus', 'offset': [2, 0, 1]},
    'HPC_CA1': {'atlas': 'subcortical', 'index': 9, 'name': 'Left Hippocampus', 'offset': [-2, 0, -1]},
    
    # Cerebellum (approximated using brainstem)
    'CB': {'atlas': 'subcortical', 'index': 8, 'name': 'Brain-Stem'},
}

# Extract coordinates for all regions
brain_coordinates = {}

print("\n=== EXTRACTING COORDINATES ===")
for region_name, info in your_regions.items():
    if info['atlas'] == 'cortical':
        data = cort_data
        img = cort_img
    else:
        data = subcort_data
        img = subcort_img
    
    # Get voxel center
    voxel_center = get_region_center(data, info['index'])
    
    if voxel_center:
        # Convert to mm
        mm_coords = voxel_to_mm(voxel_center, img)
        
        # Apply offset if specified (for subregions like CA3/CA1, CeA)
        if 'offset' in info:
            mm_coords = mm_coords + np.array(info['offset'])
        
        brain_coordinates[region_name] = mm_coords
        print(f"{region_name:12}: {mm_coords}")
    else:
        print(f"{region_name:12}: Not found in atlas")

print(f"\n=== EXTRACTED {len(brain_coordinates)} REGIONS ===")

# Calculate distances between all regions
def calculate_distance(coord1, coord2):
    return np.sqrt(np.sum((coord1 - coord2)**2))

print("\n=== INTER-REGION DISTANCES (mm) ===")
region_names = list(brain_coordinates.keys())
distance_matrix = {}

for i, region1 in enumerate(region_names):
    for j, region2 in enumerate(region_names):
        if i < j:  # Avoid duplicates and self-distances
            dist = calculate_distance(brain_coordinates[region1], brain_coordinates[region2])
            distance_matrix[f"{region1}-{region2}"] = dist
            print(f"{region1:12} to {region2:12}: {dist:6.1f} mm")

# Scale coordinates for neural simulation
scale_factors = {
    'micro': 0.01,   # Very small brain (1% of real size)
    'small': 0.05,   # Small brain (5% of real size)  
    'medium': 0.1,   # Medium brain (10% of real size)
    'large': 0.2     # Large brain (20% of real size)
}

print("\n=== SCALED COORDINATES FOR SIMULATION ===")
for scale_name, scale_factor in scale_factors.items():
    print(f"\n{scale_name.upper()} scale (factor: {scale_factor}):")
    scaled_coordinates = {}
    for region, coord in brain_coordinates.items():
        scaled_coordinates[region] = coord * scale_factor
        print(f"  {region:12}: {scaled_coordinates[region]}")

# Calculate conduction delays based on distances
def calculate_conduction_delay(distance_mm, velocity_ms=60.0):
    """Calculate conduction delay in milliseconds
    
    Args:
        distance_mm: Distance in millimeters
        velocity_ms: Conduction velocity in m/s (default 60 m/s for myelinated axons)
    
    Returns:
        delay in milliseconds
    """
    return distance_mm / velocity_ms

print("\n=== CONDUCTION DELAYS (ms) ===")
print("Using 60 m/s conduction velocity for myelinated axons:")
for connection, distance in distance_matrix.items():
    delay = calculate_conduction_delay(distance)
    print(f"{connection:25}: {delay:6.2f} ms")

# Export data for use in CORTEX 4.2
print("\n=== EXPORT FOR CORTEX 4.2 ===")
print("\n# Brain coordinates (mm) - paste into your config:")
print("BRAIN_COORDINATES_MM = {")
for region, coord in brain_coordinates.items():
    print(f"    '{region}': [{coord[0]:8.2f}, {coord[1]:8.2f}, {coord[2]:8.2f}],")
print("}")

print("\n# Distance matrix (mm) - for connection delays:")
print("DISTANCE_MATRIX_MM = {")
for connection, distance in distance_matrix.items():
    print(f"    '{connection}': {distance:6.2f},")
print("}")

print("\n# Scaled coordinates (scale factor 0.1) - for neural positions:")
scale_factor = 0.1
print("SCALED_COORDINATES = {")
for region, coord in brain_coordinates.items():
    scaled = coord * scale_factor
    print(f"    '{region}': [{scaled[0]:8.3f}, {scaled[1]:8.3f}, {scaled[2]:8.3f}],")
print("}")

print("\n=== SUMMARY ===")
print(f"Total regions mapped: {len(brain_coordinates)}")
print(f"Total connections: {len(distance_matrix)}")
print(f"Average distance: {np.mean(list(distance_matrix.values())):6.1f} mm")
print(f"Min distance: {np.min(list(distance_matrix.values())):6.1f} mm")
print(f"Max distance: {np.max(list(distance_matrix.values())):6.1f} mm")

# Check which CORTEX 4.2 regions are covered
cortex_regions = ['PFC', 'HPC_CA3', 'HPC_CA1', 'AMY_LA', 'AMY_CeA', 'BG', 'THAL', 'S1', 'M1', 'INS', 'PAR', 'CB']
mapped_regions = list(brain_coordinates.keys())

print(f"\nCORTEX 4.2 region coverage:")
for region in cortex_regions:
    if any(region in mapped for mapped in mapped_regions):
        print(f"  ✓ {region}")
    else:
        print(f"  ✗ {region} - needs manual coordinates")

print("\nReady for biological brain geometry implementation!")