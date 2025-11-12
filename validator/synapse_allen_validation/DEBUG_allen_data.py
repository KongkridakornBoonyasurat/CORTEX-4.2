#!/usr/bin/env python3
"""
DEBUG_allen_data.py - FIXED

Quick script to see what's ACTUALLY in the Allen unionizes data
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache

CACHE_DIR = Path(__file__).resolve().parent / "mouse_connectivity"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("DEBUGGING ALLEN DATA STRUCTURE")
print("="*80)

mcc = MouseConnectivityCache(
    manifest_file=str(CACHE_DIR / "manifest.json"),
    resolution=100
)

structure_tree = mcc.get_structure_tree()

# Get ONE experiment
print("\n[1] Getting example experiments...")
pl_structs = structure_tree.get_structures_by_acronym(['PL'])  # Prefrontal
if pl_structs:
    pl_id = pl_structs[0]['id']
    experiments = mcc.get_experiments(injection_structure_ids=[pl_id])
    
    if experiments:
        exp_id = experiments[0]['id']
        print(f"Using experiment {exp_id}")
        
        # Get unionizes (IT'S A DATAFRAME!)
        print("\n[2] Getting unionizes...")
        unionizes = mcc.get_structure_unionizes([exp_id])
        
        print(f"Type: {type(unionizes)}")
        print(f"Shape: {unionizes.shape}")
        
        if not unionizes.empty:
            print(f"Found {len(unionizes)} unionize records")
            
            # Print ALL column names
            print("\n[3] ALL AVAILABLE COLUMNS:")
            print("="*80)
            for col in unionizes.columns:
                print(f"  - {col}")
            
            # Print FIRST ROW to see data
            print("\n[4] FIRST ROW (ALL FIELDS):")
            print("="*80)
            first_row = unionizes.iloc[0]
            for col in unionizes.columns:
                print(f"  {col}: {first_row[col]}")
            
            # Check for our target fields
            print("\n[5] CHECKING TARGET FIELDS:")
            print("="*80)
            target_fields = ['projection_density', 'normalized_projection_volume', 
                           'projection_volume', 'sum_projection_pixels', 
                           'is_injection', 'hemisphere_id', 'structure_id']
            
            for field in target_fields:
                if field in unionizes.columns:
                    sample_val = first_row[field]
                    print(f"  ✓ {field}: {sample_val} (type: {type(sample_val)})")
                else:
                    print(f"  ✗ {field}: MISSING")
            
            # Find non-injection records
            print("\n[6] NON-INJECTION RECORDS:")
            print("="*80)
            non_inj = unionizes[unionizes['is_injection'] == False]
            print(f"Found {len(non_inj)} non-injection records")
            
            if len(non_inj) > 0:
                sample = non_inj.iloc[0]
                print(f"\nSample non-injection record:")
                print(f"  Structure ID: {sample['structure_id']}")
                print(f"  Hemisphere ID: {sample['hemisphere_id']}")
                print(f"  projection_density: {sample['projection_density']}")
                
                # Check if normalized_projection_volume exists and has value
                if 'normalized_projection_volume' in unionizes.columns:
                    print(f"  normalized_projection_volume: {sample['normalized_projection_volume']}")
                
                if 'projection_volume' in unionizes.columns:
                    print(f"  projection_volume: {sample['projection_volume']}")
                
                if 'sum_projection_pixels' in unionizes.columns:
                    print(f"  sum_projection_pixels: {sample['sum_projection_pixels']}")
                
                # Check for NON-ZERO values
                print("\n[7] NON-ZERO DENSITY RECORDS:")
                print("="*80)
                nonzero = non_inj[non_inj['projection_density'] > 0]
                print(f"Found {len(nonzero)} records with projection_density > 0")
                
                if len(nonzero) > 0:
                    top5 = nonzero.nlargest(5, 'projection_density')
                    print("\nTop 5 by projection_density:")
                    for idx, row in top5.iterrows():
                        print(f"  Structure {row['structure_id']}: density={row['projection_density']:.6f}")
                        if 'normalized_projection_volume' in unionizes.columns:
                            print(f"    normalized_volume={row.get('normalized_projection_volume', 'N/A')}")
            else:
                print("NO non-injection records found!")
                
        else:
            print("Unionizes DataFrame is EMPTY!")
    else:
        print("NO experiments found!")
else:
    print("Could not find PL structure!")

print("\n" + "="*80)
print("DEBUGGING COMPLETE!")
print("="*80)