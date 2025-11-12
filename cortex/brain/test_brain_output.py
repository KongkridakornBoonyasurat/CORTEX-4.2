"""
Test script - put this in your cortex/brain/ folder and run it
This will show you EXACTLY what each region is returning
"""
import torch
import sys
sys.path.insert(0, '.')  # Adjust path as needed

from cortex_brain import create_small_brain_for_testing

print("Creating brain...")
brain = create_small_brain_for_testing(device=torch.device('cpu'))

print("Running forward pass...")
dummy_input = torch.randn(64)
output = brain.forward(dummy_input, reward=0.0)

print("\n" + "=" * 70)
print("CHECKING WHAT EACH REGION RETURNS:")
print("=" * 70)

regions_to_check = ['sensory', 'cerebellum', 'hippocampus', 'insula']

for region_name in regions_to_check:
    print(f"\n{region_name.upper()}:")
    if region_name in output:
        region_dict = output[region_name]
        print(f"  Type: {type(region_dict)}")
        if isinstance(region_dict, dict):
            print(f"  Keys: {list(region_dict.keys())}")
            
            # Check for neural_activity
            if 'neural_activity' in region_dict:
                val = region_dict['neural_activity']
                print(f"  ✅ HAS 'neural_activity': {val} (type: {type(val)})")
            else:
                print(f"  ❌ MISSING 'neural_activity'")
            
            # Check for neural_activity_norm
            if 'neural_activity_norm' in region_dict:
                val = region_dict['neural_activity_norm']
                print(f"  ✅ HAS 'neural_activity_norm': {val} (type: {type(val)})")
            else:
                print(f"  ❌ MISSING 'neural_activity_norm'")
                
            # Show first few keys and their types
            print(f"  First 3 items:")
            for i, (k, v) in enumerate(list(region_dict.items())[:3]):
                print(f"    {k}: {type(v)}")
        else:
            print(f"  Not a dict! Value: {region_dict}")
    else:
        print(f"  ❌ KEY '{region_name}' NOT IN OUTPUT!")
        print(f"  Available keys: {list(output.keys())}")

print("\n" + "=" * 70)
print("DONE! If you see 'neural_activity' with value 0.0, that's the problem!")
print("=" * 70)