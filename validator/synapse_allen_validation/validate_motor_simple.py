# validate_motor_simple.py
"""
Simple Motor Cortex Validation - Firing Rates Only
Compares your CORTEX motor cortex to Dura-Bernal et al. 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# LOAD YOUR DATA
# ============================================================================

print("="*60)
print("MOTOR CORTEX VALIDATION - Firing Rates")
print("="*60)

# Path to your data
DATA_PATH = r"C:\Users\User\Desktop\Brain AI\cortex 4.2\cortex 4.2 v42\pacman_run_20251013_103209"

# Load region activity (motor is column 0)
region_activity = np.load(f"{DATA_PATH}/overall_region_activity.npy")
motor_activity = region_activity[:, 0]  # Motor cortex only

# Load episodes data
episodes = pd.read_csv(f"{DATA_PATH}/episodes_overall.csv")

print(f"\nLoaded {len(motor_activity)} timesteps")
print(f"Loaded {len(episodes)} episodes")

# ============================================================================
# IDENTIFY BASELINE VS MOVEMENT PERIODS
# ============================================================================

# Load dopamine (high dopamine = reward/movement)
dopamine = np.load(f"{DATA_PATH}/overall_dopamine.npy")

# Simple classification:
# Baseline = low dopamine (quiet periods)
# Movement = high dopamine (active periods)

dopamine_threshold = np.median(dopamine)

baseline_mask = dopamine < dopamine_threshold
movement_mask = dopamine >= dopamine_threshold

# ============================================================================
# CALCULATE FIRING RATES
# ============================================================================

baseline_firing = motor_activity[baseline_mask].mean()
movement_firing = motor_activity[movement_mask].mean()
firing_ratio = movement_firing / baseline_firing if baseline_firing > 0 else 0

print(f"\n{'RESULTS':^60}")
print("="*60)
print(f"Baseline Firing:  {baseline_firing:.2f} Hz")
print(f"Movement Firing:  {movement_firing:.2f} Hz")
print(f"Ratio:            {firing_ratio:.2f}×")

# ============================================================================
# COMPARE TO DURA-BERNAL 2023
# ============================================================================

print(f"\n{'COMPARISON TO DURA-BERNAL 2023':^60}")
print("="*60)

# Targets from Dura-Bernal et al. 2023 Figure 5
TARGET_BASELINE = (5, 15)    # Hz
TARGET_MOVEMENT = (10, 30)   # Hz
TARGET_RATIO = (1.5, 3.0)    # fold increase

# Check if passing
pass_baseline = TARGET_BASELINE[0] <= baseline_firing <= TARGET_BASELINE[1]
pass_movement = TARGET_MOVEMENT[0] <= movement_firing <= TARGET_MOVEMENT[1]
pass_ratio = TARGET_RATIO[0] <= firing_ratio <= TARGET_RATIO[1]

print(f"Baseline:  {'PASS ✓' if pass_baseline else 'FAIL ✗'}")
print(f"  Target: {TARGET_BASELINE[0]}-{TARGET_BASELINE[1]} Hz")
print(f"  Yours:  {baseline_firing:.2f} Hz")

print(f"\nMovement:  {'PASS ✓' if pass_movement else 'FAIL ✗'}")
print(f"  Target: {TARGET_MOVEMENT[0]}-{TARGET_MOVEMENT[1]} Hz")
print(f"  Yours:  {movement_firing:.2f} Hz")

print(f"\nRatio:     {'PASS ✓' if pass_ratio else 'FAIL ✗'}")
print(f"  Target: {TARGET_RATIO[0]}-{TARGET_RATIO[1]}×")
print(f"  Yours:  {firing_ratio:.2f}×")

overall_pass = sum([pass_baseline, pass_movement, pass_ratio])
print(f"\n{'='*60}")
print(f"OVERALL: {overall_pass}/3 tests passing ({overall_pass/3*100:.0f}%)")
print(f"{'='*60}")

# ============================================================================
# CREATE SIMPLE COMPARISON FIGURE
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Bar comparison - Baseline
ax = axes[0]
ax.bar(['Dura-Bernal\n2023', 'Your\nCORTEX'], 
       [np.mean(TARGET_BASELINE), baseline_firing],
       color=['steelblue', 'orange'])
ax.axhspan(TARGET_BASELINE[0], TARGET_BASELINE[1], alpha=0.2, color='steelblue')
ax.set_ylabel('Firing Rate (Hz)')
ax.set_title('Baseline Firing Rate')
ax.set_ylim(0, 20)

# Plot 2: Bar comparison - Movement
ax = axes[1]
ax.bar(['Dura-Bernal\n2023', 'Your\nCORTEX'], 
       [np.mean(TARGET_MOVEMENT), movement_firing],
       color=['steelblue', 'orange'])
ax.axhspan(TARGET_MOVEMENT[0], TARGET_MOVEMENT[1], alpha=0.2, color='steelblue')
ax.set_ylabel('Firing Rate (Hz)')
ax.set_title('Movement Firing Rate')
ax.set_ylim(0, 40)

# Plot 3: Bar comparison - Ratio
ax = axes[2]
ax.bar(['Dura-Bernal\n2023', 'Your\nCORTEX'], 
       [np.mean(TARGET_RATIO), firing_ratio],
       color=['steelblue', 'orange'])
ax.axhspan(TARGET_RATIO[0], TARGET_RATIO[1], alpha=0.2, color='steelblue')
ax.set_ylabel('Fold Increase')
ax.set_title('Movement/Baseline Ratio')
ax.set_ylim(0, 4)

plt.suptitle('Motor Cortex Validation: Firing Rates\nYour CORTEX vs Dura-Bernal et al. 2023', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
output_file = f"{DATA_PATH}/motor_validation_firing_rates.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nFigure saved: motor_validation_firing_rates.png")

plt.show()

print("\nValidation complete!")