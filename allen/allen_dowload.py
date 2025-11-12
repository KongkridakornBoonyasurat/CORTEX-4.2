import numpy as np
import matplotlib.pyplot as plt
from allensdk.core.cell_types_cache import CellTypesCache

ctc = CellTypesCache()
cell_id = 565871768

print(f"Analyzing cell {cell_id}")

# Get electrophysiology data
data_set = ctc.get_ephys_data(cell_id)
sweeps = ctc.get_ephys_sweeps(cell_id)

print(f"Found {len(sweeps)} sweeps")
print("Available stimulus types:", set([s['stimulus_name'] for s in sweeps]))

# Look for "Long Square" sweeps (these are current step injections)
long_square_sweeps = [s for s in sweeps if s['stimulus_name'] == 'Long Square']
print(f"Found {len(long_square_sweeps)} Long Square sweeps")

if long_square_sweeps:
    # Try several sweeps to find one that works
    for sweep_info in long_square_sweeps[:5]:
        sweep_number = sweep_info['sweep_number']
        try:
            print(f"Trying sweep {sweep_number}...")
            
            # Get sweep data
            sweep_data = data_set.get_sweep(sweep_number)
            
            # Extract voltage and current
            voltage = sweep_data['response'] * 1e3  # Convert to mV  
            current = sweep_data['stimulus'] * 1e12  # Convert to pA
            sampling_rate = sweep_data['sampling_rate']
            time = np.arange(len(voltage)) / sampling_rate
            
            print(f"Success! Sweep {sweep_number}:")
            print(f"  Duration: {time[-1]:.2f} seconds")
            print(f"  Current range: {np.min(current):.0f} to {np.max(current):.0f} pA")
            print(f"  Voltage range: {np.min(voltage):.1f} to {np.max(voltage):.1f} mV")
            
            # Simple spike detection
            spike_threshold = -20  # mV
            spikes = []
            for i in range(1, len(voltage)):
                if voltage[i-1] < spike_threshold and voltage[i] >= spike_threshold:
                    spikes.append(time[i])
            
            print(f"  Detected {len(spikes)} spikes")
            firing_rate = len(spikes) / time[-1] if time[-1] > 0 else 0
            print(f"  Firing rate: {firing_rate:.1f} Hz")
            
            # Calculate ISI statistics
            if len(spikes) > 1:
                isis = np.diff(spikes) * 1000  # Convert to ms
                isi_mean = np.mean(isis)
                isi_cv = np.std(isis) / np.mean(isis)
                print(f"  Mean ISI: {isi_mean:.1f} ms")
                print(f"  ISI CV (variability): {isi_cv:.3f}")
            
            # Plot the neural data
            plt.figure(figsize=(12, 8))
            
            # Voltage trace
            plt.subplot(3, 1, 1)
            plt.plot(time, voltage, 'b-', linewidth=0.8)
            plt.axhline(spike_threshold, color='r', linestyle='--', alpha=0.7, label='Spike threshold')
            if spikes:
                plt.scatter(spikes, [spike_threshold]*len(spikes), color='red', s=30, zorder=5, label=f'{len(spikes)} spikes')
            plt.ylabel('Voltage (mV)')
            plt.title(f'Allen Institute Cortical Neuron - Cell {cell_id}, Sweep {sweep_number}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Current injection
            plt.subplot(3, 1, 2)
            plt.plot(time, current, 'g-', linewidth=1.5)
            plt.ylabel('Current (pA)')
            plt.grid(True, alpha=0.3)
            
            # ISI histogram
            plt.subplot(3, 1, 3)
            if len(spikes) > 1:
                plt.hist(isis, bins=min(20, len(isis)), alpha=0.7, color='orange', edgecolor='black')
                plt.xlabel('Inter-Spike Interval (ms)')
                plt.ylabel('Count')
                plt.title(f'ISI Distribution (n={len(isis)})')
            else:
                plt.text(0.5, 0.5, f'Only {len(spikes)} spikes detected', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.xlabel('Time (s)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'allen_neuron_validation.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"\n{'='*60}")
            print("BIOLOGICAL VALIDATION BENCHMARKS FOR YOUR MODEL")
            print(f"{'='*60}")
            print(f"Real neuron current injection: {np.min(current):.0f} to {np.max(current):.0f} pA")
            print(f"Real neuron firing rate: {firing_rate:.1f} Hz")
            print(f"Real neuron voltage range: {np.min(voltage):.1f} to {np.max(voltage):.1f} mV")
            if len(spikes) > 1:
                print(f"Real neuron ISI mean: {isi_mean:.1f} ms")
                print(f"Real neuron ISI variability: {isi_cv:.3f}")
            
            print(f"\nFOR YOUR THESIS:")
            print(f"- Compare your model neuron's firing rate vs current curve")
            print(f"- Check if your model produces similar ISI variability ({isi_cv:.3f})")
            print(f"- Verify voltage ranges are realistic ({np.min(voltage):.1f} to {np.max(voltage):.1f} mV)")
            print(f"- Show your model responds to current injection like real neurons")
            
            # Save the data for later comparison
            np.save('real_neuron_benchmarks.npy', {
                'firing_rate': firing_rate,
                'current_range': (np.min(current), np.max(current)),
                'voltage_range': (np.min(voltage), np.max(voltage)),
                'isi_mean': isi_mean if len(spikes) > 1 else None,
                'isi_cv': isi_cv if len(spikes) > 1 else None,
                'spike_times': spikes,
                'voltage_trace': voltage,
                'current_trace': current,
                'time': time
            })
            print(f"\nBenchmark data saved to 'real_neuron_benchmarks.npy'")
            
            break  # Success! Stop trying other sweeps
            
        except Exception as e:
            print(f"Sweep {sweep_number} failed: {e}")
            continue
            
else:
    print("No Long Square sweeps found. Cannot get current injection data.")
    
    # Try any available sweep type
    if sweeps:
        print("Trying first available sweep...")
        sweep_info = sweeps[0]
        sweep_number = sweep_info['sweep_number']
        
        try:
            sweep_data = data_set.get_sweep(sweep_number)
            voltage = sweep_data['response'] * 1e3
            current = sweep_data['stimulus'] * 1e12
            time = np.arange(len(voltage)) / sweep_data['sampling_rate']
            
            print(f"Got data from sweep {sweep_number} ({sweep_info['stimulus_name']})")
            print(f"This shows basic neural activity patterns for comparison")
            
            # Basic plot
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.plot(time, voltage, 'b-', linewidth=0.8)
            plt.ylabel('Voltage (mV)')
            plt.title(f'Allen Institute Neuron Activity - Cell {cell_id}')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(time, current, 'g-', linewidth=1)
            plt.ylabel('Current (pA)')
            plt.xlabel('Time (s)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'allen_neuron_basic.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Could not access any sweep data: {e}")