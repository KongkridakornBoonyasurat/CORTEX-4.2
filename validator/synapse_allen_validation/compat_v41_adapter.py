
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import matplotlib.pyplot as plt


root = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(root, "cortex")) and root not in sys.path:
    sys.path.insert(0, root)

import os, sys
_here = os.path.dirname(os.path.abspath(__file__))               
_project_root = os.path.abspath(os.path.join(_here, os.pardir))   
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from cortex import config as C
C.DEVICE_PREFERENCE = "cpu"

from cortex.cells.enhanced_neurons_42 import EnhancedNeuron42PyTorch 
import torch
from cortex.cells import enhanced_neurons_42 as EN
EN.DEVICE = torch.device("cpu")
EN.CONSTANTS = EN.make_constants(EN.DEVICE)

NEURON_ID = "L23_Pyr"  

def run_fi(neuron: EnhancedNeuron42PyTorch,
           amps_nA=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
           hold_ms=500.0,
           dt_ms=0.1,
           v_threshold_mV=0.0):
    """
    Drive your neuron with constant current steps and count spikes via neuron.step(...).
    Returns arrays: currents (nA), firing_rates (Hz).
    """
    dt = dt_ms  # pass milliseconds; EnhancedNeuron42PyTorch.forward expects ms
    steps = int(hold_ms / dt_ms)
    currents = np.array(amps_nA, dtype=float)
    rates = []

    for I in currents:
        # reset between sweeps (your class keeps state)
        if hasattr(neuron, "reset"):
            neuron.reset()

        spikes = 0
        t = 0.0
        constant_input = np.array([I], dtype=np.float32)  # shape matches single-compartment input

        for _ in range(steps):
            spk, v = neuron.step(constant_input, dt=dt, current_time=t)
            # your step returns (bool, float)
            if spk:
                spikes += 1
            t += dt

        rate_hz = spikes / (hold_ms / 1000.0)
        rates.append(rate_hz)

    return currents, np.array(rates, dtype=float)

def main():
    # Instantiate your neuron with its defaults; if you normally pass a config dict,
    # replace this with your usual constructor usage.
    neuron = EnhancedNeuron42PyTorch(neuron_id=NEURON_ID, device="cpu")

    try:
        import torch
        if hasattr(neuron, "to"):
            neuron.to("cpu")

        # Move known buffers to CPU (correct names)
        for name in (
            "ampa_conductance", "nmda_conductance",
            "gaba_a_conductance", "gaba_b_conductance",
            "dendritic_voltages", "nmda_gating_variables", "voltage"
        ):
            if hasattr(neuron, name):
                buf = getattr(neuron, name)
                if isinstance(buf, torch.Tensor):
                    setattr(neuron, name, buf.detach().cpu())
    except Exception as _e:
        print("CPU shim warning:", _e)
    # Choose a reasonable current range for pyramidal vs interneuron; adjust as needed
    currents_nA = np.linspace(0.0, 0.4, 9)  # 0..400 pA in 50 pA steps
    I, R = run_fi(neuron, amps_nA=currents_nA, hold_ms=1000.0, dt_ms=0.1)

    # Save CSV for downstream compare against Allen FI (your validator will read this)
    out_csv = "fi_model_v41.csv"
    np.savetxt(out_csv, np.c_[I, R], delimiter=",", header="current_nA,firing_rate_Hz", comments="")
    print(f"Saved: {out_csv}")

    # Quick plot
    plt.figure()
    plt.plot(I*1e3, R, marker="o")  # x in pA
    plt.xlabel("Injected current (pA)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("F–I (v41 EnhancedNeuron42PyTorch)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fi_model_v41.png", dpi=150)
    print("Saved: fi_model_v41.png")

if __name__ == "__main__":
    main()
