# CORTEX 4.2 — Biophysically Grounded Brain Simulation Platform
---
This project was developed as part of the **Undergraduate Senior Thesis** in  
**Biomedical Engineering, King Mongkut’s Institute of Technology Ladkrabang (KMITL), Thailand.**

**Project Title:** *CORTEX 4.2: A Biophysically Grounded Brain Simulation Platform for Biomedical Neural Modeling*  
**Authors:** Kongkridakorn Boonyasurat, Punnut Phoungmalai   
**Advisor:** Prof. Dr. Chuchart Pintavirooj

The CORTEX 4.2 framework was designed to explore biologically grounded spiking neural networks and synthetic EEG generation for biomedical applications.  
This open-source repository accompanies the official thesis submission and demonstration presented to the KMITL Biomedical Engineering Department, 2025.
---

**CORTEX 4.2** is an open-source, GPU-accelerated brain simulation and neuromorphic AI framework designed for **biomedical neural modeling**, **synthetic EEG generation**, and **reinforcement-learning-driven behavior**.
It integrates **multi-compartment neuron models**, **astrocyte–neuron coupling**, and **tri-modulator (DA/ACh/NE) plasticity** inside a scalable, PyTorch-based architecture.

---

## Key Features

* **Biophysically grounded neurons:** Conductance-based multi-compartment model (CAdEx-derived) with dendritic processing.
* **Enhanced synapses 4.2:** STDP + tri-modulator plasticity with dopamine, acetylcholine, and norepinephrine control.
* **Astrocyte coupling:** Optional Ca²⁺-mediated modulation for slow plasticity regulation.
* **GPU acceleration:** Fully implemented in PyTorch for real-time training and visualization.
* **Regional brain architecture:** Modular simulation of sensory, motor, prefrontal, hippocampal, limbic, cerebellar, and thalamic systems.
* **Neuromodulation & oscillations:** Built-in oscillators and dynamic modulators for realistic activity states.
* **Synthetic EEG generation:** Multi-scale activity exported through MNE + PyVista to 3-D cortical visualization.
* **Behavioral learning:** Functional testing in classic reinforcement-learning games (Pac-Man, Pong, Snake).
* **Biomedical validation:** EEG-like outputs, functional-connectivity matrices, power-spectral analysis, and regional correlation benchmarking.
* **Open-source & educational:** Licensed under Apache 2.0 for reproducible research, teaching, and neuromorphic AI prototyping.

---

## System Architecture Overview

```
cortex/
├── cells/               # Enhanced neurons, synapses, astrocytes
├── modulation/          # Dopamine, acetylcholine, norepinephrine, oscillators
├── regions/             # Cortical & subcortical modules (motor, limbic, PFC, etc.)
├── connectivity/        # Biological connection patterns & wiring
├── sensory/             # Spike-based visual & sensory front-ends
└── brain/               # High-level orchestration, data collection, demos
```

**Core engine:**

* `EnhancedNeuron42PyTorch` & `EnhancedSynapse42PyTorch` — biologically accurate, GPU-optimized cell models.
* `CortexBrain.py` — assembles regional modules into a functional brain network.
* `BMEDataCollector.py` — logs spikes, synaptic weights, EEG spectra, and connectivity matrices.
* `MNE_EEG_3D_Regions_Patched.py` — visualizes synthetic EEG with MNE + PyVista.

---

## Supported Brain Regions

Sensory Cortex (42) – Responsible for perception, input encoding, and sensory feature extraction.
Influenced mainly by acetylcholine (ACh) and glutamate, enhancing attention and sensory gain.

Motor Cortex (42) – Generates movement output and supports motor learning and procedural plasticity.
Modulated by dopamine (DA) and acetylcholine (ACh) to reinforce skill acquisition.

Prefrontal Cortex (42) – Governs working memory, planning, and top-down decision control.
Driven by combined modulation of dopamine (DA), norepinephrine (NE), and acetylcholine (ACh).

Hippocampus (42) – Encodes episodic memory, spatial navigation, and sequential replay.
Primarily shaped by dopamine (DA) and acetylcholine (ACh) activity.

Limbic / Amygdala System (42) – Processes emotion, reward, and fear conditioning.
Regulated by dopamine (DA) and norepinephrine (NE) for adaptive behavioral responses.

Basal Ganglia System (42) – Implements action selection, reinforcement learning, and habit formation.
Heavily dependent on dopamine (DA) signaling through D1/D2 pathways.

Cerebellum (42) – Handles coordination, timing, and predictive correction of movement.
Influenced by acetylcholine (ACh) and excitatory glutamate circuits.

Thalamus System (42) – Acts as a relay and attentional gate between sensory and cortical regions.
Modulated by norepinephrine (NE) and acetylcholine (ACh) to regulate arousal and focus.

Parietal Cortex (42) – Integrates spatial awareness, proprioception, and sensorimotor alignment.
Modulated by ACh and NE for attention and spatial mapping.

Insula Cortex (42) – Monitors internal body states, emotion, and risk or pain evaluation.
Receives modulation from norepinephrine (NE) and dopamine (DA).

Unified Neocortex (42) – Provides a generalized cortical microcircuit scaffold used by higher regions.
Controlled by the tri-modulator system of dopamine (DA), acetylcholine (ACh), and norepinephrine (NE).

---

## Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/<yourusername>/CORTEX-4.2.git
cd CORTEX-4.2
pip install -r requirements.txt
```

*(For smaller environments, use the minimal requirements list if provided.)*

---

## Running Simulations

### 1. Pac-Man Behavior Demo

```bash
python -m cortex.brain.pacman_3
```

Simulates a closed-loop brain → environment → reward cycle with adaptive spiking activity and neuromodulated learning.

### 2. Minimal Smoke Test

```bash
python -m cortex.brain.brain42_smoke
```

Checks region integration, GPU setup, and “proof-of-life” firing activity.

### 3. EEG Visualization

```bash
python -m cortex.brain.mne_eeg_3d_regions_patched
```

Generates synthetic EEG, applies inverse mapping, and renders 3-D cortical activation.

---

## Validation

CORTEX 4.2 produces:

* Testing the Synapse, Neurons and Astrocyte agaist Allen brain.
* Showing the synthetic EEG when playing pacman game.

---

## Core Dependencies

* **PyTorch**, NumPy, SciPy, Pandas
* **Matplotlib**, OpenCV, ImageIO
* **MNE**, PyVista, PyVistaQt (for EEG 3-D visualization)
* **pygame** (for interactive environments)
* *(Optional)* braincog, spikingjelly, gym for research extensions

---

## License

Licensed under the **Apache License 2.0** (see [LICENSE](LICENSE))
© 2025 Kongkridakorn Boonyasurat.
You are free to use, modify, and distribute this software for academic or commercial purposes, provided that proper attribution is maintained.

---

Would you like me to append a short “Quick Project Structure” diagram (with emojis + descriptions for each folder) or keep this README summary clean and academic-style like above?
