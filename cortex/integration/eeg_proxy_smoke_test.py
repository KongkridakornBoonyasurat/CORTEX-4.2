# cortex/integration/eeg_proxy_smoke_test.py
from cortex.integration.thalamo_cortical_loop_42 import ThalamoCorticalLoop42
import torch

loop = ThalamoCorticalLoop42(resolution=(84,84), thal_neurons=32, thal_channels=8)
frame = torch.rand(84,84)  # or provide your own [H,W] or [H,W,3] tensor in 0..1

for k in range(200):
    out = loop.step(frame, dt=0.001, t=0.001*k)
    if k % 20 == 0:
        print(f"step {k}  EEG proxy = {out['eeg_proxy']:.4f}")
