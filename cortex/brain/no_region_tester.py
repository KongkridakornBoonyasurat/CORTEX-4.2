from cortex.brain.cortex_brain_no_region import CortexBrainNoRegion42
import torch

obs_dim = 128   # set to your sensory size
act_dim = 6     # set to your action count

brain = CortexBrainNoRegion42(
    n_inputs=obs_dim,
    n_actions=act_dim,
    n_neurons=512,       # 256–512 is a good speed/learning tradeoff
    dt=1.0,              # ms/step; affects the AMPA delay steps for RWS
    use_amp=True,
    use_torch_compile=True,
    device="cuda",
    spikes_into_policy=True
)
brain.reset_state(batch_size=1)

obs = torch.zeros(1, obs_dim)
out = brain.step(obs, epsilon=0.05)
action = int(out["actions"][0])
lfp = float(out["lfp_proxy"][0])
