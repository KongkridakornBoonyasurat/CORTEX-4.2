# pong_AI_runner.py
# AI Brain + Pong Integration
# Fast training with biological output compatibility

import os, sys, math, json, time, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

# Import your AI brain and Pong environment
sys.path.append(os.path.join(os.path.dirname(__file__), "game"))
from pong_env import PongEnv
from cortex.brain.AI.AI_brain import create_ai_brain_for_pong, EEGSynthesizer

class PongBrainAdapter:
    """Adapter for AI Brain + Pong Environment"""
    
    def __init__(self, device=None, verbose=False):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # Create Pong-optimized brain
        self.brain = create_ai_brain_for_pong(device=self.device, verbose=self.verbose)
        self.eeg = EEGSynthesizer(n_regions=10, device=self.device)
        self.prev_V = 0.0
        
    def _safe_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
        
    def forward_and_learn(self, obs: np.ndarray, reward: float) -> Dict[str, Any]:
        """Process Pong observation and return action"""
        
        # Convert Pong observation (84x84x3) to grayscale tensor
        if len(obs.shape) == 3:
            gray = np.mean(obs, axis=2)  # RGB to grayscale
        else:
            gray = obs
            
        # Ensure 84x84 and normalize
        if gray.shape != (84, 84):
            # Simple resize if needed
            gray = np.array(gray, dtype=np.float32)
            
        # Normalize and convert to tensor
        gray = np.clip(gray / 255.0, 0.0, 1.0)
        sensory_t = self._safe_tensor(gray)
        
        # Two-pass brain processing
        out1 = self.brain(sensory_t, reward=0.0)  # Baseline pass
        
        # Extract value estimate
        bg = out1.get('basal_ganglia', {})
        if isinstance(bg, dict) and 'value_estimate' in bg:
            V_t = float(bg['value_estimate'])
        else:
            V_t = 0.0
            
        # TD error for dopamine
        gamma = 0.99
        delta = reward + gamma * V_t - self.prev_V
        self.prev_V = V_t
        
        # Second pass with real reward
        out2 = self.brain(sensory_t, reward=reward)
        
        # Select action (0=up, 1=stay, 2=down)
        action = self._select_pong_action(out2)
        
        # Collect region activities
        region_activities = self._collect_region_activity(out2)
        
        # Get modulators
        modulators = out2.get('neuromodulators', {
            'dopamine': 0.5, 'acetylcholine': 0.5, 'norepinephrine': 0.5
        })
        
        return {
            "outputs": out2,
            "action": action,
            "V_t": V_t,
            "delta": delta,
            "modulators": modulators,
            "region_activity": region_activities
        }
    
    def _select_pong_action(self, out: Dict[str, Any]) -> int:
        """Select Pong action (0=up, 1=stay, 2=down)"""
        
        # Try basal ganglia first
        bg = out.get('basal_ganglia', {})
        if isinstance(bg, dict) and 'action_probabilities' in bg:
            probs = bg['action_probabilities']
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            
            if len(probs) >= 3:
                # Ensure valid probabilities
                probs = np.array(probs[:3], dtype=np.float32)
                probs = probs / (np.sum(probs) + 1e-9)
                
                try:
                    return int(np.random.choice(3, p=probs))
                except:
                    return int(np.argmax(probs))
        
        # Try motor cortex
        motor = out.get('motor', {})
        if isinstance(motor, dict) and 'selected_action' in motor:
            action = int(motor['selected_action'])
            return min(2, max(0, action))  # Clamp to 0-2
            
        # Fallback: random action
        return np.random.randint(0, 3)
        
    def _collect_region_activity(self, out: Dict[str, Any]) -> Dict[str, float]:
        """Collect neural activity from each region"""
        regions = ['motor', 'basal_ganglia', 'pfc', 'sensory', 'thalamus', 
                  'hippocampus', 'amygdala', 'limbic', 'insula', 'parietal']
        
        activities = {}
        for region in regions:
            r_out = out.get(region, {})
            if isinstance(r_out, dict) and 'neural_activity' in r_out:
                activities[region] = float(r_out['neural_activity'])
            else:
                activities[region] = 0.0
                
        return activities

def run_pong_ai(args):
    """Main Pong AI training loop"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize environment and brain
    env = PongEnv(width=84, height=84, render_mode=args.render)
    brain_adapter = PongBrainAdapter(device=device, verbose=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    
    print(f"Starting Pong AI training for {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < args.max_steps:
            # Brain processes observation and selects action
            brain_output = brain_adapter.forward_and_learn(obs, reward=total_reward)
            action = brain_output["action"]
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Optional rendering
            if args.render:
                env.render()
                
        # Episode complete
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Progress reporting
        if episode % 10 == 0 or episode == args.episodes - 1:
            recent_rewards = episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(episode_lengths[-10:])
            
            modulators = brain_output["modulators"]
            
            print(f"Episode {episode:3d}: "
                  f"Reward={total_reward:5.1f}, "
                  f"Avg10={avg_reward:5.1f}, "
                  f"Steps={steps:3d}, "
                  f"DA={modulators.get('dopamine', 0):.2f}")
    
    env.close()
    
    # Final statistics
    print(f"\nTraining Complete!")
    print(f"Total episodes: {args.episodes}")
    print(f"Final average reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Best single episode: {max(episode_rewards):.2f}")

def main():
    parser = argparse.ArgumentParser(description="Pong AI Training")
    parser.add_argument("--episodes", type=int, default=100, 
                       help="Number of episodes to train")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                       help="Show game window")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    run_pong_ai(args)

if __name__ == "__main__":
    main()