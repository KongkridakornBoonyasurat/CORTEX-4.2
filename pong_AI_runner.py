# pong_AI_runner.py
# AI Brain + Pong Integration
# Fast training with biological output compatibility + Adaptive Guidance

import os, sys, math, json, time, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import cv2
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
from datetime import datetime
import cv2
import os
os.environ['OPENCV_OPENCL_RUNTIME'] = ''  # Disable OpenCL
import matplotlib
import pandas as pd
from scipy import stats

# Import your AI brain and Pong environment
sys.path.append(os.path.join(os.path.dirname(__file__), "game"))
from game.pong_env import PongEnv
from cortex.brain.cortex_brain import CortexBrain42
import torch.nn as nn

# ---- Fallbacks: brain + EEG stubs so this file runs as-is ----
def create_ai_brain_for_pong(device=None, verbose=False):
    """
    Minimal brain that matches what the runner expects:
      - forward(x, reward=float) -> dict with:
          motor.action_logits (3,), basal_ganglia.value_estimate (float),
          basal_ganglia.action_probabilities (3,), and region blocks with 'neural_activity'
    """
    class SimplePongBrain(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU()
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 11 * 11, 128), nn.ReLU()
            )
            self.value = nn.Linear(128, 1)
            self.logits = nn.Linear(128, 3)
            # region “probes”
            self.reg_m = nn.Linear(128, 1)
            self.reg_bg = nn.Linear(128, 1)
            self.reg_pfc = nn.Linear(128, 1)
            self.reg_s1 = nn.Linear(128, 1)
            self.reg_th = nn.Linear(128, 1)
            self.reg_hpc = nn.Linear(128, 1)
            self.reg_am = nn.Linear(128, 1)
            self.reg_limb = nn.Linear(128, 1)
            self.reg_ins = nn.Linear(128, 1)
            self.reg_par = nn.Linear(128, 1)

        def forward(self, x, reward: float = 0.0):
            # x: (84,84) float or (1,84,84); ensure NCHW
            if x.ndim == 2:
                x = x[None, None, :, :]
            elif x.ndim == 3:
                x = x[None, :, :, :]
            feats = self.head(self.conv(x))
            val = self.value(feats).squeeze(-1)
            logits = self.logits(feats).squeeze(0)
            probs = torch.softmax(logits, dim=0)

            # region activities in 0..1
            def _act(l):
                return torch.sigmoid(l(feats)).mean()

            out = {
                'motor': {'action_logits': logits},
                'basal_ganglia': {
                    'value_estimate': val.item(),
                    'action_probabilities': probs.detach()
                },
                'pfc':          {'neural_activity': _act(self.reg_pfc).item()},
                'motor':        {'action_logits': logits, 'neural_activity': _act(self.reg_m).item()},
                'basal_ganglia_region': {'neural_activity': _act(self.reg_bg).item()},
                'sensory':      {'neural_activity': _act(self.reg_s1).item()},
                'thalamus':     {'neural_activity': _act(self.reg_th).item()},
                'hippocampus':  {'neural_activity': _act(self.reg_hpc).item()},
                'amygdala':     {'neural_activity': _act(self.reg_am).item()},
                'limbic':       {'neural_activity': _act(self.reg_limb).item()},
                'insula':       {'neural_activity': _act(self.reg_ins).item()},
                'parietal':     {'neural_activity': _act(self.reg_par).item()},
                'neuromodulators': {'dopamine': 0.5, 'acetylcholine': 0.5, 'norepinephrine': 0.5}
            }
            return out

    brain = SimplePongBrain().to(device or torch.device('cpu'))
    return brain

class EEGSynthesizer:
    """Tiny EEG stub that returns band powers so plots don’t break."""
    def __init__(self, n_regions=10, device=None): pass
    def synthesize(self, region_tensor, modulators):
        # region_tensor: torch.Tensor of shape (R,) or (N,R)
        x = region_tensor.float().mean().item() if hasattr(region_tensor, 'float') else float(np.mean(region_tensor))
        d = float(modulators.get('dopamine', 0.5)) if isinstance(modulators, dict) else 0.5
        return {
            'theta_power': 0.3 + 0.3*x,
            'alpha_power': 0.4 + 0.2*d,
            'beta_power':  0.2 + 0.4*(1.0 - d),
            'gamma_power': 0.1 + 0.3*x*d
        }

class AdaptiveGuidanceSystem:
    """Adaptive guidance system that fades based on AI-guidance alignment and removes permanently after independence test"""
    
    def __init__(self):
        self.recent_agreements = []  # Last 100 decisions
        self.guidance_strength = 1.0  # Start at full strength
        self.min_guidance_strength = 0.0  # Allow complete removal
        self.max_guidance_strength = 1.0  # Maximum strength
        self.alignment_threshold_high = 0.8  # Reduce guidance when >80% agreement
        self.alignment_threshold_low = 0.5   # Increase guidance when <50% agreement
        self.independence_threshold = 0.9    # 90% agreement triggers independence test
        
        # Independence testing
        self.consecutive_good_episodes = 0   # Count consecutive good episodes without guidance
        self.required_solo_episodes = 10    # Must succeed 10 episodes solo
        self.is_independent = False          # Final independence flag
        self.testing_independence = False    # Currently in independence test phase
        
    def update_alignment(self, ai_action: int, guidance_action: int, episode_performance: float = 0.0):
        """Update alignment tracking and adjust guidance strength"""
        if self.is_independent:
            # AI achieved independence - no more guidance forever
            return
            
        agreement = 1 if ai_action == guidance_action else 0
        self.recent_agreements.append(agreement)
        
        # Keep only recent decisions
        if len(self.recent_agreements) > 100:
            self.recent_agreements.pop(0)
            
        # Calculate alignment rate and adjust guidance
        if len(self.recent_agreements) >= 20:
            alignment_rate = sum(self.recent_agreements) / len(self.recent_agreements)
            
            # Check if ready for independence test
            if alignment_rate > self.independence_threshold and not self.testing_independence:
                self.testing_independence = True
                self.guidance_strength = 0.0  # Remove all guidance for testing
                self.consecutive_good_episodes = 0
                print(f"HIGH ALIGNMENT ({alignment_rate:.2f}) - Starting independence test (10 episodes without guidance)")
                
            elif self.testing_independence:
                # Currently testing independence - check episode performance
                if episode_performance > 0.0:  # Good episode performance
                    self.consecutive_good_episodes += 1
                    print(f"Independence test: {self.consecutive_good_episodes}/{self.required_solo_episodes} episodes passed")
                    
                    if self.consecutive_good_episodes >= self.required_solo_episodes:
                        self.is_independent = True
                        print("*** AI ACHIEVED FULL INDEPENDENCE - GUIDANCE PERMANENTLY DISABLED ***")
                        
                else:  # Poor episode performance
                    print(f"Independence test failed (poor performance) - Restoring guidance")
                    self.testing_independence = False
                    self.consecutive_good_episodes = 0
    def should_provide_guidance(self) -> bool:
        """Decide whether to provide guidance this step"""
        if self.is_independent:
            return False  # Never provide guidance after independence achieved
        return random.random() < self.guidance_strength

    def update_episode_completion(self, episode_reward: float, episode_hits: int, episode_num: int = 0):
        """Call this at the end of each episode to update independence testing"""
        if not self.is_independent and self.testing_independence:
            # Consider episode successful if reward > 0 or hits > 0
            episode_success = episode_reward > 0.0 or episode_hits > 0
            
            if episode_success:
                self.consecutive_good_episodes += 1
                print(f"Independence test: {self.consecutive_good_episodes}/{self.required_solo_episodes} episodes passed")
                
                if self.consecutive_good_episodes >= self.required_solo_episodes:
                    self.is_independent = True
                    self.independence_achieved_episode = episode_num  # Track when independence achieved
                    print("*** AI ACHIEVED FULL INDEPENDENCE - GUIDANCE PERMANENTLY DISABLED ***")
                    
            else:
                print(f"Independence test failed (poor episode performance) - Restoring guidance")
                self.testing_independence = False
                self.consecutive_good_episodes = 0
                self.guidance_strength = 0.3  # Restore partial guidance
        
    def get_alignment_stats(self) -> Dict[str, float]:
        """Get current alignment statistics"""
        if len(self.recent_agreements) >= 10:
            alignment_rate = sum(self.recent_agreements) / len(self.recent_agreements)
            return {
                'alignment_rate': alignment_rate,
                'guidance_strength': self.guidance_strength,
                'decisions_tracked': len(self.recent_agreements),
                'testing_independence': self.testing_independence,
                'consecutive_good_episodes': self.consecutive_good_episodes,
                'is_independent': self.is_independent,
                'independence_achieved_episode': getattr(self, 'independence_achieved_episode', None)
            }
        
        return {
            'alignment_rate': 0.0, 
            'guidance_strength': self.guidance_strength, 
            'decisions_tracked': 0, 
            'testing_independence': self.testing_independence, 
            'consecutive_good_episodes': self.consecutive_good_episodes, 
            'is_independent': self.is_independent,
            'independence_achieved_episode': getattr(self, 'independence_achieved_episode', None)
        }
    
class PongBrainAdapter:
    """Adapter for AI Brain + Pong Environment with Adaptive Guidance"""
    
    def __init__(self, device=None, verbose=False):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # Create Pong-optimized brain
        self.brain = create_ai_brain_for_pong(device=self.device, verbose=self.verbose)
        self.eeg = EEGSynthesizer(n_regions=10, device=self.device)
        self.prev_V = 0.0
        self.step_counter = 0
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=0.001)  # 100x stronger
        self.guidance_system = type('DummyGuidance', (), {
            'update_episode_completion': lambda *args: None,
            'get_alignment_stats': lambda: {'alignment_rate': 0.0, 'guidance_strength': 0.0, 'is_independent': False, 'testing_independence': False, 'consecutive_good_episodes': 0}
        })()
        # Add adaptive guidance system
        self.guidance_system = AdaptiveGuidanceSystem()

    def _safe_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)
        
    def _calculate_guidance_action(self, obs: np.ndarray, ball_y: float, paddle_y: float, paddle_height: float) -> int:
        """Calculate optimal action for guidance"""
        ball_center = ball_y
        paddle_center = paddle_y + paddle_height / 2
        
        diff = ball_center - paddle_center
        
        if diff < -15:  # Ball above paddle
            return 0  # Move up
        elif diff > 15:  # Ball below paddle  
            return 2  # Move down
        else:
            return 1  # Stay in place
        
    def forward_and_learn(self, obs: np.ndarray, reward: float, game_info: Dict = None) -> Dict[str, Any]:
        """Process Pong observation and return action with adaptive guidance"""
        
        # Convert Pong observation (84x84x3) to grayscale tensor
        if len(obs.shape) == 3:
            gray = np.mean(obs, axis=2, dtype=np.float32)  # RGB to grayscale with float32
        else:
            gray = obs
            
        # Resize to 84x84 for brain processing
        if gray.shape != (84, 84):
            gray = cv2.resize(gray, (84, 84)).astype(np.float32)

        # Normalize and convert to tensor
        gray = np.clip(gray / 255.0, 0.0, 1.0)
        sensory_t = self._safe_tensor(gray)
        
        # Calculate guidance suggestion
        guidance_action = None
        provide_guidance = False
        
        if game_info:
            ball_y = game_info.get('ball_y', 0)
            paddle_y = game_info.get('paddle_y', 0) 
            paddle_height = game_info.get('paddle_height', 50)
            
            # Always calculate guidance action for tracking
            guidance_action = self._calculate_guidance_action(obs, ball_y, paddle_y, paddle_height)
            
            # Decide whether to provide it as input
            provide_guidance = self.guidance_system.should_provide_guidance()
        
        # Prepare brain input with optional guidance
        if provide_guidance and guidance_action is not None:
            # Add guidance as additional input channel
            guidance_tensor = torch.tensor([guidance_action / 2.0], device=self.device)  # Normalize to [0, 1]
            enhanced_input = torch.cat([sensory_t.flatten(), guidance_tensor])
        else:
            # Pad with neutral guidance signal when not providing guidance
            neutral_guidance = torch.tensor([0.5], device=self.device)  # Neutral signal
            enhanced_input = torch.cat([sensory_t.flatten(), neutral_guidance])
        
        # The AI brain expects single tensor, not concatenated guidance input
        out1 = self.brain(sensory_t, reward=0.0)
        out2 = self.brain(sensory_t, reward=reward)
        # Extract value estimate
        bg = out1.get('basal_ganglia', {})
        if isinstance(bg, dict) and 'value_estimate' in bg:
            V_t = float(bg['value_estimate'])
        else:
            V_t = 0.0
            
        # TD error for dopamine
        gamma = 0.99        
        delta = reward + gamma * V_t - self.prev_V
        dopamine_level = max(0.01, 0.5 + 0.5 * np.tanh(delta))  # Scale to 0.01-1.0
        
        # Debug prints (remove these later)
        print(f"DEBUG: reward={reward:.3f}, V_t={V_t:.3f}, prev_V={self.prev_V:.3f}")
        print(f"DEBUG: delta={delta:.3f}, tanh(delta)={np.tanh(delta):.3f}, dopamine_level={dopamine_level:.6f}")
        
        self.prev_V = V_t
        
        # Second pass with real reward
        out2 = self.brain(sensory_t, reward=reward)

        # Select action (0=up, 1=stay, 2=down)
        action = self._select_pong_action(out2)
        
        # Update guidance alignment tracking - step level only
        if guidance_action is not None:
            self.guidance_system.update_alignment(action, guidance_action)
        
        # AGGRESSIVE LEARNING with strong exploration
        if True:  # Learn from all experiences
            motor_logits = out2['motor']['action_logits'].clone()
            
            # MUCH STRONGER exploration noise that decays slower
            exploration_strength = max(0.5, 1.0 - self.step_counter/5000)  # Decay much slower
            exploration_noise = 0.5 * torch.randn_like(motor_logits) * exploration_strength
            motor_logits = motor_logits + exploration_noise
            
            # Calculate probabilities from noisy logits
            log_probs = F.log_softmax(motor_logits, dim=0)
            probs = F.softmax(motor_logits, dim=0)
            
            # Calculate proper advantage baseline
            if hasattr(self, 'recent_hits') and len(self.recent_hits) > 5:
                baseline = np.mean(self.recent_hits) * 2.0  # Scale baseline 
                advantage = reward - baseline
            else:
                advantage = reward
            
            # Much gentler policy gradient to prevent collapse
            policy_loss = -log_probs[action] * advantage * 0.1  # Much gentler learning

            # Strong entropy bonus to prevent collapse
            entropy = -torch.sum(probs * log_probs)
            entropy_bonus = 0.5 * entropy  # Strong entropy bonus

            # Total loss
            loss = policy_loss - entropy_bonus
            
            # SLOWER learning to prevent collapse
            for param in self.brain.parameters():
                param.grad = None  # Clear gradients
            
            loss.backward()
            
            # CLIP GRADIENTS to prevent instability
            torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_norm=1.0)
            
            self.optimizer.step()

        # Collect region activities
        region_activities = self._collect_region_activity(out2)
        
        # Generate EEG signals from brain regions
        region_tensor = torch.tensor(list(region_activities.values()), device=self.device)
        modulators = {'dopamine': dopamine_level, 'acetylcholine': 0.5, 'norepinephrine': 0.5}
        eeg_data = self.eeg.synthesize(region_tensor, modulators)
        # Get modulators
        modulators = out2.get('neuromodulators', {
            'dopamine': dopamine_level,  # Use the calculated value!
            'acetylcholine': 0.5, 
            'norepinephrine': 0.5
        })
        
        # Track recent hit performance for adaptive exploration
        if not hasattr(self, 'recent_hits'):
            self.recent_hits = []
        
        # Record if this was a hit (reward > 0.5 means ball hit)
        self.recent_hits.append(1 if reward > 0.5 else 0)
        
        # Keep only last 50 attempts
        if len(self.recent_hits) > 50:
            self.recent_hits.pop(0)
            
        # Calculate recent hit rate
        if len(self.recent_hits) >= 10:
            self.recent_hit_rate = sum(self.recent_hits) / len(self.recent_hits)
        
        self.step_counter += 1

        # Clean up memory (only delete variables that exist)
        try:
            if 'sensory_t' in locals():
                del sensory_t
            if 'out1' in locals():
                del out1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        return {
            "outputs": out2,
            "action": action,
            "V_t": V_t,
            "delta": delta,
            "modulators": modulators,
            "dopamine_level": dopamine_level,
            "region_activity": region_activities,
            "eeg_signals": eeg_data,
            "guidance_action": guidance_action,
            "guidance_provided": provide_guidance,
            "guidance_stats": self.guidance_system.get_alignment_stats()
        }

    def _select_pong_action(self, out: Dict[str, Any]) -> int:
        """Select Pong action with FORCED exploration"""
        
        # FORCE random exploration for first steps only
        if self.step_counter < 50:
            action = np.random.randint(0, 3)
            print(f"EXPLORING: Random action {action} (step {self.step_counter})")
            return action
        
        # Try basal ganglia first
        bg = out.get('basal_ganglia', {})
        if isinstance(bg, dict) and 'action_probabilities' in bg:
            probs = bg['action_probabilities']
            if isinstance(probs, torch.Tensor):
                probs = probs.detach().cpu().numpy()
            
            if len(probs) >= 3:
                # Ensure valid probabilities
                probs = np.array(probs[:3], dtype=np.float32)
                print(f"Raw BG probs before processing: {probs}")

                # 5% chance of random action to maintain exploration
                if np.random.random() < 0.05:
                    action = np.random.randint(0, 3)
                    print(f"FORCED RANDOM: {action} (rate: 0.05)")
                    return action

                try:
                    probs = np.clip(probs, 1e-8, None)
                    probs = probs / probs.sum()
                    action = int(np.random.choice(3, p=probs))
                    print(f"Action probs: {probs}, Selected: {action}")
                    return action
                except Exception as e:
                    action = np.random.randint(0, 3)
                    print(f"Fallback random (probs invalid: {e}): {action}")
                    return action
        
        # Fallback: random action
        return np.random.randint(0, 3)
        
    def _collect_region_activity(self, out: Dict[str, Any]) -> Dict[str, float]:
        """Collect neural activity from each region"""
        regions = ['motor', 'basal_ganglia_region', 'pfc', 'sensory', 'thalamus',
                   'hippocampus', 'amygdala', 'limbic', 'insula', 'parietal']

        activities = {}
        for region in regions:
            r_out = out.get(region, {})
            if isinstance(r_out, dict) and 'neural_activity' in r_out:
                activities[region] = float(r_out['neural_activity'])
            else:
                activities[region] = 0.0
                
        return activities
    

class GameplayRecorder:
    """Records actual gameplay frames"""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.gameplay_frames = []
        
    def record_frame(self, env):
        """Capture current game frame with GUIDANCE overlay"""
        
        try:
            frame = env.render(mode="rgb_array")
        except:
            frame = env.render()  # Fallback to default render
            if frame is None:
                return  # Skip if no frame available
        
        # Resize to standard size for video
        if frame.shape != (480, 640, 3):
            import cv2
            frame = cv2.resize(frame, (640, 480))

        # Overlay ON/OFF text
        import cv2
        overlay = frame.copy()
        status = "GUIDANCE ON" if getattr(env, "guidance_active", False) else "FREE PLAY"
        color = (255, 80, 80) if getattr(env, "guidance_active", False) else (120, 220, 120)
        cv2.rectangle(overlay, (10, 10), (275, 55), (0, 0, 0), -1)
        cv2.putText(overlay, status, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        frame = overlay

        self.gameplay_frames.append(frame)
    
    def save_gameplay_video(self):
        """Save gameplay as MP4"""
        if not self.gameplay_frames:
            return
            
        video_path = os.path.join(self.output_dir, "gameplay.mp4")
        print(f"Saving gameplay video with {len(self.gameplay_frames)} frames...")
        
        height, width, _ = self.gameplay_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 60.0, (width, height))  # 30 FPS for smooth gameplay
        
        for frame in self.gameplay_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Gameplay saved as: {video_path}")
    
    def save_incremental_gameplay(self, episode):
        """Save gameplay video up to current episode"""
        if not self.gameplay_frames:
            return None
            
        # Save current gameplay
        video_path = os.path.join(self.output_dir, f"gameplay_ep{episode:03d}.mp4")
        
        height, width, _ = self.gameplay_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 60.0, (width, height))
        
        for frame in self.gameplay_frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        return video_path

class LearningJourneyVisualizer:
    """Records and creates MP4 of AI learning journey"""
    
    def __init__(self, output_path="ai_learning_journey.mp4"):
        self.output_path = output_path
        self.frames = []
        
        # Learning metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.hit_rates = []
        self.dopamine_levels = []
        self.learning_progress = []
        self.guidance_flags = []
        self.eeg_history = []
        
        # Setup matplotlib figure
        self.fig, self.axes = plt.subplots(3, 3, figsize=(18, 12))
        self.fig.suptitle("CORTEX 4.2 AI Learning Journey - Pong", fontsize=16)
        
    def record_episode(self, episode, reward, steps, hit_rate, dopamine, brain_output, guidance=False):
        """Record episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(steps)
        self.hit_rates.append(hit_rate)
        self.dopamine_levels.append(dopamine)
        self.guidance_flags.append(bool(guidance))
        # Convert EEG data to JSON-serializable format
        eeg_data = brain_output.get('eeg_signals', {'theta_power': 0.0, 'alpha_power': 0.0, 'beta_power': 0.0, 'gamma_power': 0.0})
        eeg_serializable = {}
        for key, value in eeg_data.items():
            try:
                if hasattr(value, 'item') and value.size == 1:  # Single element tensor/array
                    eeg_serializable[key] = float(value.item())
                elif hasattr(value, 'mean'):  # Multi-element array - take mean
                    eeg_serializable[key] = float(value.mean())
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    eeg_serializable[key] = float(np.mean(value))
                else:
                    eeg_serializable[key] = float(value) if value is not None else 0.0
            except:
                eeg_serializable[key] = 0.0  # Fallback
        self.eeg_history.append(eeg_serializable)
        # Calculate learning progress (moving average of rewards)
        window = min(10, len(self.episode_rewards))
        recent_avg = np.mean(self.episode_rewards[-window:])
        self.learning_progress.append(recent_avg)

        # Always record ep0, then every 5th episode
        if episode == 0 or (episode + 1) % 5 == 0:
            self.create_frame(episode, brain_output)
    
    def save_incremental_data(self, episode):
        """Save data every few episodes as backup"""
        data = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'hit_rates': self.hit_rates,
            'dopamine_levels': self.dopamine_levels,
            'learning_progress': self.learning_progress,
            'guidance_flags': self.guidance_flags,
            'eeg_history': self.eeg_history,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save as JSON
        backup_path = self.output_path.replace('.mp4', f'_backup_ep{episode:03d}.json')
        with open(backup_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Always create a quick progress image (even if only 1 point)
        plt.figure(figsize=(10, 6))
        episodes = list(range(1, len(self.episode_rewards) + 1))

        # ---------- Subplot 1: Reward + MA ----------
        plt.subplot(2, 2, 1)
        if self.guidance_flags:
            for i, on in enumerate(self.guidance_flags, start=1):
                if on:
                    plt.axvspan(i - 0.5, i + 0.5, color='yellow', alpha=0.15)
        if len(episodes) == 1:
            plt.plot(episodes, self.episode_rewards, 'bo')
            plt.plot(episodes, self.learning_progress, 'ro')
        else:
            plt.plot(episodes, self.episode_rewards, 'b-', alpha=0.7)
            plt.plot(episodes, self.learning_progress, 'r-', linewidth=2)
        plt.title(f'Learning Progress (Episode {episode})')
        plt.xlabel('Episode'); plt.ylabel('Reward'); plt.grid(True, alpha=0.3)

        # ---------- Subplot 2: Dopamine ----------
        plt.subplot(2, 2, 2)
        if self.guidance_flags:
            for i, on in enumerate(self.guidance_flags, start=1):
                if on:
                    plt.axvspan(i - 0.5, i + 0.5, color='yellow', alpha=0.15)
        if len(episodes) == 1:
            plt.plot(episodes, self.dopamine_levels, 'o', color='purple')
        else:
            plt.plot(episodes, self.dopamine_levels, 'purple', linewidth=2)
        plt.title('Dopamine Levels'); plt.xlabel('Episode'); plt.ylabel('Level'); plt.grid(True, alpha=0.3)

        # ---------- Subplot 3: Hit Rate ----------
        plt.subplot(2, 2, 3)
        if self.guidance_flags:
            for i, on in enumerate(self.guidance_flags, start=1):
                if on:
                    plt.axvspan(i - 0.5, i + 0.5, color='yellow', alpha=0.15)
        if len(episodes) == 1:
            plt.plot(episodes, self.hit_rates, 'go')
        else:
            plt.plot(episodes, self.hit_rates, 'g-', linewidth=2)
        plt.title('Hit Rates'); plt.xlabel('Episode'); plt.ylabel('Hit Rate'); plt.grid(True, alpha=0.3)

        # ---------- Subplot 4: Stats box ----------
        plt.subplot(2, 2, 4)
        recent_avg = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
        best_reward = max(self.episode_rewards) if self.episode_rewards else 0.0
        latest = self.episode_rewards[-1] if self.episode_rewards else 0.0
        dopamine = self.dopamine_levels[-1] if self.dopamine_levels else 0.0
        plt.text(0.1, 0.7, f'Episode: {episode}\n'
                           f'Latest: {latest:.1f}\n'
                           f'Best: {best_reward:.1f}\n'
                           f'Avg(10): {recent_avg:.1f}\n'
                           f'Dopamine: {dopamine:.2f}',
                 transform=plt.gca().transAxes, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue'))
        plt.axis('off'); plt.title('Current Stats')

        plt.tight_layout()
        progress_path = self.output_path.replace('.mp4', f'_progress_ep{episode:03d}.png')
        plt.savefig(progress_path, dpi=150, bbox_inches='tight')
        plt.close()

        return backup_path, progress_path


    
    def create_frame(self, episode, brain_output):
        """Create single frame of learning visualization"""
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            
        episodes = list(range(1, len(self.episode_rewards) + 1))
        current_on = self.guidance_flags[-1] if self.guidance_flags else False
        status_label = "GUIDANCE ON (teacher forcing)" if current_on else "FREE PLAY"
        status_color = 'tab:red' if current_on else 'tab:green'
        self.fig.suptitle(f"CORTEX 4.2 AI Learning Journey - Pong  —  {status_label}",
                        fontsize=16, color=status_color)
       
        # 1. Learning Curve (top-left)
        ax1 = self.axes[0, 0]
        if len(episodes) > 1:
            # Shade ON episodes
            if self.guidance_flags:
                for i, on in enumerate(self.guidance_flags, start=1):
                    if on:
                        ax1.axvspan(i - 0.5, i + 0.5, color='yellow', alpha=0.15)
            ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.3, label='Episode Reward')
            ax1.plot(episodes, self.learning_progress, 'r-', linewidth=2, label='Learning Progress')
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax1.set_title(f'Learning Progress (Episode {episode})')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Performance Metrics (top-middle)
        ax2 = self.axes[0, 1]
        if len(episodes) > 1:
            # Shade ON episodes
            if self.guidance_flags:
                for i, on in enumerate(self.guidance_flags, start=1):
                    if on:
                        ax2.axvspan(i - 0.5, i + 0.5, color='yellow', alpha=0.15)
            ax2.plot(episodes, self.hit_rates, 'g-', linewidth=2, label='Hit Rate')
            ax2.plot(episodes, self.episode_lengths, 'orange', alpha=0.7, label='Episode Length')
            ax2.set_title('Performance Metrics')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Neuromodulator Levels (top-right)
        ax3 = self.axes[0, 2]
        if len(episodes) > 1:
            ax3.plot(episodes, self.dopamine_levels, 'purple', linewidth=2, label='Dopamine')
            # Add other modulators if available
            modulators = brain_output.get('modulators', {})
            if 'acetylcholine' in modulators:
                ach_levels = [modulators['acetylcholine']] * len(episodes)
                ax3.plot(episodes[-len(ach_levels):], ach_levels, 'blue', alpha=0.7, label='ACh')
            if 'norepinephrine' in modulators:
                ne_levels = [modulators['norepinephrine']] * len(episodes)
                ax3.plot(episodes[-len(ne_levels):], ne_levels, 'red', alpha=0.7, label='NE')
            
            ax3.set_title('Neuromodulator Levels')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Level')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Brain Region Activity (bottom-left)
        ax4 = self.axes[1, 0]
        region_activity = brain_output.get('region_activity', {})
        if region_activity:
            regions = list(region_activity.keys())[:8]  # Top 8 regions
            activities = [region_activity[r] for r in regions]
            
            bars = ax4.bar(range(len(regions)), activities, 
                          color=['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray'][:len(regions)])
            ax4.set_title('Brain Region Activity')
            ax4.set_xlabel('Brain Region')
            ax4.set_ylabel('Activity Level')
            ax4.set_xticks(range(len(regions)))
            ax4.set_xticklabels(regions, rotation=45, ha='right')
            
            # Add activity values on bars
            for i, (bar, activity) in enumerate(zip(bars, activities)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{activity:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Learning Statistics (bottom-middle)
        ax5 = self.axes[1, 1]
        stats_text = [
            f'Current Episode: {episode}',
            f'Latest Reward: {self.episode_rewards[-1]:.1f}',
            f'Best Reward: {max(self.episode_rewards):.1f}',
            f'Average Reward: {np.mean(self.episode_rewards):.1f}',
            f'Learning Trend: {"â†—" if len(self.learning_progress) > 10 and self.learning_progress[-1] > self.learning_progress[-10] else "â†’"}',
            f'Hit Rate: {self.hit_rates[-1]:.1%}',
            f'Total Episodes: {len(self.episode_rewards)}',
            f'',
            f'AI Brain Status:',
            f'Dopamine: {self.dopamine_levels[-1]:.2f}',
            f'Decision Quality: {"Good" if self.hit_rates[-1] > 0.3 else "Learning"}',
            f'Exploration: {"Active" if self.dopamine_levels[-1] > 0.6 else "Focused"}'
        ]
        
        ax5.text(0.1, 0.9, '\n'.join(stats_text), transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax5.set_title('Learning Statistics')
        ax5.axis('off')
        
        # 6. EEG Visualization (bottom-right)
        ax6 = self.axes[1, 2]

        if len(self.eeg_history) > 1:
            episodes_range = list(range(1, len(self.eeg_history) + 1))
            
            theta_powers = [eeg.get('theta_power', 0.0) for eeg in self.eeg_history]
            alpha_powers = [eeg.get('alpha_power', 0.0) for eeg in self.eeg_history]
            beta_powers = [eeg.get('beta_power', 0.0) for eeg in self.eeg_history]
            gamma_powers = [eeg.get('gamma_power', 0.0) for eeg in self.eeg_history]
            
            ax6.plot(episodes_range, theta_powers, 'b-', label='Theta (4-8 Hz)', alpha=0.7)
            ax6.plot(episodes_range, alpha_powers, 'g-', label='Alpha (8-12 Hz)', alpha=0.7)
            ax6.plot(episodes_range, beta_powers, 'orange', label='Beta (12-30 Hz)', alpha=0.7)
            ax6.plot(episodes_range, gamma_powers, 'r-', label='Gamma (30-100 Hz)', alpha=0.7)
            
            ax6.set_title('EEG Band Powers')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Power')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'EEG Data\nCollecting...', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('EEG Band Powers')
            ax6.axis('off')    
                
        # 7. Neural Network Visualization (new subplot)
        ax7 = self.axes[2, 0]

        # Simple neural network diagram  
        layers = [3, 8, 5, 3]  # Input, hidden1, hidden2, output
        layer_names = ['Sensory', 'Processing', 'Decision', 'Action']

        # Draw network nodes
        for i, (layer_size, name) in enumerate(zip(layers, layer_names)):
            x = i / (len(layers) - 1)
            for j in range(layer_size):
                y = (j + 0.5) / layer_size
                
                # Color nodes based on activity
                if i == 0:  # Input layer
                    color = 'lightblue'
                elif i == len(layers) - 1:  # Output layer
                    color = 'lightcoral' 
                else:  # Hidden layers
                    activity = region_activity.get(list(region_activity.keys())[i % len(region_activity)], 0)
                    intensity = min(1.0, abs(activity))
                    color = plt.cm.viridis(intensity)
                
                circle = plt.Circle((x, y), 0.03, color=color, ec='black', linewidth=0.5)
                ax7.add_patch(circle)
            
            # Add layer labels
            ax7.text(x, -0.1, name, ha='center', va='top', fontsize=8, rotation=0)

        # Draw connections (simplified)
        for i in range(len(layers) - 1):
            x1 = i / (len(layers) - 1)
            x2 = (i + 1) / (len(layers) - 1)
            for j in range(min(layers[i], 3)):  # Limit connections for clarity
                y1 = (j + 0.5) / layers[i]
                for k in range(min(layers[i+1], 3)):
                    y2 = (k + 0.5) / layers[i+1]
                    ax7.plot([x1, x2], [y1, y2], 'k-', alpha=0.2, linewidth=0.5)

        ax7.set_xlim(-0.1, 1.1)
        ax7.set_ylim(-0.2, 1.1)
        ax7.set_title('Neural Network Activity')
        ax7.axis('off')
        
        plt.tight_layout()
        
        # Convert to frame
        self.fig.canvas.draw()
        # Handle different matplotlib backends
        try:
            frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        except AttributeError:
            # Newer matplotlib versions
            frame = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]  # Drop alpha channel
        
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)
    
    def create_mp4(self):
        """Create MP4 from recorded frames"""
        if not self.frames:
            print("No frames recorded. Cannot create video.")
            return
            
        print(f"Creating learning journey MP4 with {len(self.frames)} frames...")
        
        # Setup video writer
        height, width, _ = self.frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, 2.0, (width, height))  # 2 FPS for slow viewing
        
        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Learning journey saved as: {self.output_path}")
        
        # Also create a summary frame
        if self.frames:
            final_frame = cv2.cvtColor(self.frames[-1], cv2.COLOR_RGB2BGR)
            summary_path = self.output_path.replace('.mp4', '_final_summary.png')
            cv2.imwrite(summary_path, final_frame)
            print(f"Final summary saved as: {summary_path}")
        
        plt.close(self.fig)
    
    def __del__(self):
        """Cleanup matplotlib figure"""
        if hasattr(self, 'fig'):
            plt.close(self.fig)

# Rest of the file (main training loop, visualizers, etc.) stays the same
# Just modify the main training loop to pass game info to the brain adapter

def run_pong_ai(args):
    """Main Pong AI training loop with adaptive guidance"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"pong_training_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    # Initialize environment and brain
    env = PongEnv(width=640, height=480, render_mode=args.render)
    brain_adapter = PongBrainAdapter(device=device, verbose=True)
    
    # Create enhanced visualizer with guidance tracking
    learning_video_path = os.path.join(output_dir, "learning_journey.mp4")
    visualizer = LearningJourneyVisualizer(learning_video_path)
    
    print(f"Starting Pong AI training with adaptive guidance for {args.episodes} episodes...")
    
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        hits_this_episode = 0
        last_reward = 0.0
        
        while not done and steps < args.max_steps:
            # Gather game info for guidance calculation
            game_info = {
                'ball_y': getattr(env, 'ball_y', 0),
                'paddle_y': getattr(env, 'player_y', 0),
                'paddle_height': getattr(env, 'paddle_height', 50)
            }
            
            # Use the *previous* step's reward to update the brain
            brain_output = brain_adapter.forward_and_learn(obs, reward=last_reward, game_info=game_info)
            action = brain_output["action"]

            # Step the environment with the chosen action
            obs, reward, done, info = env.step(action)
            
            if info.get("hit", False):
                hits_this_episode += 1
            
            # Update trackers
            last_reward = reward
            total_reward += reward
            steps += 1
            
            # Optional rendering
            if args.render:
                env.render()
        
        # Episode completed - update independence testing
        brain_adapter.guidance_system.update_episode_completion(total_reward, hits_this_episode, episode)

        # Get guidance stats for recording
        guidance_stats = brain_adapter.guidance_system.get_alignment_stats()
        
        # Record episode with enhanced guidance tracking
        misses = max(0, -int(total_reward))
        total_contacts = max(1, hits_this_episode + misses)
        hit_rate = hits_this_episode / total_contacts
        
        visualizer.record_episode(
            episode=episode,
            reward=total_reward,
            steps=steps,
            hit_rate=hit_rate,
            dopamine=brain_output.get("dopamine_level", 0.5),
            brain_output=brain_output,
            guidance=False
        )
                
        # Progress reporting and incremental saving
        if episode % 10 == 0 or episode == args.episodes - 1:
            print(f"Saving backup data for episode {episode}...")
            backup_json, progress_img = visualizer.save_incremental_data(episode)
            print(f"  Enhanced data backup: {os.path.basename(backup_json)}")
            print(f"  Progress image: {os.path.basename(progress_img)}")
            
            # Also mention EEG plots if they exist
            eeg_plot_path = visualizer.output_path.replace('.mp4', f'_eeg_analysis_ep{episode:03d}.png')
            if os.path.exists(eeg_plot_path):
                print(f"  EEG analysis: {os.path.basename(eeg_plot_path)}")
        
        # Print episode summary with enhanced guidance stats
        independence_status = ""
        if guidance_stats['is_independent']:
            independence_status = " [INDEPENDENT]"
        elif guidance_stats['testing_independence']:
            independence_status = f" [TESTING {guidance_stats['consecutive_good_episodes']}/10]"
            
        print(f"Episode {episode:3d}: "
              f"Reward={total_reward:5.1f}, "
              f"Steps={steps:3d}, "
              f"Hits={hits_this_episode}, "
              f"Alignment={guidance_stats['alignment_rate']:.2f}, "
              f"Guidance={guidance_stats['guidance_strength']:.3f}"
              f"{independence_status}")
    
    env.close()
    
    # Create final summary report
    final_guidance_stats = brain_adapter.guidance_system.get_alignment_stats()
    
    print(f"\nAdaptive Guidance Training Complete!")
    print(f"Results saved in folder: {output_dir}")
    print(f"Total episodes: {args.episodes}")
    
    # Enhanced final report with guidance system summary
    if final_guidance_stats['is_independent']:
        independence_ep = final_guidance_stats['independence_achieved_episode']
        print(f"*** AI ACHIEVED FULL INDEPENDENCE at episode {independence_ep} ***")
        print(f"Final status: Fully autonomous gameplay (no guidance)")
        print(f"Independence milestone: Completed 10 consecutive successful episodes solo")
    elif final_guidance_stats['testing_independence']:
        consec = final_guidance_stats['consecutive_good_episodes']
        print(f"AI currently testing independence: {consec}/10 consecutive good episodes")
        print(f"Final guidance strength: {final_guidance_stats['guidance_strength']:.3f}")
    else:
        print(f"AI still learning with guidance assistance")
        print(f"Final guidance strength: {final_guidance_stats['guidance_strength']:.3f}")
        print(f"Final alignment rate: {final_guidance_stats['alignment_rate']:.2f}")
        
    print(f"Enhanced visualizations and data saved with complete guidance system tracking!")
    # Create learning journey MP4
    print("Creating learning visualization...")
    visualizer.create_mp4()

def main():
    parser = argparse.ArgumentParser(description="Pong AI Training with Adaptive Guidance")
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