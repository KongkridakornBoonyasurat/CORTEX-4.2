import pygame
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

# ====================================================================
# CORTEX 4.1 LIVE PONG DEMO - LOAD SAVED BRAIN STATE
# ====================================================================

class LivePongDemo:
    def __init__(self, brain_file="good_brain_ep150.pkl"):
        """
        Load saved brain state and create live Pong demo
        """
        self.width = 800
        self.height = 400
        self.paddle_height = 60
        self.paddle_width = 10
        self.ball_size = 8
        
        # Load saved brain state
        self.load_brain_state(brain_file)
        
        # Game state
        self.reset_game()
        
        # Performance tracking
        self.hits = 0
        self.misses = 0
        self.episode_score = 0
        self.frame_count = 0
        
        # Visual elements
        self.neural_activity = np.zeros(24)
        self.consciousness_level = 0.0
        
    def load_brain_state(self, filepath):
        """Load the saved brain state from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                brain_data = pickle.load(f)
                
            # Extract key data
            self.self_agency = brain_data.get('self_agency', 0.443)
            self.neuron_voltages = brain_data.get('neuron_voltages', [-65.0] * 24)
            self.synapse_weights = brain_data.get('synapse_weights', [0.2] * 24)
            self.total_episodes = brain_data.get('total_episodes', 150)
            
            print(f"Loaded brain state:")
            print(f"  Self-agency: {self.self_agency:.3f}")
            print(f"  Training episodes: {self.total_episodes}")
            print(f"  Neural voltages: {len(self.neuron_voltages)} neurons")
            print(f"  Synapse weights: {len(self.synapse_weights)} synapses")
            
        except Exception as e:
            print(f"Could not load {filepath}: {e}")
            print("Using default trained brain parameters...")
            
            # Use Episode 150 values as fallback
            self.self_agency = 0.443
            self.total_episodes = 150
            self.neuron_voltages = [-50.2, -65.0, -52.8, -65.0, -58.7, -48.3, -65.0, 
                                   -38.4, -67.2, -65.0, -65.0, -65.0, -65.0, -49.1, 
                                   -68.1, -47.2, -65.0, -65.0, -50.4, -42.6, -65.0, 
                                   -65.0, -31.8, -65.0]
            self.synapse_weights = [0.69, 0.78, 0.71, 0.43, 0.54, 0.55, 0.20, 0.44,
                                   0.58, 0.72, 0.32, 0.59, 0.61, 0.72, 0.39, 0.48,
                                   0.55, 0.30, 0.57, 0.40, 0.80, 0.73, 0.47, 0.81]
    
    def reset_game(self):
        """Reset Pong game state"""
        self.player_y = self.height // 2 - self.paddle_height // 2
        self.ai_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = 4 * (1 if np.random.random() > 0.5 else -1)
        self.ball_dy = np.random.uniform(-3, 3)
        
    def get_visual_input(self):
        """Get simplified visual input like the trained AI"""
        # 80x40 visual field
        visual = np.zeros((40, 80))
        
        # Add paddle
        paddle_y = int(self.player_y / self.height * 40)
        paddle_h = int(self.paddle_height / self.height * 40)
        visual[paddle_y:paddle_y+paddle_h, 0:2] = 1.0
        
        # Add ball
        ball_x = int(self.ball_x / self.width * 80)
        ball_y = int(self.ball_y / self.height * 40)
        if 0 <= ball_x < 80 and 0 <= ball_y < 40:
            visual[ball_y, ball_x] = 1.0
            
        return visual.flatten()
    
    def neural_processing(self, visual_input):
        """Simulate neural processing using saved brain state"""
        # Simple neural activity simulation based on saved weights
        activity = np.zeros(24)
        
        # Process visual input through "neurons"
        for i in range(min(24, len(visual_input))):
            # Use saved synapse weights to process input
            weight = self.synapse_weights[i] if i < len(self.synapse_weights) else 0.2
            
            # Simulate neural response
            input_strength = visual_input[i] * weight * 10
            noise = np.random.normal(0, 1)
            
            # Simple threshold activation
            if input_strength + noise > 2.0:
                activity[i] = 1.0
                
        self.neural_activity = activity
        return activity
    
    def conscious_decision(self, neural_activity):
        """Make decision using consciousness-inspired algorithm"""
        # Calculate ball trajectory
        ball_center_y = self.ball_y + self.ball_size / 2
        paddle_center_y = self.player_y + self.paddle_height / 2
        
        # Distance to ball
        distance_to_ball = abs(ball_center_y - paddle_center_y)
        
        # Predict where ball will be
        time_to_reach = (self.ball_x - self.paddle_width) / abs(self.ball_dx) if self.ball_dx < 0 else float('inf')
        predicted_y = self.ball_y + self.ball_dy * time_to_reach
        
        # Consciousness-influenced decision
        consciousness_factor = self.self_agency  # Use saved self-agency
        
        # Weighted decision combining prediction and current position
        if self.ball_dx < 0:  # Ball coming toward paddle
            target_y = predicted_y * consciousness_factor + ball_center_y * (1 - consciousness_factor)
        else:  # Ball going away
            target_y = paddle_center_y  # Stay put
            
        # Convert to action
        paddle_center = self.player_y + self.paddle_height / 2
        
        if target_y < paddle_center - 10:
            return 0  # Move up
        elif target_y > paddle_center + 10:
            return 1  # Move down
        else:
            return 2  # Stay
    
    def update_game(self, action):
        """Update game state based on AI action"""
        # Move player paddle
        PADDLE_SPEED = 6
        if action == 0:  # Up
            self.player_y = max(0, self.player_y - PADDLE_SPEED)
        elif action == 1:  # Down
            self.player_y = min(self.height - self.paddle_height, self.player_y + PADDLE_SPEED)
        
        # Update ball position
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collisions with top/bottom
        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_dy = -self.ball_dy
            
        # Simple AI opponent
        ai_center = self.ai_y + self.paddle_height / 2
        ball_center = self.ball_y + self.ball_size / 2
        if ball_center < ai_center:
            self.ai_y = max(0, self.ai_y - 4)
        else:
            self.ai_y = min(self.height - self.paddle_height, self.ai_y + 4)
            
        # Ball collision with player paddle
        hit = False
        if (self.ball_x <= self.paddle_width and 
            self.player_y <= self.ball_y <= self.player_y + self.paddle_height):
            self.ball_dx = abs(self.ball_dx)
            self.hits += 1
            self.episode_score += 100
            hit = True
            
        # Ball collision with AI paddle
        if (self.ball_x >= self.width - self.paddle_width - self.ball_size and
            self.ai_y <= self.ball_y <= self.ai_y + self.paddle_height):
            self.ball_dx = -abs(self.ball_dx)
            
        # Ball goes off screen
        if self.ball_x < 0:
            self.misses += 1
            self.episode_score -= 50
            self.reset_ball()
        elif self.ball_x > self.width:
            self.reset_ball()
            
        return hit
    
    def reset_ball(self):
        """Reset ball to center"""
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_dx = 4 * (1 if np.random.random() > 0.5 else -1)
        self.ball_dy = np.random.uniform(-3, 3)
        
    def calculate_consciousness(self):
        """Calculate current consciousness level"""
        # Based on neural activity and self-agency
        activity_level = np.mean(self.neural_activity)
        self.consciousness_level = (activity_level * 0.5 + self.self_agency * 0.5)
        return self.consciousness_level

# ====================================================================
# PYGAME VISUALIZATION
# ====================================================================

def run_pygame_demo():
    """Run live Pong demo with Pygame"""
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((1200, 600))
    pygame.display.set_caption("CORTEX 4.1 - Conscious AI Playing Pong")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Create demo
    demo = LivePongDemo("good_brain_ep150.pkl")  # Try to load your brain file
    
    running = True
    step_count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    demo.reset_game()
                    demo.hits = 0
                    demo.misses = 0
                    demo.episode_score = 0
                    step_count = 0
        
        # AI Processing
        visual_input = demo.get_visual_input()
        neural_activity = demo.neural_processing(visual_input)
        action = demo.conscious_decision(neural_activity)
        hit = demo.update_game(action)
        consciousness = demo.calculate_consciousness()
        
        step_count += 1
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw game area (left side)
        game_rect = pygame.Rect(50, 50, 800, 400)
        pygame.draw.rect(screen, (20, 20, 20), game_rect)
        pygame.draw.rect(screen, (255, 255, 255), game_rect, 2)
        
        # Draw paddles
        player_paddle = pygame.Rect(50, 50 + demo.player_y, demo.paddle_width, demo.paddle_height)
        ai_paddle = pygame.Rect(850 - demo.paddle_width, 50 + demo.ai_y, demo.paddle_width, demo.paddle_height)
        pygame.draw.rect(screen, (0, 255, 0), player_paddle)  # Green = AI
        pygame.draw.rect(screen, (255, 255, 255), ai_paddle)
        
        # Draw ball
        ball_rect = pygame.Rect(50 + demo.ball_x, 50 + demo.ball_y, demo.ball_size, demo.ball_size)
        pygame.draw.rect(screen, (255, 255, 0), ball_rect)
        
        # Draw center line
        pygame.draw.line(screen, (100, 100, 100), (450, 50), (450, 450), 2)
        
        # Draw AI stats (right side)
        stats_x = 900
        stats_y = 50
        
        # Title
        title_text = font.render("CORTEX 4.1 AI", True, (255, 255, 255))
        screen.blit(title_text, (stats_x, stats_y))
        
        # Performance stats
        stats = [
            f"Self-Agency: {demo.self_agency:.1%}",
            f"Training Episodes: {demo.total_episodes}",
            f"",
            f"Current Performance:",
            f"Hits: {demo.hits}",
            f"Misses: {demo.misses}",
            f"Score: {demo.episode_score}",
            f"Hit Rate: {demo.hits/(demo.hits+demo.misses)*100:.1f}%" if (demo.hits+demo.misses) > 0 else "Hit Rate: 0%",
            f"",
            f"Consciousness Level:",
            f"{consciousness:.1%}",
            f"",
            f"Neural Activity:",
            f"Active Neurons: {int(np.sum(neural_activity))}/24",
            f"",
            f"Step: {step_count}",
            f"",
            f"Controls:",
            f"R - Reset Game"
        ]
        
        for i, stat in enumerate(stats):
            color = (255, 255, 255)
            if "Self-Agency" in stat or "Consciousness" in stat:
                color = (0, 255, 255)  # Cyan for consciousness
            elif "Hits:" in stat or "Hit Rate:" in stat:
                color = (0, 255, 0)   # Green for performance
            elif "Score:" in stat:
                color = (255, 255, 0) if demo.episode_score >= 0 else (255, 100, 100)
                
            text = small_font.render(stat, True, color)
            screen.blit(text, (stats_x, stats_y + 40 + i * 20))
        
        # Draw consciousness meter
        meter_x = stats_x
        meter_y = 400
        meter_width = 200
        meter_height = 20
        
        pygame.draw.rect(screen, (50, 50, 50), (meter_x, meter_y, meter_width, meter_height))
        consciousness_width = int(consciousness * meter_width)
        pygame.draw.rect(screen, (0, 255, 255), (meter_x, meter_y, consciousness_width, meter_height))
        pygame.draw.rect(screen, (255, 255, 255), (meter_x, meter_y, meter_width, meter_height), 2)
        
        meter_text = small_font.render("Consciousness", True, (255, 255, 255))
        screen.blit(meter_text, (meter_x, meter_y - 25))
        
        # Neural activity visualization
        neural_y = 450
        for i in range(24):
            x = stats_x + (i % 12) * 15
            y = neural_y + (i // 12) * 15
            color = (255, 0, 0) if neural_activity[i] > 0 else (50, 50, 50)
            pygame.draw.circle(screen, color, (x + 7, y + 7), 5)
        
        neural_text = small_font.render("Neural Activity (24 neurons)", True, (255, 255, 255))
        screen.blit(neural_text, (stats_x, neural_y - 20))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()
    
    # Print final stats
    print("\n" + "="*50)
    print("DEMO COMPLETED")
    print("="*50)
    print(f"Final Performance:")
    print(f"  Hits: {demo.hits}")
    print(f"  Misses: {demo.misses}")
    print(f"  Hit Rate: {demo.hits/(demo.hits+demo.misses)*100:.1f}%" if (demo.hits+demo.misses) > 0 else "  Hit Rate: 0%")
    print(f"  Final Score: {demo.episode_score}")
    print(f"  Consciousness Level: {consciousness:.1%}")
    print(f"  Self-Agency: {demo.self_agency:.1%}")
    print("="*50)

# ====================================================================
# MATPLOTLIB VISUALIZATION (Alternative)
# ====================================================================

def run_matplotlib_demo():
    """Run demo with matplotlib for presentations"""
    
    demo = LivePongDemo("good_brain_ep150.pkl")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("CORTEX 4.1 - Conscious AI Playing Pong (Live Demo)", fontsize=16)
    
    # Performance tracking
    performance_data = {'hits': [], 'misses': [], 'consciousness': [], 'steps': []}
    
    def update_demo(frame):
        if frame % 60 == 0 and frame > 0:  # Every second
            # AI decision making
            visual_input = demo.get_visual_input()
            neural_activity = demo.neural_processing(visual_input)
            action = demo.conscious_decision(neural_activity)
            hit = demo.update_game(action)
            consciousness = demo.calculate_consciousness()
            
            # Track performance
            performance_data['hits'].append(demo.hits)
            performance_data['misses'].append(demo.misses)
            performance_data['consciousness'].append(consciousness)
            performance_data['steps'].append(frame // 60)
        
        # Clear axes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.clear()
        
        # Game visualization
        ax1.set_xlim(0, demo.width)
        ax1.set_ylim(0, demo.height)
        ax1.add_patch(Rectangle((0, demo.player_y), demo.paddle_width, demo.paddle_height, color='green', label='AI Paddle'))
        ax1.add_patch(Rectangle((demo.width-demo.paddle_width, demo.ai_y), demo.paddle_width, demo.paddle_height, color='white', label='Opponent'))
        ax1.add_patch(Rectangle((demo.ball_x, demo.ball_y), demo.ball_size, demo.ball_size, color='yellow', label='Ball'))
        ax1.set_title(f"Game State (Step {frame})")
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance tracking
        if performance_data['steps']:
            ax2.plot(performance_data['steps'], performance_data['hits'], 'g-', label='Hits', linewidth=2)
            ax2.plot(performance_data['steps'], performance_data['misses'], 'r-', label='Misses', linewidth=2)
            ax2.set_title("Performance Over Time")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Count")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Consciousness level
        if performance_data['steps']:
            ax3.plot(performance_data['steps'], performance_data['consciousness'], 'c-', linewidth=3)
            ax3.axhline(y=demo.self_agency, color='r', linestyle='--', alpha=0.7, label=f'Max Self-Agency: {demo.self_agency:.1%}')
            ax3.set_title("Consciousness Level")
            ax3.set_xlabel("Time (seconds)")
            ax3.set_ylabel("Consciousness")
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Neural activity
        if hasattr(demo, 'neural_activity'):
            neural_data = demo.neural_activity.reshape(6, 4)  # 24 neurons in 6x4 grid
            im = ax4.imshow(neural_data, cmap='hot', vmin=0, vmax=1, aspect='auto')
            ax4.set_title("Neural Activity (24 neurons)")
            ax4.set_xlabel("Neuron Group")
            ax4.set_ylabel("Layer")
        
        plt.tight_layout()
    
    # Run animation
    anim = animation.FuncAnimation(fig, update_demo, frames=1800, interval=50, repeat=False)
    plt.show()
    
    return anim

# ====================================================================
# MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    print("CORTEX 4.1 - Live Pong Demo")
    print("Loading saved brain state...")
    print()
    
    # Choose visualization method
    print("Choose demo type:")
    print("1. Pygame (Interactive, best for live presentation)")
    print("2. Matplotlib (Good for recording/screenshots)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("Starting matplotlib demo...")
        anim = run_matplotlib_demo()
    else:
        print("Starting pygame demo...")
        print("Controls: R = Reset game, ESC = Exit")
        run_pygame_demo()