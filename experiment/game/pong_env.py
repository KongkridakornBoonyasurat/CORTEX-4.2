import pygame
import numpy as np
import random

class PongEnv:
    def __init__(self, width=84, height=84, render_mode=False):
        pygame.init()
        self.width, self.height = width, height
        self.render_mode = render_mode
        self.display = pygame.display.set_mode((self.width, self.height)) if render_mode else pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.paddle_height = self.height // 6
        self.paddle_width = 4
        self.ball_size = 4

        # Player paddle (left)
        self.player_y = self.height // 2 - self.paddle_height // 2
        self.player_vel = 0

        # Opponent paddle (right, simple AI)
        self.opponent_y = self.height // 2 - self.paddle_height // 2
        self.opponent_vel = 0

        # Ball
        self.ball_x = self.width // 2 - self.ball_size // 2
        self.ball_y = self.height // 2 - self.ball_size // 2
        self.ball_vel_x = random.choice([-2, 2])
        self.ball_vel_y = random.choice([-2, 2])

        self.done = False
        self.score = 0
        self.opponent_score = 0
        return self._get_frame()

    def step(self, action):
        # action: 0=up, 1=stay, 2=down
        if action == 0:
            self.player_vel = -4
        elif action == 2:
            self.player_vel = 4
        else:
            self.player_vel = 0

        self.player_y += self.player_vel
        self.player_y = np.clip(self.player_y, 0, self.height - self.paddle_height)

        # Opponent AI: track ball with some noise
        if self.ball_y > self.opponent_y + self.paddle_height // 2:
            self.opponent_vel = 2
        elif self.ball_y < self.opponent_y + self.paddle_height // 2:
            self.opponent_vel = -2
        else:
            self.opponent_vel = 0
        self.opponent_y += self.opponent_vel
        self.opponent_y = np.clip(self.opponent_y, 0, self.height - self.paddle_height)

        # Ball movement
        self.ball_x += self.ball_vel_x
        self.ball_y += self.ball_vel_y

        # Ball collision with top/bottom
        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_vel_y *= -1

        # Ball collision with paddles
        reward = 0
        # Player paddle
        if (self.ball_x <= self.paddle_width and
            self.player_y <= self.ball_y <= self.player_y + self.paddle_height):
            self.ball_vel_x *= -1
            reward = 0.1  # Small reward for paddle hit
        # Opponent paddle
        if (self.ball_x >= self.width - self.paddle_width - self.ball_size and
            self.opponent_y <= self.ball_y <= self.opponent_y + self.paddle_height):
            self.ball_vel_x *= -1

        # Ball out of bounds (scoring) - CONTINUE PLAYING
        if self.ball_x < 0:
            reward = -1
            self.opponent_score += 1
            # Reset ball but DON'T end episode
            self.ball_x = self.width // 2
            self.ball_y = self.height // 2
            self.ball_vel_x = random.choice([-2, 2])
            self.ball_vel_y = random.choice([-2, 2])
        elif self.ball_x > self.width - self.ball_size:
            reward = 1  # Player scored!
            self.score += 1
            # Reset ball but DON'T end episode
            self.ball_x = self.width // 2
            self.ball_y = self.height // 2
            self.ball_vel_x = random.choice([-2, 2])
            self.ball_vel_y = random.choice([-2, 2])

        # Only end episode after time limit or score limit
        if self.score >= 5 or self.opponent_score >= 5:
            self.done = True

        frame = self._get_frame()
        return frame, reward, self.done, {}

    def render(self, mode="human"):
        if mode == "human":
            surface = pygame.display.get_surface()
        else:
            surface = self.display

        # Draw everything
        surface.fill((0, 0, 0))
        # Player paddle
        pygame.draw.rect(surface, (255, 255, 255), (0, self.player_y, self.paddle_width, self.paddle_height))
        # Opponent paddle
        pygame.draw.rect(surface, (255, 255, 255), (self.width - self.paddle_width, self.opponent_y, self.paddle_width, self.paddle_height))
        # Ball
        pygame.draw.rect(surface, (255, 255, 255), (self.ball_x, self.ball_y, self.ball_size, self.ball_size))
        # Optionally show score
        # pygame.display.set_caption(f"Player: {self.score}  Opponent: {self.opponent_score}")

        if mode == "human":
            pygame.display.flip()
            self.clock.tick(60)
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    def _get_frame(self):
        return self.render(mode="rgb_array")

    def close(self):
        pygame.quit()

# === Minimal test if you run this file directly ===
if __name__ == "__main__":
    env = PongEnv(render_mode=True)
    obs = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        # For testing: agent always 'stay'
        obs, reward, done, info = env.step(1)
        env.render()
    env.close()
