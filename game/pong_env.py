# pong_env.py  (REPLACE ENTIRE FILE WITH THIS)

import os, random, math, numpy as np
import pygame

class PongEnv:
    def __init__(self, width=640, height=480, render_mode=True):
        pygame.init()
        self.width, self.height = width, height
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((self.width, self.height)) if render_mode else pygame.Surface((self.width, self.height))
        pygame.display.set_caption("CORTEX AI Playing Pong")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 48)

        self.paddle_width = 10
        self.paddle_height = 65
        self.ball_radius = 7

        # ON/OFF guidance schedule
        self.guidance_on_length = 3
        self.guidance_off_length = 7
        self.guidance_cutoff_episode = 500

        self.opponent_skill = 0.6
        self.ai_performance_history = []

        self.episode_count = 0
        self.done = False

        self.reset()

    def reset(self):
        self.episode_count += 1
        self.done = False

        # paddles
        self.player_x = 10.0
        self.player_y = (self.height - self.paddle_height) / 2
        self.opponent_x = self.width - 10.0 - self.paddle_width
        self.opponent_y = (self.height - self.paddle_height) / 2

        # ball
        self.ball_x = 307.5
        self.ball_y = 232.5
        # serve TOWARD the AI
        self.speed_x = -150.0
        self.speed_y = 150.0

        # speeds
        self.speed_circ = 250.0

        # score & rally
        self.player_score = 0 if self.episode_count == 1 else self.player_score
        self.opponent_score = 0 if self.episode_count == 1 else self.opponent_score
        self.rally_length = 0

        # on/off guidance
        if self.episode_count > self.guidance_cutoff_episode:
            self.guidance_active = False
        else:
            cycle = self.guidance_on_length + self.guidance_off_length
            pos = (self.episode_count - 1) % cycle
            self.guidance_active = (pos < self.guidance_on_length)

        return self._get_frame()

    def _get_frame(self):
        surf = pygame.Surface((self.width, self.height))
        surf.fill((0, 0, 0))
        # center line
        pygame.draw.line(surf, (120, 120, 120), (self.width // 2, 0), (self.width // 2, self.height), 2)
        # paddles
        pygame.draw.rect(surf, (40, 120, 255), pygame.Rect(int(self.player_x), int(self.player_y), self.paddle_width, self.paddle_height))
        pygame.draw.rect(surf, (255, 60, 60), pygame.Rect(int(self.opponent_x), int(self.opponent_y), self.paddle_width, self.paddle_height))
        # ball
        pygame.draw.circle(surf, (60, 255, 60), (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        # score
        score_text = self.font.render(str(self.player_score), True, (220, 220, 220))
        self._blit_center(surf, score_text, (self.width * 0.33, 80))
        score_text2 = self.font.render(str(self.opponent_score), True, (220, 220, 220))
        self._blit_center(surf, score_text2, (self.width * 0.66, 80))
        # tag
        tag_font = pygame.font.SysFont("arial", 54)
        tag = tag_font.render("GUIDANCE ON" if self.guidance_active else "FREE PLAY", True,
                              (255, 80, 80) if self.guidance_active else (120, 220, 120))
        surf.blit(tag, (14, 10))
        return pygame.surfarray.array3d(surf).swapaxes(0, 1)

    def _blit_center(self, surf, text, pos):
        rect = text.get_rect(center=(int(pos[0]), int(pos[1])))
        surf.blit(text, rect)

    def render(self, mode="human"):
        if not self.render_mode and mode == "human":
            return
        self.screen.fill((0, 0, 0))
        # center line
        pygame.draw.line(self.screen, (120,120,120), (self.width // 2, 0), (self.width // 2, self.height), 2)
        # paddles
        pygame.draw.rect(self.screen, (40,120,255), pygame.Rect(int(self.player_x), int(self.player_y), self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (255,60,60), pygame.Rect(int(self.opponent_x), int(self.opponent_y), self.paddle_width, self.paddle_height))
        # ball
        pygame.draw.circle(self.screen, (60,255,60), (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        # score
        score_text = self.font.render(str(self.player_score), True, (220,220,220))
        self._blit_center(self.screen, score_text, (self.width * 0.33, 80))
        score_text2 = self.font.render(str(self.opponent_score), True, (220,220,220))
        self._blit_center(self.screen, score_text2, (self.width * 0.66, 80))
        # tag
        tag = pygame.font.SysFont("arial", 54).render("GUIDANCE ON" if self.guidance_active else "FREE PLAY", True,
                                                      (255,80,80) if self.guidance_active else (120,220,120))
        self.screen.blit(tag, (14, 10))
        if mode == "human":
            pygame.display.flip()
        return pygame.surfarray.array3d(self.screen).swapaxes(0,1)

    def step(self, action):
        time_sec = self.clock.tick(30) / 1000.0
        ai_speed = self.speed_circ * time_sec

        # input -> player_move
        if action == 0:
            self.player_move = -ai_speed
        elif action == 2:
            self.player_move = ai_speed
        else:
            self.player_move = 0.0

        # GUIDANCE assist (smooth, capped)
        if self.guidance_active:
            ball_c   = self.ball_y
            paddle_c = self.player_y + self.paddle_height / 2
            guide_move = np.clip(ball_c - paddle_c, -ai_speed * 1.2, ai_speed * 1.2)
            self.player_move = 0.9 * self.player_move + 0.1 * guide_move  # 90% AI, 10% guidance

        self.player_y += self.player_move
        self.player_y = float(np.clip(self.player_y, 0.0, self.height - self.paddle_height))

        # opponent controller (always active; smooth)
        accuracy = 0.6 + (self.opponent_skill * 0.3)
        reaction_speed = 0.4 + (self.opponent_skill * 0.6)
        target_y = self.ball_y + random.uniform(-15, 15) * (1.0 - accuracy)
        diff = (target_y - (self.opponent_y + self.paddle_height / 2))
        if abs(diff) > 4.0:
            speed = self.speed_circ * time_sec * reaction_speed
            self.opponent_y += np.clip(diff, -speed, speed)
        self.opponent_y = float(np.clip(self.opponent_y, 0.0, self.height - self.paddle_height))

        # ball physics
        self.ball_x += self.speed_x * time_sec
        self.ball_y += self.speed_y * time_sec

        # wall bounce
        if self.ball_y <= 5.0:
            self.ball_y = 5.0
            self.speed_y = abs(self.speed_y)
        elif self.ball_y >= self.height - 5.0:
            self.ball_y = self.height - 5.0
            self.speed_y = -abs(self.speed_y)

        reward = -0.001
        hit_flag = False

        # collision: player
        if self.ball_x - self.ball_radius <= self.player_x + self.paddle_width:
            if self.player_y - 5.0 <= self.ball_y <= self.player_y + self.paddle_height + 5.0:
                self.ball_x = self.player_x + self.paddle_width + self.ball_radius + 1.0
                self.speed_x = abs(self.speed_x)
                self.rally_length += 1
                reward = 1.0
                hit_flag = True

        # collision: opponent
        if self.ball_x + self.ball_radius >= self.opponent_x:
            if self.opponent_y - 5.0 <= self.ball_y <= self.opponent_y + self.paddle_height + 5.0:
                self.ball_x = self.opponent_x - self.ball_radius - 1.0
                self.speed_x = -abs(self.speed_x)

        # scoring
        if self.ball_x < 5.0:
            self.opponent_score += 1
            reward = -0.1
            self._update_opponent_skill()
            self.rally_length = 0
            self.ball_x, self.ball_y = 307.5, 232.5
            self.speed_x = -150.0      # serve toward AI
            self.speed_y = 150.0
            self.player_y, self.opponent_y = 215.0, 215.0

        elif self.ball_x > self.width - 5.0:
            self.player_score += 1
            reward = 1.0
            self.rally_length = 0
            self.ball_x, self.ball_y = 307.5, 232.5
            self.speed_x = -150.0      # still serve toward AI
            self.speed_y = 150.0
            self.player_y, self.opponent_y = 215.0, 215.0

        frame = self.render(mode="rgb_array") if self.render_mode else self._get_frame()
        return frame, float(reward), self.done, {
            "hit": hit_flag,
            "rally": self.rally_length,
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "guidance": self.guidance_active
        }

    def _update_opponent_skill(self):
        if self.episode_count <= 1:
            return
        self.ai_performance_history.append(self.rally_length)
        if len(self.ai_performance_history) > 20:
            self.ai_performance_history.pop(0)
        if len(self.ai_performance_history) >= 10:
            avg_rally = sum(self.ai_performance_history) / len(self.ai_performance_history)
            if avg_rally > 8:
                self.opponent_skill = min(0.95, self.opponent_skill + 0.03)
            elif avg_rally < 3:
                self.opponent_skill = max(0.5, self.opponent_skill - 0.02)

    def close(self):
        pygame.quit()
