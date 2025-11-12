# cortex.brain.pacman_3.py
# python -m cortex.brain.pacman_3
# Pac-Man–like environment with pygame + CORTEX 4.2 brain + logging + video
import os, random, argparse
import numpy as np
import pygame
import torch
from datetime import datetime
from cortex.brain.cortex_brain import create_small_brain_for_testing
from cortex.brain.brain_position import get_brain_coordinator
from cortex.brain.bme_data_collector import BMEDataCollector
import time
import json
import math
# ---------- Logging + Video ----------
import cv2
import csv
from collections import deque
import warnings
import builtins as _builtins
import matplotlib
matplotlib.use("Agg")  # headless backend for saving PNGs
import matplotlib.pyplot as plt
try:
    import mne
    _HAVE_MNE = True
except Exception:
    mne = None
    _HAVE_MNE = False
import sys

# Hide the tensor copy-construct warnings coming from enhanced_synapses_42.py
warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor",
    category=UserWarning,
    module=r".*enhanced_synapses_42"
)

_orig_print = _builtins.print
def _quiet_print(*args, **kwargs):
    s = " ".join(str(a) for a in args)


    # Drop EnhancedSynapse spam
    if s.startswith("[PRE] EnhancedSynapse42") or s.startswith("[POST] EnhancedSynapse42"):
        return

    # Drop EnhancedSynapse42 PyTorch device banners
    if s.startswith("EnhancedSynapse42 PyTorch:"):
        return
    # Drop EnhancedNeuron and Population init spam
    if s.startswith("Enhanced Neuron ") or s.startswith("Enhanced Population "):
        return
    
    # Drop CORTEX module banners (extended list)
    if s.startswith("EnhancedSynapticSystem42 PyTorch") \
       or s.startswith("Enhanced AstrocyteNetwork CORTEX") \
       or s.startswith("Enhanced Modulator CORTEX") \
       or s.startswith("Enhanced DopamineModulator CORTEX") \
       or s.startswith("Enhanced AchModulator CORTEX") \
       or s.startswith("ModulatorSystem42 ") \
       or s.startswith("Enhanced NeModulator ") \
       or s.startswith(" Enhanced Oscillator ") \
       or s.startswith("LimbicAmygdala42PyTorch") \
       or s.startswith(" Interoceptive Processor initialized") \
       or s.startswith(" Emotional Processor initialized") \
       or s.startswith(" Pain/Temperature Processor initialized") \
       or s.startswith(" Risk Assessment Processor initialized") \
       or s.startswith(" BiologicalNeuralCorrelation ") \
       or s.startswith(" BiologicalSelfBoundaryDetector ") \
       or s.startswith("BiologicalSpatialIntegration "):
        return

    # Drop GPU banner duplicates
    if s.lstrip().startswith("Using GPU:"):
        return

    # Drop motor + neural_output spam
    if s.startswith("MOTOR DEBUG:") or s.startswith("neural_output type:"):
        return

    _orig_print(*args, **kwargs)

_builtins.print = _quiet_print

class PacmanLogger:
    """
    Writes an overall CSV in the run root AND a per-episode CSV inside each episode folder.
    Also writes dopamine traces into the episode folder.
    """
    def __init__(self, outdir):
        self.root = outdir
        self.overall_path = os.path.join(outdir, "episodes_overall.csv")
        is_new = not os.path.exists(self.overall_path)
        self.f = open(self.overall_path, "a", newline="")
        self.writer = csv.writer(self.f)
        if is_new:
            self.writer.writerow([
                "episode","return","steps","ate","dopamine_mean","d_star",
                "efficiency","turns",
                "spike_count","firing_rate","E_rate","I_rate",
                "ATP_proxy","DA_mean","ACh_mean","E_I_balance"
            ])
            self.f.flush()

    def log_episode(self, ep, ret, steps, ate, dopamine_trace, d_star=None, turns=0, ep_dir=None):
        dmean = float(np.mean(dopamine_trace)) if len(dopamine_trace) > 0 else 0.0
        eff = (float(d_star)/float(steps)) if (d_star is not None and steps > 0) else 0.0
        # Biological proxies (safe means to avoid empty-slice warnings)
        def _safe_mean(x, default=0.0):
            arr = np.asarray(x)
            if arr.size == 0 or not np.isfinite(arr).any():
                return float(default)
            return float(np.nanmean(arr))

        spike_count = int(np.sum(getattr(self, "_last_spikes", [0])))   # total spikes in network
        firing_rate = float(spike_count) / max(1, steps)                # avg firing per step
        E_rate = _safe_mean(getattr(self, "_last_E", []))               # pyramidal mean firing
        I_rate = _safe_mean(getattr(self, "_last_I", []))               # interneuron mean firing
        ATP_proxy = spike_count * 1.0 + (E_rate + I_rate) * 0.1         # simple energy proxy
        DA_mean = _safe_mean(getattr(self, "_last_DA", []))             # dopamine trace
        ACh_mean = _safe_mean(getattr(self, "_last_ACh", []))           # acetylcholine trace
        E_I_balance = (E_rate / max(1e-6, I_rate)) if I_rate > 0 else 0

        # 1) Overall CSV row
        self.writer.writerow([
            ep, ret, steps, int(ate), dmean,
            d_star if d_star is not None else -1,
            eff, int(turns),
            spike_count, firing_rate, E_rate, I_rate,
            ATP_proxy, DA_mean, ACh_mean, E_I_balance
        ])
        self.f.flush()
            
        if ep_dir:
            os.makedirs(ep_dir, exist_ok=True)

            # 2) Per-episode CSV (single row)
            ep_csv = os.path.join(ep_dir, f"episode_{ep:03d}.csv")
            with open(ep_csv, "w", newline="") as ef:
                ew = csv.writer(ef)
                ew.writerow([
                    "episode","return","steps","ate","dopamine_mean","d_star","efficiency","turns",
                    "spike_count","firing_rate","E_rate","I_rate",
                    "ATP_proxy","DA_mean","ACh_mean","E_I_balance"
                ])
                ew.writerow([
                    ep, ret, steps, int(ate), dmean,
                    d_star if d_star is not None else -1, eff, int(turns),
                    spike_count, firing_rate, E_rate, I_rate,
                    ATP_proxy, DA_mean, ACh_mean, E_I_balance
                ])

            # 3) Dopamine trace (per episode) + PNG plot
            dop_path = os.path.join(ep_dir, f"dopamine_ep{ep:03d}.npy")
            dop = np.asarray(dopamine_trace, dtype=np.float32)
            np.save(dop_path, dop)
            try:
                fig = plt.figure(figsize=(6, 2))
                ax = fig.add_subplot(111)
                ax.plot(dop, linewidth=1)

                # robust axes for very short traces
                if dop.size > 1:
                    ax.set_xlim(0, len(dop) - 1)
                else:
                    ax.set_xlim(-0.5, 0.5)

                ymin, ymax = ax.get_ylim()
                pad = 0.05 * max(1e-6, abs(ymax - ymin))
                ax.set_ylim(ymin - pad, ymax + pad)

                ax.set_title(f"Dopamine ep{ep:03d}")
                ax.set_xlabel("t")
                ax.set_ylabel("reward/DA")
                fig.tight_layout()
                fig.savefig(os.path.join(ep_dir, f"dopamine_ep{ep:03d}.png"), dpi=150)
                plt.close(fig)

            except Exception:
                pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class GameplayRecorder:
    """
    One VideoWriter per episode, stored inside that episode's folder.
    Call start_episode(ep_dir), then record(...), then end_episode().
    """
    def __init__(self, fps=20.0, basename="gameplay"):
        self.fps = float(fps)
        self.basename = basename
        self.writer = None
        self.frame_size = None
        self.cur_path = None

    def start_episode(self, ep_dir):
        self.end_episode()  # just in case
        os.makedirs(ep_dir, exist_ok=True)
        self.cur_path = os.path.join(ep_dir, f"{self.basename}.mp4")
        self.writer = None
        self.frame_size = None

    def _ensure_writer(self, frame):
        h, w, _ = frame.shape
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.cur_path, fourcc, self.fps, (w, h))
            # Fallback to AVI/XVID if MP4 fails to open
            if (self.writer is None) or (not self.writer.isOpened()):
                self.cur_path = os.path.splitext(self.cur_path)[0] + ".avi"
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.writer = cv2.VideoWriter(self.cur_path, fourcc, self.fps, (w, h))
            self.frame_size = (w, h)

    def record(self, frame):
        if self.cur_path is None:
            return  # not started
        # frame is RGB; cv2 wants BGR
        self._ensure_writer(frame)
        self.writer.write(frame[:, :, ::-1])

    def end_episode(self):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.frame_size = None
        self.cur_path = None

    def close(self):
        self.end_episode()

# ---------- Pacman Env ----------
PALETTE = {
    "background": (10,10,20),
    "wall": (0,120,255),    # BLUE walls
    "pacman": (240,220,50),
    "food": (255,120,0),
    "food_glow": (255,180,60),
}

# ===== BUILT-IN MAZES (embedded; no files needed) =====
# Legend we use here:
#   walls: X | - n 0 1 6   (everything else is passable)
#   spawn: P  (first P found; if none, we keep default)
#   dots/pellets: we ignore here; pellets still placed by env logic
# ===== NUMERIC MAZE FORMAT (like your React code) =====
MAZES = {
    "maze1": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1],
        [1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1],
        [1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1],
        [1, 2, 2, 2, 1, 1, 5, 1, 1, 2, 2, 2, 1],
        [1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1],
        [1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 3, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "maze2": [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1],
        [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 2, 1, 1, 5, 1, 1, 2, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
        [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1],
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
}

class PacmanEnv:
    def __init__(self, H=10, W=10, seed=42, max_steps=None, step_cost=0.0, pellets=1, maze_name: str="", deterministic=False, fixed_spawn=None, fixed_pellets=None):
        self.H, self.W = H, W
        self.rng = random.Random(seed)
        self.max_steps = int(max_steps) if (max_steps is not None and max_steps > 0) else 4 * self.H * self.W
        self.step_cost = float(step_cost)
        self.t = 0
        self.prev_dist = None
        self._prev_action = None
        self.d0 = None

        # design toggles (initialized; can be overwritten by runner)
        self.afterglow_steps = 0
        self.afterglow_reward = 0.0
        self._afterglow = 0
        self.halo_radius = 0
        self.halo_reward = 0.0

        # --- speed-based reward shaping (new) ---
        self.fast_pellet_bonus = 0.0     # per-pellet bonus grows when eaten earlier
        self.fast_finish_bonus = 0.0     # big completion bonus grows with speed
        self.finish_power = 1.0          # exponent shaping for finish bonus

        # --- proximity (“smell”) shaping (new defaults; overridable by runner) ---
        self.smell_gain = 0.5
        self.smell_lam  = 2.0

        # --- Pacman 3: multi-pellet state (required by reset/_place_pellets/step) ---
        self.pellets_target = int(max(1, pellets))
        self.pellet_set = set()
        self.pellet_times = []
        self.visited = set()
        self.walls = set()
        # Deterministic mode for validation
        self.deterministic = deterministic
        self.fixed_spawn = fixed_spawn
        self.fixed_pellets = fixed_pellets
        # embedded-maze selector
        self._maze_name = (maze_name or "").strip().lower()  # "", "maze1", "maze2"

        self._make_simple_maze()
        self.reset()

    def _parse_embedded_maze(self, s: str):
        """Fill walls and (optionally) spawn + pellet set from an embedded maze string."""
        self.walls.clear()
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
        rows = [ln.split() for ln in lines]
        H = len(rows)
        W = max((len(r) for r in rows), default=0)  # handle ragged rows

        if H > 0 and W > 0:
            self.H, self.W = H, W

        spawn = None
        # treat any digit as a wall too (the maze uses 2,3,7,8,9, …)
        WALL = set(["X","|","-","n"]) | set(list("0123456789"))
        PEL  = {".", "+"}  # pellets from map tokens
        pellets = set()

        for y, row in enumerate(rows):
            for x, tok in enumerate(row):
                if x >= self.W:
                    continue
                if tok in WALL:
                    self.walls.add((y, x))
                elif tok.upper() == "P" and spawn is None:
                    spawn = (y, x)
                elif tok in PEL:
                    pellets.add((y, x))

        if spawn is not None:
            self._forced_spawn = spawn
        # store pellet positions parsed from the map
        self._pellets_from_maze = pellets

    def _parse_numeric_maze(self, maze_array):
        """Parse numeric array maze (like React format).
        
        Legend:
            1 = wall
            2 = pellet/coin
            3 = empty ground
            5 = Pacman spawn
        """
        self.walls.clear()
        H = len(maze_array)
        W = max(len(row) for row in maze_array) if H > 0 else 0
        
        if H > 0 and W > 0:
            self.H, self.W = H, W
        
        spawn = None
        pellets = set()
        
        for y, row in enumerate(maze_array):
            for x, cell in enumerate(row):
                if cell == 1:  # Wall
                    self.walls.add((y, x))
                elif cell == 2:  # Pellet
                    pellets.add((y, x))
                elif cell == 5:  # Pacman spawn
                    if spawn is None:
                        spawn = (y, x)
                # cell == 3 is empty ground, do nothing
        
        if spawn is not None:
            self._forced_spawn = spawn
        self._pellets_from_maze = pellets

    def _make_simple_maze(self):
        """Populate walls either from embedded maze or fallback simple maze."""
        self.walls.clear()
        name = getattr(self, "_maze_name", "")
        if name in MAZES:
            maze_data = MAZES[name]
            # Check if it's numeric array (list) or string
            if isinstance(maze_data, list):
                self._parse_numeric_maze(maze_data)
            else:
                self._parse_embedded_maze(maze_data)
            return

        # --- fallback simple layout (old behavior) ---
        H, W = self.H, self.W
        if H < 5 or W < 5:
            return

        # Perimeter walls
        for y in range(H):
            self.walls.add((y, 0))
            self.walls.add((y, W - 1))
        for x in range(W):
            self.walls.add((0, x))
            self.walls.add((H - 1, x))

        # Interior grid bars (leave every other cell open to keep paths)
        cx, cy = W // 2, H // 2
        for y in range(1, H - 1):
            if y % 2 == 0:
                self.walls.add((y, cx))
        for x in range(1, W - 1):
            if x % 2 == 0:
                self.walls.add((cy, x))

        # Ensure center pocket is open (good default spawn zone)
        for dy, dx in [(0,0), (-1,0), (1,0), (0,-1), (0,1)]:
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < H and 0 <= xx < W and (yy, xx) in self.walls:
                self.walls.remove((yy, xx))

    def reset(self):
        self.done = False
        self.t = 0
        
        # Deterministic spawn takes priority
        if self.deterministic and self.fixed_spawn is not None:
            self.pos = tuple(self.fixed_spawn)
            self.spawn_pos = self.pos
        elif hasattr(self, "_forced_spawn") and self._forced_spawn is not None:
            self.pos = tuple(self._forced_spawn)
            self.spawn_pos = self.pos
        else:
            self.pos = (self.H//2, self.W//2)
            self.spawn_pos = self.pos

        # If spawn is inside a wall, relocate to nearest free cell
        if self.pos in self.walls:
            from collections import deque
            H, W = self.H, self.W
            q = deque([self.pos])
            seen = {self.pos}
            free = None
            while q:
                y, x = q.popleft()
                if (y, x) not in self.walls:
                    free = (y, x); break
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    yy, xx = y+dy, x+dx
                    if 0 <= yy < H and 0 <= xx < W and (yy, xx) not in seen:
                        seen.add((yy, xx)); q.append((yy, xx))
            if free is not None:
                self.pos = free
                self.spawn_pos = free

        # Pacman 3: pellets
        if self.deterministic:
            if self.fixed_pellets is not None:
                self.pellet_set = set(self.fixed_pellets)
            elif hasattr(self, "_pellets_from_maze") and self._pellets_from_maze:
                self.pellet_set = set(self._pellets_from_maze)
            else:
                self._place_pellets()
        elif hasattr(self, "_pellets_from_maze") and self._pellets_from_maze:
            src = [p for p in self._pellets_from_maze if p not in self.walls and p != self.pos]
            if self.pellets_target > 0 and self.pellets_target < len(src):
                self.pellet_set = set(self.rng.sample(src, self.pellets_target))
            else:
                self.pellet_set = set(src)
                self.pellets_target = len(self.pellet_set)
        else:
            self._place_pellets()

        # cache optimal steps from start to nearest pellet
        if self.pellet_set:
            self.d0 = min(self._L1(self.pos, p) for p in self.pellet_set)
        else:
            self.d0 = 0

        # initialize previous distance (for shaping)
        self.prev_dist = self.d0
        self._afterglow = 0
        self.pellet_times.clear()
        self.visited.clear()
        return self.render()
    
    def _place_pellets(self):
        """Place pellets_target many pellets; avoid spawning on Pac-Man or spawn cell."""
        self.pellet_set.clear()
        # If a stage policy exists, seed one near spawn, then fill the rest randomly.
        # (Keeps compatibility with your previous policies but now supports many pellets.)
        def _ok(y, x):
            if (y, x) == self.pos: return False
            if hasattr(self, "spawn_pos") and self.spawn_pos is not None and (y, x) == self.spawn_pos: return False
            if (y, x) in self.walls: return False
            return True

        # Optional seed near spawn if a policy is specified
        if hasattr(self, "_food_policy") and self._food_policy:
            ay, ax = (self.spawn_pos if hasattr(self, "spawn_pos") and self.spawn_pos is not None else self.pos)
            if self._food_policy == "adjacent_right":
                fy, fx = ay, min(self.W-1, ax+1)
                if _ok(fy, fx):
                    self.pellet_set.add((fy, fx))
            elif self._food_policy == "left_or_right":
                cand = [(ay, max(0, ax-1)), (ay, min(self.W-1, ax+1))]
                cand = [p for p in cand if _ok(*p)]
                if cand:
                    self.pellet_set.add(random.choice(cand))
            elif self._food_policy == "neighbor":
                nbr = []
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    y, x = ay+dy, ax+dx
                    if 0 <= y < self.H and 0 <= x < self.W and _ok(y, x):
                        nbr.append((y, x))
                if nbr:
                    self.pellet_set.add(random.choice(nbr))

        # Fill remaining pellets randomly
        while len(self.pellet_set) < self.pellets_target:
            fy = self.rng.randrange(0, self.H)
            fx = self.rng.randrange(0, self.W)
            if _ok(fy, fx):
                self.pellet_set.add((fy, fx))
            
    def _L1(self, a, b):
        """Standard Manhattan distance on a bounded grid."""
        (y1, x1), (y2, x2) = a, b
        return abs(y2 - y1) + abs(x2 - x1)
    
    def render(self):
        """Return an HxWx3 RGB array for the current grid state."""
        rgb = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        # background
        rgb[:, :, 0] = PALETTE["background"][0]
        rgb[:, :, 1] = PALETTE["background"][1]
        rgb[:, :, 2] = PALETTE["background"][2]

        # walls first (guard indices)
        if hasattr(self, "walls"):
            for (wy, wx) in self.walls:
                if 0 <= wy < self.H and 0 <= wx < self.W:
                    rgb[wy, wx, 0] = PALETTE["wall"][0]
                    rgb[wy, wx, 1] = PALETTE["wall"][1]
                    rgb[wy, wx, 2] = PALETTE["wall"][2]

        # pellets (guard + skip if overlapping walls)
        for (fy, fx) in self.pellet_set:
            if 0 <= fy < self.H and 0 <= fx < self.W and (fy, fx) not in self.walls:
                rgb[fy, fx, 0] = PALETTE["food"][0]
                rgb[fy, fx, 1] = PALETTE["food"][1]
                rgb[fy, fx, 2] = PALETTE["food"][2]

        # pacman (guard)
        py, px = self.pos
        if 0 <= py < self.H and 0 <= px < self.W:
            rgb[py, px, 0] = PALETTE["pacman"][0]
            rgb[py, px, 1] = PALETTE["pacman"][1]
            rgb[py, px, 2] = PALETTE["pacman"][2]

        return rgb

    def step(self, action:int):
        # move
        y, x = self.pos
        if action == 0: y -= 1
        elif action == 1: y += 1
        elif action == 2: x -= 1
        elif action == 3: x += 1

        y = max(0, min(self.H - 1, y))
        x = max(0, min(self.W - 1, x))
        # if target is a wall, stay in place
        if (y, x) in self.walls:
            y, x = self.pos
        self.pos = (y, x)

        # time advances
        self.t += 1
        # smooth proximity shaping (exponential “smell”) to the NEAREST pellet
        if self.pellet_set:
            d_prev = min(self._L1(getattr(self, "_prev_pos", self.pos), p) for p in self.pellet_set)
            d_curr = min(self._L1(self.pos, p) for p in self.pellet_set if p not in self.walls) if self.pellet_set else 0
        else:
            d_prev = d_curr = 0
        lam = float(getattr(self, "smell_lam", 2.0))
        phi_prev = math.exp(- d_prev / lam)
        phi_curr = math.exp(- d_curr / lam)
        gain = float(getattr(self, "smell_gain", 0.5))
        r_smell = gain * (phi_curr - phi_prev)

        # per-step metabolic cost
        reward = r_smell - self.step_cost

        # remember previous pos for next delta
        self._prev_pos = self.pos
        # appetitive halo around food (dense gradient, distance 1..R)
        if self.halo_radius and d_curr > 0 and d_curr <= self.halo_radius:    
            # linear decay with distance: 1.0 at dist=1, ~0 at dist=R
            scale = max(0.0, (self.halo_radius - (d_curr - 1)) / max(1, self.halo_radius))
            reward += self.halo_reward * float(scale)
            # Optional: penalize turning only when moving away from food
            turn_pen = getattr(self, "turn_penalty", 0.0)
            if turn_pen > 0.0 and self._prev_action is not None and action != self._prev_action:
                d_next = (min(self._L1(self.pos, p) for p in self.pellet_set) if self.pellet_set else 0)
                if d_next > d_curr:
                    reward -= float(turn_pen)
            self._prev_action = action

        # eat pellet by stepping onto it (no sprite mouth; just position match)
        ate = False
        if self._afterglow == 0 and self.pos in self.pellet_set:
            self.pellet_set.remove(self.pos)
            ate = True

            # base pellet reward
            reward += 0.05

            # === per-pellet speed bonus (earlier step -> bigger) ===
            Tcap = float(self.max_steps or (4 * self.H * self.W))
            t_frac = min(1.0, self.t / max(1.0, Tcap))   # 0 early … 1 late
            early = 1.0 - t_frac                          # 1 early … 0 late
            reward += float(self.fast_pellet_bonus) * early

            self.pellet_times.append(self.t)

            # if all pellets are gone, add clear bonus and end after afterglow
            if len(self.pellet_set) == 0:
                # base clear bonus (keeps old behavior)
                reward += 1.0

                # === fast-finish bonus (more remaining time -> bigger) ===
                rem_frac = 1.0 - t_frac
                shaped = max(0.0, rem_frac) ** float(self.finish_power)
                reward += float(self.fast_finish_bonus) * shaped

                self._afterglow = int(self.afterglow_steps)
                if self._afterglow <= 0:
                    self.done = True
        else:
            if self.max_steps and self.t >= self.max_steps:
                self.done = True

        
        # Afterglow payout + graceful end after N steps
        if self._afterglow > 0:
            reward += float(self.afterglow_reward) * (self._afterglow / max(1, self.afterglow_steps))
            self._afterglow -= 1
            if self._afterglow == 0:
                self.done = True

        return self.render(), reward, self.done, {
            "ate": ate,
            "d0": self.d0,
            "pellets_left": len(self.pellet_set),
            "pellets_total": self.pellets_target
        }        

# ---------- Brain Adapter ----------
class BrainAdapter:
    def save(self, path: str):
        """Save the brain weights (state_dict if available, else pickle the object)."""
        try:
            if hasattr(self.brain, "state_dict"):
                torch.save({"state_dict": self.brain.state_dict()}, path)
            else:
                torch.save({"pickled": self.brain}, path)
        except Exception as e:
            print(f"[BRAIN] save failed: {e}")

    def load(self, path: str):
        """Load brain weights (state_dict if available, else unpickle)."""
        try:
            data = torch.load(path, map_location=self.device)
            if isinstance(data, dict) and "state_dict" in data and hasattr(self.brain, "load_state_dict"):
                self.brain.load_state_dict(data["state_dict"])
                # move to device if module supports .to()
                if hasattr(self.brain, "to"):
                    self.brain.to(self.device)
            elif isinstance(data, dict) and "pickled" in data:
                self.brain = data["pickled"]
                if hasattr(self.brain, "to"):
                    self.brain.to(self.device)
            print(f"[BRAIN] loaded checkpoint: {path}")
        except Exception as e:
            print(f"[BRAIN] load failed from {path}: {e}")

    def __init__(self, device=None, run_dir=None):

        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.run_dir = run_dir or os.getcwd()
        self.brain = create_small_brain_for_testing(device=self.device)
        self.step_counter = 0
        self._prev_action = None  # for turn penalty


        # === EEG GEOMETRY & LEADFIELD (precompute) ===
        try:
            coord = get_brain_coordinator(device=self.device)
            geom = coord.export_coordinates_for_eeg()
            # (E,3) electrode points (torch -> np)
            self._eeg_electrodes = np.asarray(
                geom["electrode_positions"].detach().cpu().numpy(),
                dtype=np.float32
            )
            # raw region centers dict -> name -> (3,)
            self._region_centers_raw = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in geom["region_centers"].items()
            }
        except Exception as e:
            print("[EEG] brain_position not available, using fallback:", e)
            th = np.linspace(0, 2*np.pi, 10, endpoint=False)
            self._eeg_electrodes = np.stack(
                [np.cos(th)*8.0, np.sin(th)*8.0, np.ones_like(th)*6.0],
                axis=1
            ).astype(np.float32)
            self._region_centers_raw = {
                "PFC": np.array([0.0, 7.0, 6.0], np.float32),
                "M1":  np.array([0.0, 0.0, 6.0], np.float32),
                "HPC": np.array([-5.0,-5.0,5.0], np.float32),
                "AMY": np.array([ 5.0,-5.0,5.0], np.float32),
                "TH":  np.array([ 0.0, 0.0,4.0], np.float32),
                "CB":  np.array([ 0.0,-8.0,4.0], np.float32),
            }

        # macro regions must match the order of region_activity you log
        self._macro_regions = [
            'motor','sensory','thalamus','cerebellum',
            'parietal','pfc','limbic','hippocampus','insula','basal_ganglia'
        ]

        self._macro_centers = self._collapse_to_macro_centers(
            self._region_centers_raw, self._macro_regions
        )
        self._leadfield = self._build_leadfield(
            self._eeg_electrodes, self._macro_centers
        )  # shape (E,R)

        self._leadfield /= (np.linalg.norm(self._leadfield, axis=0, keepdims=True) + 1e-6)

        # --- sanity guard: if leadfield sums to ~0, fall back to uniform weights ---
        if float(np.abs(self._leadfield).sum()) == 0.0 or not np.isfinite(self._leadfield).any():
            R = int(self._leadfield.shape[1])
            self._leadfield[:] = 1.0 / max(1, R)

        np.save(os.path.join(self.run_dir, 'leadfield.npy'), self._leadfield)
        np.savetxt(os.path.join(self.run_dir, 'leadfield.csv'), self._leadfield, delimiter=',')
        print(f"[EEG] leadfield shape={self._leadfield.shape} "
            f"sum={float(np.abs(self._leadfield).sum()):.3e} "
            f"min={float(self._leadfield.min()):.3e} max={float(self._leadfield.max()):.3e}")

        self._eeg_buffer = []  # per-episode EEG samples (T,E)
        # --- LFP emergence: state buffers + tiny per-electrode phase mixing ---
        self._lfp_state = None           # (R,) low-pass region activity (current)
        self._lfp_prev  = None           # (R,) previous-step LFP (for slight phase offsets)
        # per-electrode mixing coefficient (creates tiny phase differences per channel)
        # 0.05..0.25 is small but enough to prevent perfect alignment
        import numpy as _np
        E = int(self._leadfield.shape[0])
        self._elec_beta = _np.linspace(0.00, 0.06, E).astype(_np.float32)  # tiny per-electrode phase mix ↑
        # keep previous eeg sample per electrode for tiny lag-mixing
        self._prev_eeg = None  # shape (E,), set on first eeg step
        # per-episode buffers for raw traces
        self._ra_buffer = []       # (T, R) region_activity
        self._bio_means = {        # scalar means per step
            "E": [], "I": [], "DA": [], "ACh": []
        }
        # Overall tracking across all episodes (incremental saves)
        self._overall_eeg_list = []
        self._overall_region_list = []
        self._overall_dopamine_list = []

    def _safe_tensor(self, arr): return torch.as_tensor(arr,dtype=torch.float32,device=self.device)
    # ---- small helpers ----
    def _to_np(self, x):
        """Convert torch tensor / list / np-like to np.float32 safely."""
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        try:
            return np.asarray(x, dtype=np.float32)
        except Exception:
            return np.array([], dtype=np.float32)
          
    # ===== EEG HELPERS =====
    def _collapse_to_macro_centers(self, raw_centers: dict, macro_keys: list) -> np.ndarray:
        """
        raw_centers: dict like {'PFC_dorsal': [x,y,z], 'HPC_CA1': [x,y,z], ...}
        macro_keys:  ['motor','unified_neocortex','thalamus', ...]
        Returns array (R,3) of averaged centers per macro region.
        """
        buckets = {k: [] for k in macro_keys}

        def add_if_match(name: str, key: str) -> bool:
            n = name.lower()
            aliases = {
                'motor': ['motor','m1','betz'],
                'unified_neocortex': ['unified','neocortex','cortex','ctx'],
                'thalamus': ['thalamus','th'],
                'cerebellum': ['cerebellum','cb'],
                'parietal': ['parietal','pctx','pc'],
                'pfc': ['pfc','prefrontal'],
                'limbic': ['limbic','amygdala','amy','la','cea'],
                'hippocampus': ['hippocampus','hpc','ca1','ca3','dg'],
                'insula': ['insula','ins'],
                'basal_ganglia': ['basal_ganglia','bg','caudate','putamen','striatum','gpe','gpi'],
                'sensory': ['sensory','s1','somato','postcentral','unified','neocortex','cortex','ctx'],
            }
            keys = aliases.get(key, [key])
            return any(k in n for k in keys)

        for name, xyz in raw_centers.items():
            for key in macro_keys:
                if add_if_match(name, key):
                    buckets[key].append(np.asarray(xyz, dtype=np.float32))

        centers = []
        for key in macro_keys:
            if buckets[key]:
                centers.append(np.mean(np.stack(buckets[key], axis=0), axis=0))
            else:
                centers.append(np.array([0.0, 0.0, 5.0], dtype=np.float32))
        return np.stack(centers, axis=0)  # (R,3)

    def _build_leadfield(self, electrodes: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        electrodes: (E,3), centers: (R,3)
        Returns L: (E,R) with weights ~ exp(-d/λ), normalized per electrode.
        """
        E, R = electrodes.shape[0], centers.shape[0]
        L = np.zeros((E, R), dtype=np.float32)
        lam = 5.0  # length scale (mm), tune 3–10 based on head size
        for e in range(E):
            for r in range(R):
                d = np.linalg.norm(electrodes[e] - centers[r])
                v = electrodes[e] - centers[r]
                if d > 1e-6:
                    v_hat = v / d
                else:
                    v_hat = np.array([0.0,0.0,1.0], dtype=np.float32)
                # Use region center as a crude dipole “normal” (better than distance-only)
                n = centers[r]
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-6:
                    n_hat = n / n_norm
                else:
                    n_hat = np.array([0.0,0.0,1.0], dtype=np.float32)
                # use absolute value so orientation doesn't kill signal
                orient = abs(float(np.dot(n_hat, v_hat)))
                L[e, r] = (0.3 + orient) * np.exp(-d / lam)  # 0.3 baseline + orientation boost
        
        L /= (L.sum(axis=1, keepdims=True) + 1e-6)
        return L

    def eeg_step_from_region_activity(self, region_activity: list):
        """
        region_activity order must be:
        ['motor','sensory','thalamus','cerebellum',
        'parietal','pfc','limbic','hippocampus','insula','basal_ganglia']
        Appends one (E,) EEG sample = leadfield @ activity.
        """
        if not hasattr(self, "_leadfield"):
            return None
        try:
            import numpy as _np
            
            a = _np.asarray(region_activity, dtype=_np.float32)
            R = self._leadfield.shape[1]
            if a.size != R:
                if a.size > R: a = a[:R]
                else: a = _np.pad(a, (0, R - a.size))

            # NEW: Temporal filtering for realistic dynamics
            if (not hasattr(self, '_lfp_state')) or (self._lfp_state is None) or (not hasattr(self, '_lfp_prev')) or (self._lfp_prev is None):
                self._lfp_state = a.copy()
                self._lfp_prev = a.copy()
            
            # Low-pass filter (simulates synaptic/volume conduction)
            alpha = 0.30  # Temporal smoothing factor
            self._lfp_state = alpha * self._lfp_prev + (1 - alpha) * a
            self._lfp_prev = self._lfp_state.copy()
            
            # Use filtered signals instead of raw
            a_filtered = self._lfp_state

            # CENTER around 0 so oscillations are visible
            a_centered = a_filtered - float(a_filtered.mean())  # center by current mean
            eeg_base = self._leadfield @ a_centered

            # per-electrode lag mixing using previous sample (creates tiny phase offsets)
            if hasattr(self, '_elec_beta'):
                if self._prev_eeg is None or len(self._prev_eeg) != len(eeg_base):
                    # first step: seed previous with current
                    self._prev_eeg = eeg_base.copy()
                # first-order per-channel lag: current mixes with last step using beta[i]
                eeg_base = (1.0 - self._elec_beta) * eeg_base + self._elec_beta * self._prev_eeg
                self._prev_eeg = eeg_base.copy()

            # Scale to microvolts ONCE
            scale_uv = 50.0
            eeg_sample = scale_uv * eeg_base

            # Common-average reference (demean per step) -> removes DC offset
            eeg_sample = eeg_sample - float(eeg_sample.mean())

            # NEW: Add 1/f pink noise for biological realism
            if not hasattr(self, '_pink_state'):
                self._pink_state = _np.random.randn(len(eeg_sample))
            white = _np.random.randn(len(eeg_sample))
            self._pink_state = 0.95 * self._pink_state + 0.05 * white
            eeg_sample = eeg_sample + self._pink_state * 0.05  # 1/f component

            # Common-average reference BEFORE sensor noise
            eeg_sample = eeg_sample - float(eeg_sample.mean())

            # Soft-clip extreme spikes so plots look sane
            import numpy as _np
            eeg_sample = _np.clip(eeg_sample, -150.0, 150.0).astype(_np.float32)

            # (optional tiny sensor noise ~3 µV RMS)
            eeg_sample = eeg_sample + _np.random.normal(0.0, 3.0, size=eeg_sample.shape).astype(_np.float32)

            # Log to CSV
            log_csv = os.path.join(self.run_dir, 'eeg_region_log.csv')
            if not hasattr(self, '_eeg_csv_initialized'):
                self._eeg_csv_initialized = True
                with open(log_csv, 'w', newline='') as f:
                    f.write('# Direct brain oscillations (no synthetic overlay)\n')
                    f.write('t,' + ','.join(self._macro_regions) + ',' + ','.join([f'Ch{i+1}' for i in range(len(eeg_sample))]) + '\n')

            with open(log_csv, 'a', newline='') as f:
                t_val = int(self._step_counter) if hasattr(self, '_step_counter') else 0
                f.write(f"{t_val}," + ",".join(map(str, a.tolist())) + "," + ",".join(map(str, eeg_sample.tolist())) + "\n")

            self._eeg_buffer.append(eeg_sample)
            if (getattr(self, '_step_counter', 0) % 50) == 0:
                print(f"[EEG dbg] step={getattr(self,'_step_counter',0)} "
                    f"min={float(eeg_sample.min()):.3e} max={float(eeg_sample.max()):.3e}")

            self._step_counter = getattr(self, '_step_counter', 0) + 1
            return eeg_sample
        except Exception:
            return None

    def eeg_reset_episode(self):
        self._eeg_buffer = []

    def eeg_save_episode(self, ep_dir: str, ep_idx: int):
        if getattr(self, "_eeg_buffer", None) and len(self._eeg_buffer) > 0:
            arr = np.nan_to_num(np.stack(self._eeg_buffer, axis=0).astype(np.float32),
                                nan=0.0, posinf=0.0, neginf=0.0)  # (T,E)
            print(f"[EEG] buffer_len={len(self._eeg_buffer)} shape={arr.shape} dir={ep_dir}")
            
            out_path = os.path.join(ep_dir, f"eeg_ep{ep_idx:03d}.npy")
            np.save(out_path, arr.astype(np.float32))

            # === ALSO SAVE AS MNE FIF ===
            try:
                E = arr.shape[1]
                _std_1020 = ['Fp1','Fp2','F3','F4','F7','F8','Fz',
                             'C3','C4','Cz','P3','P4','Pz','O1','O2',
                             'T7','T8','P7','P8','POz','Oz','FC1','FC2','CP1','CP2']
                ch_names = _std_1020[:E] if E <= len(_std_1020) else [f'EEG{i+1}' for i in range(E)]
                info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types='eeg')
                raw = mne.io.RawArray(arr.T * 1e-6, info)  # µV -> V, shape (ch, time)
                try:
                    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
                except Exception:
                    raw.set_montage(mne.channels.make_standard_montage('standard_1005'))
                fif_path = os.path.join(ep_dir, f"eeg_ep{ep_idx:03d}.fif")
                raw.save(fif_path, overwrite=True)
                
                # === MNE 3D VIEWER ===
                try:
                    here = os.path.dirname(os.path.abspath(__file__))
                    viewer_py = os.path.join(here, "mne_eeg_3d_regions_patched.py")
                    if os.path.exists(viewer_py):
                        cmd = f'"{sys.executable}" "{viewer_py}" --eeg "{fif_path}"'
                        os.system(cmd)
                    else:
                        print(f"[EEG] Viewer script not found at: {viewer_py}")
                except Exception as e:
                    print(f"[EEG] Could not open MNE viewer: {e}")

                print(f"[EEG] FIF saved -> {fif_path}")
            except Exception as e:
                print(f"[EEG] FIF save failed: {e}")
                
            # PNG line plot (each electrode as a trace)
            try:
                fig = plt.figure(figsize=(6, 2.5))
                ax = fig.add_subplot(111)
                T = arr.shape[0]
                if T < 2:
                    ax.text(0.5, 0.5, f'Episode too short: {T} step(s)',
                            ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlim(-0.5, 0.5)
                else:
                    for e in range(arr.shape[1]):
                        ax.plot(arr[:, e], label=f"ch{e}", linewidth=1)
                    ax.set_xlim(0, T - 1)
                ax.set_title(f"EEG ep{ep_idx:03d} (channels)")
                ax.set_xlabel("time step")
                ax.set_ylabel("amplitude (µV)")
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(loc="upper right", fontsize=6, ncol=4) 
                fig.tight_layout()
                fig.savefig(os.path.join(ep_dir, f"eeg_ep{ep_idx:03d}.png"), dpi=150)
                plt.close(fig)

            except Exception:
                pass
            print(f"[EEG] saved -> {out_path} shape={arr.shape}")

    def trace_step(self, region_activity, E, I, DA, ACh, brain_output):
        """Enhanced trace collection for thesis-quality data"""
        bg = None  # ensure defined even if 'basal_ganglia' is missing
        # store region activity (10-d expected) and scalar means per step
        try:
            ra = np.asarray(region_activity, dtype=np.float32)
            if ra.ndim == 0 or ra.size == 0:
                return
            # pad/trim to macro region count
            R = len(self._macro_regions) if hasattr(self, "_macro_regions") else 10
            if ra.size < R:
                ra = np.pad(ra, (0, R - ra.size))
            elif ra.size > R:
                ra = ra[:R]
            self._ra_buffer.append(ra.astype(np.float32))
            
            # means of populations (fallback to 0 if empty)
            def _m(x):
                # Handle scalar values (floats, single tensors)
                if isinstance(x, (int, float)):
                    return float(x)
                if torch.is_tensor(x):
                    if x.ndim == 0:  # scalar tensor
                        return float(x.item())
                    return float(x.mean().item())
                # Handle arrays/lists
                if not hasattr(x, "__len__"):
                    return 0.0
                if len(x) == 0:
                    return 0.0
                arr = []
                for t in x:
                    if torch.is_tensor(t):
                        arr.append(float(t.detach().cpu().item() if t.numel() == 1 else t.detach().cpu().numpy().mean()))
                    else:
                        arr.append(float(t))
                return float(np.mean(arr)) if arr else 0.0
            
            self._bio_means["E"].append(_m(E))
            self._bio_means["I"].append(_m(I))
            self._bio_means["DA"].append(_m(DA))
            self._bio_means["ACh"].append(_m(ACh))
            
            # === NEW: Capture comprehensive brain state ===
            # TD Learning signals
            if 'basal_ganglia' in brain_output:
                bg = brain_output['basal_ganglia']
                if isinstance(bg, dict):
                    self._bio_means.setdefault("td_error", []).append(float(bg.get('td_error', 0.0)))
                    self._bio_means.setdefault("value", []).append(float(bg.get('state_value', 0.0)))
                    
                    # Action values (Q-values) - store as array
                    if 'action_values' in bg:
                        av = bg['action_values']
                        if torch.is_tensor(av):
                            av = av.detach().cpu().numpy()
                        else:
                            av = np.asarray(av, dtype=np.float32)
                        av = np.nan_to_num(av, nan=0.0, posinf=0.0, neginf=0.0)
                        if not hasattr(self, '_action_values_buffer'):
                            self._action_values_buffer = []
                        self._action_values_buffer.append(av)
            # Also capture top-level learning signals if present
            if 'td_error' in brain_output:
                self._bio_means.setdefault("td_error", []).append(float(brain_output.get('td_error', 0.0)))
            if 'value_estimate' in brain_output:
                self._bio_means.setdefault("value", []).append(float(brain_output.get('value_estimate', 0.0)))

            # Capture action-values whether top-level or nested
            av_top = brain_output.get('action_values', None)
            if av_top is not None:
                import numpy as _np, torch as _torch
                if not hasattr(self, '_action_values_buffer'):
                    self._action_values_buffer = []
                if _torch.is_tensor(av_top):
                    av_np = av_top.detach().cpu().numpy()
                else:
                    av_np = _np.asarray(av_top, dtype=_np.float32)
                av_np = _np.nan_to_num(av_np, nan=0.0, posinf=0.0, neginf=0.0)
                self._action_values_buffer.append(av_np)

            # Policy entropy (safe)
            if isinstance(bg, dict) and 'policy_entropy' in bg:
                self._bio_means.setdefault("entropy", []).append(float(bg['policy_entropy']))

            # Neuromodulators (full tri-modulation)
            if 'neuromodulators' in brain_output or 'modulators' in brain_output:
                mods = brain_output.get('neuromodulators', brain_output.get('modulators', {}))
                if isinstance(mods, dict):
                    self._bio_means.setdefault("NE", []).append(float(mods.get('norepinephrine', 0.0)))
            
            # Per-region spike counts
            if 'spikes_per_region' in brain_output:
                spr = brain_output['spikes_per_region']
                # Fix: Handle case where spikes_per_region is a list instead of dict
                if isinstance(spr, list):
                    # Skip if it's just region names, not spike data
                    pass
                elif isinstance(spr, dict):

                    for region_name, spike_arr in spr.items():
                        key = f"spikes_{region_name}"
                        if torch.is_tensor(spike_arr):
                            count = float(spike_arr.sum().item())
                        else:
                            count = float(np.sum(spike_arr))
                        self._bio_means.setdefault(key, []).append(count)
            
            # Oscillatory bands (if available)
            if 'oscillatory_state' in brain_output:
                osc = brain_output['oscillatory_state']
                if isinstance(osc, dict):
                    for region_name, bands in osc.items():
                        if isinstance(bands, dict):
                            for band_name, value in bands.items():
                                key = f"osc_{region_name}_{band_name}"
                                self._bio_means.setdefault(key, []).append(float(value))
        
        except Exception as e:
            print(f"[trace_step] error: {e}")
            pass

    def traces_reset_episode(self):
        self._ra_buffer = []
        self._bio_means = {"E": [], "I": [], "DA": [], "ACh": []}
        self._action_values_buffer = []  # Q-values per step
   
    def traces_save_episode(self, ep_dir: str, ep_idx: int):
        try:
            if len(self._ra_buffer) > 0:
                ra = np.stack(self._ra_buffer, axis=0)  # (T,R)
                ra_path = os.path.join(ep_dir, f"region_activity_ep{ep_idx:03d}.npy")
                np.save(ra_path, ra.astype(np.float32))

                # PNG line plot (each region as a trace)
                try:
                    fig = plt.figure(figsize=(6, 2.5))
                    ax = fig.add_subplot(111)
                    T = ra.shape[0]

                    if T < 2:
                        # Skip plotting for episodes with <2 steps (can't draw lines)
                        ax.text(0.5, 0.5, f'Episode too short: {T} step(s)',
                                ha='center', va='center', transform=ax.transAxes)
                        ax.set_xlim(-0.5, 0.5)
                    else:
                        for r in range(ra.shape[1]):
                            ax.plot(ra[:, r], label=f"R{r}", linewidth=1)
                        ax.set_xlim(0, T - 1)
                        handles, labels = ax.get_legend_handles_labels()
                        if labels:
                            ax.legend(loc="upper right", fontsize=6, ncol=5)
                    ax.set_title(f"Region Activity ep{ep_idx:03d}")
                    ax.set_xlabel("time step")
                    ax.set_ylabel("activity")
                    fig.tight_layout()
                    fig.savefig(os.path.join(ep_dir, f"region_activity_ep{ep_idx:03d}.png"), dpi=150)
                    plt.close(fig)
                except Exception:
                    pass

                # --- NEW: per-episode region-to-region correlation (R x R) ---
                try:
                    # correlate columns (regions) across time
                    with np.errstate(invalid="ignore", divide="ignore"):
                        corr = np.corrcoef(ra, rowvar=False)
                    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                    np.save(os.path.join(ep_dir, f"region_corr_ep{ep_idx:03d}.npy"), corr)
                except Exception as _e:
                    print(f"[traces_save] corr failed: {_e}")

                # PNG timeseries plot of E, I, DA, ACh
                try:
                    E  = np.asarray(self._bio_means.get("E",  []), dtype=np.float32)
                    I  = np.asarray(self._bio_means.get("I",  []), dtype=np.float32)
                    DA = np.asarray(self._bio_means.get("DA", []), dtype=np.float32)
                    ACh= np.asarray(self._bio_means.get("ACh",[]), dtype=np.float32)

                    fig = plt.figure(figsize=(6, 2.5))
                    ax = fig.add_subplot(111)
                    if E.size:   ax.plot(E,   label="E",   linewidth=1)
                    if I.size:   ax.plot(I,   label="I",   linewidth=1)
                    if DA.size:  ax.plot(DA,  label="DA",  linewidth=1)
                    if ACh.size: ax.plot(ACh,label="ACh", linewidth=1)

                    L = max(E.size, I.size, DA.size, ACh.size)
                    if L > 1:
                        ax.set_xlim(0, L - 1)
                    else:
                        ax.set_xlim(-0.5, 0.5)

                    ax.set_title(f"Biophys signals ep{ep_idx:03d}")
                    ax.set_xlabel("t")
                    ax.set_ylabel("mean value")
                    handles, labels = ax.get_legend_handles_labels()
                    if labels:
                        ax.legend(loc="upper right", fontsize=8, ncol=2)
                                        
                    
                    fig.tight_layout()
                    fig.savefig(os.path.join(ep_dir, f"biophys_ep{ep_idx:03d}.png"), dpi=150)
                    plt.close(fig)
                except Exception:
                    pass

            # Save comprehensive biophysical traces
            if any(len(v) > 0 for v in self._bio_means.values()):
                biopath = os.path.join(ep_dir, f"biophys_comprehensive_ep{ep_idx:03d}.npz")
                save_dict = {key: np.asarray(val, dtype=np.float32) 
                            for key, val in self._bio_means.items() if len(val) > 0}
                np.savez_compressed(biopath, **save_dict)
                
                # Generate comprehensive plots
                try:
                    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
                    
                    # Plot 1: TD Learning
                    if 'td_error' in save_dict and 'value' in save_dict:
                        ax = axes[0]
                        ax.plot(save_dict['td_error'], label='TD Error', linewidth=1, alpha=0.8)
                        ax2 = ax.twinx()
                        ax2.plot(save_dict['value'], label='Value', color='purple', linewidth=1, alpha=0.8)
                        ax.set_ylabel('TD Error')
                        ax2.set_ylabel('Value')
                        ax.set_title(f'TD Learning ep{ep_idx:03d}')
                        ax.legend(loc='upper left')
                        ax2.legend(loc='upper right')
                        ax.grid(True, alpha=0.2)
                    
                    # Plot 2: Neuromodulators
                    ax = axes[1]
                    if 'DA' in save_dict: ax.plot(save_dict['DA'], label='DA', linewidth=1)
                    if 'ACh' in save_dict: ax.plot(save_dict['ACh'], label='ACh', linewidth=1)
                    if 'NE' in save_dict: ax.plot(save_dict['NE'], label='NE', linewidth=1)
                    ax.set_ylabel('Level')
                    ax.set_title(f'Neuromodulators ep{ep_idx:03d}')
                    ax.legend(loc='upper right', ncol=3)
                    ax.grid(True, alpha=0.2)
                    
                    # Plot 3: E/I Balance
                    ax = axes[2]
                    if 'E' in save_dict and 'I' in save_dict:
                        ax.plot(save_dict['E'], label='E (Excitatory)', linewidth=1, alpha=0.8)
                        ax.plot(save_dict['I'], label='I (Inhibitory)', linewidth=1, alpha=0.8)
                        ax.set_ylabel('Population activity')
                        ax.set_title(f'E/I Balance ep{ep_idx:03d}')
                        ax.legend(loc='upper right')
                        ax.grid(True, alpha=0.2)
                    
                    # Plot 4: Entropy (exploration measure)
                    ax = axes[3]
                    if 'entropy' in save_dict:
                        ax.plot(save_dict['entropy'], label='Policy Entropy', color='red', linewidth=1, alpha=0.8)
                        ax.set_ylabel('Entropy')
                        ax.set_xlabel('time step')
                        ax.set_title(f'Exploration ep{ep_idx:03d}')
                        ax.legend(loc='upper right')
                        ax.grid(True, alpha=0.2)
                    
                    fig.tight_layout()
                    fig.savefig(os.path.join(ep_dir, f"biophys_comprehensive_ep{ep_idx:03d}.png"), dpi=150)
                    plt.close(fig)
                except Exception as plot_err:
                    print(f"[traces_save] comprehensive plot failed: {plot_err}")
            
            # Save action values (Q-values) if collected
            if hasattr(self, '_action_values_buffer') and len(self._action_values_buffer) > 0:
                av_path = os.path.join(ep_dir, f"action_values_ep{ep_idx:03d}.npy")
                av_arr = np.stack(self._action_values_buffer, axis=0)
                np.save(av_path, av_arr.astype(np.float32))

        except Exception:
            pass

    def _prep_obs(self, rgb: np.ndarray):
        """ALWAYS output exactly 64 dimensions (8x8 grayscale, boosted)."""
        H, W, _ = rgb.shape
        ys = (np.linspace(0, H - 1, 8)).astype(np.int32)
        xs = (np.linspace(0, W - 1, 8)).astype(np.int32)
        resized = rgb[ys][:, xs]               # 8x8x3
        gray    = resized.mean(axis=2).astype(np.float32) / 255.0
        boosted = gray * 5.0                   # simple contrast boost
        flat    = boosted.reshape(-1)          # 64
        assert flat.shape[0] == 64, f"Expected 64 dims, got {flat.shape[0]}"
        return self._safe_tensor(flat)

    # ====== FORCED REGION + SIGNAL HELPERS ======
    def _find_obj(self, rgb, target, tol=24):
        """Return the (y,x) of the brightest pixel matching a target color (within tol)."""
        r,g,b = target
        diff = (rgb[:,:,0].astype(np.int16)-r)**2 + (rgb[:,:,1].astype(np.int16)-g)**2 + (rgb[:,:,2].astype(np.int16)-b)**2
        mask = diff <= (tol*tol*3)
        if not mask.any():  # fallback: brightest channel heuristic
            y,x = np.unravel_index(np.argmax(rgb.mean(axis=2)), mask.shape)
            return y, x, False
        yy, xx = np.where(mask)
        # pick the most saturated match
        idx = np.argmax(rgb[yy, xx].sum(axis=1))
        return int(yy[idx]), int(xx[idx]), True

    def _region_activity_from_obs(self, obs_rgb):
        """Build a 10-D region vector from task features (+tiny noise)."""
        H, W, _ = obs_rgb.shape
        py, px, _ = self._find_obj(obs_rgb, (240,220,50))    # Pac-Man
        fy, fx, _ = self._find_obj(obs_rgb, (255,120,0))     # Food

        # normalized positions
        p = np.array([py/H, px/W], dtype=np.float32)
        f = np.array([fy/H, fx/W], dtype=np.float32)

        # vector to goal (no wrap here; good enough to break symmetry)
        v = f - p
        dist = float(np.linalg.norm(v) + 1e-6)
        ang  = float(np.arctan2(v[0], v[1]))  # y,x order

        # features
        approach = 1.0 / (1.0 + dist*8.0)           # larger when close
        dir_x = float(np.cos(ang))
        dir_y = float(np.sin(ang))
        sal   = float(np.clip(obs_rgb[:,:,0].mean()/255.0, 0, 1))  # crude 'salience'

        # Hippocampal novelty via simple hashing of position
        hp = float((np.sin(13*py)+np.cos(7*px))*0.5 + 0.5)

        # Build 10 macro regions in your order
        motor        = 0.65*approach + 0.20*abs(dir_x) + 0.15*abs(dir_y)
        sensory      = 0.60*sal + 0.40*approach
        thalamus     = 0.50*(sensory + motor)
        cerebellum   = 0.40*motor + 0.10
        parietal     = 0.80*(1.0 - approach) + 0.10
        pfc          = 0.70*approach + 0.20*sal + 0.10
        limbic       = 0.50*(1.0 - dist) + 0.20*sal + 0.10
        hippocampus  = 0.6*hp + 0.2*(1.0 - approach)
        insula       = 0.5*(sensory) + 0.05
        basal_ganglia= 0.6*motor + 0.2*pfc + 0.05

        ra = np.array([
            motor, sensory, thalamus, cerebellum,
            parietal, pfc, limbic, hippocampus, insula, basal_ganglia
        ], dtype=np.float32)

        # squish to ~[0.05, 1.05] and add tiny, observation-dependent jitter
        ra = 0.05 + 0.95*np.clip(ra, 0.0, 1.2)
        return ra.astype(np.float32)

    def _ensure_learning_signals(self, out_dict, reward, ra_vec):
        """Make sure td_error/state_value/action_probabilities exist."""
        if not hasattr(self, '_v'):     # simple value EMA
            self._v, self._v_prev = 0.0, 0.0
        gamma, beta = 0.98, 0.10
        self._v_prev = float(self._v)
        self._v = (1.0 - beta)*self._v + beta*float(reward)

        td = float(reward + gamma*self._v - self._v_prev)

        # ensure BG dict
        if not isinstance(out_dict.get('basal_ganglia', None), dict):
            out_dict['basal_ganglia'] = {}

        bg = out_dict['basal_ganglia']
        bg.setdefault('td_error', td)
        bg.setdefault('state_value', self._v)

        # action values -> softmax policy; else bias toward goal direction from RA
        if 'action_probabilities' not in bg:
            if 'action_values' in bg:
                av = bg['action_values']
                av = av.detach().cpu().numpy() if torch.is_tensor(av) else np.asarray(av, np.float32)
                av = av - np.max(av)
                ex = np.exp(av)
                p = ex / (ex.sum() + 1e-9)
            else:
                # use RA to create a small directional prior (U,D,L,R)
                dir_x = float(ra_vec[0] - ra_vec[4])  # motor vs parietal proxy
                dir_y = float(ra_vec[0] - ra_vec[3])  # motor vs cerebellum proxy
                base = np.array([max(-dir_y,0), max(dir_y,0), max(-dir_x,0), max(dir_x,0)], np.float32)
                base = base + 0.25  # keep exploration
                p = base / base.sum()
            bg['action_probabilities'] = p.astype(np.float32)
    
    def _lfp_scalar_from_region_dict(self, d):
            import numpy as _np, torch as _torch
            if not isinstance(d, dict): 
                return 0.001
            
            # NEW: Extract synaptic currents FIRST (real EEG source!)
            if 'neural_dynamics' in d and 'synaptic_currents' in d['neural_dynamics']:
                currents = d['neural_dynamics']['synaptic_currents']
                if _torch.is_tensor(currents):
                    return float(currents.mean().item()) * 10.0  # Scale up
            
            # NEW: Try voltages (better than spikes for EEG)
            if 'neural_dynamics' in d and 'voltages' in d['neural_dynamics']:
                voltages = d['neural_dynamics']['voltages']
                if _torch.is_tensor(voltages):
                    return float(voltages.mean().item() + 70.0) * 0.5  # Shift from -70mV baseline
            
            # PRIORITY 0: normalized activity if provided (best for learning/logging)
            na_norm = d.get('neural_activity_norm', None)
            if na_norm is not None:
                try:
                    if _torch.is_tensor(na_norm):
                        return float(na_norm.mean().item())
                    else:
                        return float(_np.asarray(na_norm).mean())
                except Exception:
                    pass

            # PRIORITY 1: raw neural_activity
            na = d.get('neural_activity', None)
            if na is not None:
                try:
                    if _torch.is_tensor(na):
                        return float(na.mean().item())
                    else:
                        return float(_np.asarray(na).mean())
                except Exception:
                    pass
            
            # PRIORITY 2: Try typical scalar-named keys
            for k in ('activity', 'activation', 'spikes', 'relay_output', 'output', 'selected_action'):
                if k in d:
                    v = d[k]
                    try:
                        if _torch.is_tensor(v):
                            return float(v.mean().item())
                        else:
                            return float(_np.asarray(v).mean())
                    except Exception:
                        pass
            
            # PRIORITY 3: try to average any tensor/array found
            for k, v in d.items():
                if _torch.is_tensor(v):
                    try:
                        return float(v.mean().item())
                    except Exception:
                        pass
                elif isinstance(v, _np.ndarray):
                    try:
                        return float(v.mean())
                    except Exception:
                        pass
            
            # fallback
            return 0.001
    
    def forward_and_learn(self, obs_rgb: np.ndarray, reward: float, train: bool = True):
        x = self._prep_obs(obs_rgb)
        self.last_obs_rgb = obs_rgb
        # Remove step_idx if brain doesn't support it
        out = self.brain(x, reward=float(reward if train else 0.0))
        # Ensure TD/value/policy exist for logging & analysis
        try:
            ra_vec_for_td = getattr(self, "_last_region_activity", None)
            if ra_vec_for_td is None and hasattr(self, "last_obs_rgb"):
                ra_vec_for_td = self._region_activity_from_obs(self.last_obs_rgb)
            self._ensure_learning_signals(
                out_dict=out if isinstance(out, dict) else {},
                reward=float(reward),
                ra_vec=(ra_vec_for_td if ra_vec_for_td is not None else np.zeros(10, np.float32))
            )
        except Exception:
            pass
        self.step_counter += 1  # Keep counter for EEG logging

        # --- populate logging buffers from brain output ---
        if isinstance(out, dict):
            # Global neuromodulators
            self._last_DA  = [ float(out.get('dopamine', 0.0)) ]
            self._last_ACh = [ float(out.get('acetylcholine', 0.0)) ]
            # Motor spikes -> split 25E/7I if vector length >=32
            m = out.get('motor', {}) if isinstance(out.get('motor', {}), dict) else {}
            sp = m.get('spikes', None)
            na = m.get('neural_activity', None)
            def _to_np(a):
                import numpy as _np, torch as _torch
                if a is None: return None
                if _torch.is_tensor(a): return a.detach().cpu().numpy()
                try: return _np.asarray(a)
                except: return None
            sp_np = _to_np(sp)
            na_np = _to_np(na)
            if sp_np is not None and sp_np.size > 0:
                self._last_spikes = sp_np.astype('float32')
                if sp_np.ndim > 0 and sp_np.shape[0] >= 32:
                    self._last_E = [ float(sp_np[:25].mean()) ]
                    self._last_I = [ float(sp_np[25:32].mean()) ]
                else:
                    self._last_E = [ float(sp_np.mean()) ]
                    self._last_I = [ 0.0 ]
            elif na_np is not None and na_np.size > 0:
                # fallback if spikes missing
                if na_np.ndim > 0 and na_np.shape[0] >= 32:
                    self._last_E = [ float(na_np[:25].mean()) ]
                    self._last_I = [ float(na_np[25:32].mean()) ]
                else:
                    self._last_E = [ float(na_np.mean()) ]
                    self._last_I = [ 0.0 ]
                self._last_spikes = na_np.astype('float32')

        # DEBUG: Print what brain actually returns
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        if self._debug_counter < 3:  # only print first 3 steps
            print(f"\n[DEBUG] Brain output keys: {list(out.keys()) if isinstance(out, dict) else type(out)}")
            if isinstance(out, dict):
                for key, val in out.items():
                    if isinstance(val, dict):
                        print(f"  {key}: {list(val.keys())}")
                    else:
                        print(f"  {key}: {type(val)}")
        
        bg=out.get('basal_ganglia',{}) if isinstance(out,dict) else {}
        
        # Extract region activity as LFP-like scalars per macro-region (synaptic currents dominate)
        region_activities = []
        for key in self._macro_regions:
            try:
                if key == 'motor':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('motor', {}))
                elif key == 'sensory':
                    u = out.get('unified_neocortex', {})
                    if isinstance(u, dict):
                        se = u.get('sensory_encoding', None)
                        if se is not None:
                            # Treat sensory_encoding as LFP-like voltages (better EEG source)
                            if torch.is_tensor(se):
                                activity_val = float(se.mean().item())
                            else:
                                _se_np = np.asarray(se, dtype=np.float32)
                                activity_val = float(_se_np.mean()) if _se_np.size else 0.001
                        else:
                            # Fallback: use generic extractor on the UCX dict
                            activity_val = self._lfp_scalar_from_region_dict(u)
                    else:
                        # Last fallback: try a dedicated 'sensory' dict if present
                        activity_val = self._lfp_scalar_from_region_dict(out.get('sensory', {}))
                elif key == 'thalamus':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('thalamus', {}))
                elif key == 'cerebellum':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('cerebellum', {}))
                elif key == 'parietal':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('parietal', {}))
                elif key == 'pfc':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('pfc', {}))
                elif key == 'limbic':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('limbic', {}))
                elif key == 'hippocampus':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('hippocampus', {}))
                elif key == 'insula':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('insula', {}))
                elif key == 'basal_ganglia':
                    activity_val = self._lfp_scalar_from_region_dict(out.get('basal_ganglia', {}))
                else:
                    activity_val = 0.001
            except Exception:
                activity_val = 0.001
            region_activities.append(activity_val)
        self._last_region_activity = np.array(region_activities, dtype=np.float32)
        # Guard: if region activity is degenerate, derive from observation to keep EEG/credit useful
        try:
            ra = self._last_region_activity
            if (not np.isfinite(ra).any()) or (float(ra.std()) < 1e-6):
                if hasattr(self, "last_obs_rgb"):
                    self._last_region_activity = self._region_activity_from_obs(self.last_obs_rgb)
        except Exception:
            pass

        if isinstance(bg, dict) and 'action_probabilities' in bg:
            p = bg['action_probabilities']
            p = p.detach().cpu().numpy() if torch.is_tensor(p) else np.asarray(p, float)
            if p.sum() <= 0:
                p = np.ones(4) / 4.0
            return (int(np.random.choice(len(p), p=p)) if train else int(np.argmax(p))), out
        return (int(np.random.randint(0, 4)) if train else 0), out

# ---------- Pygame Viewer ----------
class PygameViewer:
    def __init__(self, scale=32, title="Pacman", window_size=None):
        """
        window_size: (width, height) in pixels. If provided, we will render at a fixed
        resolution regardless of grid size, scaling the game surface to fit.
        """
        self.scale = scale
        self.window_size = window_size  # (W_px, H_px) or None
        pygame.init()
        pygame.display.set_caption(title)

    def open(self, H, W):
        if self.window_size and self.window_size[0] > 0 and self.window_size[1] > 0:
            self.screen = pygame.display.set_mode(self.window_size)
        else:
            self.screen = pygame.display.set_mode((W * self.scale, H * self.scale))
        self.clock = pygame.time.Clock()
        self._last_grid_size = (H, W)

    def draw(self, rgb, overlay=""):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                pygame.quit(); raise SystemExit

        # Make a game surface from the grid (width=x, height=y)
        game_surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        # Compute integer pixel-perfect scale and letterbox/pillarbox
        win_w, win_h = self.screen.get_width(), self.screen.get_height()
        grid_h, grid_w = rgb.shape[0], rgb.shape[1]
        s = max(1, min(win_w // grid_w, win_h // grid_h))  # integer scale >=1
        scaled_w, scaled_h = grid_w * s, grid_h * s
        x0 = (win_w - scaled_w) // 2
        y0 = (win_h - scaled_h) // 2

        self._screen_meta = {"x0": x0, "y0": y0, "scale": s, "grid": (grid_h, grid_w)}

        # fill background (letterbox area)
        self.screen.fill(PALETTE["background"])

        # nearest-neighbour upscale (crisp cells)
        scaled = pygame.transform.scale(game_surf, (scaled_w, scaled_h))
        self.screen.blit(scaled, (x0, y0))

        # Overlay text
        if overlay:
            font = pygame.font.SysFont(None, 24)
            txt = font.render(overlay, True, (200, 200, 200))
            self.screen.blit(txt, (10, 10))

        pygame.display.flip()
        # NOTE: FPS is controlled by GameplayRecorder's fps for video and here for screen refresh.
        self.clock.tick( max(1, int(getattr(self, "_fps_hint", 20))) )

        # Return a numpy frame (RGB) for recording
        frame = pygame.surfarray.array3d(self.screen).swapaxes(0, 1)
        return frame
   
    def draw_dot(self, cell_yx_float, color=PALETTE["pacman"], radius_px=None):
        if not hasattr(self, "_screen_meta"):
            return
        y, x = cell_yx_float
        m = self._screen_meta
        s = m["scale"]; x0 = m["x0"]; y0 = m["y0"]
        r = radius_px if radius_px is not None else max(2, s // 3)
        cx = int(x0 + (x + 0.5) * s)
        cy = int(y0 + (y + 0.5) * s)
        pygame.draw.circle(self.screen, color, (cx, cy), r)
        pygame.display.flip()
    
    def close(self):
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
# ---------- Runner ----------
def run(args):
    BASE_ROOT = os.getenv("PACMAN_OUTDIR", os.path.join(os.getcwd(), "runs"))
    os.makedirs(BASE_ROOT, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"pacman_run_{stamp}"
    outdir = os.path.join(BASE_ROOT, run_name)
    i = 1
    while os.path.exists(outdir):
        outdir = os.path.join(BASE_ROOT, f"{run_name}_{i:03d}")
        i += 1
    os.makedirs(outdir, exist_ok=True)
    env = PacmanEnv(
        H=args.H, W=args.W, seed=args.seed,
        max_steps=args.max_steps,
        step_cost=args.step_cost,
        pellets=int(args.pellets),
        maze_name=args.use_maze,
        # smell shaping
        )
    # set smell params (explicit to keep backward compat)
    env.smell_gain = float(args.smell_gain)
    env.smell_lam  = float(args.smell_lam)

    # design toggles
    env.afterglow_steps = int(args.afterglow_steps)
    env.afterglow_reward = float(args.afterglow_reward)
    env.halo_radius = int(args.halo_radius)
    env.halo_reward = float(args.halo_reward)
    env.turn_penalty = float(args.turn_penalty)
    env.fast_pellet_bonus = float(args.fast_pellet_bonus)
    env.fast_finish_bonus = float(args.fast_finish_bonus)
    env.finish_power = float(args.finish_power)
    brain=BrainAdapter(run_dir=outdir)

    # force higher exploration early
    if hasattr(brain, "epsilon_explore"):
        brain.epsilon_explore[...] = 0.20

    # === BME Data Collector ===
    bme_collector = BMEDataCollector(
        output_dir=outdir,
        sampling_rate_hz=1000.0
    )
    print("[BME] Comprehensive data collector initialized")
    brain_ckpt = os.path.join(outdir, "brain.pt")
    # Initialize overall tracking file paths
    brain._overall_eeg_path = os.path.join(outdir, "overall_eeg.npy")
    brain._overall_region_path = os.path.join(outdir, "overall_region_activity.npy")
    brain._overall_dopamine_path = os.path.join(outdir, "overall_dopamine.npy")    
    brain._overall_bio_path = os.path.join(outdir, "overall_biophys_comprehensive.npz")
    brain._overall_action_values_path = os.path.join(outdir, "overall_action_values.npy")
    brain._overall_bio_lists = {}            # dict: key -> [episode arrays]
    brain._overall_action_values_list = []   # list of episode stacks

    # save EEG geometry once for reference
    try:
        if hasattr(brain, "_leadfield"):
            np.save(os.path.join(outdir, "leadfield.npy"), brain._leadfield.astype(np.float32))
        if hasattr(brain, "_eeg_electrodes"):
            np.save(os.path.join(outdir, "electrodes.npy"), brain._eeg_electrodes.astype(np.float32))
        if hasattr(brain, "_macro_regions"):
            with open(os.path.join(outdir, "macro_regions.txt"), "w") as f:
                for name in brain._macro_regions:
                    f.write(str(name) + "\n")
    except Exception:
        pass

    if getattr(args, "load_brain", ""):
        brain.load(args.load_brain)

    _win = (args.video_width, args.video_height) if (args.video_width > 0 and args.video_height > 0) else None
    viewer = PygameViewer(scale=args.scale, title="Pacman 2", window_size=_win)
    # let the viewer know our preferred fps for smoother preview
    viewer._fps_hint = args.fps
    viewer.open(env.H, env.W)

    logger = PacmanLogger(outdir)
    rec = GameplayRecorder(fps=float(args.fps))

    stages = [
        {"name":"S1_three", "H":1, "W":3, "spawn":(0,1), "food_policy":"left_or_right"},
        {"name":"S2_cross", "H":3, "W":3, "spawn":(1,1), "food_policy":"neighbor"},
        {"name":"S3_free5", "H":5, "W":5, "spawn":None,  "food_policy":""},
        {"name":"S4_free7", "H":7, "W":7, "spawn":None,  "food_policy":""},
    ]
    def _apply_stage(si):
        s = stages[si]
        nonlocal env, viewer
        env = PacmanEnv(
            H=s["H"], W=s["W"], seed=args.seed,
            max_steps=args.max_steps,
            step_cost=args.step_cost,
            pellets=int(args.pellets),
            maze_name=args.use_maze,
        )
        # smell shaping
        env.smell_gain = float(args.smell_gain)
        env.smell_lam  = float(args.smell_lam)

        # design toggles for the new stage env
        env.afterglow_steps = int(args.afterglow_steps)
        env.afterglow_reward = float(args.afterglow_reward)
        env.halo_radius = int(args.halo_radius)
        env.halo_reward = float(args.halo_reward)
        env.turn_penalty = float(args.turn_penalty)
        env.fast_pellet_bonus = float(args.fast_pellet_bonus)
        env.fast_finish_bonus = float(args.fast_finish_bonus)
        env.finish_power = float(args.finish_power)
        env._forced_spawn = s["spawn"]
        env._food_policy  = s["food_policy"]
        
        if viewer:
            viewer.close()
            _win = (args.video_width, args.video_height) if (args.video_width > 0 and args.video_height > 0) else None
            viewer = PygameViewer(scale=args.scale, title=f"Pacman 2 - {s['name']}", window_size=_win)
            viewer._fps_hint = args.fps
            viewer.open(env.H, env.W)
        return s
    
    if args.auto_curriculum:
        # Always start at Stage 1 (1x2) exactly as requested
        _start = 0  # S1_two
        curr_stage = _apply_stage(_start)
        stage_idx  = _start
        recent = deque(maxlen=int(args.curr_window))          # successes
        recent_steps = deque(maxlen=int(args.curr_window))    # steps
        recent_dstars = deque(maxlen=int(args.curr_window))   # d* optimal
        ep_in_stage = 0
    else:
        # honor fixed size (no curriculum)
        env = PacmanEnv(
            H=args.H, W=args.W, seed=args.seed,
            max_steps=args.max_steps,
            step_cost=args.step_cost,
            pellets=int(args.pellets),
            maze_name=args.use_maze
        )
        # smell shaping
        env.smell_gain = float(args.smell_gain)
        env.smell_lam  = float(args.smell_lam)

        # design toggles
        env.afterglow_steps = int(args.afterglow_steps)
        env.afterglow_reward = float(args.afterglow_reward)
        env.halo_radius = int(args.halo_radius)
        env.halo_reward = float(args.halo_reward)

        env.turn_penalty = float(args.turn_penalty)
        env.fast_pellet_bonus = float(args.fast_pellet_bonus)
        env.fast_finish_bonus = float(args.fast_finish_bonus)
        env.finish_power = float(args.finish_power)

        env._forced_spawn = None
        env._food_policy  = ""

    for ep in range(args.episodes):
        # Anneal exploration if supported: 0.30 -> 0.05 over full run
        if hasattr(brain, "epsilon_explore"):
            frac = (ep + 1) / max(1, args.episodes)
            try:
                brain.epsilon_explore[...] = float(0.30 - 0.25 * frac)
            except Exception:
                # allow scalar or list
                try:
                    brain.epsilon_explore = float(0.30 - 0.25 * frac)
                except Exception:
                    pass
        # pick starting stage
        obs=env.reset(); ret=0.0; steps=0; done=False
        
        # --- EEG: keep episode alive for minimum steps to accumulate meaningful signals ---
        MIN_STEPS_FOR_EEG = 0  # you can tune this
        dopamine_trace=[]
        d_star = getattr(env, "d0", None)
        turns = 0
        prev_action = None

        # create per-episode folder
        ep_dir = os.path.join(outdir, f"ep{ep:03d}")
        os.makedirs(ep_dir, exist_ok=True)

        # start per-episode video
        rec.start_episode(ep_dir)
        # reset EEG buffer for this episode
        if hasattr(brain, "eeg_reset_episode"):
            brain.eeg_reset_episode()
        # reset raw trace buffers
        if hasattr(brain, "traces_reset_episode"):
            brain.traces_reset_episode()
        
        # === BME: Reset episode buffers ===
        bme_collector.reset_episode_buffers()
        # draw initial frame so you can see the start BEFORE the first move
        pel_left = len(env.pellet_set)
        pel_tot  = env.pellets_target
        frame = viewer.draw(obs, f"EP{ep} STEP{steps} R{ret:.1f}  PEL {pel_left}/{pel_tot}")

        rec.record(frame)

        if args.step_delay_ms > 0:
            pygame.time.wait(int(args.step_delay_ms))
        
        prev_reward = 0.0

        # Brain learns from PREVIOUS reward
        while not done or steps < MIN_STEPS_FOR_EEG:
            step_start_time = time.time()
            
            # Brain learns from PREVIOUS reward (proper TD learning!)
            train_flag = (not args.eval) and (not args.no_learning) and (steps >= args.warmup_no_learn)
            action, out = brain.forward_and_learn(obs, reward=prev_reward, train=train_flag)

            old_pos = env.pos

            # Take the step
            next_obs, reward, done, info = env.step(action)
            ret += float(reward)
            steps += 1
            
            # Save reward for NEXT iteration
            prev_reward = reward
            
            # Record dopamine
            _dop = float(out.get("dopamine", 0.0)) if isinstance(out, dict) else 0.0
            dopamine_trace.append(_dop)

            # Advance state
            obs = next_obs
                       
            # EEG: append one sample based on region activity (per-step)
            # Apply oscillations for realistic EEG
            # Prefer EEG-specific activity if brain exported it
            use_eeg = None
            if isinstance(out, dict) and 'eeg_activity' in out:
                order = ['motor','sensory','thalamus','cerebellum','parietal','pfc','limbic','hippocampus','insula','basal_ganglia']
                vec = []
                for k in order:
                    v = out['eeg_activity'].get(k, 0.0)
                    try:
                        vv = float(v.detach().cpu().mean().item()) if torch.is_tensor(v) else float(v)
                    except Exception:
                        vv = 0.0
                    vec.append(vv)
                use_eeg = np.asarray(vec, dtype=np.float32)

            if use_eeg is None:
                # fallback to last region vector (no extra synthetic modulation here)
                _ra = getattr(brain, "_last_region_activity", [])
                use_eeg = np.asarray(_ra, dtype=np.float32)

            R = 10
            use_ra = (use_eeg[:R] if use_eeg.size >= R else np.pad(use_eeg, (0, R - use_eeg.size))).astype(np.float32)
            if hasattr(brain, "eeg_step_from_region_activity"):
                brain.eeg_step_from_region_activity(use_ra)

            # record raw traces for later saving
            _E = getattr(brain, "_last_E", [])
            _I = getattr(brain, "_last_I", [])
            _DA = getattr(brain, "_last_DA", [])
            _ACh = getattr(brain, "_last_ACh", [])

            if hasattr(brain, "trace_step"):
                brain.trace_step(use_ra, _E, _I, _DA, _ACh, out)

            # draw the new grid frame
            pel_left = info.get("pellets_left", 0) if isinstance(info, dict) else 0
            pel_tot  = info.get("pellets_total", 0) if isinstance(info, dict) else 0
            frame = viewer.draw(obs, f"EP{ep} STEP{steps} R{ret:.1f}  PEL {pel_left}/{pel_tot}")
            rec.record(frame)  # record exact pygame-rendered frame

            # Visual feedback when eating
            if info.get("ate", False):
                for _ in range(3):  # 3 flash frames
                    flash_frame = viewer.draw(obs, f"EP{ep} STEP{steps} R{ret:.1f} *ATE*")
                    rec.record(flash_frame)
                    if args.step_delay_ms > 0:
                        pygame.time.wait(max(50, args.step_delay_ms // 3))

            # micro-tween: overlay a moving dot between old and new cell (visual only)
            try:
                oy, ox = old_pos
                ny, nx = env.pos
                K = 3  # number of in-between dot frames

                # simple straight tween (no wrap)
                for k in range(1, K + 1):
                    a = k / (K + 1.0)
                    iy = (1 - a) * oy + a * ny
                    ix = (1 - a) * ox + a * nx
                    viewer.draw_dot((iy, ix))
                    rec.record(pygame.surfarray.array3d(viewer.screen).swapaxes(0, 1))
                    if args.step_delay_ms > 0:
                        pygame.time.wait(max(1, int(args.step_delay_ms) // (K + 1)))
            
            except Exception:
                pass

            # generic per-step delay
            if args.step_delay_ms > 0:
                pygame.time.wait(int(args.step_delay_ms))

            # if we just ate, hold the “chomp” frame a bit longer so it’s visible
            if info.get("ate", False):
                pygame.time.wait(max(int(args.step_delay_ms), 300))

        # attach d* and turns for the logger
        logger._last_d_star = d_star if d_star is not None else info.get("d0", None)
        logger._last_turns = turns

        # hand off biological metrics from brain -> logger for this episode
        logger._last_spikes = getattr(brain, "_last_spikes", [])
        logger._last_E = getattr(brain, "_last_E", [])
        logger._last_I = getattr(brain, "_last_I", [])
        logger._last_DA = getattr(brain, "_last_DA", [])
        logger._last_ACh = getattr(brain, "_last_ACh", [])
        logger._last_region_activity = getattr(brain, "_last_region_activity", [])

        # write logs both overall and per-episode
        logger.log_episode(
            ep=ep,
            ret=ret,
            steps=steps,
            ate=int(info.get("ate", 0)),
            dopamine_trace=dopamine_trace,
            d_star=logger._last_d_star,
            turns=turns,
            ep_dir=ep_dir
        )
       
        # periodic brain checkpoint
        # save brain checkpoint INSIDE this episode folder (brain_epXYZ.pt)
        brain_ep_path = os.path.join(ep_dir, f"brain_ep{ep:03d}.pt")
        brain.save(brain_ep_path)
        print(f"[BRAIN] saved -> {brain_ep_path}")

        # optional periodic checkpoint in run root (unchanged behavior, if you want it)
        if int(getattr(args, "save_every", 0)) and (ep + 1) % int(args.save_every) == 0:
            brain.save(os.path.join(outdir, "brain.pt"))
            print(f"[BRAIN] periodic root save -> {os.path.join(outdir,'brain.pt')} at ep {ep+1}")

        print(f"[Episode {ep}] return={ret:.2f} steps={steps}")
        rec.end_episode()  # close this episode's gameplay.mp4
        # save EEG for this episode
        if hasattr(brain, "eeg_save_episode"):
            brain.eeg_save_episode(ep_dir, ep)
        # save raw region & biophys traces
        if hasattr(brain, "traces_save_episode"):
            brain.traces_save_episode(ep_dir, ep)
        
        # === BME: Save comprehensive episode data ===
        bme_collector.save_episode_data(ep, ep_dir)
        # Append to overall cumulative files (crash-safe incremental saves)
        if hasattr(brain, "_eeg_buffer") and len(brain._eeg_buffer) > 0:
            brain._overall_eeg_list.append(np.stack(brain._eeg_buffer, axis=0))
            np.save(brain._overall_eeg_path, np.concatenate(brain._overall_eeg_list, axis=0))
        # === OVERALL FIF ===
        try:
            eeg_all = np.load(brain._overall_eeg_path)  # (T_all, E)
            E_all = eeg_all.shape[1]
            _std_1020 = ['Fp1','Fp2','F3','F4','F7','F8','Fz',
                         'C3','C4','Cz','P3','P4','Pz','O1','O2',
                         'T7','T8','P7','P8','POz','Oz','FC1','FC2','CP1','CP2']
            ch_names = _std_1020[:E_all] if E_all <= len(_std_1020) else [f'EEG{i+1}' for i in range(E_all)]
            info = mne.create_info(ch_names=ch_names, sfreq=256.0, ch_types='eeg')
            raw_all = mne.io.RawArray(eeg_all.T * 1e-6, info)
            try:
                raw_all.set_montage(mne.channels.make_standard_montage('standard_1020'))
            except Exception:
                raw_all.set_montage(mne.channels.make_standard_montage('standard_1005'))
            fif_all = os.path.join(outdir, "eeg_overall.fif")
            raw_all.save(fif_all, overwrite=True)
            print(f"[EEG] overall FIF saved -> {fif_all}")
            # optional: open viewer once
            # os.system(f'python "mne_eeg_3d_regions_patched.py" --eeg "{fif_all}"')
        except Exception as e:
            print(f"[EEG] overall FIF save failed: {e}")
        if hasattr(brain, "_ra_buffer") and len(brain._ra_buffer) > 0:
            brain._overall_region_list.append(np.stack(brain._ra_buffer, axis=0))
            np.save(brain._overall_region_path, np.concatenate(brain._overall_region_list, axis=0))
        
        if len(dopamine_trace) > 0:
            brain._overall_dopamine_list.append(np.array(dopamine_trace, dtype=np.float32))
            np.save(brain._overall_dopamine_path, np.concatenate(brain._overall_dopamine_list, axis=0))
        # === NEW: Overall biophys (E, I, DA, ACh, NE, td_error, value, entropy, spikes_*, osc_*) ===
        if hasattr(brain, "_bio_means") and any(len(v) > 0 for v in brain._bio_means.values()):
            if not hasattr(brain, "_overall_bio_lists") or brain._overall_bio_lists is None:
                brain._overall_bio_lists = {}
            for k, seq in brain._bio_means.items():
                arr = np.asarray(seq, dtype=np.float32)
                if arr.size == 0:
                    continue
                brain._overall_bio_lists.setdefault(k, []).append(arr)
            # write compressed NPZ with concatenated arrays for each key
            save_dict = {k: np.concatenate(v, axis=0) for k, v in brain._overall_bio_lists.items() if len(v) > 0}
            if save_dict:
                np.savez_compressed(brain._overall_bio_path, **save_dict)

        # === NEW: Overall action-values (Q) over all steps ===
        if hasattr(brain, "_action_values_buffer") and len(brain._action_values_buffer) > 0:
            av_arr = np.stack(brain._action_values_buffer, axis=0).astype(np.float32)  # (T, A)
            av_arr = np.nan_to_num(av_arr, nan=0.0, posinf=0.0, neginf=0.0)
            brain._overall_action_values_list.append(av_arr)
            np.save(
                brain._overall_action_values_path,
                np.nan_to_num(np.concatenate(brain._overall_action_values_list, axis=0),
                            nan=0.0, posinf=0.0, neginf=0.0)
            )
        # === NEW: Overall region-to-region correlation (R x R) on the fly
        if os.path.exists(brain._overall_region_path):
            try:
                _reg_all = np.load(brain._overall_region_path)  # (T, R)
                if _reg_all.ndim == 2 and _reg_all.shape[0] >= 2 and _reg_all.shape[1] >= 2:
                    X = np.nan_to_num(_reg_all, nan=0.0, posinf=0.0, neginf=0.0)
                    # First: full corr for reference
                    with np.errstate(invalid="ignore", divide="ignore"):
                        _corr = np.corrcoef(X, rowvar=False)
                    _corr = np.nan_to_num(_corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

                    # keep columns with finite data and non-zero variance
                    keep = (np.isfinite(X).all(axis=0)) & (np.std(X, axis=0) > 1e-9)
                    R = X.shape[1]
                    corr_full = np.zeros((R, R), dtype=np.float32)

                    if keep.any():
                        with np.errstate(invalid="ignore", divide="ignore"):
                            corr = np.corrcoef(X[:, keep], rowvar=False)
                        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                        idx = np.where(keep)[0]
                        for i, ii in enumerate(idx):
                            for j, jj in enumerate(idx):
                                corr_full[ii, jj] = corr[i, j]
                        for ii in idx:
                            corr_full[ii, ii] = 1.0
                    else:
                        corr_full = _corr

                    np.save(os.path.join(outdir, "overall_region_corr.npy"), corr_full)
            except Exception as e:
                print(f"[OVERALL] corr save failed: {e}")

        # === BME: Generate thesis report after every episode ===
        print(f"\n[BME] Generating thesis report for episode {ep}...")
        generate_thesis_report(bme_collector, outdir)

        if args.episode_delay_ms > 0:
            
            pygame.time.wait(int(args.episode_delay_ms))
        if args.auto_curriculum:
            ep_in_stage += 1

            # Define d* for this episode (optimal steps from start to food on the grid)
            d_star_win = d_star if d_star is not None else info.get("d0", None)

            # Steps-based success: count an episode as success if its steps are within (d* + slack).
            # If d* is unknown for some reason, fall back to a loose grid-based heuristic.
            if d_star_win is not None:
                is_steps_success = (steps <= (float(d_star_win) + float(args.curr_median_slack)))
            else:
                is_steps_success = (steps <= max(1, env.H + env.W))

            recent.append(1.0 if is_steps_success else 0.0)
            recent_steps.append(steps)
            recent_dstars.append(d_star_win)

            # adaptive slack: start looser, tighten as stages increase
            adaptive_slack = max(1.0, float(args.curr_median_slack) - 0.2 * stage_idx)

            # only try to advance if enough episodes seen
            if ep_in_stage >= int(args.curr_min) and len(recent) == recent.maxlen:
                rate = sum(recent) / float(len(recent))  # % of near-optimal episodes
                med_steps = float(np.median(np.asarray(recent_steps, dtype=np.float32)))

                # compute window median d* (ignore None)
                _ds = [d for d in recent_dstars if d is not None]
                med_dstar = float(np.median(np.asarray(_ds, dtype=np.float32))) if _ds else med_steps

                near_optimal_window = (med_steps <= (med_dstar + adaptive_slack))
                
                # regression guard: if rate collapses, stay; if previously promoted too early, consider reverting
                if rate < 0.5 and stage_idx > 0:
                    # simple backoff once; clear recent history to avoid oscillations
                    stage_idx -= 1
                    curr_stage = _apply_stage(stage_idx)
                    recent.clear(); recent_steps.clear(); recent_dstars.clear()
                    ep_in_stage = 0
                    print(f"[CURRICULUM] regression -> back to {curr_stage['name']}")
                    continue
                
                if (rate >= float(args.curr_pass)
                    and near_optimal_window
                    and stage_idx + 1 < len(stages)):
                    stage_idx += 1
                    curr_stage = _apply_stage(stage_idx)
                    recent.clear(); recent_steps.clear(); recent_dstars.clear()
                    ep_in_stage = 0
                    print(f"[CURRICULUM] advancing to {curr_stage['name']} "
                        f"(H={curr_stage['H']} W={curr_stage['W']}) "
                        f"[pass_rate={rate:.2f}, med_steps={med_steps:.1f}, med_d*={med_dstar:.1f}]")
    
    # nothing to finalize for per-episode recorder; just ensure closed
    rec.close()
    logger.close()

    # final root brain save (optional)
    brain_ckpt = os.path.join(outdir, "brain_final.pt")
    brain.save(brain_ckpt)
    print(f"[BRAIN] final save -> {brain_ckpt}")
    
    # Generate overall summary plots - LINE PLOTS, not heatmaps
    try:
        if os.path.exists(brain._overall_eeg_path):
            eeg_all = np.load(brain._overall_eeg_path)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            for e in range(eeg_all.shape[1]):
                ax.plot(eeg_all[:, e], label=f"ch{e}", linewidth=0.8, alpha=0.7)
            ax.set_title(f"Overall EEG (all {args.episodes} episodes, {eeg_all.shape[0]} steps)")
            ax.set_xlabel("time step (concatenated)")
            ax.set_ylabel("amplitude (µV)")
            ax.legend(loc='upper right', fontsize=7, ncol=5)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "overall_eeg.png"), dpi=200)
            plt.close(fig)
        
        if os.path.exists(brain._overall_region_path):
            region_all = np.load(brain._overall_region_path)
            region_names = brain._macro_regions if hasattr(brain, "_macro_regions") else [f"R{i}" for i in range(region_all.shape[1])]
            
            fig, ax = plt.subplots(figsize=(12, 4))
            for r in range(region_all.shape[1]):
                label = region_names[r] if r < len(region_names) else f"R{r}"
                ax.plot(region_all[:, r], label=label, linewidth=1.0, alpha=0.8)
            ax.set_title(f"Overall Region Activity (all {args.episodes} episodes, {region_all.shape[0]} steps)")
            ax.set_xlabel("time step (concatenated)")
            ax.set_ylabel("activity")            
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(loc='upper right', fontsize=7, ncol=5)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "overall_region_activity.png"), dpi=200)
            plt.close(fig)
        
        if os.path.exists(brain._overall_dopamine_path):
            dop_all = np.load(brain._overall_dopamine_path)
            
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(dop_all, linewidth=1.0, color='orange', alpha=0.8)
            ax.set_title(f"Overall Dopamine (all {args.episodes} episodes, {len(dop_all)} steps)")
            ax.set_xlabel("time step (concatenated)")
            ax.set_ylabel("dopamine/reward")
            ax.axhline(y=0, linestyle="--", color="gray", alpha=0.5)
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "overall_dopamine.png"), dpi=200)
            plt.close(fig)

        if os.path.exists(brain._overall_bio_path):
            try:
                bio = np.load(brain._overall_bio_path)
                fig, ax = plt.subplots(figsize=(12, 4))
                # Plot what we have; DA/ACh/NE first, then E/I if present
                for key in ["DA", "ACh", "NE", "E", "I", "entropy"]:
                    if key in bio.files:
                        ax.plot(bio[key], label=key, linewidth=1.0)
                ax.set_title("Overall Biophys (concatenated across episodes)")
                ax.set_xlabel("time step (concatenated)")
                ax.set_ylabel("level / mean activity")
                ax.legend(loc='upper right', fontsize=8, ncol=3)
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                fig.savefig(os.path.join(outdir, "overall_biophys.png"), dpi=200)
                plt.close(fig)
            except Exception as e:
                print(f"[OVERALL] Biophys plot failed: {e}")

        print("[OVERALL] Saved cumulative line plots")
    except Exception as e:
        print(f"[OVERALL] Plot generation failed: {e}")
    print("Done. Outputs in:", outdir)

    # === BME: Generate comprehensive thesis report ===
    print("\n[BME] Generating comprehensive thesis report...")
    generate_thesis_report(bme_collector, outdir)
    
    print("Done. Outputs in:", outdir)

def generate_thesis_report(bme_collector: BMEDataCollector, output_dir: str):
    """Generate comprehensive thesis report with all statistics"""
    
    report_dir = os.path.join(output_dir, "thesis_report")
    os.makedirs(report_dir, exist_ok=True)
    
    all_stats = bme_collector.episode_stats
    
    # Fallback sources written by the main loop
    overall_npz  = os.path.join(output_dir, "overall_biophys_comprehensive.npz")
    episodes_csv = os.path.join(output_dir, "episodes_overall.csv")

    # Build from BME if present
    td_errors    = [ep.get('learning', {}).get('mean_td_error', 0.0) for ep in all_stats] if all_stats else []
    values       = [ep.get('learning', {}).get('final_value', 0.0) for ep in all_stats]   if all_stats else []
    entropies    = [ep.get('learning', {}).get('mean_entropy', 0.0) for ep in all_stats]  if all_stats else []
    firing_rates = [ep.get('neural_dynamics', {}).get('mean_firing_rate', 0.0) for ep in all_stats] if all_stats else []

    # Helper: treat empty or all-zeros as unusable
    def _all_zero(a):
        try:
            arr = np.asarray(a, dtype=np.float32)
            return arr.size == 0 or np.all(arr == 0)
        except Exception:
            return True

    # If BME stats are missing/zero, pull real arrays from disk
    DA = ACh = NE = np.array([], dtype=np.float32)  # ensure defined
    if (not all_stats) or (_all_zero(td_errors) and _all_zero(values) and _all_zero(entropies)):
        if os.path.exists(overall_npz):
            bio = np.load(overall_npz)
            td_errors = bio.get("td_error", np.array([], dtype=np.float32)).tolist()
            values    = bio.get("value",    np.array([], dtype=np.float32)).tolist()
            entropies = bio.get("entropy",  np.array([], dtype=np.float32)).tolist()
            DA  = bio.get("DA",  np.array([], dtype=np.float32))
            ACh = bio.get("ACh", np.array([], dtype=np.float32))
            NE  = bio.get("NE",  np.array([], dtype=np.float32))
        if os.path.exists(episodes_csv):
            fr = []
            with open(episodes_csv, "r", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        fr.append(float(row.get("firing_rate", "0") or 0))
                    except Exception:
                        fr.append(0.0)
            firing_rates = fr

    # Ensure neuromodulator arrays exist even if BME stats were used
    if DA.size == 0:
        DA  = np.array([ep.get('neuromodulation', {}).get('mean_dopamine',       0.0) for ep in all_stats], dtype=np.float32)
    if ACh.size == 0:
        ACh = np.array([ep.get('neuromodulation', {}).get('mean_acetylcholine',  0.0) for ep in all_stats], dtype=np.float32)
    if NE.size == 0:
        NE  = np.array([ep.get('neuromodulation', {}).get('mean_norepinephrine', 0.0) for ep in all_stats], dtype=np.float32)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].plot(td_errors, linewidth=2)
        axes[0, 0].set_title('TD Error Convergence')
        axes[0, 0].set_xlabel('step')
        axes[0, 0].set_ylabel('Mean TD Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(values, linewidth=2, color='blue')
        axes[0, 1].set_title('Value Function Learning')
        axes[0, 1].set_xlabel('step')
        axes[0, 1].set_ylabel('Final Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(entropies, linewidth=2, color='purple')
        axes[1, 0].set_title('Exploration Dynamics')
        axes[1, 0].set_xlabel('step')
        axes[1, 0].set_ylabel('Mean Entropy')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(firing_rates, linewidth=2, color='orange')
        axes[1, 1].set_title('Neural Activity')
        axes[1, 1].set_xlabel('Episode')  # firing_rate is per-episode
        axes[1, 1].set_ylabel('Firing Rate (Hz)')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(os.path.join(report_dir, 'learning_curves.png'), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[BME] Learning curves failed: {e}")
    
    # === 2. Neuromodulation ===
    try:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        da  = DA.tolist()
        ach = ACh.tolist()
        ne  = NE.tolist()

        axes[0].plot(da, linewidth=2, color='orange')
        
        axes[0].set_ylabel('Dopamine')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(ach, linewidth=2, color='green')
        axes[1].set_ylabel('Acetylcholine')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(ne, linewidth=2, color='purple')
        axes[2].set_ylabel('Norepinephrine')
        axes[2].set_xlabel('Episode')
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle('Neuromodulator Dynamics')
        fig.tight_layout()
        fig.savefig(os.path.join(report_dir, 'neuromodulators.png'), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"[BME] Neuromodulator plot failed: {e}")
    
    # === 3. Summary JSON ===
    try:
        summary = {
            'total_episodes': len(all_stats),
            'final_td_error': float(td_errors[-1]) if len(td_errors) else 0.0,
            'final_value': float(values[-1]) if len(values) else 0.0,
            'mean_firing_rate': float(np.mean(firing_rates)) if len(firing_rates) else 0.0,
            'mean_dopamine': float(np.mean(DA)) if DA.size else 0.0,
            'total_spikes': int(sum([ep.get('neural_dynamics', {}).get('spike_count', 0) for ep in all_stats]))
        }

        with open(os.path.join(report_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[BME] Thesis report saved to: {report_dir}")
    except Exception as e:
        print(f"[BME] Summary save failed: {e}")

def main():
    ap=argparse.ArgumentParser("Pacman with Cortex Brain")
    ap.add_argument("--episodes",type=int,default=10)
    ap.add_argument("--H",type=int,default=7)
    ap.add_argument("--W",type=int,default=7)
    ap.add_argument("--pellets", type=int, default=40, help="number of pellets in the maze (Pacman 3)")
    ap.add_argument("--use-maze", type=str, default="", choices=["","maze1","maze2"],
                    help="use built-in embedded maze layout")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--save-every",type=int,default=10)
    ap.add_argument("--auto-curriculum", action="store_true", help="auto-advance stages: 1x2 -> 1x3 -> 3x3 -> 5x5 ...")
    ap.add_argument("--curr-window", type=int, default=60, help="episodes window for pass rate check")
    ap.add_argument("--curr-pass", type=float, default=0.95, help="required success rate (0..1) within window to advance")
    ap.add_argument("--curr-min", type=int, default=100, help="minimum episodes per stage before advancing")
    ap.add_argument("--scale", type=int, default=32, help="pixel size per grid cell for window & video")
    ap.add_argument("--load-brain", type=str, default="", help="path to a saved brain.pt to resume from")
    
    # video/output controls
    ap.add_argument("--video-width", type=int, default=1280, help="fixed output/window width in pixels (same across stages)")
    ap.add_argument("--video-height", type=int, default=720, help="fixed output/window height in pixels (same across stages)")
    ap.add_argument("--fps", type=int, default=30, help="preview and video FPS")

    # bio-friendly efficiency shaping
    ap.add_argument("--step-cost", type=float, default=0.01, help="per-step metabolic cost (small negative reward each step)")
    ap.add_argument("--max-steps", type=int, default=0, help="episode time limit; 0 uses 4*H*W")

    # curriculum: require near-optimal median steps before promotion
    ap.add_argument("--curr-median-slack", type=float, default=1.0, help="allow promotion when median steps <= median(d*) + slack")
   
    # preview pacing
    ap.add_argument("--step-delay-ms", type=int, default=0, help="extra delay after each step (milliseconds)")
    ap.add_argument("--eval", action="store_true", help="evaluation mode: no learning, deterministic policy")
    ap.add_argument("--no-learning", action="store_true", help="force no learning even outside --eval")
    ap.add_argument("--warmup-no-learn", type=int, default=0, help="disable learning for first N steps each episode")

    # reward shaping (design toggles)
    ap.add_argument("--afterglow-steps", type=int, default=0, help="keep episode alive N steps after eating")
    ap.add_argument("--afterglow-reward", type=float, default=0.0, help="per-step bonus during afterglow (decays)")
    ap.add_argument("--halo-radius", type=int, default=0, help="cells within this L1 distance get small appetitive reward")
    ap.add_argument("--halo-reward", type=float, default=0.0, help="base reward at distance 1 (scales down linearly)")
    ap.add_argument("--turn-penalty", type=float, default=0.01, help="penalize turns only when moving away from food")
    ap.add_argument("--episode-delay-ms", type=int, default=0, help="extra delay after each episode (milliseconds)")

    # --- speed-based reward shaping (new) ---
    ap.add_argument("--fast-pellet-bonus", type=float, default=0.0,
                    help="extra reward per pellet scaled by how early it was eaten")
    ap.add_argument("--fast-finish-bonus", type=float, default=0.0,
                    help="completion bonus scaled by how fast the episode finished")
    ap.add_argument("--finish-power", type=float, default=1.0,
                    help="exponent for finish scaling (1=linear, 2=quadratic, ...)")
    # --- proximity (“smell”) shaping controls ---
    ap.add_argument("--smell-gain", type=float, default=0.5,
                    help="scale for proximity shaping (Δexp)")
    ap.add_argument("--smell-lam",  type=float, default=2.0,
                    help="length-scale for smell shaping")

    args=ap.parse_args(); run(args)

if __name__=="__main__":
    main()
