#!/usr/bin/env python3
"""
# Code is inspired from : https://github.com/datawhalechina/easy-rl/blob/master/notebooks/envs/simple_grid.py

A structured and extensible version of the 'Drunken Walk' environment.
Inspired by OpenAI Gym's FrozenLake 
Author: OpenAI ChatGPT(GPT-5)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding, colorize
from io import StringIO


# ---- Environment Constants ----
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
DEFAULT_REWARD = 10
POTHOLE_PROB = 0.2
BROKEN_LEG_PENALTY = -5
SLEEP_DEPRIVATION_PENALTY = 0.0


# ---- Map Definitions ----
MAPS = {
    "theAlley": ["S...H...H...G"],
    "walkInThePark": [
        "S.......",
        ".....H..",
        "........",
        "......H.",
        "........",
        "...H...G",
    ],
    "4x4": [
        "S...",
        ".H.H",
        "...H",
        "H..G",
    ],
}


# ---- Utilities ----
def categorical_sample(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    """Sample from a categorical distribution."""
    return np.searchsorted(np.cumsum(probabilities), rng.random())


def generate_random_map(size=8, p=0.8) -> List[str]:
    """Generates a random valid map."""
    valid = False

    def is_valid(grid):
        frontier, visited = [(0, 0)], set()
        while frontier:
            r, c = frontier.pop()
            if (r, c) not in visited:
                visited.add((r, c))
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        if grid[nr][nc] == "G":
                            return True
                        if grid[nr][nc] not in "#H":
                            frontier.append((nr, nc))
        return False

    while not valid:
        grid = np.random.choice([".", "H"], (size, size), p=[p, 1 - p])
        grid[0, 0], grid[-1, -1] = "S", "G"
        valid = is_valid(grid)
    return ["".join(r) for r in grid]


# ---- Main Environment ----
@dataclass
class DrunkenWalkConfig:
    map_name: Optional[str] = "4x4"
    custom_map: Optional[List[str]] = None
    slip_prob: float = 0.8
    pothole_prob: float = POTHOLE_PROB
    reward_goal: float = DEFAULT_REWARD
    penalty_fall: float = BROKEN_LEG_PENALTY
    penalty_idle: float = SLEEP_DEPRIVATION_PENALTY


class DrunkenWalkEnv(gym.Env):
    """A stochastic gridworld environment with potholes and drunk walking."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, config: DrunkenWalkConfig = DrunkenWalkConfig()):
        super().__init__()
        self.config = config
        desc = config.custom_map or MAPS.get(config.map_name, generate_random_map())

        self.desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = self.desc.shape
        self.nS = self.nrow * self.ncol
        self.nA = 4

        # Define spaces
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        # Build transitions
        self.P = self._build_transitions()

        # Initial state distribution
        self.isd = (self.desc == b"S").astype(np.float64).ravel()
        self.isd /= self.isd.sum()

        self._np_random, _ = seeding.np_random(None)
        self.state = categorical_sample(self.isd, self._np_random)
        self.last_action = None

    # ---- Core Environment Methods ----
    def _to_s(self, r: int, c: int) -> int:
        return r * self.ncol + c

    def _intended_dest(self, r: int, c: int, a: int) -> Tuple[int, int]:
        if a == LEFT:
            c = max(c - 1, 0)
        elif a == DOWN:
            r = min(r + 1, self.nrow - 1)
        elif a == RIGHT:
            c = min(c + 1, self.ncol - 1)
        elif a == UP:
            r = max(r - 1, 0)
        return r, c

    def _build_transitions(self) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        """Construct full transition probability dictionary."""
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        for r in range(self.nrow):
            for c in range(self.ncol):
                s = self._to_s(r, c)
                letter = self.desc[r, c]

                for a in range(self.nA):
                    li = P[s][a]

                    if letter in b"G":
                        li.append((1.0, s, 0, True))
                        continue

                    if letter in b"H":
                        li.append((self.config.pothole_prob, s, self.config.penalty_fall, True))
                        nr, nc = self._intended_dest(r, c, a)
                        ns = self._to_s(nr, nc)
                        li.append((1 - self.config.pothole_prob, ns, self.config.penalty_idle, False))
                        continue

                    # Normal pavement
                    for prob, dir_ in zip(
                        [self.config.slip_prob, 0.1, 0.1],
                        [a, (a - 1) % 4, (a + 1) % 4],
                    ):
                        nr, nc = self._intended_dest(r, c, dir_)
                        ns = self._to_s(nr, nc)
                        nl = self.desc[nr, nc]
                        done = bytes(nl) in b"G"
                        reward = self.config.reward_goal if done else self.config.penalty_idle
                        li.append((prob, ns, reward, done))
        return P

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = categorical_sample(self.isd, self._np_random)
        self.last_action = None
        return self.state, {}

    def step(self, action: int):
        transitions = self.P[self.state][action]
        probs = [t[0] for t in transitions]
        i = categorical_sample(probs, self._np_random)
        p, s, r, d = transitions[i]
        self.state = s
        self.last_action = action
        return s, r, d, False, {"prob": p}

    def render(self, mode="human"):
        desc = [[c.decode("utf-8") for c in row] for row in self.desc]
        r, c = divmod(self.state, self.ncol)
        desc[r][c] = colorize(desc[r][c], "red", highlight=True)

        output = "\n".join("".join(line) for line in desc)
        if mode == "ansi":
            return output
        print(output)


if __name__ == "__main__":
    env = DrunkenWalkEnv(DrunkenWalkConfig(map_name="walkInThePark"))
    obs, _ = env.reset()
    env.render()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            break
