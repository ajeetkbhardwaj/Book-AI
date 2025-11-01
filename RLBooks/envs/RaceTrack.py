"""
# Code is inspired from : https://github.com/datawhalechina/easy-rl/blob/master/notebooks/envs/
@author : GPT-5 OpenAI
"""
import time
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from IPython.display import clear_output
from gym.spaces import Discrete, Box
from gym import Env
from matplotlib import colors

class RacetrackEnv(Env):
    """
    Custom Racetrack Environment based on Sutton & Barto (2018).
    The track layout is loaded from 'track.txt'.
    """

    ACTIONS_DICT = {
        0: (1, -1),  # Acc Vert., Brake Horiz.
        1: (1, 0),   # Acc Vert., Hold Horiz.
        2: (1, 1),   # Acc Vert., Acc Horiz.
        3: (0, -1),  # Hold Vert., Brake Horiz.
        4: (0, 0),   # Hold Vert., Hold Horiz.
        5: (0, 1),   # Hold Vert., Acc Horiz.
        6: (-1, -1), # Brake Vert., Brake Horiz.
        7: (-1, 0),  # Brake Vert., Hold Horiz.
        8: (-1, 1)   # Brake Vert., Acc Horiz.
    }

    CELL_TYPES_DICT = {
        0: "track",
        1: "wall",
        2: "start",
        3: "goal"
    }

    metadata = {'render_modes': ['human'], "render_fps": 4}

    def __init__(self, render_mode='human'):
        # Load racetrack map from file
        track_path = os.path.join(os.path.dirname(__file__), "track.txt")
        self.track = np.flip(np.loadtxt(track_path, dtype=int), axis=0)

        # Discover start grid squares
        self.initial_states = [(y, x) for y in range(self.track.shape[0])
                               for x in range(self.track.shape[1])
                               if self.CELL_TYPES_DICT[self.track[y, x]] == "start"]

        # Observation and action spaces
        high = np.array([np.finfo(np.float32).max] * 4)
        self.observation_space = Box(low=-high, high=high, shape=(4,), dtype=np.float32)
        self.action_space = Discrete(9)
        self.is_reset = False

    def step(self, action: int):
        if not self.is_reset:
            raise RuntimeError("Call .reset() before .step().")

        if not (isinstance(action, int) or isinstance(action, np.integer)):
            raise TypeError("Action must be an integer.")
        if action < 0 or action > 8:
            raise ValueError("Action out of range [0-8].")

        # With prob 0.8 apply intended acceleration, else 0 (slip)
        if np.random.uniform() < 0.8:
            d_y, d_x = self.ACTIONS_DICT[action]
        else:
            d_y, d_x = (0, 0)

        # Update velocity
        self.velocity = (self.velocity[0] + d_y, self.velocity[1] + d_x)
        self.velocity = (np.clip(self.velocity[0], -10, 10),
                         np.clip(self.velocity[1], -10, 10))

        # Update position
        new_position = (self.position[0] + self.velocity[0],
                        self.position[1] + self.velocity[1])

        reward = -1
        done = False

        # Out of bounds
        if (new_position[0] < 0 or new_position[1] < 0 or
            new_position[0] >= self.track.shape[0] or
            new_position[1] >= self.track.shape[1]):
            self.position = random.choice(self.initial_states)
            self.velocity = (0, 0)
            reward -= 10
        elif self.CELL_TYPES_DICT[self.track[new_position]] == "wall":
            self.position = random.choice(self.initial_states)
            self.velocity = (0, 0)
            reward -= 10
        elif self.CELL_TYPES_DICT[self.track[new_position]] in ["track", "start"]:
            self.position = new_position
        elif self.CELL_TYPES_DICT[self.track[new_position]] == "goal":
            self.position = new_position
            reward += 10
            done = True
            self.is_reset = False
        else:
            raise RuntimeError("Invalid cell encountered in track map.")

        return np.array([self.position[0], self.position[1],
                         self.velocity[0], self.velocity[1]]), reward, done, {}

    def reset(self, seed=None):
        self.position = random.choice(self.initial_states)
        self.velocity = (0, 0)
        self.is_reset = True
        return np.array([self.position[0], self.position[1],
                         self.velocity[0], self.velocity[1]])

    def render(self, render_mode='human'):
        plt.ion()
        fig = plt.figure(num="Racetrack Render")
        ax = plt.gca()
        ax.clear()
        clear_output(wait=True)

        env_plot = np.copy(self.track)
        env_plot[self.position] = 4
        env_plot = np.flip(env_plot, axis=0)

        cmap = colors.ListedColormap(["white", "black", "green", "red", "yellow"])
        norm = colors.BoundaryNorm(range(6), cmap.N)
        ax.imshow(env_plot, cmap=cmap, norm=norm, zorder=0)

        if self.velocity != (0, 0):
            ax.arrow(self.position[1],
                     self.track.shape[0] - 1 - self.position[0],
                     self.velocity[1], -self.velocity[0],
                     path_effects=[pe.Stroke(linewidth=1, foreground='black')],
                     color="yellow", width=0.1, length_includes_head=True, zorder=2)

        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, self.track.shape[1], 1))
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, self.track.shape[0], 1))
        ax.set_yticklabels([])

        plt.show()
        time.sleep(0.1)

    def get_actions(self):
        return [*self.ACTIONS_DICT]


# ----------------------------
# Main Simulation Loop
# ----------------------------
if __name__ == "__main__":
    env = RacetrackEnv()
    state = env.reset()
    print("Initial State:", state)

    for step in range(1000):
        action = random.choice(env.get_actions())
        next_state, reward, done, _ = env.step(action)
        print(f"Step {step}: State={next_state}, Reward={reward}, Done={done}")
        env.render()

        if done:
            print("\n🏁 Goal reached! Resetting...\n")
            state = env.reset()
