from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np

SEED = 42


class Actions():

    def __init__(self) -> None:
        # Execute RI, do nothing, get out of RI

        self.action_space = spaces.Dict(
            {
                "RI": spaces.Discrete(3, start=-1, seed=SEED)
            }
        )


class BessEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode=None, time_length=24):
        self.time_length = time_length  # The size of the square grid
        self.duration = 24*60  # The size of the PyGame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "dispatch": spaces.MultiDiscrete([3 for _ in range(24)], seed=SEED),
                "price": spaces.Box(low=-9999, high=9999, shape=(1, time_length), dtype=np.int16, seed=SEED),
            }
        )

        # We have 3 actions, corresponding to execute RI, do nothing, get out of RI
        self.action_space = spaces.Dict(
            {
                "RI": spaces.Discrete(3, start=-1, seed=SEED)
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"dispatch": self._dispatch, "price": self._price}

    def _get_info(self):
        return {
            "time": self.time_now,
            "avg_price": np.mean(
                self._price
            ),
            "spread": max(self._price)-min(self._price)
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.time_now = 0

        # Choose the agent's location uniformly at random
        self._dispatch = np.array([1 for _ in range(self.time_length)])

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._price = self.np_random.integers(0,
                                              150, size=self.time_length, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # Compute RI
        min_index = np.argmin(self._price)
        max_index = np.argmax(self._price)
        spread = self._price[max_index] - self._price[min_index]

        if 0 in self._dispatch or 2 in self._dispatch:
            buy_index = self._dispatch.tolist().index(0)
            sell_index = self._dispatch.tolist().index(2)

            if action['RI'] == -1:
                self._dispatch[buy_index] = 1
                self._dispatch[sell_index] = 1
                spread = self._price[sell_index] - self._price[buy_index]

            else:
                spread = 0

        elif action['RI'] == 1:
            self._dispatch[min_index] -= action['RI']
            self._dispatch[max_index] += action['RI']

        elif action['RI'] == -1:
            spread = 0

        # An episode is done iff the agent has reached the target
        terminated = self.time_now >= self.duration
        reward = action['RI'] * spread
        self.time_now += 1
        observation = self._get_obs()
        info = self._get_info()

        self._price += self.np_random.integers(-2,
                                               2, size=self.time_length, dtype=int)

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):
        print(self._get_info())
