import numpy as np
import copy
from gym import spaces, Wrapper
    

class FlattenObservationSpace(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space["pov"]
    
    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs["pov"]

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs["pov"], reward, done, info


class FlattenActionSpace(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        # Count non-camera action keys and create mapping
        button_keys = [k for k in self.action_space.keys() if k not in ("camera", "ESC") and "hotbar" not in k]
        self.n_buttons = len(button_keys) - 1 # exclude ESC button
        self.buttons_mapping = {i: key for i, key in enumerate(button_keys)}
        
        # Create camera mapping
        self.n_camera_bins = 11
        normalized_camera_bins = np.arange(self.n_camera_bins) - self.n_camera_bins // 2
        self.camera_mapping = (10 ** (np.abs(normalized_camera_bins) / (self.n_camera_bins - 1) * 2) - 1) * np.sign(normalized_camera_bins)

        # Create flattened action space
        self.action_space = spaces.Discrete(2 ** self.n_buttons * self.n_camera_bins ** 2 * 9) # 9 hotbar slots

    def step(self, action):
        # Extract binary representation of each button
        action_copy = copy.copy(action)
        buttons_binary = []
        for _ in range(self.n_buttons):
            buttons_binary.append(action_copy & 1)
            action_copy >>= 1
        hotbar_one_hot = action_copy // 9
        action_copy %= 9
        camera_bin_x = action_copy // self.n_camera_bins
        action_copy %= self.n_camera_bins
        camera_bin_y = action_copy // self.n_camera_bins
        action_copy %= self.n_camera_bins
        
        # Map binary encoding back to button dictionary
        buttons = {
            self.buttons_mapping[i]: 1 if buttons_binary[i] else 0 
            for i in range(self.n_buttons)
        }

        # Map camera bin to action space
        camera_val_x = self.camera_mapping[camera_bin_x]
        camera_val_y = self.camera_mapping[camera_bin_y]
        
        # Execute action with reconstructed button mapping
        obs, reward, done, info = super().step({
            "camera": (camera_val_x, camera_val_y), 
            "hotbar": hotbar_one_hot,
            "ESC": 0,
            **buttons
        })
        return obs, reward, done, info


# class ResetNoParam(Wrapper):
#     def reset(self, *, seed=None, options=None):
#         return self.env.reset()


# class CleanStatisticsInfo(Wrapper):
#     def __init__(self, env, stats_key: str = "episode"):
#         super().__init__(env)
#         self._stats_key = stats_key

#     def reset(self, *, seed = None, options = None):
#         obs, info = super().reset(seed=seed, options=options)
#         if self._stats_key in info:
#             del info[self._stats_key]
#         return obs, info
