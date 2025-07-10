import numpy as np
import copy
from gym import spaces, Wrapper
from gym.spaces import Box


class CustomObservationSpace(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=1, shape=(3, env.observation_space["pov"].shape[0], env.observation_space["pov"].shape[1])
        )

    def _process_obs(self, obs):
        obs = obs["pov"] / 255.0  # normalize the observation to be between 0 and 1
        obs = np.transpose(obs, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        return obs

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return self._process_obs(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self._process_obs(obs), reward, done, info


class CustomActionSpace(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.action_groups = [
            ["forward", "back"],
            ["left", "right"],
            ["sneak", "sprint"],
            [f"hotbar.{i}" for i in range(1, 10)],
            ["use"],
            ["drop"],
            ["attack"],
            ["jump"],
            # ["pickItem"],
            # ["swapHand"],
        ]

        # create action templates
        self.no_action = {key: 0 for key in env.action_space.keys()}
        self.no_action["camera"] = (0, 0)
        self.inventory_only_action = copy.deepcopy(self.no_action)
        self.inventory_only_action["inventory"] = 1

        # Create camera mapping
        self.n_camera_bins = 11
        normalized_camera_bins = np.arange(self.n_camera_bins) - self.n_camera_bins // 2
        self.camera_mapping = (11 ** (np.abs(normalized_camera_bins) / (self.n_camera_bins - 1) * 2) - 1) * np.sign(
            normalized_camera_bins
        )

        self.action_space = spaces.Dict(
            {
                "main_head": spaces.Discrete(
                    np.prod([len(group) + 1 for group in self.action_groups])
                ),  # product of the number of actions in each group
                "inventory": spaces.Discrete(2),
                "camera_enabled": spaces.Discrete(2),
                "camera_head": spaces.Discrete(
                    self.n_camera_bins**2
                ),  # n_camera_bins^2 camera movements, only enabled when the camera is enabled in the main_head
            }
        )

    def step(self, original_action):
        main_head = original_action["main_head"]
        camera_head = original_action["camera_head"]
        if original_action["inventory"] == True:  # if inventory is selected, return the inventory only action
            return super().step(self.inventory_only_action)
        else:
            action = self.no_action.copy()
            for group in self.action_groups:
                n_choices = len(group) + 1  # +1 for the no action option
                selection = main_head % n_choices
                main_head //= n_choices
                if selection != 0:
                    action[group[selection - 1]] = 1

            if original_action["camera_enabled"] == True:
                camera_bin_x = camera_head // self.n_camera_bins
                camera_bin_y = camera_head % self.n_camera_bins
                action["camera"] = (self.camera_mapping[camera_bin_x], self.camera_mapping[camera_bin_y])
            return super().step(action)

class RenderWrapper(Wrapper):
    def __init__(self, env, render_mode="human"):
        super().__init__(env)
        self.env = env
        self.render_mode = render_mode
        
    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.env.render(mode=self.render_mode)
        return obs, reward, done, info
