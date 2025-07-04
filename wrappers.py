import numpy as np
import copy
from gym import spaces, Wrapper


class CustomObservationSpace(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.observation_space["pov"]

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs["pov"]

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs["pov"], reward, done, info


class CustomActionSpace(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.action_groups = [
            ["forward", "back"],
            ["left", "right"],
            ["sneak", "sprint"],
            [f"hotbar.{i}" for i in range(9)],
            ["use"],
            ["drop"],
            ["attack"],
            ["jump"],
            ["pickItem"],
            ["swapHand"],
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
                "main_head": spaces.Discrete(34561),
                # 3^3 × 10 × 2^6 × 2 + 1 = 34561
                # 3^3 for the 3 sets of 2 mutually exclusive keys above where taking neither in the set is an option
                # 10 for the 9 hotbar keys or no hotbar keypress
                # 2^4 for the remaining binary 4 keys: use, drop, attack, and jump
                # 2 for whether or not to move the mouse(camera)
                # +1 for the inventory button which is mutually exclusive with all other actions
                "camera_head": spaces.Discrete(121),
                # 11^2 camera movements, only enabled when the camera is enabled in the main_head
            }
        )

    def step(self, action):
        main_head = action["main_head"]
        camera_head = action["camera_head"]
        if main_head == 0:  # if inventory is selected, return the inventory only action
            print("inventory")
            return super().step(self.inventory_only_action)
        else:
            main_head -= 1  # remove the inventory offset
            action = self.no_action.copy()
            for group in self.action_groups:
                n_choices = len(group) + 1  # +1 for the no action option
                selection = main_head % n_choices
                main_head //= n_choices
                if selection != 0:
                    action[group[selection - 1]] = 1
            if main_head == 0:
                camera_bin_x = camera_head // self.n_camera_bins
                camera_bin_y = camera_head % self.n_camera_bins
                action["camera"] = (self.camera_mapping[camera_bin_x], self.camera_mapping[camera_bin_y])
            print(action)
            return super().step(action)
