# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gym
import minerl
import numpy as np
import tyro

from wrappers import CustomObservationSpace, CustomActionSpace, RenderWrapper


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    render: bool = True
    env_id: str = "MineRLObtainDiamondShovel-v0"
    total_timesteps: int = 500000
    num_envs: int = 1
    num_steps: int = 128


def make_env(env_id, idx, capture_video, render, run_name):
    def thunk():
        env = gym.make(env_id)
        if capture_video and idx == 0:
            env = gym.wrappers.record_video.RecordVideo(env, f"videos/{run_name}")
        if render and idx == 0:
            env = RenderWrapper(env)
        env = gym.wrappers.record_episode_statistics.RecordEpisodeStatistics(env)
        env = CustomObservationSpace(env)
        env = CustomActionSpace(env)
        return env

    return thunk


if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_name = args.exp_name

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.render, run_name) for i in range(args.num_envs)],
    )

    obs = envs.reset()
    print(f"Initial obs shape: {obs.shape}")
    for step in range(args.total_timesteps):
        # Take random actions for debugging
        action = envs.action_space.sample()
        next_obs, rewards, dones, infos = envs.step(action)
        print(f"Step {step}")
        if any(dones):
            print(f"Episode finished for envs: {[i for i, d in enumerate(dones) if d]}")
        obs = next_obs
    envs.close()
