# %%
import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
import minerl
from wrappers import *

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = RecordEpisodeStatistics(env)
        return env

    return thunk
env = gym.make('MineRLBasaltFindCave-v0')



# %%
env = CustomObservationSpace(env)
env = CustomActionSpace(env)

# %%
print(env.observation_space)

# %%
print(env.action_space)

# %%
# dict({k:0 for k in env.action_space.keys()})

# %%
print("Environment created successfully.")
obs = env.reset()
done = False

# %%
step_count = 0
env.reset()
done = False
while not done:
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    # action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    env.render()
    step_count += 1
    print(f"Action: {action}")
    print(f"Step count: {step_count}")
    print(f"Done: {done}")

# %%
env.close()


