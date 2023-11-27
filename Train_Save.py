import gymnasium as gym
import os

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback


models_dir = "models/PPO"
log_dir = "logs"

TIMESTEPS = 1000000
SAVE_FREQ = 10000


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


env = gym.make("LunarLander-v2", render_mode="human")


# Create the callback
callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=models_dir, name_prefix="PPO",save_replay_buffer=True,
    save_vecnormalize=True,)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cuda")

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", progress_bar=True, callback=callback,)



# episodes = 10

# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         env.render()
#         obs, reward, done, trunc, info = env.step(env.action_space.sample())
#         print(reward)


env.close()