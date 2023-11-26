import gymnasium as gym
import os
from stable_baselines3 import PPO, A2C, DQN


models_dir = "models/A2C"
log_dir = "logs"



if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


env = gym.make('LunarLander-v2', render_mode='human')

env.reset()


model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device='cuda')

TIMESTEPS = 100000


for i in range(1,30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C", progress_bar=True)
    model.save(f"{models_dir}/{TIMESTEPS*i}")












# episodes = 10


# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:    
#         env.render()
#         obs, reward, done, trunc, info = env.step(env.action_space.sample())
#         print(reward)


# env.close()