import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv



env = gym.make('LunarLander-v2', render_mode='human')

env = DummyVecEnv([lambda: env]) 

models_dir = "models/PPO"
model_path = f"{models_dir}/20000.zip"


model = PPO.load(model_path, env=env)

episodes = 10




for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:    
        env.render()

        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        print(reward)


env.close()