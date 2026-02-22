import glob
import os

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.rl.FastFoodEnv import FastFoodEnv


def mask_fn(env: gym.Env):
    return env.unwrapped.action_masks()


def train_kitchen_ai():
    print("Booting up Vectorized Environments...")
    os.makedirs("stats", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for f in glob.glob("stats/*.monitor.csv"):
        try:
            os.remove(f)
        except OSError:
            pass

    num_cpu = 4
    env = make_vec_env(
        FastFoodEnv,
        n_envs=num_cpu,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        vec_env_cls=SubprocVecEnv,
        monitor_dir="stats",
    )

    model = MaskablePPO(
        "MlpPolicy", env, verbose=0, learning_rate=0.0003, ent_coef=0.05
    )

    print("Beginning Training...")
    model.learn(total_timesteps=5_000_000, progress_bar=True)

    # Save to the new models folder
    model.save("models/fast_food_manager_ai")
    print("Training complete! Model saved to models/fast_food_manager_ai.zip")


if __name__ == "__main__":
    train_kitchen_ai()
