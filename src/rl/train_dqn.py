import os

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.FastFoodEnv import FastFoodEnv


class FlattenActionWrapper(gym.ActionWrapper):
    """Tricks DQN into controlling a MultiDiscrete environment using a single Discrete integer (0-7)."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, act):
        # Converts integer 0-7 into a binary array [Burger, Fries, Ice Cream]
        return [act // 4, (act // 2) % 2, act % 2]


def make_env():
    def _init():
        env = FastFoodEnv()
        env = FlattenActionWrapper(env)  # Apply our custom DQN fix
        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    print("Initializing DQN Training (Lunch Rush + Dessert Station)...")

    models_dir = "models"
    log_dir = "tb_logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # DQN relies heavily on Replay Buffers and prefers DummyVecEnv over SubprocVecEnv
    env = DummyVecEnv([make_env()])

    # Build the DQN Agent
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=log_dir,
        learning_rate=0.0001,
        buffer_size=100000,
        exploration_fraction=0.1,
    )

    total_timesteps = 500_000
    print(f"Beginning DQN training. Open TensorBoard to watch progress!")

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="DQN_Lunch_Rush_Agent",
        progress_bar=True,
    )

    model_path = f"{models_dir}/fast_food_manager_dqn"
    model.save(model_path)
    print(f"Training Complete! Model saved successfully to {model_path}.zip")

    env.close()
