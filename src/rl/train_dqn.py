import os

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.FastFoodEnv import FastFoodEnv


class FlattenActionWrapper(gym.ActionWrapper):
    """A wrapper that flattens a MultiDiscrete action space into a single Discrete space.

    This allows algorithms that strictly require a 1D Discrete action space, such as DQN,
    to operate within an environment originally designed with MultiDiscrete actions.

    Args:
        env (gym.Env): The environment to wrap.
    """

    def __init__(self, env):
        """Initializes the wrapper and overwrites the action space to Discrete(8)."""
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, act):
        """Translates a single discrete integer into a binary list for the base environment.

        Args:
            act (int): The discrete action integer ranging from 0 to 7.

        Returns:
            list: A binary list representing the cooking commands for the burger,
                fries, and ice cream stations respectively.
        """
        return [act // 4, (act // 2) % 2, act % 2]


def make_env():
    """Creates a callable that initializes the environment for DQN training.

    Returns:
        callable: A function that returns a wrapped and monitored FastFoodEnv instance.
    """

    def _init():
        env = FastFoodEnv()
        env = FlattenActionWrapper(env)
        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    print("Initializing DQN Training (Lunch Rush + Dessert Station)...")

    models_dir = "models"
    log_dir = "tb_logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_env()])

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
    print("Beginning DQN training. Open TensorBoard to watch progress.")

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="DQN_Lunch_Rush_Agent",
        progress_bar=True,
    )

    model_path = f"{models_dir}/fast_food_manager_dqn"
    model.save(model_path)
    print(f"Training Complete. Model saved successfully to {model_path}.zip")

    env.close()
