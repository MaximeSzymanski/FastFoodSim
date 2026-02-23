import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.rl.FastFoodEnv import FastFoodEnv


def mask_fn(env):
    """Extracts action masks from the unwrapped environment.

    Args:
        env (gym.Env): The wrapped environment instance.

    Returns:
        np.ndarray: A boolean array indicating valid actions for the agent.
    """
    return env.unwrapped.action_masks()


def make_env():
    """Creates a callable that initializes the environment for multiprocessing.

    Returns:
        callable: A function that returns a monitored FastFoodEnv instance
            wrapped with action masking capabilities.
    """

    def _init():
        env = FastFoodEnv()
        env = Monitor(env)
        return ActionMasker(env, mask_fn)

    return _init


if __name__ == "__main__":
    print("Initializing MaskablePPO Training (Lunch Rush + Dessert Station)...")

    models_dir = "models"
    log_dir = "tb_logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    num_cpu = 4
    env = SubprocVecEnv([make_env() for i in range(num_cpu)])

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=log_dir,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
    )

    total_timesteps = 500_000
    print(
        f"Beginning training for {total_timesteps} steps. Open TensorBoard to watch progress."
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="PPO_Lunch_Rush_Agent",
        progress_bar=True,
    )

    model_path = f"{models_dir}/fast_food_manager_elite"
    model.save(model_path)
    print(f"Training Complete. Model saved successfully to {model_path}.zip")

    env.close()
