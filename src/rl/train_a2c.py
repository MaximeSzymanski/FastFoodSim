import os

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.rl.FastFoodEnv import FastFoodEnv


def make_env():
    """Creates a callable that initializes the environment for multiprocessing.

    Returns:
        callable: A function that returns a monitored FastFoodEnv instance.
    """

    def _init():
        env = FastFoodEnv()
        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    print("Initializing A2C Training (Lunch Rush + Dessert Station)...")

    models_dir = "models"
    log_dir = "tb_logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    num_cpu = 4
    env = SubprocVecEnv([make_env() for i in range(num_cpu)])

    model = A2C(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=log_dir,
        learning_rate=0.0007,
    )

    total_timesteps = 500_000
    print("Beginning A2C training. Open TensorBoard to watch progress!")

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="A2C_Lunch_Rush_Agent",
        progress_bar=True,
    )

    model_path = f"{models_dir}/fast_food_manager_a2c"
    model.save(model_path)
    print(f"Training Complete! Model saved successfully to {model_path}.zip")

    env.close()
