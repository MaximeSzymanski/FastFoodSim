import os

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.rl.FastFoodEnv import FastFoodEnv


def mask_fn(env):
    """A wrapper function to pull the action masks from our environment."""
    # THE FIX: We must use .unwrapped to bypass the Monitor wrapper!
    return env.unwrapped.action_masks()


def make_env():
    """Utility function for multiprocessing."""

    def _init():
        env = FastFoodEnv()
        env = Monitor(env)  # TensorBoard hook
        # Wrap the environment so MaskablePPO knows where to find the valid actions
        return ActionMasker(env, mask_fn)

    return _init


if __name__ == "__main__":
    print("Initializing Hardcore AI Training (Lunch Rush + Dessert Station)...")

    # 1. Setup Directories
    models_dir = "models"
    log_dir = "tb_logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Vectorize the environment
    num_cpu = 4
    env = SubprocVecEnv([make_env() for i in range(num_cpu)])

    # 3. Build the MaskablePPO Agent
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,  # Keep console clean for tqdm
        tensorboard_log=log_dir,
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=64,
    )

    # 4. Train the AI!
    total_timesteps = 500_000
    print(
        f"Beginning training for {total_timesteps} steps. Open TensorBoard to watch progress!"
    )

    # progress_bar=True activates the clean tqdm display
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="PPO_Lunch_Rush_Agent",
        progress_bar=True,
    )

    # 5. Save the final elite manager
    model_path = f"{models_dir}/fast_food_manager_elite"
    model.save(model_path)
    print(f"Training Complete! Model saved successfully to {model_path}.zip")

    env.close()
