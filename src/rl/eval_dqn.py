import os

import gymnasium as gym
import imageio
from stable_baselines3 import DQN

from src import config
from src.rl.FastFoodEnv import FastFoodEnv


class FlattenActionWrapper(gym.ActionWrapper):
    """A wrapper that flattens a MultiDiscrete action space into a single Discrete space."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, act):
        return [act // 4, (act // 2) % 2, act % 2]


def main():
    """Evaluates a trained DQN model, records the simulation, and saves it as a GIF.

    This function initializes the FastFoodEnv wrapped with a FlattenActionWrapper,
    loads the DQN model, runs a simulation episode, and captures each frame.
    It compiles the frames into a 10 FPS GIF and prints the final financial statistics.
    """
    print("Initializing DQN Evaluation Environment and GIF Recorder...")

    # We must wrap the environment exactly how we did during training
    base_env = FastFoodEnv(render_mode="rgb_array")
    env = FlattenActionWrapper(base_env)

    model_path = "models/fast_food_manager_dqn"
    try:
        model = DQN.load(model_path, env=env)
        print(f"Successfully loaded DQN model from {model_path}")
    except Exception as e:
        print(f"Could not load model. Did you finish training it yet? Error: {e}")
        return

    os.makedirs("videos", exist_ok=True)
    gif_path = "videos/restaurant_ai_dqn.gif"
    frames = []

    obs, info = env.reset()
    done = False

    print("Recording DQN management shift to GIF... Please wait.")

    while not done:
        frames.append(env.render())

        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    frames.append(env.render())

    # Use env.unwrapped to reliably access the core environment's statistics
    stats = env.unwrapped.stats
    final_revenue = sum(stats["captured_revenue"])

    final_waste = (
        (len(stats["wasted_burgers"]) * config.COST_WASTED_BURGER)
        + (len(stats["wasted_fries"]) * config.COST_WASTED_FRIES)
        + (len(stats["wasted_ice_cream"]) * config.COST_WASTED_ICE_CREAM)
    )
    lost_customers = len(stats["balked"]) + len(stats["reneged"])

    env.close()

    print(f"Compiling {
            len(frames)} frames into a GIF... (this might take a few moments)")

    imageio.mimsave(gif_path, frames, duration=100)

    print("\n--- DQN LUNCH RUSH SHIFT COMPLETE ---")
    print(f"Total Revenue Captured: ${final_revenue:.2f}")
    print(f"Total Food Wasted: ${final_waste:.2f}")
    print(f"Total Walk-Outs: {lost_customers}")
    print(f"GIF saved successfully to {gif_path}!")


if __name__ == "__main__":
    main()
