import os

import imageio
from stable_baselines3 import A2C

from src import config
from src.rl.FastFoodEnv import FastFoodEnv


def main():
    """Evaluates a trained A2C model, records the simulation, and saves it as a GIF.

    This function initializes the FastFoodEnv in RGB array mode, loads the A2C
    model, runs a complete simulation episode, and captures each frame.
    Upon completion, it compiles the frames into a 10 FPS GIF and prints the final
    financial statistics of the shift.
    """
    print("Initializing A2C Evaluation Environment and GIF Recorder...")

    env = FastFoodEnv(render_mode="rgb_array")

    model_path = "models/fast_food_manager_a2c"
    try:
        model = A2C.load(model_path, env=env)
        print(f"Successfully loaded A2C model from {model_path}")
    except Exception as e:
        print(f"Could not load model. Did you finish training it yet? Error: {e}")
        return

    os.makedirs("videos", exist_ok=True)
    gif_path = "videos/restaurant_ai_a2c.gif"
    frames = []

    obs, info = env.reset()
    done = False

    print("Recording A2C management shift to GIF... Please wait.")

    while not done:
        frames.append(env.render())

        # A2C does not support action masking
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    frames.append(env.render())

    stats = env.stats
    final_revenue = sum(stats["captured_revenue"])

    final_waste = (
        (len(stats["wasted_burgers"]) * config.COST_WASTED_BURGER)
        + (len(stats["wasted_fries"]) * config.COST_WASTED_FRIES)
        + (len(stats["wasted_ice_cream"]) * config.COST_WASTED_ICE_CREAM)
    )
    lost_customers = len(stats["balked"]) + len(stats["reneged"])

    env.close()

    print(
        f"Compiling {len(frames)} frames into a GIF... (this might take a few moments)"
    )

    imageio.mimsave(gif_path, frames, duration=100)

    print("\n--- A2C LUNCH RUSH SHIFT COMPLETE ---")
    print(f"Total Revenue Captured: ${final_revenue:.2f}")
    print(f"Total Food Wasted: ${final_waste:.2f}")
    print(f"Total Walk-Outs: {lost_customers}")
    print(f"GIF saved successfully to {gif_path}!")


if __name__ == "__main__":
    main()
