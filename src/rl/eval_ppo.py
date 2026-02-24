import os

import imageio
from sb3_contrib import MaskablePPO

from src import config
from src.rl.FastFoodEnv import FastFoodEnv


def main():
    """Evaluates a trained RL model, records the simulation, and saves it as a GIF.

    This function initializes the FastFoodEnv in RGB array mode, loads the elite
    MaskablePPO model, runs a complete simulation episode, and captures each frame.
    Upon completion, it compiles the frames into a 10 FPS GIF and prints the final
    financial statistics of the shift.
    """
    print("Initializing Hardcore Environment and GIF Recorder...")

    env = FastFoodEnv(render_mode="rgb_array")

    model_path = "models/fast_food_manager_elite"
    try:
        model = MaskablePPO.load(model_path, env=env)
        print(f"Successfully loaded elite model from {model_path}")
    except Exception as e:
        print(f"Could not load model. Did you finish training it yet? Error: {e}")
        return

    os.makedirs("videos", exist_ok=True)
    gif_path = "videos/restaurant_ai.gif"
    frames = []

    obs, info = env.reset()
    done = False

    print(
        "Recording AI management shift to GIF... Please wait (this may take a minute)."
    )

    while not done:
        frames.append(env.render())

        action, _states = model.predict(
            obs, action_masks=env.action_masks(), deterministic=True
        )

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

    print(f"Compiling {
            len(frames)} frames into a GIF... (this might take a few moments)")

    imageio.mimsave(gif_path, frames, duration=100)

    print("\n--- LUNCH RUSH SHIFT COMPLETE ---")
    print(f"Total Revenue Captured: ${final_revenue:.2f}")
    print(f"Total Food Wasted: ${final_waste:.2f}")
    print(f"Total Walk-Outs: {lost_customers}")
    print(f"GIF saved successfully to {gif_path}! Ready for your README.")


if __name__ == "__main__":
    main()
