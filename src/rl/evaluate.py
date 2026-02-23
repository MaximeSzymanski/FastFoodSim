import os

import imageio
from sb3_contrib import MaskablePPO

from src import config
from src.rl.FastFoodEnv import FastFoodEnv


def main():
    print("Initializing Hardcore Environment and GIF Recorder...")

    # 1. Create the environment in RGB ARRAY mode
    env = FastFoodEnv(render_mode="rgb_array")

    # 2. Load your newly trained, elite model
    model_path = "models/fast_food_manager_elite"
    try:
        model = MaskablePPO.load(model_path, env=env)
        print(f"Successfully loaded elite model from {model_path}")
    except Exception as e:
        print(f"Could not load model. Did you finish training it yet? Error: {e}")
        return

    # 3. Setup GIF recording
    os.makedirs("videos", exist_ok=True)
    gif_path = "videos/hardcore_lunch_rush_ai.gif"
    frames = []

    # 4. Watch it work!
    obs, info = env.reset()
    done = False

    print(
        "Recording AI management shift to GIF... Please wait (this may take a minute)."
    )

    while not done:
        # Capture the current frame BEFORE taking the step
        frames.append(env.render())

        # Get the AI's action based on the current state and action masks
        action, _states = model.predict(
            obs, action_masks=env.action_masks(), deterministic=True
        )

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Capture the very last frame showing the final score
    frames.append(env.render())

    # 5. Extract Final Stats to see how it did
    stats = env.stats
    final_revenue = sum(stats["captured_revenue"])

    # Calculate the total waste using the config costs
    final_waste = (
        (len(stats["wasted_burgers"]) * config.COST_WASTED_BURGER)
        + (len(stats["wasted_fries"]) * config.COST_WASTED_FRIES)
        + (len(stats["wasted_ice_cream"]) * config.COST_WASTED_ICE_CREAM)
    )
    lost_customers = len(stats["balked"]) + len(stats["reneged"])

    env.close()

    # 6. Save the GIF!
    print(
        f"Compiling {len(frames)} frames into a GIF... (this might take a few moments)"
    )

    # We use duration=100 (milliseconds per frame) to achieve 10 FPS
    imageio.mimsave(gif_path, frames, duration=100)

    print("\n--- LUNCH RUSH SHIFT COMPLETE ---")
    print(f"Total Revenue Captured: ${final_revenue:.2f}")
    print(f"Total Food Wasted: ${final_waste:.2f}")
    print(f"Total Walk-Outs: {lost_customers}")
    print(f"GIF saved successfully to {gif_path}! Ready for your README.")


if __name__ == "__main__":
    main()
