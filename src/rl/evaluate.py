import os

from gymnasium.wrappers import RecordVideo
from sb3_contrib import MaskablePPO

from src.rl.FastFoodEnv import FastFoodEnv


def main():
    print("Initializing Environment and Video Recorder...")

    # 1. Create the environment in RGB ARRAY mode so it can be recorded
    base_env = FastFoodEnv(render_mode="rgb_array")

    # 2. Wrap it to record an MP4! (It will save in a folder called 'videos')
    os.makedirs("videos", exist_ok=True)
    env = RecordVideo(
        base_env,
        video_folder="videos",
        name_prefix="lunch_rush_ai",
        episode_trigger=lambda x: True,  # Record every episode we run
    )

    # 3. Load your newly trained, profitable model
    model_path = "models/fast_food_manager_ai"
    try:
        model = MaskablePPO.load(model_path, env=env)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Could not load model. Did you train it yet? Error: {e}")
        return

    # 4. Watch it work!
    obs, info = env.reset()
    done = False

    print("Recording AI management shift... Please wait (this may take a minute).")

    while not done:
        # Get the AI's action based on the current state and action masks
        action, _states = model.predict(
            obs, action_masks=env.unwrapped.action_masks(), deterministic=True
        )

        # Step the environment (the wrapper automatically grabs the frame and saves it!)
        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

    # 5. Extract Final Stats to see how it did
    final_revenue = sum(env.unwrapped.stats["captured_revenue"])
    final_waste = (len(env.unwrapped.stats["wasted_burgers"]) * 3.50) + (
        len(env.unwrapped.stats["wasted_fries"]) * 1.00
    )
    lost_customers = len(env.unwrapped.stats["balked"]) + len(
        env.unwrapped.stats["reneged"]
    )

    # Safely close the window and finalize the MP4 file
    env.close()

    print("\n--- SHIFT COMPLETE ---")
    print(f"Total Revenue Captured: ${final_revenue:.2f}")
    print(f"Total Food Wasted: ${final_waste:.2f}")
    print(f"Total Walk-Outs: {lost_customers}")
    print("Video saved successfully in the 'videos' folder!")


if __name__ == "__main__":
    main()
