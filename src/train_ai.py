import os

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from FastFoodEnv import FastFoodEnv


def mask_fn(env: gym.Env):
    """Extracts the valid actions from the custom environment."""
    # We use .unwrapped to get to the FastFoodEnv class specifically
    return env.unwrapped.action_masks()


def train_kitchen_ai():
    print("Booting up the Fast Food Environment...")
    os.makedirs("stats", exist_ok=True)

    # 1. Create base env
    base_env = FastFoodEnv()

    # 2. Wrap with Monitor (Essential for the CSV logs)
    # Use a absolute or clean path to avoid write issues
    log_path = "stats/rl_training_logs.monitor.csv"
    if os.path.exists(log_path):
        os.remove(log_path)  # Clear old logs for a fresh chart
    monitored_env = Monitor(base_env, "stats/rl_training_logs")

    # 3. Wrap with ActionMasker
    env = ActionMasker(monitored_env, mask_fn)

    # 4. Initialize MaskablePPO
    print("Initializing the Deep Reinforcement Learning Agent...")
    model = MaskablePPO("MlpPolicy", env, verbose=0, learning_rate=0.005)

    # 5. Train with Progress Bar
    print("Beginning Training (Watch the progress bar!)...")
    model.learn(total_timesteps=500000, progress_bar=True)

    # 6. Save
    model.save("fast_food_manager_ai")
    print("Training complete! AI brain saved.")

    return model, env


def test_trained_ai(model, env):
    """Watch the trained AI manage a single 1-hour shift."""
    print("\n" + "=" * 40)
    print("🍔 RUNNING A TEST SHIFT WITH THE TRAINED AI 🍔")
    print("=" * 40)

    obs, info = env.reset()
    terminated = False
    total_reward = 0

    action_meanings = {
        0: "Did nothing",
        1: "Ordered Burger",
        2: "Ordered Fries",
        3: "Ordered BOTH",
    }

    # Access the base env class once for cleaner code
    inner_env = env.unwrapped

    while not terminated:
        # Get masks directly from the unwrapped environment
        action_masks = inner_env.action_masks()

        # Predict
        action, _states = model.predict(
            obs, action_masks=action_masks, deterministic=True
        )
        action = int(action)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print logic
        if action != 0:
            # obs[0] is Queue Length based on our FastFoodEnv observation space
            queue_len = obs[0]
            print(
                f"[Time: {inner_env.env.now:.0f}s] Queue: {queue_len:.0f} | AI Decision: {action_meanings[action]}"
            )

    print("=" * 40)
    print(f"Shift Complete! Total Net Reward: ${total_reward:.2f}")

    # Access stats safely from the unwrapped class
    stats = inner_env.stats
    print(f"Customers Served: {len(stats['wait_times'])}")
    print(f"Lost Customers: {len(stats['balked']) + len(stats['reneged'])}")
    print(f"Wasted Burgers: {len(stats['wasted_burgers'])}")
    print(f"Wasted Fries: {len(stats['wasted_fries'])}")
    print("=" * 40)


def plot_learning_curve():
    log_file = "stats/rl_training_logs.monitor.csv"

    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    try:
        # Monitor files have a comment on the first line, pandas handles this well with skiprows
        df = pd.read_csv(log_file, skiprows=1)
    except Exception as e:
        print(f"Could not read log file: {e}")
        return

    if df.empty:
        print("Log file is empty! Ensure training completed at least one full episode.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["r"], alpha=0.3, color="#66b3ff", label="Raw Episode Reward")

    if len(df) >= 5:
        # Use a window relative to the amount of data we actually have
        window = min(len(df), 20)
        df["rolling_reward"] = df["r"].rolling(window=window).mean()
        plt.plot(
            df.index,
            df["rolling_reward"],
            color="#000080",
            linewidth=2,
            label=f"Trend ({window}-Ep Moving Avg)",
        )

    plt.title("AI Manager Training Progress (Learning Curve)")
    plt.xlabel("Training Episode (1-Hour Shift)")
    plt.ylabel("Net Reward ($)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig("stats/rl_learning_curve.png")
    print(f"\nSuccessfully plotted {len(df)} episodes to 'stats/rl_learning_curve.png'")
    plt.show()


if __name__ == "__main__":
    # 1. Train
    trained_model, test_env = train_kitchen_ai()

    # 2. Plot
    plot_learning_curve()

    # 3. Test
    test_trained_ai(trained_model, test_env)
