import glob

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.rl.env import FastFoodEnv

from src.rl.train import mask_fn


def plot_learning_curve():
    """Generates the training progress chart from stats/ logs."""
    log_files = glob.glob("stats/*.monitor.csv")
    if not log_files:
        print("Error: No logs found.")
        return

    dataframes = [pd.read_csv(f, skiprows=1) for f in log_files]
    df = (
        pd.concat(dataframes, ignore_index=True).sort_values("t").reset_index(drop=True)
    )

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["r"], alpha=0.3, color="#66b3ff", label="Raw Episode Reward")

    if len(df) >= 50:
        window = min(len(df), 100)
        df["rolling_reward"] = df["r"].rolling(window=window).mean()
        plt.plot(
            df.index, df["rolling_reward"], color="#000080", linewidth=2, label="Trend"
        )

    plt.title(f"AI Manager Training Progress ({len(df)} Episodes)")
    plt.xlabel("Training Episode")
    plt.ylabel("Net Reward (Scaled)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("stats/rl_learning_curve.png")
    print(f"Chart saved to stats/rl_learning_curve.png")


def test_trained_ai():
    """Runs a visual shift using the saved model."""
    try:
        model = MaskablePPO.load("models/fast_food_manager_ai")
    except Exception:
        print("Model not found! Run train.py first.")
        return

    test_env = FastFoodEnv()
    env = ActionMasker(test_env, mask_fn)
    obs, _ = env.reset(seed=42)
    terminated = False
    total_reward = 0

    action_meanings = {
        0: "Did nothing",
        1: "Ordered Burger",
        2: "Ordered Fries",
        3: "Ordered BOTH",
    }
    inner_env = env.unwrapped

    while not terminated:
        action_masks = inner_env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        action = int(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if action != 0:
            print(
                f"[Time: {obs[5] * 3600:.0f}s] Queue: {obs[0] * 20:.0f} | AI Decision: {action_meanings[action]}"
            )

    print(f"Shift Complete! Scaled Reward: {total_reward:.2f}")


if __name__ == "__main__":
    plot_learning_curve()
    test_trained_ai()
