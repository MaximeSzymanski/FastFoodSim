import os

import matplotlib.pyplot as plt
import pandas as pd


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
    # plt.show()


if __name__ == "__main__":
    plot_learning_curve()
