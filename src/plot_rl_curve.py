import glob

import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve():
    """Reads all parallel CSV logs and combines them into one chart."""
    log_files = glob.glob("stats/*.monitor.csv")

    if not log_files:
        print("Error: No .monitor.csv files found in the 'stats' folder.")
        return

    dataframes = []
    for f in log_files:
        try:
            # skiprows=1 skips the JSON metadata header that Monitor adds
            df = pd.read_csv(f, skiprows=1)
            dataframes.append(df)
        except Exception as e:
            print(f"Could not read {f}: {e}")

    if not dataframes:
        print("No valid data found in logs.")
        return

    # Combine all CPU logs and sort them chronologically by wall-clock time ('t')
    df = pd.concat(dataframes, ignore_index=True)
    df = df.sort_values("t").reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    # Plot the raw data (light blue)
    plt.plot(df.index, df["r"], alpha=0.3, color="#66b3ff", label="Raw Episode Reward")

    # Plot the trendline (dark blue)
    if len(df) >= 50:
        window = min(len(df), 100)
        df["rolling_reward"] = df["r"].rolling(window=window).mean()
        plt.plot(
            df.index,
            df["rolling_reward"],
            color="#000080",
            linewidth=2,
            label=f"Trend ({window}-Ep Moving Avg)",
        )

    plt.title(f"AI Manager Training Progress ({len(df)} Total Episodes)")
    plt.xlabel("Training Episode (Chronological across all CPUs)")
    plt.ylabel("Net Reward (Scaled)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig("stats/rl_learning_curve.png")
    print(f"\nSuccessfully plotted {len(df)} episodes to 'stats/rl_learning_curve.png'")
    # plt.show()


if __name__ == "__main__":
    plot_learning_curve()
