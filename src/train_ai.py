import glob
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from FastFoodEnv import FastFoodEnv


def mask_fn(env: gym.Env):
    """Extracts the valid actions from the unwrapped custom environment."""
    return env.unwrapped.action_masks()


def train_kitchen_ai():
    print("Booting up the Vectorized Fast Food Environments...")
    os.makedirs("stats", exist_ok=True)

    # Clear old multiprocess logs so our graph is clean
    for f in glob.glob("stats/*.monitor.csv"):
        try:
            os.remove(f)
        except OSError:
            pass

    # 1. Setup Parallel Environments (4 CPU Cores)
    num_cpu = 4
    env = make_vec_env(
        FastFoodEnv,
        n_envs=num_cpu,
        wrapper_class=ActionMasker,
        wrapper_kwargs={"action_mask_fn": mask_fn},
        vec_env_cls=SubprocVecEnv,
        monitor_dir="stats",  # Saves logs as 0.monitor.csv, 1.monitor.csv, etc.
    )

    # 2. Initialize MaskablePPO with High Entropy
    print(f"Initializing AI on {num_cpu} parallel CPU cores...")
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=0.0003,
        ent_coef=0.05,  # Forces the AI to stay curious!
    )

    # 3. Train for 1 Million Steps
    print("Beginning Training (Go grab a coffee, it's blazing fast now!)...")
    model.learn(total_timesteps=5_000_000, progress_bar=True)

    # 4. Save
    model.save("fast_food_manager_ai")
    print("Training complete! AI brain saved.")

    return model


def test_trained_ai(model):
    """Watch the trained AI manage a single 1-hour shift."""
    print("\n" + "=" * 40)
    print("🍔 RUNNING A TEST SHIFT WITH THE TRAINED AI 🍔")
    print("=" * 40)

    # We create a single standard environment just for the visual test
    test_env = FastFoodEnv()
    env = ActionMasker(test_env, mask_fn)

    obs, info = env.reset(seed=42)
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
        action, _states = model.predict(
            obs, action_masks=action_masks, deterministic=True
        )
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if action != 0:
            queue_len = obs[0] * 20.0  # Un-scale for printing
            time_passed = obs[5] * 3600  # Un-scale for printing
            print(
                f"[Time: {time_passed:.0f}s] Queue: {queue_len:.0f} | AI Decision: {action_meanings[action]}"
            )

    print("=" * 40)
    print(f"Shift Complete! Total Net Reward (Scaled): {total_reward:.2f}")

    stats = inner_env.stats
    print(f"Customers Served: {len(stats['wait_times'])}")
    print(f"Lost Customers: {len(stats['balked']) + len(stats['reneged'])}")
    print(f"Wasted Burgers: {len(stats['wasted_burgers'])}")
    print(f"Wasted Fries: {len(stats['wasted_fries'])}")
    print("=" * 40)


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
    plt.plot(df.index, df["r"], alpha=0.3, color="#66b3ff", label="Raw Episode Reward")

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
    plt.show()


if __name__ == "__main__":
    # 1. Train the model using all CPU cores
    trained_model = train_kitchen_ai()

    # 2. Generate the combined learning curve
    plot_learning_curve()

    # 3. Test the final brain on a single shift
    test_trained_ai(trained_model)
