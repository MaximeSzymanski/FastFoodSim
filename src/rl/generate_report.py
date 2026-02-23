import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C, DQN

from src import config
from src.rl.FastFoodEnv import FastFoodEnv


def generate_business_report(model_path, model_type="PPO", num_seeds=5):
    """Generates a business report by evaluating an RL model across multiple random seeds.

    Args:
        model_path (str): The file path to the trained model.
        model_type (str): The type of RL algorithm used ('PPO', 'A2C', or 'DQN').
        num_seeds (int): The number of simulation days (seeds) to average across.
    """
    print(
        f"📊 Generating Business Report for {model_type} AI across {num_seeds} seeds..."
    )

    env = FastFoodEnv(render_mode=None)

    try:
        if model_type == "PPO":
            model = MaskablePPO.load(model_path, env=env)
        elif model_type == "A2C":
            model = A2C.load(model_path, env=env)
        elif model_type == "DQN":
            model = DQN.load(model_path, env=env)
    except Exception as e:
        print(f"❌ Could not load {model_path}. Did it finish training? Error: {e}")
        return

    all_data = []

    for seed in range(num_seeds):
        obs, info = env.reset(seed=seed)
        done = False

        while not done:
            if model_type == "PPO":
                action, _ = model.predict(
                    obs, action_masks=env.unwrapped.action_masks(), deterministic=True
                )
            else:
                action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            stats = env.unwrapped.stats
            current_rev = sum(stats["captured_revenue"])
            current_waste = (
                (len(stats["wasted_burgers"]) * config.COST_WASTED_BURGER)
                + (len(stats["wasted_fries"]) * config.COST_WASTED_FRIES)
                + (len(stats["wasted_ice_cream"]) * config.COST_WASTED_ICE_CREAM)
            )
            current_lost = sum(stats["lost_revenue"])

            # Collect all extended stats for the 3 graphs
            all_data.append(
                {
                    "seed": seed,
                    "time_min": env.unwrapped.env.now / 60.0,
                    "revenue": current_rev,
                    "waste": current_waste,
                    "lost_revenue": current_lost,
                    "net_profit": current_rev - current_waste,
                    "burger_inv": len(env.unwrapped.restaurant.burger_shelf.items),
                    "fries_inv": len(env.unwrapped.restaurant.fries_shelf.items),
                    "ice_cream_inv": len(
                        env.unwrapped.restaurant.ice_cream_shelf.items
                    ),
                    "queue_len": len(env.unwrapped.restaurant.cashier.queue),
                    "waiting_food": env.unwrapped.restaurant.customers_waiting_for_food,
                }
            )

    df = pd.DataFrame(all_data)

    df_mean = df.groupby("time_min").mean().reset_index()
    df_std = df.groupby("time_min").std().reset_index().fillna(0)

    plt.style.use("ggplot")

    # Created 3 subplots instead of 2, expanded the height
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), dpi=150)
    fig.suptitle(
        f"AI Restaurant Manager: Lunch Rush Performance ({num_seeds} Days)",
        fontsize=16,
        fontweight="bold",
    )

    # --- Plot 1: Financial Health ---
    metrics_ax1 = [
        ("revenue", "Cumulative Revenue", "green"),
        ("net_profit", "Net Profit (After Waste)", "blue"),
        ("waste", "Food Waste Cost", "orange"),
        ("lost_revenue", "Lost Revenue (Walk-outs)", "red"),
    ]

    for col, label, color in metrics_ax1:
        ax1.plot(
            df_mean["time_min"], df_mean[col], label=label, color=color, linewidth=2
        )
        ax1.fill_between(
            df_mean["time_min"],
            df_mean[col] - df_std[col],
            df_mean[col] + df_std[col],
            color=color,
            alpha=0.15,
        )

    ax1.axvspan(20, 40, color="gray", alpha=0.2, label="Lunch Rush Wave")
    ax1.set_title("Financial Growth Over 1-Hour Shift", fontsize=12)
    ax1.set_ylabel("USD ($)")
    ax1.legend(loc="upper left")

    # --- Plot 2: Predictive Stockpiling Behavior ---
    metrics_ax2 = [
        ("burger_inv", "Burgers on Prep Table", "brown", "-"),
        ("fries_inv", "Fries under Heat Lamp", "goldenrod", "-"),
        ("ice_cream_inv", "Ice Cream Ready", "teal", "-"),
    ]

    for col, label, color, style in metrics_ax2:
        # THE FIX: This is now correctly ax2.plot instead of ax1.plot
        ax2.plot(
            df_mean["time_min"],
            df_mean[col],
            label=label,
            color=color,
            linewidth=2,
            linestyle=style,
        )
        ax2.fill_between(
            df_mean["time_min"],
            np.maximum(0, df_mean[col] - df_std[col]),  # Floor at 0 for inventories
            df_mean[col] + df_std[col],
            color=color,
            alpha=0.15,
        )

    ax2.axvspan(20, 40, color="gray", alpha=0.2)
    ax2.set_title("Predictive Inventory Control (Shelves)", fontsize=12)
    ax2.set_ylabel("Item Count")
    ax2.legend(loc="upper left")

    # --- Plot 3: Customer Crowding / Bottlenecks ---
    metrics_ax3 = [
        ("queue_len", "Cashier Line (Ordering)", "purple", "--"),
        ("waiting_food", "Pickup Area Crowding", "magenta", "-."),
    ]

    for col, label, color, style in metrics_ax3:
        ax3.plot(
            df_mean["time_min"],
            df_mean[col],
            label=label,
            color=color,
            linewidth=2,
            linestyle=style,
        )
        ax3.fill_between(
            df_mean["time_min"],
            np.maximum(0, df_mean[col] - df_std[col]),  # Floor at 0 for people
            df_mean[col] + df_std[col],
            color=color,
            alpha=0.15,
        )

    ax3.axvspan(20, 40, color="gray", alpha=0.2)
    ax3.set_title("Customer Flow & Bottlenecks", fontsize=12)
    ax3.set_xlabel("Simulation Time (Minutes)")
    ax3.set_ylabel("People Count")
    ax3.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("business_report.png")
    print("\n✅ Dashboard saved as 'business_report.png'!")

    # 4. Print the Concrete Text Summary
    final_steps = df[df["time_min"] == df_mean["time_min"].max()]

    final_rev_mean = final_steps["revenue"].mean()
    final_rev_std = final_steps["revenue"].std()

    final_waste_mean = final_steps["waste"].mean()
    final_waste_std = final_steps["waste"].std()

    final_lost_mean = final_steps["lost_revenue"].mean()
    final_lost_std = final_steps["lost_revenue"].std()

    final_profit_mean = final_steps["net_profit"].mean()
    final_profit_std = final_steps["net_profit"].std()

    print("\n" + "=" * 50)
    print(f"📈 CONCRETE BUSINESS KPIs (Avg over {num_seeds} seeds)")
    print("=" * 50)
    print(f"Total Revenue:       ${final_rev_mean:.2f} ± ${final_rev_std:.2f}")
    print(f"Total Waste Cost:    ${final_waste_mean:.2f} ± ${final_waste_std:.2f}")
    print(f"Lost to Walkouts:    ${final_lost_mean:.2f} ± ${final_lost_std:.2f}")
    print(f"NET SHIFT PROFIT:    ${final_profit_mean:.2f} ± ${final_profit_std:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    run_model = "models/fast_food_manager_elite.zip"
    generate_business_report(run_model, model_type="PPO", num_seeds=5)
