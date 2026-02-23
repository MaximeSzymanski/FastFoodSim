import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3 import A2C, DQN

from src import config
from src.rl.FastFoodEnv import FastFoodEnv


def generate_business_report(model_path, model_type="PPO"):
    print(f"📊 Generating Business Report for {model_type} AI...")

    # 1. Setup Environment
    env = FastFoodEnv(render_mode=None)  # No UI needed, we just want the data

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

    obs, info = env.reset()
    done = False

    # Data collectors
    history = {
        "time_min": [],
        "revenue": [],
        "waste": [],
        "lost_revenue": [],
        "net_profit": [],
        "burger_inv": [],
        "queue_len": [],
    }

    # 2. Simulate the 1-Hour Shift
    while not done:
        if model_type == "PPO":
            action, _ = model.predict(
                obs, action_masks=env.unwrapped.action_masks(), deterministic=True
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Extract real-time metrics
        stats = env.unwrapped.stats
        current_rev = sum(stats["captured_revenue"])
        current_waste = (
            (len(stats["wasted_burgers"]) * config.COST_WASTED_BURGER)
            + (len(stats["wasted_fries"]) * config.COST_WASTED_FRIES)
            + (len(stats["wasted_ice_cream"]) * config.COST_WASTED_ICE_CREAM)
        )
        current_lost = sum(stats["lost_revenue"])

        # Log it
        history["time_min"].append(env.unwrapped.env.now / 60.0)
        history["revenue"].append(current_rev)
        history["waste"].append(current_waste)
        history["lost_revenue"].append(current_lost)
        history["net_profit"].append(current_rev - current_waste)
        history["burger_inv"].append(len(env.unwrapped.restaurant.burger_shelf.items))
        history["queue_len"].append(len(env.unwrapped.restaurant.cashier.queue))

    df = pd.DataFrame(history)

    # 3. Create the Visual Dashboard
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150)
    fig.suptitle(
        "AI Restaurant Manager: Lunch Rush Performance", fontsize=16, fontweight="bold"
    )

    # --- Plot 1: Financial Health ---
    ax1.plot(
        df["time_min"],
        df["revenue"],
        label="Cumulative Revenue",
        color="green",
        linewidth=2,
    )
    ax1.plot(
        df["time_min"],
        df["net_profit"],
        label="Net Profit (After Waste)",
        color="blue",
        linewidth=2,
    )
    ax1.plot(
        df["time_min"],
        df["waste"],
        label="Food Waste Cost",
        color="orange",
        linewidth=2,
    )
    ax1.plot(
        df["time_min"],
        df["lost_revenue"],
        label="Lost Revenue (Walk-outs)",
        color="red",
        linewidth=2,
    )

    # Highlight the Lunch Rush (mins 20 to 40)
    ax1.axvspan(20, 40, color="gray", alpha=0.2, label="Lunch Rush Wave")
    ax1.set_title("Financial Growth Over 1-Hour Shift", fontsize=12)
    ax1.set_ylabel("USD ($)")
    ax1.legend(loc="upper left")

    # --- Plot 2: Predictive Stockpiling Behavior ---
    ax2.plot(
        df["time_min"],
        df["burger_inv"],
        label="Burgers on Prep Table",
        color="brown",
        linewidth=2,
    )
    ax2.plot(
        df["time_min"],
        df["queue_len"],
        label="Customer Queue Length",
        color="purple",
        linestyle="--",
    )

    ax2.axvspan(20, 40, color="gray", alpha=0.2)
    ax2.set_title("Predictive Inventory Control", fontsize=12)
    ax2.set_xlabel("Simulation Time (Minutes)")
    ax2.set_ylabel("Count (Items / People)")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("business_report.png")
    print("\n✅ Dashboard saved as 'business_report.png'!")

    # 4. Print the Concrete Text Summary
    final_rev = df["revenue"].iloc[-1]
    final_waste = df["waste"].iloc[-1]
    final_lost = df["lost_revenue"].iloc[-1]
    final_profit = df["net_profit"].iloc[-1]

    print("\n" + "=" * 40)
    print("📈 CONCRETE BUSINESS KPIs")
    print("=" * 40)
    print(f"Total Revenue:       ${final_rev:.2f}")
    print(f"Total Waste Cost:    ${final_waste:.2f}")
    print(f"Lost to Walkouts:    ${final_lost:.2f}")
    print(f"NET SHIFT PROFIT:    ${final_profit:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    # Change this to whatever model performed the best in your tournament!
    run_model = "models/fast_food_manager_elite.zip"
    generate_business_report(run_model, model_type="PPO")
