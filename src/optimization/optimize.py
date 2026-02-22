import random
import statistics

import optuna
import simpy

# --- UPDATED IMPORTS ---
from src import config
from src.sim.processes import (
    burger_cook_loop,
    customer_arrivals,
    fry_cook_loop,
    inventory_manager,  # Added this so waste gets tracked!
)
from src.sim.restaurant import FastFoodRestaurant

# --- OPTIMIZER SETTINGS ---
N_SEEDS = (
    10  # How many random days to test EACH setup (Higher = more accurate, but slower)
)
N_TRIALS = 50  # How many totally different setups Optuna will try


def run_sim_for_optuna(cashiers, burger_cooks, fries_cooks, n_seeds):
    """Runs a mini-batch of simulations and returns the average profit."""
    profits = []

    # Run the exact same setup across 'n' different random days
    for seed in range(n_seeds):
        random.seed(seed)
        env = simpy.Environment()
        stats = {
            "wait_times": [],
            "wasted_burgers": [],
            "wasted_fries": [],
            "balked": [],
            "reneged": [],
            "captured_revenue": [],
            "lost_revenue": [],
        }

        # UPDATED: Pass all staff counts to the restaurant
        restaurant = FastFoodRestaurant(
            env,
            num_cashiers=cashiers,
            num_burger_cooks=burger_cooks,
            num_fries_cooks=fries_cooks,
        )

        # Start all the background processes
        for _ in range(burger_cooks):
            env.process(burger_cook_loop(env, restaurant))
        for _ in range(fries_cooks):
            env.process(fry_cook_loop(env, restaurant))

        env.process(customer_arrivals(env, restaurant, stats))

        # Start the inventory manager so waste is actually calculated!
        env.process(inventory_manager(env, restaurant, stats))

        env.run(until=config.SIM_TIME)

        # Calculate the Profit for this specific run
        revenue = sum(stats["captured_revenue"])
        sim_hours = config.SIM_TIME / 3600.0
        cost_staff = (
            cashiers * config.WAGE_CASHIER
            + burger_cooks * config.WAGE_BURGER_COOK
            + fries_cooks * config.WAGE_FRIES_COOK
        ) * sim_hours
        waste_cost = (len(stats["wasted_burgers"]) * config.COST_WASTED_BURGER) + (
            len(stats["wasted_fries"]) * config.COST_WASTED_FRIES
        )

        profit = revenue - (cost_staff + waste_cost)
        profits.append(profit)

    # Return the average profit across all 'n' seeds
    return statistics.mean(profits)


def objective(trial):
    """The function Optuna tries to maximize."""

    # 1. Let Optuna suggest the number of employees
    cashiers = trial.suggest_int("cashiers", 1, 5)
    burger_cooks = trial.suggest_int("burger_cooks", 1, 5)
    fries_cooks = trial.suggest_int("fries_cooks", 1, 3)

    # 2. Let Optuna suggest the inventory limits
    config.TARGET_BURGER_INV = trial.suggest_int("target_burger_inv", 2, 15)
    config.TARGET_FRIES_INV = trial.suggest_int("target_fries_inv", 4, 20)

    # 3. Run the simulation, passing in our N_SEEDS parameter!
    avg_profit = run_sim_for_optuna(cashiers, burger_cooks, fries_cooks, N_SEEDS)

    return avg_profit


if __name__ == "__main__":
    print(f"Starting AI Hyperparameter Optimization...")
    print(
        f"Testing {N_TRIALS} combinations, averaging {N_SEEDS} seeds per combination."
    )

    # Optuna will suppress the massive wall of text and just show a nice progress bar
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")

    # Optional: show a progress bar in the terminal so you know it hasn't crashed!
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n" + "=" * 40)
    print("🏆 OPTIMIZATION COMPLETE 🏆")
    print(f"Maximum Expected Profit (Avg over {N_SEEDS} days): ${study.best_value:.2f}")
    print("Optimal Restaurant Setup:")
    for key, value in study.best_params.items():
        print(f"  - {key}: {value}")
    print("=" * 40)
