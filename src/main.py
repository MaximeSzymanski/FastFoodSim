import csv
import os
import random

import simpy

from config import *
from processes import customer_arrivals
from restaurant import FastFoodRestaurant

NUM_RUNS = 10
SCENARIOS = [
    (1, 1),  # 1 Cashier, 1 Cook
    (2, 2),  # 2 Cashiers, 2 Cooks
    (3, 3),  # 3 Cashiers, 3 Cooks
    (5, 5),
    (10, 10),
]


def run_simulation(seed, num_cashiers, num_cooks, wait_writer, fin_writer):
    random.seed(seed)
    env = simpy.Environment()
    stats = {"wait_times": []}

    restaurant = FastFoodRestaurant(env, num_cashiers, num_cooks)
    env.process(customer_arrivals(env, restaurant, stats))
    env.run(until=SIM_TIME)

    # --- 1. Save Wait Times ---
    for customer_id, wait_time in enumerate(stats["wait_times"]):
        wait_writer.writerow(
            [num_cashiers, num_cooks, seed, customer_id + 1, wait_time]
        )

    # --- 2. Calculate Financials ---
    customers_served = len(stats["wait_times"])
    revenue = customers_served * ORDER_VALUE

    # Calculate costs (Sim time is in seconds, so we convert to hours)
    sim_hours = SIM_TIME / 3600.0
    cost_cashiers = num_cashiers * WAGE_CASHIER * sim_hours
    cost_cooks = num_cooks * WAGE_COOK * sim_hours
    total_cost = cost_cashiers + cost_cooks

    profit = revenue - total_cost

    # Save the financial summary for this specific run
    fin_writer.writerow(
        [num_cashiers, num_cooks, seed, customers_served, revenue, total_cost, profit]
    )


if __name__ == "__main__":
    print("Starting Financial Scenario Analysis...")
    os.makedirs("stats", exist_ok=True)

    # Open both CSV files
    with (
        open("stats/results.csv", mode="w", newline="") as wait_file,
        open("stats/financials.csv", mode="w", newline="") as fin_file,
    ):
        wait_writer = csv.writer(wait_file)
        wait_writer.writerow(["Cashiers", "Cooks", "Seed", "CustomerID", "WaitTime"])

        fin_writer = csv.writer(fin_file)
        fin_writer.writerow(
            [
                "Cashiers",
                "Cooks",
                "Seed",
                "CustomersServed",
                "Revenue",
                "TotalCost",
                "Profit",
            ]
        )

        for cashiers, cooks in SCENARIOS:
            print(f"\nTesting Setup: {cashiers} Cashiers, {cooks} Cooks")
            for seed in range(NUM_RUNS):
                run_simulation(seed, cashiers, cooks, wait_writer, fin_writer)
                print(f"  Run {seed + 1}/{NUM_RUNS} complete.")

    print("\nAll scenarios finished! Financial data saved to stats/financials.csv")
