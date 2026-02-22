import csv
import os
import random

import simpy

from config import *
from processes import burger_cook_loop, customer_arrivals, fry_cook_loop
from restaurant import FastFoodRestaurant

NUM_RUNS = 10

SCENARIOS = [(1, 1, 1), (2, 2, 1), (2, 2, 2), (3, 3, 2)]


def run_simulation(seed, cashiers, burger_cooks, fries_cooks, wait_writer, fin_writer):
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

    restaurant = FastFoodRestaurant(env, cashiers)

    for _ in range(burger_cooks):
        env.process(burger_cook_loop(env, restaurant))
    for _ in range(fries_cooks):
        env.process(fry_cook_loop(env, restaurant))

    env.process(customer_arrivals(env, restaurant, stats))
    env.run(until=SIM_TIME)

    # Data Export
    for customer_id, wait_time in enumerate(stats["wait_times"]):
        wait_writer.writerow(
            [cashiers, burger_cooks, fries_cooks, seed, customer_id + 1, wait_time]
        )

    customers_served = len(stats["wait_times"])
    total_lost_customers = len(stats["balked"]) + len(stats["reneged"])

    revenue = sum(stats["captured_revenue"])
    lost_revenue = sum(stats["lost_revenue"])

    sim_hours = SIM_TIME / 3600.0
    cost_cashiers = cashiers * WAGE_CASHIER * sim_hours
    cost_burger = burger_cooks * WAGE_BURGER_COOK * sim_hours
    cost_fries = fries_cooks * WAGE_FRIES_COOK * sim_hours

    total_wasted_burgers = len(stats["wasted_burgers"])
    total_wasted_fries = len(stats["wasted_fries"])
    waste_cost = (total_wasted_burgers * COST_WASTED_BURGER) + (
        total_wasted_fries * COST_WASTED_FRIES
    )

    total_cost = cost_cashiers + cost_burger + cost_fries + waste_cost
    profit = revenue - total_cost

    fin_writer.writerow(
        [
            cashiers,
            burger_cooks,
            fries_cooks,
            seed,
            customers_served,
            total_lost_customers,
            total_wasted_burgers,
            total_wasted_fries,
            revenue,
            lost_revenue,
            total_cost,
            profit,
        ]
    )


if __name__ == "__main__":
    print("Starting Advanced Fast-Food Simulation...")
    os.makedirs("stats", exist_ok=True)

    with (
        open("stats/results.csv", mode="w", newline="") as wait_file,
        open("stats/financials.csv", mode="w", newline="") as fin_file,
    ):
        wait_writer = csv.writer(wait_file)
        wait_writer.writerow(
            ["Cashiers", "BurgerCooks", "FriesCooks", "Seed", "CustomerID", "WaitTime"]
        )

        fin_writer = csv.writer(fin_file)
        fin_writer.writerow(
            [
                "Cashiers",
                "BurgerCooks",
                "FriesCooks",
                "Seed",
                "CustomersServed",
                "LostCustomers",
                "WastedBurgers",
                "WastedFries",
                "Revenue",
                "LostRevenue",
                "TotalCost",
                "Profit",
            ]
        )

        for cashiers, burger_cooks, fries_cooks in SCENARIOS:
            print(f"\nTesting Setup: {cashiers}C, {burger_cooks}B, {fries_cooks}F")
            for seed in range(NUM_RUNS):
                run_simulation(
                    seed, cashiers, burger_cooks, fries_cooks, wait_writer, fin_writer
                )
                print(f"  Run {seed + 1}/{NUM_RUNS} complete.")

    print("\nAll scenarios finished! Data saved.")
