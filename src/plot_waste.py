import os

import matplotlib.pyplot as plt
import pandas as pd

from config import COST_WASTED_BURGER, COST_WASTED_FRIES


def plot_waste_costs():
    csv_path = "stats/financials.csv"
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    df["Setup"] = (
        df["Cashiers"].astype(str)
        + "C, "
        + df["BurgerCooks"].astype(str)
        + "B, "
        + df["FriesCooks"].astype(str)
        + "F"
    )

    df["BurgerWasteCost"] = df["WastedBurgers"] * COST_WASTED_BURGER
    df["FriesWasteCost"] = df["WastedFries"] * COST_WASTED_FRIES
    grouped = (
        df.groupby("Setup")[["BurgerWasteCost", "FriesWasteCost"]].mean().reset_index()
    )

    plt.figure(figsize=(10, 6))
    bar1 = plt.bar(
        grouped["Setup"],
        grouped["BurgerWasteCost"],
        color="#ff9999",
        edgecolor="black",
        label=f"Burger Waste (${COST_WASTED_BURGER}/ea)",
    )
    bar2 = plt.bar(
        grouped["Setup"],
        grouped["FriesWasteCost"],
        bottom=grouped["BurgerWasteCost"],
        color="#ffd700",
        edgecolor="black",
        label=f"Fries Waste (${COST_WASTED_FRIES}/ea)",
    )

    plt.title("Average Financial Loss Due to Expired Food")
    plt.xlabel("Restaurant Configuration")
    plt.ylabel("Money Lost ($)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(grouped)):
        total_waste = (
            grouped.loc[i, "BurgerWasteCost"] + grouped.loc[i, "FriesWasteCost"]
        )
        plt.text(
            i,
            total_waste + (total_waste * 0.02) + 0.5,
            f"${total_waste:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.ylim(0, plt.ylim()[1] * 1.15)
    plt.savefig("stats/waste_costs_stacked.png")
    print("Plot saved to stats/waste_costs_stacked.png")
    plt.show()


if __name__ == "__main__":
    plot_waste_costs()
