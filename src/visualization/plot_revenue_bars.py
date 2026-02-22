import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_revenue_with_std():
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
    grouped = df.groupby("Setup")["Revenue"].agg(["mean", "std"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        grouped["Setup"],
        grouped["mean"],
        yerr=grouped["std"],
        capsize=8,
        color=["#ff9999", "#66b3ff", "#99ff99", "#ffd700"],
        edgecolor="black",
        alpha=0.8,
    )

    plt.title("Average Captured Revenue per Scenario")
    plt.xlabel("Restaurant Configuration")
    plt.ylabel("Revenue ($)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        std_val = grouped.loc[i, "std"]
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + std_val + (yval * 0.02),
            f"${yval:.0f}\n± ${std_val:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.ylim(0, plt.ylim()[1] * 1.15)
    plt.savefig("stats/revenue_std_bars.png")
    print("Plot saved to stats/revenue_std_bars.png")
    plt.show()


if __name__ == "__main__":
    plot_revenue_with_std()
