import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_missed_revenue():
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
    grouped = df.groupby("Setup")[["Revenue", "LostRevenue"]].mean().reset_index()

    plt.figure(figsize=(10, 6))
    bar1 = plt.bar(
        grouped["Setup"],
        grouped["Revenue"],
        color="#66b3ff",
        edgecolor="black",
        label="Captured Revenue ($)",
    )
    bar2 = plt.bar(
        grouped["Setup"],
        grouped["LostRevenue"],
        bottom=grouped["Revenue"],
        color="#ff9999",
        edgecolor="black",
        label="Lost Revenue (Angry Customers)",
    )

    plt.title("Captured Revenue vs. Missed Opportunity (Variable Menu)")
    plt.xlabel("Restaurant Configuration")
    plt.ylabel("Total Potential Revenue ($)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(grouped)):
        lost_val = grouped.loc[i, "LostRevenue"]
        rev_val = grouped.loc[i, "Revenue"]
        total_val = rev_val + lost_val

        if lost_val > 0:
            plt.text(
                i,
                rev_val + (lost_val / 2),
                f"-${lost_val:.0f}",
                ha="center",
                va="center",
                fontweight="bold",
                color="black",
            )
        plt.text(
            i,
            total_val + (total_val * 0.02),
            f"Potential: ${total_val:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.ylim(0, plt.ylim()[1] * 1.15)
    plt.savefig("stats/missed_opportunity.png")
    print("Plot saved to stats/missed_opportunity.png")
    plt.show()


if __name__ == "__main__":
    plot_missed_revenue()
