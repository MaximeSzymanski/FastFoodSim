import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_revenue_with_std():
    csv_path = "stats/financials.csv"

    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. Run main.py first!")
        return

    # Load the financial data we saved earlier
    df = pd.read_csv(csv_path)

    # Create a readable label for the X-axis
    df["Setup"] = df["Cashiers"].astype(str) + "C, " + df["Cooks"].astype(str) + "K"

    # Group by the Setup and calculate both the Mean and the Standard Deviation of Revenue
    grouped = df.groupby("Setup")["Revenue"].agg(["mean", "std"]).reset_index()

    # If you only ran 1 seed, std will be NaN (Not a Number). Let's fill it with 0 just in case.
    grouped["std"] = grouped["std"].fillna(0)

    # Set up the plot
    plt.figure(figsize=(9, 6))

    # Create a bar chart with error bars (yerr represents the standard deviation)
    bars = plt.bar(
        grouped["Setup"],
        grouped["mean"],
        yerr=grouped["std"],  # This draws the standard deviation lines!
        capsize=8,  # This adds the little flat caps to the top/bottom of the error bars
        color=["#ff9999", "#66b3ff", "#99ff99"],
        edgecolor="black",
        alpha=0.8,
    )

    # Format the chart
    plt.title("Average Revenue per Scenario (with Standard Deviation)")
    plt.xlabel("Restaurant Configuration (Cashiers, Cooks)")
    plt.ylabel("Revenue ($)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Add the exact average dollar amount on top of each bar (adjusted to sit above the error bar)
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        std_val = grouped.loc[i, "std"]

        # Place the text slightly above the top of the standard deviation line
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + std_val + 5,
            f"${yval:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Save and show
    plt.savefig("stats/revenue_std_bars.png")
    print("Plot saved to stats/revenue_std_bars.png")
    plt.show()


if __name__ == "__main__":
    plot_revenue_with_std()
