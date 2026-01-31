import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot num_bars distribution from summary CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/preprocess_data/dollar_bars/BTCUSDT/dollar_bars_summary_BTCUSDT.csv"),
        help="Path to dollar_bars_summary CSV",
    )
    parser.add_argument(
        "--compare-csv",
        type=Path,
        default=Path("reports/dynamic_threshold_num_bars.csv"),
        help="Path to dynamic threshold output CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/num_bars_distribution.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Histogram bins",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    if "num_bars" not in df.columns:
        raise ValueError("Column 'num_bars' not found in CSV")

    if "year_month" not in df.columns:
        raise ValueError("Column 'year_month' not found in CSV")

    df = df.dropna(subset=["year_month", "num_bars"]).copy()
    df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["year_month"]).sort_values("year_month")

    plt.figure(figsize=(11, 5))
    plt.plot(df["year_month"], df["num_bars"], color="#4C72B0", linewidth=1.5, label="Original")
    plt.scatter(df["year_month"], df["num_bars"], color="#4C72B0", s=12, alpha=0.7)

    if args.compare_csv.exists():
        comp = pd.read_csv(args.compare_csv)
        if "year_month" in comp.columns and "expected_num_bars" in comp.columns:
            comp = comp.dropna(subset=["year_month", "expected_num_bars"]).copy()
            comp["year_month"] = pd.to_datetime(
                comp["year_month"], format="%Y-%m", errors="coerce"
            )
            comp = comp.dropna(subset=["year_month"]).sort_values("year_month")
            plt.plot(
                comp["year_month"],
                comp["expected_num_bars"],
                color="#DD8452",
                linewidth=1.5,
                label="Dynamic Threshold",
            )
            plt.scatter(
                comp["year_month"],
                comp["expected_num_bars"],
                color="#DD8452",
                s=12,
                alpha=0.7,
            )

    plt.title("Monthly num_bars comparison")
    plt.xlabel("Time")
    plt.ylabel("num_bars")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
