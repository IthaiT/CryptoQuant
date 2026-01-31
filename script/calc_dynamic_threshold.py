import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    # 解析命令行参数，允许用户自定义输入/输出路径与参数
    parser = argparse.ArgumentParser(
        description="Calculate dynamic dollar-bar threshold and expected num_bars"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/preprocess_data/dollar_bars/BTCUSDT/dollar_bars_summary_BTCUSDT.csv"),
        help="Path to dollar_bars_summary CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/dynamic_threshold_num_bars.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--target-bars",
        type=float,
        default=6000.0,
        help="Target monthly bars (used to compute base threshold)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=12,
        help="Months window for EMA",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.6,
        help="EMA smoothing factor",
    )
    return parser.parse_args()


def ema(values: list[float], alpha: float) -> float | None:
    # 计算一组数值的指数移动平均（EMA）
    # alpha = k，取值 (0, 1]，越大越重视近期
    if not values:
        return None
    # 初始值使用序列首值
    ema_value = values[0]
    for v in values[1:]:
        # EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
        ema_value = alpha * v + (1.0 - alpha) * ema_value
    return ema_value


def main() -> None:
    args = parse_args()
    # 读取汇总数据（每月一行）
    df = pd.read_csv(args.csv)

    # 检查必要字段是否齐全
    required = {"year_month", "num_bars", "total_dollar_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    # 处理缺失值与时间字段，按月份升序排列
    df = df.dropna(subset=["year_month", "num_bars", "total_dollar_volume"]).copy()
    df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["year_month"]).sort_values("year_month").reset_index(drop=True)

    # base_thresholds[i]：用“上个月成交额/目标bar数”得到的阈值
    # 第一个月没有上月数据，因此为 None
    base_thresholds: list[float | None] = [None] * len(df)
    for i in range(1, len(df)):
        base_thresholds[i] = df.loc[i - 1, "total_dollar_volume"] / args.target_bars

    # ema_thresholds[i]：对过去 window 个月的 base_thresholds 做 EMA 后得到的阈值
    # expected_num_bars[i]：使用 ema_thresholds[i] 推算的当月 bar 数
    ema_thresholds: list[float | None] = [None] * len(df)
    expected_num_bars: list[float | None] = [None] * len(df)

    for i in range(len(df)):
        # 取过去 window 个月（不含当月）的 base_thresholds 做 EMA
        start = max(0, i - args.window)
        window_vals = [v for v in base_thresholds[start:i] if v is not None]
        threshold = ema(window_vals, args.k)
        ema_thresholds[i] = threshold
        if threshold:
            # 用当月成交额除以阈值，得到预测 bar 数量
            expected_num_bars[i] = df.loc[i, "total_dollar_volume"] / threshold

    # 汇总输出字段，方便与原始 num_bars 对比
    out_df = pd.DataFrame(
        {
            "year_month": df["year_month"].dt.strftime("%Y-%m"),
            "original_num_bars": df["num_bars"],
            "total_dollar_volume": df["total_dollar_volume"],
            "base_threshold_prev_month": base_thresholds,
            "ema_threshold": ema_thresholds,
            "expected_num_bars": expected_num_bars,
        }
    )

    # 写出 CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
