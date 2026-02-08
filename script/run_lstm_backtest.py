"""
LSTM 策略回测脚本

使用训练好的 LSTM 模型进行比特币交易回测。

用法:
    python script/run_lstm_backtest.py                              # 使用默认参数
    python script/run_lstm_backtest.py --buy_threshold 0.6          # 自定义买入阈值
    python script/run_lstm_backtest.py --model_dir models/my_model  # 指定模型目录
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.backtest import (
    BacktestEngine,
    LSTMBacktraderStrategy,
    LSTMModelLoader,
    LSTMPredictor,
    load_dollar_bar_lstm_data,
)
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocess_data"
    / "factor"
    / "BTCUSDT"
    / "BTCUSDT_2025-01-01_2025-12-31_dollar_bars_4m_labeled.csv"
)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = ["ffd_close", "log_return", "volume", "dollar_volume"]
LOOKBACK = 60


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用 LSTM 模型进行 BTC/USDT 回测"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Dollar-bar CSV 数据路径",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="模型检查点目录",
    )
    parser.add_argument(
        "--initial_cash",
        type=float,
        default=10000.0,
        help="初始资金（USDT）",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0004,
        help="手续费率（币安现货默认 0.04%）",
    )
    parser.add_argument(
        "--buy_threshold",
        type=float,
        default=0.55,
        help="买入信号阈值（Up 概率）",
    )
    parser.add_argument(
        "--sell_threshold",
        type=float,
        default=0.45,
        help="卖出信号阈值（Up 概率）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "reports"),
        help="报告输出目录",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 数据加载与预处理
# ---------------------------------------------------------------------------

def load_and_prepare_scaler(csv_path: str | Path) -> MinMaxScaler:
    """
    加载数据并拟合 MinMaxScaler（用于特征归一化）。

    必须与训练脚本使用相同的方式拟合，确保推理时的特征范围一致。

    Args:
        csv_path: Dollar-bar CSV 文件路径

    Returns:
        已在训练数据上拟合的 MinMaxScaler
    """
    logger.info(f"加载数据用于 Scaler 拟合: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["datetime"])

    # 删除 NaN 标签和标签 == 0
    before = len(df)
    df = df.dropna(subset=["label"])
    df = df[df["label"] != 0].copy()
    logger.info(f"删除了 {before - len(df)} 行 NaN/label==0")

    # 计算 log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 删除 NaN 特征
    required_cols = FEATURE_COLUMNS + ["label"]
    before = len(df)
    df = df.dropna(subset=required_cols)
    logger.info(f"删除了 {before - len(df)} 行 NaN 特征")

    # 用于拟合的特征（跟训练脚本一致）
    features = df[FEATURE_COLUMNS].values

    # 拟合 Scaler（在所有数据上）
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(features)
    logger.info("✅ MinMaxScaler 已拟合")

    return scaler


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    """主入口函数"""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("LSTM 策略回测")
    logger.info("=" * 70)

    # ---- 1. 加载 LSTM 模型 ----
    logger.info("\n[1/4] 加载 LSTM 模型...")
    try:
        model_loader = LSTMModelLoader(model_dir=args.model_dir)
        model = model_loader.load(input_size=len(FEATURE_COLUMNS))
    except FileNotFoundError as e:
        logger.error(f"❌ 模型加载失败: {e}")
        sys.exit(1)

    # ---- 2. 准备 Scaler 和 Predictor ----
    logger.info("\n[2/4] 准备特征归一化器...")
    scaler = load_and_prepare_scaler(args.data)

    logger.info("\n[3/4] 初始化 LSTM 预测器...")
    predictor = LSTMPredictor(model, lookback=LOOKBACK, feature_names=FEATURE_COLUMNS)
    predictor.set_scaler(scaler)

    # ---- 3. 设置回测引擎 ----
    logger.info("\n[4/4] 设置回测引擎...")
    engine = BacktestEngine(
        initial_cash=args.initial_cash,
        commission=args.commission,
        enable_realtime_chart=False,  # 关闭实时图表，加速回测
    )

    # 策略参数
    strategy_params = {
        "lstm_predictor": predictor,
        "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold,
        "printlog": False,
    }

    engine.setup(
        strategy_class=LSTMBacktraderStrategy,
        strategy_params=strategy_params,
    )

    # 加载数据
    data = load_dollar_bar_lstm_data(csv_path=args.data)
    engine.cerebro.adddata(data)

    # 运行回测
    logger.info("\n" + "=" * 70)
    logger.info("开始回测...")
    logger.info("=" * 70 + "\n")

    results = engine.run()

    # 获取分析结果
    analysis = engine.get_analysis()

    logger.info("\n" + "=" * 70)
    logger.info("回测完成！")
    logger.info("=" * 70)
    logger.info(f"\n📊 回测统计:")
    logger.info(f"  初始资金: {args.initial_cash:.2f} USDT")
    logger.info(f"  期末资金: {engine.cerebro.broker.getvalue():.2f} USDT")
    logger.info(f"  收益: {engine.cerebro.broker.getvalue() - args.initial_cash:.2f} USDT")
    logger.info(f"  收益率: {(engine.cerebro.broker.getvalue() / args.initial_cash - 1) * 100:.2f}%")
    logger.info(f"\n📈 策略参数:")
    logger.info(f"  买入阈值: {args.buy_threshold}")
    logger.info(f"  卖出阈值: {args.sell_threshold}")
    logger.info(f"  Lookback 窗口: {LOOKBACK}")
    logger.info(f"\n📁 输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
