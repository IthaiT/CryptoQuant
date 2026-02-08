"""
BTC/USDT 策略回测 - 支持 RSI 和 LSTM 两种策略

使用方法:
    python script/run_backtest.py                    # 默认 RSI 策略
    python script/run_backtest.py --strategy lstm    # LSTM 策略
    python script/run_backtest.py --strategy lstm --buy_threshold 0.6   # LSTM 自定义参数
"""
from pathlib import Path

# 添加项目路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

# 设置 matplotlib 后端
import matplotlib
matplotlib.use('TkAgg')

import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

from src.backtest import (
    quick_backtest,
    RSIBacktraderStrategy,
    LSTMBacktraderStrategy,
    LSTMModelLoader,
    LSTMPredictor,
    load_dollar_bar_lstm_data,
)
from src.backtest.engine import BacktestEngine



def main():
    """主函数 - 支持 RSI 和 LSTM 策略选择"""
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="BTC/USDT 策略回测")
    parser.add_argument("--strategy", type=str, default="rsi", 
                       choices=["rsi", "lstm"],
                       help="选择策略: rsi (默认) 或 lstm")
    parser.add_argument("--model_dir", type=str, 
                       default=str(project_root / "models"),
                       help="LSTM 模型目录（strategy=lstm 时需要）")
    parser.add_argument("--buy_threshold", type=float, default=0.55,
                       help="LSTM 策略买入阈值（概率）")
    parser.add_argument("--sell_threshold", type=float, default=0.45,
                       help="LSTM 策略卖出阈值（概率）")
    parser.add_argument("--initial_cash", type=float, default=10000.0,
                       help="初始资金 (USDT)")
    
    args = parser.parse_args()
    
    # ========== RSI 策略 ==========
    if args.strategy == "rsi":
        candidates = [
            project_root / "Data" / "btc-usdt-5m.csv",
            project_root / "data" / "btc-usdt-5m.csv",
        ]
        data_path = None
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        
        if not data_path:
            print("❌ 未找到数据文件: 期望位置为以下之一：")
            for p in candidates:
                print(f" - {p}")
            print("请先运行 Data/BTCUSDT_data_download.py 下载真实数据")
            return
        
        print("=" * 70)
        print("BTC/USDT RSI 策略回测")
        print("=" * 70)
        print(f"📊 数据文件: {data_path}")
        print(f"⏱️  时间周期: 5 分钟")
        print(f"💰 初始资金: {args.initial_cash:.2f} USDT")
        print("=" * 70)
        
        strategy_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'printlog': False,
        }
        
        results = quick_backtest(
            csv_path=data_path,
            strategy_class=RSIBacktraderStrategy,
            strategy_params=strategy_params,
            initial_cash=args.initial_cash,
            commission=0.0004,
            output_dir=str(project_root / "reports"),
            strategy_name="BTC_RSI_5m"
        )
        
        print("\n" + "=" * 70)
        print("回测完成！报告已生成在 ./reports 目录")
        print("=" * 70)
        print("\n🌐 浏览器打开地址: http://127.0.0.1:8765")
        print("⏹️  按 Ctrl+C 退出")
        print("=" * 70)
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n✅ 服务器已关闭")
            import sys
            sys.exit(0)
    
    # ========== LSTM 策略 ==========
    elif args.strategy == "lstm":
        print("=" * 70)
        print("BTC/USDT LSTM 神经网络策略回测")
        print("=" * 70)
        
        # LSTM 数据路径
        data_path = str(project_root / "data" / "preprocess_data" / "factor" / "BTCUSDT" / 
                       "BTCUSDT_2025-01-01_2025-12-31_dollar_bars_4m_labeled.csv")
        
        if not Path(data_path).exists():
            print(f"❌ 未找到 LSTM 数据文件: {data_path}")
            return
        
        print(f"📊 数据文件: {data_path}")
        print(f"📈 数据类型: Dollar-Bar")
        print(f"⏱️  时间周期: 4-5 分钟")
        print(f"💰 初始资金: {args.initial_cash:.2f} USDT")
        print(f"🎯 买入阈值: {args.buy_threshold}")
        print(f"🎯 卖出阈值: {args.sell_threshold}")
        print("=" * 70)
        
        # 加载 LSTM 模型
        print("\n[1/5] 加载 LSTM 模型...")
        try:
            model_loader = LSTMModelLoader(model_dir=args.model_dir)
            model = model_loader.load(input_size=4)
        except FileNotFoundError as e:
            print(f"❌ 模型加载失败: {e}")
            print("请先运行: python script/train_lstm.py 进行模型训练")
            return
        
        # 准备 Scaler
        print("[2/5] 准备特征归一化器...")
        df = pd.read_csv(data_path, parse_dates=["datetime"])
        df = df.dropna(subset=["label"])
        df = df[df["label"] != 0].copy()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        feature_cols = ["ffd_close", "log_return", "volume", "dollar_volume"]
        df = df.dropna(subset=feature_cols + ["label"])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[feature_cols].values)
        print("✅ Scaler 已准备完成")
        
        # 初始化预测器
        print("[3/5] 初始化 LSTM 预测器...")
        predictor = LSTMPredictor(model, lookback=60, feature_names=feature_cols)
        predictor.set_scaler(scaler)
        print("✅ LSTM 预测器已初始化")
        
        # 策略参数
        strategy_params = {
            'lstm_predictor': predictor,
            'buy_threshold': args.buy_threshold,
            'sell_threshold': args.sell_threshold,
            'printlog': False,
        }
        
        # 设置回测引擎
        print("[4/5] 设置回测引擎...")
        engine = BacktestEngine(
            initial_cash=args.initial_cash,
            commission=0.0004,
            enable_realtime_chart=True
        )
        
        engine.setup(strategy_class=LSTMBacktraderStrategy, strategy_params=strategy_params)
        
        lstm_data = load_dollar_bar_lstm_data(csv_path=data_path)
        engine.cerebro.adddata(lstm_data)
        
        # 动态设置图表最大显示bars（根据数据量）
        if engine.plotter:
            df_count = len(df)  # df 已经在前面加载过
            max_bars = int(df_count * 1.2)  # 数据量 + 20% 缓冲
            engine.plotter.set_max_bars(max_bars)
            logger.info(f'数据行数: {df_count}, 设置图表最大显示: {max_bars}')
        
        print("✅ 回测引擎已配置，数据已加载")
        
        # 运行回测
        print("[5/5] 运行回测...")
        print("=" * 70 + "\n")
        results = engine.run()
        print("\n" + "=" * 70)
        print("✅ 回测完成！")
        print("=" * 70)
        print("\n🌐 浏览器打开地址: http://127.0.0.1:8765")
        print("⏹️  按 Ctrl+C 退出")
        print("=" * 70)
        
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n✅ 服务器已关闭")
            import sys
            sys.exit(0)
    

if __name__ == "__main__":
    main()
