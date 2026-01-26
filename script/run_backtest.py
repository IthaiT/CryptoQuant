"""
BTC/USDT RSI 策略回测示例
使用 Backtrader 框架
"""
import os
import sys
from pathlib import Path

# 添加项目路径（项目根目录：CryptoQuant）
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent  # script/ 的上一级就是项目根目录
sys.path.insert(0, str(project_root))

# 设置 matplotlib 后端以支持实时图表
import matplotlib
matplotlib.use('TkAgg')

from src.backtest import quick_backtest, RSIBacktraderStrategy


def main():
    """主函数"""
    
    # 数据文件路径 (使用下载的 5 分钟数据)
    # 同时兼容 Data/ 与 data/ 两种目录名
    candidates = [
        project_root / "Data" / "btc-usdt-5m.csv",
        project_root / "data" / "btc-usdt-5m.csv",
    ]
    data_path = None
    for p in candidates:
        if p.exists():
            data_path = str(p)
            break
    
    # 检查数据文件是否存在；如不存在尝试生成演示数据
    if not data_path:
        print("❌ 未找到数据文件: 期望位置为以下之一：")
        for p in candidates:
            print(f" - {p}")

        print("请先运行 Data/BTCUSDT_data_download.py 下载真实数据")
        return
    
    print("=" * 60)
    print("BTC/USDT RSI 策略回测")
    print("=" * 60)
    print(f"数据文件: {data_path}")
    print(f"时间周期: 5 分钟")
    print("=" * 60)
    
    # 策略参数
    strategy_params = {
        'rsi_period': 14,       # RSI 周期
        'rsi_oversold': 30,     # 超卖线 (买入)
        'rsi_overbought': 70,   # 超买线 (卖出)
        'printlog': False,      # 是否打印日志
    }
    
    # 运行回测
    results = quick_backtest(
        csv_path=data_path,
        strategy_class=RSIBacktraderStrategy,
        strategy_params=strategy_params,
        initial_cash=10000.0,      # 初始资金 10000 USDT
        commission=0.0004,          # 手续费 0.04% (币安现货)
        output_dir=str(project_root / "reports"),  # 报告输出目录
        strategy_name="BTC_RSI_5m"  # 策略名称
    )
    
    # 打印关键指标
    print("\n" + "=" * 60)
    print("回测完成！报告已生成在 ./reports 目录")
    print("=" * 60)
    print("\n浏览器打开地址: http://127.0.0.1:8765")
    print("按 Ctrl+C 退出")
    print("=" * 60)
    
    # 保持服务器运行，让用户能够查看图表
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n服务器已关闭")
        import sys
        sys.exit(0)
    

if __name__ == "__main__":
    main()
