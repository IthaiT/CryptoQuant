# CryptoQuant - 加密货币量化框架

基于 Backtrader 的加密货币量化回测系统，支持实时 Web 图表可视化和高效数据处理。

## ✨ 核心特性

- 🚀 **TradingView 实时图表** - Web 端交互式 K 线图，支持缩放、平移、逐根播放
- 📊 **完整回测引擎** - 基于 Backtrader，内置 RSI、均线等策略
- 📈 **性能分析报告** - 自动生成收益率、夏普比率、最大回撤等指标
- 💾 **高效数据工具** - Binance 官方数据下载 + 多种 Bar 生成 (Dollar/Volume/Tick/Custom)
- ⚡ **流式压缩处理** - Zstandard 压缩，内存占用低

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 下载数据

```python
from src.data_loader import RawDataDownloader

# 下载原始交易数据
downloader = RawDataDownloader()
downloader.download_agg_trades(
    symbol="BTCUSDT",
    start_date="2026-01-20",
    end_date="2026-01-26"
)

# 或下载 K 线数据
downloader.download_klines(
    symbol="BTCUSDT",
    start_date="2026-01-20",
    end_date="2026-01-26",
    interval="1m"
)
```

### 3. 生成 Bar 数据

```python
from src.data_loader import BarGenerator

generator = BarGenerator()

# Dollar Bar (固定成交额)
df_bars = generator.generate_dollar_bars(
    symbol="BTCUSDT",
    start_date="2026-01-20",
    end_date="2026-01-26",
    threshold=500_000.0
)

# Volume Bar (固定成交量)
df_bars = generator.generate_volume_bars(
    symbol="BTCUSDT",
    start_date="2026-01-20",
    end_date="2026-01-26",
    threshold=50.0
)

# Tick Bar (固定成交笔数)
df_bars = generator.generate_tick_bars(
    symbol="BTCUSDT",
    start_date="2026-01-20",
    end_date="2026-01-26",
    threshold=1000
)
```

### 4. 运行回测

```bash
python script/run_backtest.py
```

回测启动后：
- 浏览器自动打开 `http://127.0.0.1:8765` 显示实时图表
- 终端显示回测进度和最终收益
- 按 `Ctrl+C` 退出

## 📊 图表功能

### 实时 Web 图表
- 🌙 深色主题，K 线 + 独立成交量图
- 🔍 鼠标滚轮缩放，拖拽平移
- 🎯 自动标记买入（青色↑）、卖出（紫色↓）、平仓（黄色■）
- ▶️ 可选逐根播放模式

### 静态报告（`reports/` 目录）
- 权益曲线、回撤图、月度热力图
- 收益分布、盈亏占比、持仓时长统计

## 📁 项目结构

```
CryptoQuant/
├── script/
│   ├── run_backtest.py              # 回测主程序
│   ├── get_btcusdt_data.py          # 数据下载脚本
│   └── simulation_plate.py          # 模拟盘
├── src/
│   ├── backtest/
│   │   ├── engine.py                # 回测引擎
│   │   ├── realtime_chart.py        # 实时图表服务器
│   │   ├── data_loader.py           # 数据加载
│   │   ├── analyzers.py             # 分析器
│   │   ├── visualizer.py            # 报告生成
│   │   └── strategies.py            # 策略示例
│   └── data_loader/
│       ├── raw_downloader.py        # Binance 数据下载器
│       ├── bar_generator.py         # Bar 生成器
│       └── __init__.py
├── data/                            # 数据存储目录
│   ├── raw_data/                    # 原始数据
│   └── bar_data/                    # 生成的 Bar 数据
├── docs/                            # 📚 完整文档
└── pyproject.toml                   # 项目依赖
```

## 💻 使用示例

### 快速回测

```python
from src.backtest.engine import quick_backtest
from src.strategy.RSIStrategy import RSIBacktraderStrategy

results = quick_backtest(
    csv_path='data/btc-usdt-5m.csv',
    strategy_class=RSIBacktraderStrategy,
    strategy_params={
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70
    },
    initial_cash=10000.0,
    commission=0.0004,
    output_dir='./reports',
    strategy_name='BTC_RSI'
)

print(f"收益率: {results['return_pct']:.2f}%")
```

### 自定义策略

```python
from src.strategy.StrategyBase import StrategyBase, Position
import talib

class MyStrategy(StrategyBase):
    def __init__(self, data, ma_period=20, **kwargs):
        super().__init__(data, **kwargs)
        self.ma_period = ma_period
    
    def next(self):
        if len(self.data.close) < self.ma_period:
            return None
        
        ma = talib.SMA(self.data.close, self.ma_period)
        
        if self.data.close[-1] > ma[-1]:
            return Position.LONG  # 价格在均线上方，做多
        elif self.data.close[-1] < ma[-1]:
            return Position.EXIT  # 价格在均线下方，平仓
        
        return None
```

## 📚 详细文档

完整教程请查看 [docs/](./docs/) 目录：

- 📖 [快速开始指南](./docs/01-快速开始指南.md)
- 💾 [数据下载指南](./docs/02-数据下载指南.md)
- 🔧 [回测框架手册](./docs/03-回测框架手册.md)
- 🧠 [策略开发教程](./docs/04-策略开发教程.md)
- 🚀 [进阶功能](./docs/05-进阶功能.md)

## ⚙️ 配置说明

### 修改策略参数

编辑 `script/run_backtest.py`：

```python
strategy_params = {
    'rsi_period': 14,       # RSI 周期
    'rsi_oversold': 30,     # 超卖线（买入）
    'rsi_overbought': 70,   # 超买线（卖出）
}
```

### 数据下载配置

编辑 `script/get_btcusdt_data.py`：

```python
SYMBOL = 'BTC/USDT'              # 交易对
START_DATE = '2025-06-20'         # 开始日期
END_DATE = '2025-12-31'           # 结束日期
TIMEFRAME_CONFIG = [              # 时间周期
    {'timeframe': '5m', 'label': '5分钟'},
]

# 国内用户配置代理
PROXIES = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}
```

## ❓ 常见问题

**Q: 图表无法显示？**  
A: 检查终端是否显示 "Web server is running at http://127.0.0.1:8765"，手动访问该地址。

**Q: 下载数据速度慢？**  
A: 国内用户需配置代理，编辑 `script/get_btcusdt_data.py` 中的 `PROXIES`。

**Q: 如何开发自己的策略？**  
A: 参考 [策略开发教程](./docs/04-策略开发教程.md)，继承 `StrategyBase` 类实现 `next()` 方法。

**Q: 回测结果与实盘不符？**  
A: 注意过拟合、交易成本、滑点等因素，详见 [进阶功能](./docs/05-进阶功能.md)。

## 📝 License

MIT License