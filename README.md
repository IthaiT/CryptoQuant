# 环境安装
```
pip install -e .
```

# 项目结构
```
CryptoQuant/
├── data/                        # 数据模块
│   └── BTCUSDT_data_download.py # 数据下载脚本
├── executor/                    # 执行器模块
│   └── SimulationPlate.py       # 模拟盘交易主程序
├── src/                         # 源码目录
│   ├── strategy/                # 策略模块
│   │   ├── __init__.py
│   │   ├── StrategyBase.py      # 策略基类与持仓枚举
│   │   └── RSIStrategy.py       # RSI 策略实现
│   └── backtest/                # 回测模块
├── .env.example                 # 环境变量配置(需自行修改)
├── .gitignore   
├── pyproject.toml               # 项目配置与依赖
└── README.md
```