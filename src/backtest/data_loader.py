"""
Backtrader Data Loader for Crypto Market (Binance CSV → PandasData)
"""
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime


def load_crypto_data(csv_path: str, fromdate=None, todate=None):
    """
    读取 Binance 风格 CSV（毫秒时间戳），转换为 PandasData 供 Backtrader 使用。

    Args:
        csv_path: CSV 文件路径
        fromdate: 开始日期 (datetime)
        todate: 结束日期 (datetime)

    Returns:
        Backtrader PandasData feed
    """
    df = pd.read_csv(csv_path)

    # 将毫秒时间戳转换为 datetime 并设为索引
    if 'Open time' not in df.columns:
        raise ValueError('CSV 缺少列: Open time')

    df['datetime'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('datetime', inplace=True)

    # 重命名列以符合 Backtrader PandasData 默认映射
    rename_map = {
        'Open price': 'open',
        'High price': 'high',
        'Low price': 'low',
        'Close price': 'close',
        'Volume': 'volume',
    }
    df = df.rename(columns=rename_map)

    # 只保留所需列（必须按照Backtrader期望的顺序）
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[required_cols].astype('float64').copy()

    # 根据 fromdate/todate 进行切片
    if fromdate is not None:
        df = df[df.index >= pd.to_datetime(fromdate)]
    if todate is not None:
        df = df[df.index <= pd.to_datetime(todate)]

    # 确保索引单调递增
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    # 构造 PandasData（注意：dtformat用于时间戳解析）
    data = bt.feeds.PandasData(
        dataname=df,
        openinterest=-1  # No open interest in crypto data
    )
    return data


def load_dollar_bar_lstm_data(csv_path: str, fromdate=None, todate=None):
    """
    加载 Dollar-Bar 数据并计算 log returns，转换为 Backtrader PandasData。

    专门用于 LSTM 策略，包含以下特征：
        - ohlcv (open, high, low, close, volume)
        - ffd_close: 分数差分收盘价
        - log_return: 对数收益率 (计算: log(close / close.shift(1)))
        - dollar_volume: 美元成交量

    Args:
        csv_path: Dollar-bar CSV 文件路径
        fromdate: 开始日期 (datetime)
        todate: 结束日期 (datetime)

    Returns:
        Backtrader PandasData feed，包含所有必要特征
    """
    
    # 定义扩展的 PandasData 类，支持额外的列
    class ExtendedPandasData(bt.feeds.PandasData):
        """扩展 PandasData 以支持额外的特征列"""
        lines = ('ffd_close', 'log_return', 'dollar_volume',)
        params = (
            ('ffd_close', -1),      # 列索引（-1 表示自动检测）
            ('log_return', -1),
            ('dollar_volume', -1),
        )
    
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    df.set_index("datetime", inplace=True)

    # 计算 log returns（平稳化）
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 确保有必要的列
    required_cols = ["open", "high", "low", "close", "volume", "ffd_close", "dollar_volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 缺少必须列: {col}")

    # 保留所需列（按照 Backtrader 需要的顺序）
    df = df[["open", "high", "low", "close", "volume", "ffd_close", "log_return", "dollar_volume"]]
    df = df.astype("float64").copy()

    # 根据 fromdate/todate 进行切片
    if fromdate is not None:
        df = df[df.index >= pd.to_datetime(fromdate)]
    if todate is not None:
        df = df[df.index <= pd.to_datetime(todate)]

    # 删除 NaN 行
    df = df.dropna()

    # 确保索引单调递增
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # 构造扩展的 PandasData，显式指定列索引
    data = ExtendedPandasData(
        dataname=df,
        openinterest=-1,
        ffd_close=5,        # DataFrame 中的列索引（0-based: o,h,l,c,v,ffd_close=5）
        log_return=6,       # log_return 在索引 6
        dollar_volume=7,    # dollar_volume 在索引 7
    )
    return data
