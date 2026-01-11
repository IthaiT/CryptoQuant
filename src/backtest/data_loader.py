"""
Backtrader Data Loader for Crypto Market (Binance CSV → PandasData)
"""
import pandas as pd
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

    # 只保留所需列
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    df = df[required_cols].copy()

    # 根据 fromdate/todate 进行切片
    if fromdate is not None:
        df = df[df.index >= pd.to_datetime(fromdate)]
    if todate is not None:
        df = df[df.index <= pd.to_datetime(todate)]

    # 构造 PandasData（索引为 datetime）
    data = bt.feeds.PandasData(dataname=df)
    return data
