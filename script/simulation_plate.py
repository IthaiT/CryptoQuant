import time
import os
import pandas as pd
from binance.client import Client
from binance.enums import *
from src.strategies.RSIStrategy import RSIStrategy
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Binance Testnet 配置
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# 验证配置
if not API_KEY or not API_SECRET:
    raise ValueError("请在 .env 文件中配置 BINANCE_API_KEY 和 BINANCE_API_SECRET")

# Binance Client
client = Client(API_KEY, API_SECRET, testnet=True)

# 获取 Binance 的历史数据
def get_binance_data(symbol, interval, limit=1000):
    klines = client.get_historical_klines(symbol, interval, limit=limit)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data = data[['close']].astype(float)
    return data

# 模拟交易执行
def execute_trade(symbol, signal, position_size=0.001):
    balance = client.get_asset_balance(asset='USDT')
    usdt_balance = float(balance['free'])
    
    # 当前市场价格
    price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    
    if signal == 1 and usdt_balance > 10:  # Long position (买入)
        quantity = usdt_balance * 0.1 / price  # 这里 0.1 是持仓量比例
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        print(f"Buy Order Executed: {order}")
        
    elif signal == -1 and usdt_balance > 10:  # Short position (卖空)
        quantity = usdt_balance * 0.1 / price
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        print(f"Sell Order Executed: {order}")
    
    elif signal == 0:  # Exit position (平仓)
        # 这里假设已经有仓位，不做平空处理，可以根据实际情况调整
        print("Exit Position")

# 主函数
def main():
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_1MINUTE  # 1分钟K线
    strategy = RSIStrategy(window_size=14, enter_long=70, exit_long=30, enter_short=30, exit_short=70)
    
    while True:
        # 获取最新数据
        data = get_binance_data(symbol, interval)
        
        # 运行策略
        positions = strategy.run(data)
        
        # 获取最后一个信号
        signal = positions[-1]
        
        # 执行交易
        execute_trade(symbol, signal)
        
        # 暂停 1 分钟后继续
        time.sleep(60)

if __name__ == "__main__":
    main()
