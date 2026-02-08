"""
Backtrader 策略适配器 - 将现有策略桥接到 Backtrader
"""
import backtrader as bt
import numpy as np
from .realtime_chart import RealtimeChartPlotter
from .lstm_backtest_helper import LSTMPredictor
from src.utils.logger import logger


class RSIBacktraderStrategy(bt.Strategy):
    """
    基于 RSI 的 Backtrader 策略
    """
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),   # 超卖阈值，买入信号
        ('rsi_overbought', 70), # 超买阈值，卖出信号
        ('printlog', False),
        ('plotter', None),      # RealtimeChartPlotter instance
    )

    def __init__(self):
        # 计算 RSI 指标
        self.rsi = bt.indicators.RSI(
            self.data.close,
            period=self.params.rsi_period
        )
        
        # 记录订单
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        
    def log(self, txt, dt=None):
        """日志输出"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            logger.info(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'买入执行, 价格: {order.executed.price:.2f}, '
                    f'成本: {order.executed.value:.2f}, '
                    f'手续费: {order.executed.comm:.2f}'
                )
                # 记录到图表
                if self.params.plotter:
                    self.params.plotter.add_buy_signal(order.executed.price)
                self.buy_price = order.executed.price
                self.buy_comm = order.executed.comm
            else:
                self.log(
                    f'卖出执行, 价格: {order.executed.price:.2f}, '
                    f'成本: {order.executed.value:.2f}, '
                    f'手续费: {order.executed.comm:.2f}'
                )
                # 记录到图表
                if self.params.plotter:
                    self.params.plotter.add_sell_signal(order.executed.price)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/拒绝')

        self.order = None

    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return

        self.log(f'交易盈亏, 毛利: {trade.pnl:.2f}, 净利: {trade.pnlcomm:.2f}')

    def next(self):
        """策略逻辑"""
        # 更新图表数据
        if self.params.plotter:
            self.params.plotter.add_bar(
                self.data.datetime.datetime(0),
                self.data.open[0],
                self.data.high[0],
                self.data.low[0],
                self.data.close[0],
                self.data.volume[0]
            )
            # 每10根K线更新一次图表（避免闪烁）
            self.params.plotter.update_chart()
        
        # 检查是否有待处理订单
        if self.order:
            return

        # 当前持仓情况
        if not self.position:
            # 无持仓，检查买入信号
            if self.rsi[0] < self.params.rsi_oversold:
                # RSI 超卖，买入
                self.log(f'买入信号, RSI: {self.rsi[0]:.2f}')
                # 使用全部可用资金买入
                cash = self.broker.getcash()
                size = cash * 0.95 / self.data.close[0]  # 保留 5% 现金
                self.order = self.buy(size=size)
        else:
            # 有持仓，检查卖出信号
            if self.rsi[0] > self.params.rsi_overbought:
                # RSI 超买，卖出
                self.log(f'卖出信号, RSI: {self.rsi[0]:.2f}')
                self.order = self.sell(size=self.position.size)

    def stop(self):
        """回测结束"""
        self.log(
            f'(RSI周期 {self.params.rsi_period}) '
            f'期末资金: {self.broker.getvalue():.2f}',
            dt=self.datas[0].datetime.date(0)
        )


class MABacktraderStrategy(bt.Strategy):
    """
    双均线策略 (示例)
    """
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('printlog', False),
    )

    def __init__(self):
        # 快速均线
        self.fast_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.fast_period
        )
        # 慢速均线
        self.slow_ma = bt.indicators.SMA(
            self.data.close,
            period=self.params.slow_period
        )
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        self.order = None
        
    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            logger.info(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}')
            else:
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}')

        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # 金叉 - 买入
            if self.crossover > 0:
                cash = self.broker.getcash()
                size = cash * 0.95 / self.data.close[0]
                self.order = self.buy(size=size)
        else:
            # 死叉 - 卖出
            if self.crossover < 0:
                self.order = self.sell(size=self.position.size)

class LSTMBacktraderStrategy(bt.Strategy):
    """
    基于 LSTM 神经网络的 Backtrader 策略

    该策略使用训练好的 LSTM 模型来预测比特币价格上升（Up）或下降（Down）的概率。
    - 当预测概率 > 阈值 → 买入信号
    - 当预测概率 < 阈值 → 卖出信号

    模型输入：滑动窗口（60 步）特征
        - ffd_close: 分数差分收盘价（平稳化）
        - log_return: 对数收益率（平稳化）
        - volume: 交易量
        - dollar_volume: 美元成交量

    Args:
        lstm_predictor: 初始化后的 LSTMPredictor 实例
        buy_threshold: 买入信号阈值（默认 0.55）
        sell_threshold: 卖出信号阈值（默认 0.45）
        printlog: 是否打印日志
        plotter: RealtimeChartPlotter 实例
    """

    params = (
        ('lstm_predictor', None),      # 必须由外部设置
        ('buy_threshold', 0.55),       # 买入概率阈值
        ('sell_threshold', 0.45),      # 卖出概率阈值
        ('printlog', False),
        ('plotter', None),             # RealtimeChartPlotter instance
    )

    def __init__(self):
        self.predictor: LSTMPredictor = self.params.lstm_predictor
        if self.predictor is None:
            raise ValueError("LSTM predictor 未初始化！请提供 lstm_predictor 参数")

        self.order = None
        self.bar_count = 0
        self.signal_log = []  # 记录所有预测信号

    def log(self, txt, dt=None):
        """日志输出"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            logger.info(f"{dt.isoformat()} {txt}")

    def notify_order(self, order):
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"买入执行, 价格: {order.executed.price:.2f}, "
                    f"成本: {order.executed.value:.2f}, "
                    f"手续费: {order.executed.comm:.2f}"
                )
                if self.params.plotter:
                    self.params.plotter.add_buy_signal(order.executed.price)
            else:
                self.log(
                    f"卖出执行, 价格: {order.executed.price:.2f}, "
                    f"成本: {order.executed.value:.2f}, "
                    f"手续费: {order.executed.comm:.2f}"
                )
                if self.params.plotter:
                    self.params.plotter.add_sell_signal(order.executed.price)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("订单取消/保证金不足/拒绝")

        self.order = None

    def notify_trade(self, trade):
        """交易完成通知"""
        if not trade.isclosed:
            return

        self.log(f"交易盈亏, 毛利: {trade.pnl:.2f}, 净利: {trade.pnlcomm:.2f}")

    def next(self):
        """策略逻辑 - 调用 LSTM 模型进行预测"""
        self.bar_count += 1

        # 更新图表数据
        if self.params.plotter:
            self.params.plotter.add_bar(
                self.data.datetime.datetime(0),
                self.data.open[0],
                self.data.high[0],
                self.data.low[0],
                self.data.close[0],
                self.data.volume[0],
            )
            self.params.plotter.update_chart()

        # 检查是否有待处理订单
        if self.order:
            return

        # 获取当前 bar 的特征（从扩展的 data lines 中获取）
        try:
            ffd_close = self.data.ffd_close[0]
            log_return = self.data.log_return[0]
            volume = self.data.volume[0]
            dollar_volume = self.data.dollar_volume[0]
        except (AttributeError, IndexError) as e:
            # 如果字段不可用，跳过这个 bar
            return

        # 检查 NaN 值
        if np.isnan(ffd_close) or np.isnan(log_return):
            return

        # 更新 LSTM 特征缓冲
        self.predictor.update_features(
            ffd_close=float(ffd_close),
            log_return=float(log_return),
            volume=float(volume),
            dollar_volume=float(dollar_volume),
        )

        # 获取预测概率
        prob_up = self.predictor.predict()
        self.signal_log.append((self.data.datetime.date(0), prob_up))

        # 生成交易信号
        if not self.position:
            # 无持仓：概率足够高 → 买入
            if prob_up > self.params.buy_threshold:
                self.log(f"🔵 买入信号, LSTM 预测 Up 概率: {prob_up:.4f}")
                cash = self.broker.getcash()
                size = cash * 0.95 / self.data.close[0]
                self.order = self.buy(size=size)
        else:
            # 有持仓：概率太低 → 卖出
            if prob_up < self.params.sell_threshold:
                self.log(f"🔴 卖出信号, LSTM 预测 Up 概率: {prob_up:.4f}")
                self.order = self.sell(size=self.position.size)

    def stop(self):
        """回测结束"""
        final_value = self.broker.getvalue()
        self.log(f"期末资金: {final_value:.2f}")
        logger.info(f"\n📊 LSTM 策略完成回测: 总 bars={self.bar_count}, 信号数={len(self.signal_log)}")