"""
Backtrader 策略适配器 - 将现有策略桥接到 Backtrader
"""
import backtrader as bt
from .realtime_chart import RealtimeChartPlotter


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
            print(f'{dt.isoformat()} {txt}')

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
            print(f'{dt.isoformat()} {txt}')

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
