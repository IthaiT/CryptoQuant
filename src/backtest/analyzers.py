"""
自定义分析器 - 计算各种量化指标
"""
import backtrader as bt
import numpy as np
from collections import defaultdict
from datetime import datetime


class TradingAnalyzer(bt.Analyzer):
    """
    综合交易分析器，计算：
    - 胜率
    - 盈亏比
    - 最大回撤及持续时间
    - 交易次数、平均持仓时长
    - 月度收益率
    """
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.peak = 0
        self.max_dd = 0
        self.max_dd_duration = 0
        self.current_dd_start = None
        
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'pnl': trade.pnl,
                'pnl_pct': trade.pnlcomm / trade.value if trade.value else 0,
                'size': trade.size,
                'price_open': trade.price,
                'price_close': trade.price + (trade.pnl / trade.size if trade.size else 0),
                'bars': int(getattr(trade, 'barlen', 0) or 0),
                'datetime_open': bt.num2date(trade.dtopen),
                'datetime_close': bt.num2date(trade.dtclose),
            })
    
    def next(self):
        # 记录权益曲线
        value = self.strategy.broker.getvalue()
        self.equity_curve.append({
            'datetime': self.strategy.datetime.datetime(),
            'value': value,
        })
        
        # 计算回撤
        if value > self.peak:
            self.peak = value
            self.current_dd_start = None
        else:
            dd = (self.peak - value) / self.peak
            if dd > self.max_dd:
                self.max_dd = dd
            
            if self.current_dd_start is None:
                self.current_dd_start = self.strategy.datetime.datetime()
            else:
                dd_duration = (self.strategy.datetime.datetime() - self.current_dd_start).days
                if dd_duration > self.max_dd_duration:
                    self.max_dd_duration = dd_duration
    
    def get_analysis(self):
        # 计算胜率
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # 计算盈亏比
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
        profit_factor = avg_win / avg_loss if avg_loss != 0 else 0
        
        # 计算平均持仓时长
        avg_bars = np.mean([t['bars'] for t in self.trades]) if self.trades else 0
        
        # 计算月度收益
        monthly_returns = self._calculate_monthly_returns()
        
        # 计算总收益和年化收益
        if self.equity_curve:
            initial_value = self.equity_curve[0]['value']
            final_value = self.equity_curve[-1]['value']
            total_return = (final_value - initial_value) / initial_value
            
            # 计算年化收益率
            days = (self.equity_curve[-1]['datetime'] - self.equity_curve[0]['datetime']).days
            years = days / 365.25 if days > 0 else 1
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            total_return = 0
            annual_return = 0
        
        # 计算夏普比率（简化版，假设无风险利率为 0）
        returns = []
        for i in range(1, len(self.equity_curve)):
            r = (self.equity_curve[i]['value'] - self.equity_curve[i-1]['value']) / self.equity_curve[i-1]['value']
            returns.append(r)
        
        if returns:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_bars_per_trade': avg_bars,
            'max_drawdown': self.max_dd,
            'max_drawdown_duration': self.max_dd_duration,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'monthly_returns': monthly_returns,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
        }
    
    def _calculate_monthly_returns(self):
        """计算月度收益率"""
        if not self.equity_curve:
            return {}
        
        monthly_data = defaultdict(list)
        for point in self.equity_curve:
            key = point['datetime'].strftime('%Y-%m')
            monthly_data[key].append(point['value'])
        
        monthly_returns = {}
        for month, values in sorted(monthly_data.items()):
            if len(values) >= 2:
                ret = (values[-1] - values[0]) / values[0]
                monthly_returns[month] = ret
        
        return monthly_returns


class CryptoCommissionScheme(bt.CommInfoBase):
    """
    加密货币手续费方案
    - 支持 Maker/Taker 费率
    - 按成交金额百分比收费
    """
    params = (
        ('commission', 0.0004),  # 0.04% (币安现货)
        ('mult', 1.0),
        ('margin', False),
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission
