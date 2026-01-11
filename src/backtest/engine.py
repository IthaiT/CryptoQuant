"""
Backtrader 回测引擎 - 主引擎封装
"""
import backtrader as bt
from datetime import datetime
from .data_loader import load_crypto_data
from .analyzers import TradingAnalyzer, CryptoCommissionScheme
from .visualizer import BacktestVisualizer
import os


class BacktestEngine:
    """
    加密货币回测引擎
    """
    
    def __init__(self, 
                 initial_cash=10000.0,
                 commission=0.0004,
                 slippage=0.0002):
        """
        Args:
            initial_cash: 初始资金 (USDT)
            commission: 手续费率 (默认 0.04%)
            slippage: 滑点 (默认 0.02%)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.cerebro = None
        self.results = None
        
    def setup(self, strategy_class, strategy_params=None):
        """
        设置回测引擎
        
        Args:
            strategy_class: Backtrader 策略类
            strategy_params: 策略参数字典
        """
        self.cerebro = bt.Cerebro()
        
        # 添加策略
        if strategy_params:
            self.cerebro.addstrategy(strategy_class, **strategy_params)
        else:
            self.cerebro.addstrategy(strategy_class)
        
        # 设置初始资金
        self.cerebro.broker.setcash(self.initial_cash)
        
        # 设置手续费
        self.cerebro.broker.addcommissioninfo(
            CryptoCommissionScheme(commission=self.commission)
        )
        
        # 添加分析器
        self.cerebro.addanalyzer(TradingAnalyzer, _name='trading')
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        
    def load_data(self, csv_path, fromdate=None, todate=None):
        """
        加载数据
        
        Args:
            csv_path: CSV 文件路径
            fromdate: 开始日期 (datetime)
            todate: 结束日期 (datetime)
        """
        if not self.cerebro:
            raise RuntimeError("请先调用 setup() 方法")
        
        data = load_crypto_data(csv_path, fromdate=fromdate, todate=todate)
        self.cerebro.adddata(data)
        
    def run(self):
        """运行回测"""
        if not self.cerebro:
            raise RuntimeError("请先调用 setup() 方法")
        
        print('=' * 60)
        print(f'初始资金: {self.initial_cash:.2f} USDT')
        print('=' * 60)
        
        # 运行回测
        self.results = self.cerebro.run()
        
        # 获取最终资金
        final_value = self.cerebro.broker.getvalue()
        print('=' * 60)
        print(f'期末资金: {final_value:.2f} USDT')
        print(f'收益: {final_value - self.initial_cash:.2f} USDT ({(final_value/self.initial_cash - 1)*100:.2f}%)')
        print('=' * 60)
        
        return self.results
    
    def get_analysis(self):
        """获取分析结果"""
        if not self.results:
            raise RuntimeError("请先运行回测")
        
        strategy = self.results[0]
        
        # 获取自定义分析器结果
        trading_analysis = strategy.analyzers.trading.get_analysis()
        
        return trading_analysis
    
    def plot(self, output_dir="./reports", strategy_name="Strategy", show=False):
        """
        生成可视化报告
        
        Args:
            output_dir: 输出目录
            strategy_name: 策略名称
            show: 是否显示图表
        """
        analysis = self.get_analysis()
        
        # 生成报告
        visualizer = BacktestVisualizer(analysis, strategy_name=strategy_name)
        visualizer.generate_full_report(output_dir=output_dir, show=show)
        
    def optimize(self, strategy_class, param_grid):
        """
        参数优化 (示例)
        
        Args:
            strategy_class: 策略类
            param_grid: 参数网格，例如 {'rsi_period': range(10, 20)}
        """
        self.cerebro = bt.Cerebro(optreturn=False)
        
        # 添加优化策略
        self.cerebro.optstrategy(strategy_class, **param_grid)
        
        # 配置 broker
        self.cerebro.broker.setcash(self.initial_cash)
        self.cerebro.broker.addcommissioninfo(
            CryptoCommissionScheme(commission=self.commission)
        )
        
        # 添加分析器
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        print('开始参数优化...')
        opt_results = self.cerebro.run()
        
        # 输出优化结果
        print('\n优化结果:')
        for result in opt_results:
            strategy = result[0]
            params = strategy.params._getkwargs()
            sharpe = strategy.analyzers.sharpe.get_analysis()
            print(f'参数: {params}, 夏普比率: {sharpe}')
        
        return opt_results


def quick_backtest(csv_path, 
                   strategy_class, 
                   strategy_params=None,
                   initial_cash=10000.0,
                   commission=0.0004,
                   output_dir="./reports",
                   strategy_name="Strategy"):
    """
    快速回测函数
    
    Args:
        csv_path: 数据文件路径
        strategy_class: 策略类
        strategy_params: 策略参数
        initial_cash: 初始资金
        commission: 手续费率
        output_dir: 报告输出目录
        strategy_name: 策略名称
    
    Returns:
        分析结果字典
    """
    # 创建引擎
    engine = BacktestEngine(
        initial_cash=initial_cash,
        commission=commission
    )
    
    # 设置策略
    engine.setup(strategy_class, strategy_params)
    
    # 加载数据
    engine.load_data(csv_path)
    
    # 运行回测
    engine.run()
    
    # 生成报告
    engine.plot(output_dir=output_dir, strategy_name=strategy_name)
    
    # 返回分析结果
    return engine.get_analysis()
