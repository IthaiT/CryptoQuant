"""
可视化模块 - 生成量化回测报表和图表
"""
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免与Qt冲突
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib import font_manager as fm
import os
import platform
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import os
from src.utils.logger import logger


class BacktestVisualizer:
    """回测结果可视化"""
    
    def __init__(self, analysis_result, strategy_name="Strategy"):
        self.analysis = analysis_result
        self.strategy_name = strategy_name
        
        # 配置字体，确保中文可显示
        self._configure_font()
        sns.set_style("whitegrid")

    def _configure_font(self):
        """在 Windows/Mac/Linux 上优先选择可显示中文的字体"""
        # 允许通过环境变量指定字体文件路径
        env_font = os.environ.get('MATPLOTLIB_CJK_FONT')
        font_name = None

        def use_font_by_path(path: str):
            nonlocal font_name
            if path and os.path.exists(path):
                fm.fontManager.addfont(path)
                try:
                    font_name = fm.FontProperties(fname=path).get_name()
                except Exception:
                    font_name = None
                return True
            return False

        # 优先使用环境变量指定的字体
        if env_font:
            use_font_by_path(env_font)

        # 其次尝试系统已注册的字体族
        if not font_name:
            candidates = [
                # Windows 常见
                'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi',
                # macOS 常见
                'PingFang SC', 'Songti SC',
                # 通用 CJK 字体
                'Noto Sans CJK SC', 'Source Han Sans SC', 'Arial Unicode MS'
            ]
            available = {f.name for f in fm.fontManager.ttflist}
            for name in candidates:
                if name in available:
                    font_name = name
                    break

        # 若仍未找到，尝试典型系统路径加载字体文件
        if not font_name:
            sysname = platform.system()
            win_paths = [
                r"C:\Windows\Fonts\msyh.ttc",   # 微软雅黑
                r"C:\Windows\Fonts\msyh.ttf",
                r"C:\Windows\Fonts\simhei.ttf", # 黑体
                r"C:\Windows\Fonts\simsun.ttc", # 宋体
            ]
            mac_paths = [
                "/System/Library/Fonts/PingFang.ttc",
                "/System/Library/Fonts/STHeiti Light.ttc",
                "/Library/Fonts/Songti.ttc",
            ]
            linux_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
            ]

            paths = win_paths if sysname == 'Windows' else mac_paths if sysname == 'Darwin' else linux_paths
            for p in paths:
                if use_font_by_path(p):
                    break

        # 最后尝试项目本地字体目录 assets/fonts
        if not font_name:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fonts_dir = os.path.join(repo_root, 'assets', 'fonts')
            candidates_files = [
                'msyh.ttc', 'msyh.ttf', 'simhei.ttf', 'simsun.ttc',
                'NotoSansCJK-Regular.ttc', 'NotoSansSC-Regular.otf', 'SourceHanSansSC-Regular.otf'
            ]
            for fname in candidates_files:
                p = os.path.join(fonts_dir, fname)
                if use_font_by_path(p):
                    break

        # 最终应用字体设置（若仍未找到，则继续使用 Arial，但会缺字形）
        plt.rcParams['font.family'] = 'sans-serif'
        # Use DejaVu Sans as safe default for English labels
        plt.rcParams['font.sans-serif'] = [font_name or 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def generate_full_report(self, output_dir="./reports", show=False):
        """Generate full backtest report (English labels)"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建主报告图
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 权益曲线和回撤
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1)
        
        # 2. 月度收益热力图
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_monthly_returns_heatmap(ax2)
        
        # 3. 关键指标卡片
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_metrics_card(ax3)
        
        # 4. 收益分布
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax4)
        
        # 5. 交易分析
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_trade_analysis(ax5)
        
        # 6. 持仓时长分布
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_holding_period(ax6)
        
        # 7. 盈亏分布
        ax7 = fig.add_subplot(gs[3, 0])
        self._plot_pnl_distribution(ax7)
        
        # 8. 累计收益对比
        ax8 = fig.add_subplot(gs[3, 1:])
        self._plot_cumulative_returns(ax8)
        
        plt.suptitle(f'{self.strategy_name} - Backtest Report', fontsize=18, fontweight='bold', y=0.995)
        
        # 保存
        output_path = os.path.join(output_dir, f'{self.strategy_name}_report.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"📊 Report saved: {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        # 生成文本报告
        self._generate_text_report(output_dir)
    
    def _plot_equity_curve(self, ax):
        """Plot equity curve and drawdown"""
        if not self.analysis['equity_curve']:
            return
        
        df = pd.DataFrame(self.analysis['equity_curve'])
        dates = df['datetime']
        values = df['value']
        
        # 计算回撤
        peak = values.expanding().max()
        drawdown = (values - peak) / peak * 100
        
        # 双坐标轴
        ax2 = ax.twinx()
        
        # 权益曲线
        ax.plot(dates, values, color='#2E86AB', linewidth=2, label='Equity')
        ax.fill_between(dates, values, alpha=0.3, color='#2E86AB')
        ax.set_ylabel('Equity (USDT)', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(dates, drawdown, 0, color='#E63946', alpha=0.4, label='Drawdown')
        ax2.plot(dates, drawdown, color='#E63946', linewidth=1.5, alpha=0.8)
        ax2.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right')
        
        ax.set_title('Equity Curve & Drawdown', fontsize=13, fontweight='bold', pad=15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_monthly_returns_heatmap(self, ax):
        """Monthly returns heatmap"""
        monthly = self.analysis.get('monthly_returns', {})
        if not monthly:
            ax.text(0.5, 0.5, 'No monthly data', ha='center', va='center', fontsize=12)
            ax.axis('off')
            return
        
        # 转换为矩阵
        data = []
        for month_str, ret in sorted(monthly.items()):
            year, month = month_str.split('-')
            data.append({'year': int(year), 'month': int(month), 'return': ret * 100})
        
        df = pd.DataFrame(data)
        pivot = df.pivot_table(values='return', index='year', columns='month', aggfunc='sum')
        
        # 绘制热力图
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
               cbar_kws={'label': 'Return (%)'}, ax=ax, linewidths=0.5)
        ax.set_title('Monthly Returns Heatmap', fontsize=13, fontweight='bold', pad=15)
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel('Year', fontsize=11)
    
    def _plot_metrics_card(self, ax):
        """Key metrics card"""
        ax.axis('off')
        
        metrics = [
            ('Total Return', f"{self.analysis['total_return']*100:.2f}%"),
            ('Annual Return', f"{self.analysis['annual_return']*100:.2f}%"),
            ('Sharpe Ratio', f"{self.analysis['sharpe_ratio']:.2f}"),
            ('Max Drawdown', f"{self.analysis['max_drawdown']*100:.2f}%"),
            ('Win Rate', f"{self.analysis['win_rate']*100:.2f}%"),
            ('Profit Factor', f"{self.analysis['profit_factor']:.2f}"),
            ('Trades', f"{self.analysis['total_trades']}"),
            ('Avg Holding', f"{self.analysis['avg_bars_per_trade']:.1f} bars"),
        ]
        
        y_start = 0.95
        for i, (label, value) in enumerate(metrics):
            y = y_start - i * 0.11
            ax.text(0.05, y, label, fontsize=11, fontweight='bold', va='top')
            ax.text(0.95, y, value, fontsize=11, ha='right', va='top',
                   color='green' if i in [0, 1, 2, 4, 5] and float(value.replace('%', '').replace('次', '').replace(' K线', '')) > 0 else 'black')
        
        ax.set_title('Key Metrics', fontsize=13, fontweight='bold', pad=15)
    
    def _plot_returns_distribution(self, ax):
        """Returns distribution"""
        if not self.analysis['equity_curve'] or len(self.analysis['equity_curve']) < 2:
            return
        
        values = [p['value'] for p in self.analysis['equity_curve']]
        returns = np.diff(values) / values[:-1] * 100
        
        ax.hist(returns, bins=50, color='#06FFA5', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.3f}%')
        ax.set_xlabel('Return (%)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trade_analysis(self, ax):
        """Trade analysis - win/lose share"""
        winning = self.analysis['winning_trades']
        losing = self.analysis['losing_trades']
        
        if winning + losing == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            ax.axis('off')
            return
        
        colors = ['#06FFA5', '#E63946']
        explode = (0.05, 0)
        ax.pie([winning, losing], labels=['Win', 'Loss'], autopct='%1.1f%%',
              colors=colors, explode=explode, startangle=90, textprops={'fontsize': 10})
        ax.set_title('Win/Loss Share', fontsize=12, fontweight='bold')
    
    def _plot_holding_period(self, ax):
        """Holding period distribution"""
        if not self.analysis['trades']:
            return
        
        bars = [t['bars'] for t in self.analysis['trades']]
        ax.hist(bars, bins=30, color='#A663CC', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(bars), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(bars):.1f}')
        ax.set_xlabel('Holding Period (bars)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Holding Period Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pnl_distribution(self, ax):
        """PnL distribution"""
        if not self.analysis['trades']:
            return
        
        pnls = [t['pnl'] for t in self.analysis['trades']]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Trade Index', fontsize=10)
        ax.set_ylabel('PnL (USDT)', fontsize=10)
        ax.set_title('Trade-by-Trade PnL', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_cumulative_returns(self, ax):
        """Cumulative PnL curve"""
        if not self.analysis['trades']:
            return
        
        cumulative = np.cumsum([t['pnl'] for t in self.analysis['trades']])
        
        ax.plot(cumulative, color='#2E86AB', linewidth=2, marker='o', markersize=3)
        ax.fill_between(range(len(cumulative)), cumulative, alpha=0.3, color='#2E86AB')
        ax.axhline(0, color='black', linewidth=1, linestyle='--')
        ax.set_xlabel('Trades', fontsize=10)
        ax.set_ylabel('Cumulative PnL (USDT)', fontsize=10)
        ax.set_title('Cumulative PnL', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _generate_text_report(self, output_dir):
        """Generate text report (English)"""
        report_path = os.path.join(output_dir, f'{self.strategy_name}_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"   {self.strategy_name} - Backtest Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("[Performance]\n")
            f.write(f"  Total Return:    {self.analysis['total_return']*100:>10.2f}%\n")
            f.write(f"  Annual Return:   {self.analysis['annual_return']*100:>10.2f}%\n")
            f.write(f"  Sharpe Ratio:    {self.analysis['sharpe_ratio']:>10.2f}\n\n")
            
            f.write("[Risk]\n")
            f.write(f"  Max Drawdown:    {self.analysis['max_drawdown']*100:>10.2f}%\n")
            f.write(f"  Max DD Duration: {self.analysis['max_drawdown_duration']:>10} days\n\n")
            
            f.write("[Trading]\n")
            f.write(f"  Total Trades:    {self.analysis['total_trades']:>10}\n")
            f.write(f"  Winning Trades:  {self.analysis['winning_trades']:>10}\n")
            f.write(f"  Losing Trades:   {self.analysis['losing_trades']:>10}\n")
            f.write(f"  Win Rate:        {self.analysis['win_rate']*100:>10.2f}%\n")
            f.write(f"  Avg Win:         {self.analysis['avg_win']:>10.2f} USDT\n")
            f.write(f"  Avg Loss:        {self.analysis['avg_loss']:>10.2f} USDT\n")
            f.write(f"  Profit Factor:   {self.analysis['profit_factor']:>10.2f}\n")
            f.write(f"  Avg Holding:     {self.analysis['avg_bars_per_trade']:>10.1f} bars\n\n")
            
            f.write("=" * 60 + "\n")
        
        logger.info(f"📝 Text report saved: {report_path}")
