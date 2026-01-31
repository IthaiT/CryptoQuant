# ============================================================================
# 导入模块
# ============================================================================
# 从本地 StrategyBase 模块导入基类 StrategyBase 和持仓状态枚举 Position
from .StrategyBase import StrategyBase, Position
# 导入 talib 库用于计算技术指标（如 RSI）
import talib
# 导入 pandas 库用于数据处理（DataFrame 结构）
import pandas as pd
# 导入 numpy 库用于数值计算和数组操作
import numpy as np
# 导入类型提示模块，用于函数返回值注解
from typing import Dict, Any


# ============================================================================
# RSI 策略类定义
# ============================================================================
class RSIStrategy(StrategyBase):
    """
    基于 RSI（相对强弱指数）的交易策略类。
    继承自 StrategyBase，实现向量化的信号生成逻辑。
    """

    # 策略名称常量
    NAME = "RSI"

    def __init__(self,
                 window_size: int = 14,
                 enter_long=None,
                 exit_long=None,
                 enter_short=None,
                 exit_short=None):
        """
        初始化 RSI 策略。

        参数说明：
        - window_size (int): RSI 计算周期，默认 14。RSI = 100 - 100/(1+RS)，其中 RS = 平均涨幅/平均跌幅
        - enter_long (float): 多头入场 RSI 阈值。当 RSI > enter_long 时，生成买入信号。如果为 None，则不生成买入信号
        - exit_long (float): 多头出场 RSI 阈值。当 RSI < exit_long 时，平多仓。如果为 None，则不设置平仓条件
        - enter_short (float): 空头入场 RSI 阈值。当 RSI < enter_short 时，生成卖出信号（做空）。如果为 None，则不做空
        - exit_short (float): 空头出场 RSI 阈值。当 RSI > exit_short 时，平空仓。如果为 None，则不设置平仓条件
        """
        # 保存 RSI 计算的时间窗口
        self.window_size = window_size
        # 保存多头入场阈值（RSI 超买的上限）
        self.enter_long = enter_long
        # 保存多头出场阈值（RSI 回落的下限）
        self.exit_long = exit_long
        # 保存空头入场阈值（RSI 超卖的下限）
        self.enter_short = enter_short
        # 保存空头出场阈值（RSI 反弹的上限）
        self.exit_short = exit_short
        # 将策略名称赋值给实例属性
        self.name = RSIStrategy.NAME

    def info(self) -> Dict[str, Any]:
        """
        返回策略的参数信息字典。

        返回值：
        - Dict[str, Any]: 包含策略名称和所有参数设置的字典，便于查询和日志记录
        """
        return {
            'strategy_name': self.name,
            'window_size': self.window_size,
            'enter_long': self.enter_long,
            'exit_long': self.exit_long,
            'enter_short': self.enter_short,
            'exit_short': self.exit_short
        }

    def run(self, data: pd.DataFrame):
        """
        执行策略，基于 RSI 生成持仓信号。

        参数说明：
        - data (pd.DataFrame): 行情数据，必须包含 'close' 列（收盘价）

        返回值：
        - np.ndarray: int32 类型的数组，长度与输入行情数据相同，值为 Position 枚举（EXIT=0, LONG=1, SHORT=-1）
        """
        # 将 DataFrame 的收盘价列转换为 numpy 数组，提高计算效率
        array = data['close'].to_numpy()

        # 使用 talib 库计算 RSI 指标。RSI 值范围为 0-100
        # RSI > 70 表示超买（可能回调），RSI < 30 表示超卖（可能反弹）
        rsi = talib.RSI(array, timeperiod=self.window_size)

        # ========================================================================
        # 信号生成逻辑
        # ========================================================================
        
        # 多头入场信号：RSI 大于 enter_long 阈值则为 True
        # 如果 self.enter_long 为 None，则使用 np.inf 使条件永远不成立
        enter_long = rsi > (self.enter_long or np.inf)

        # 多头出场信号：RSI 小于 exit_long 阈值则为 True
        # 如果 self.exit_long 为 None，则使用 -np.inf 使条件永远不成立
        exit_long = rsi < (self.exit_long or -np.inf)

        # 空头入场信号：RSI 小于 enter_short 阈值则为 True
        # 如果 self.enter_short 为 None，则使用 -np.inf 使条件永远不成立
        enter_short = rsi < (self.enter_short or -np.inf)

        # 空头出场信号：RSI 大于 exit_short 阈值则为 True
        # 如果 self.exit_short 为 None，则使用 np.inf 使条件永远不成立
        exit_short = rsi > (self.exit_short or np.inf)

        # ========================================================================
        # 持仓状态数组初始化与赋值
        # ========================================================================

        # 创建与 RSI 相同形状的数组，初始值全为 NaN（未定义状态）
        positions = np.full(rsi.shape, np.nan)

        # 优先级 1（最高）：平仓信号。同时满足 exit_long 或 exit_short 时，设置为 EXIT (0)
        positions[exit_long | exit_short] = Position.EXIT

        # 优先级 2：多头入场信号。设置为 LONG (1)
        positions[enter_long] = Position.LONG

        # 优先级 3：空头入场信号。设置为 SHORT (-1)
        positions[enter_short] = Position.SHORT

        # ========================================================================
        # 前向填充处理：填充 NaN 值
        # ========================================================================
        # 说明：若某个 K 线没有生成新信号，则继承前一个信号，形成持续持仓的效果

        # 如果第一个 K 线的持仓状态仍为 NaN（无信号），则初始化为 EXIT（平仓状态，即无持仓）
        if np.isnan(positions[0]):
            positions[0] = Position.EXIT

        # 创建布尔掩码：标记所有 NaN 位置
        mask = np.isnan(positions)

        # 创建索引数组：记录每个位置最近的非 NaN 值的索引位置
        # np.where(~mask, ...) 对于非 NaN 位置，记录其索引；对于 NaN 位置，初始化为 0
        idx = np.where(~mask, np.arange(mask.size), 0)

        # np.maximum.accumulate()：前向累积最大值
        # 实现前向填充的关键：每个 NaN 位置都被赋予最近的非 NaN 位置的索引
        np.maximum.accumulate(idx, out=idx)

        # 根据填充后的索引，将每个 NaN 位置的值替换为最近的非 NaN 值
        # 这样实现了"未生成新信号时，保持上一个持仓状态"的效果
        positions[mask] = positions[idx[mask]]

        # ========================================================================
        # 返回结果
        # ========================================================================
        # 将持仓数组转换为 int32 类型并返回，便于存储和下游使用
        return positions.astype(np.int32)
