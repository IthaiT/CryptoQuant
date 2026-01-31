# 这个脚本用来测试RSIStrategy文件中run函数的返回值

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from src.strategies.RSIStrategy import RSIStrategy, Position


def test_rsi_strategy_run_return_type():
    """测试 RSIStrategy.run() 返回类型为 int32 数组"""
    # 加载测试数据
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'btc-usdt-5m.csv'))
    # 重命名列以匹配策略期望的列名
    data.rename(columns={'Close price': 'close'}, inplace=True)
    
    # 创建策略实例
    strategy = RSIStrategy(window_size=14, enter_long=70, exit_long=30, 
                          enter_short=30, exit_short=70)
    
    # 执行策略
    result = strategy.run(data)
    
    # 验证返回类型为 numpy 数组且数据类型为 int32
    assert isinstance(result, np.ndarray), "返回值应该是 numpy 数组"
    assert result.dtype == np.int32, f"返回值类型应该是 int32，实际为 {result.dtype}"
    print("✓ 返回类型测试通过：返回值为 int32 数组")


def test_rsi_strategy_run_return_length():
    """测试 RSIStrategy.run() 返回值长度与输入数据一致"""
    # 加载测试数据
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'btc-usdt-5m.csv'))
    data.rename(columns={'Close price': 'close'}, inplace=True)
    
    # 创建策略实例
    strategy = RSIStrategy(window_size=14, enter_long=70, exit_long=30)
    
    # 执行策略
    result = strategy.run(data)
    
    # 验证返回值长度与输入数据长度相同
    assert len(result) == len(data), f"返回值长度 {len(result)} 应该等于输入数据长度 {len(data)}"
    print(f"✓ 返回长度测试通过：返回数组长度 {len(result)} 等于输入数据长度")


def test_rsi_strategy_run_return_values():
    """测试 RSIStrategy.run() 返回值包含有效的持仓状态"""
    # 加载测试数据
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'btc-usdt-5m.csv'))
    data.rename(columns={'Close price': 'close'}, inplace=True)
    
    # 创建策略实例
    strategy = RSIStrategy(window_size=14, enter_long=70, exit_long=30, 
                          enter_short=30, exit_short=70)
    
    # 执行策略
    result = strategy.run(data)
    
    # 验证所有返回值都是有效的持仓状态
    valid_positions = {Position.EXIT, Position.LONG, Position.SHORT}
    result_set = set(result)
    assert result_set.issubset(valid_positions), \
        f"返回值包含无效的持仓状态：{result_set - valid_positions}。有效状态为 {valid_positions}"
    print(f"✓ 返回值测试通过：所有值都是有效的持仓状态 {valid_positions}")
    print(f"  实际返回的持仓状态：{result_set}")


def test_rsi_strategy_no_parameters():
    """测试 RSIStrategy.run() 当所有参数为 None 时的行为"""
    # 加载测试数据
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'btc-usdt-5m.csv'))
    data.rename(columns={'Close price': 'close'}, inplace=True)
    
    # 创建策略实例，所有阈值为 None
    strategy = RSIStrategy(window_size=14, enter_long=None, exit_long=None,
                          enter_short=None, exit_short=None)
    
    # 执行策略
    result = strategy.run(data)
    
    # 验证返回值为 int32 数组
    assert isinstance(result, np.ndarray), "返回值应该是 numpy 数组"
    assert result.dtype == np.int32, f"返回值类型应该是 int32，实际为 {result.dtype}"
    
    # 当没有入场条件时，应该都是 EXIT 状态
    assert np.all(result == Position.EXIT), \
        "当所有阈值为 None 时，所有持仓状态应该为 EXIT (0)"
    print("✓ 无参数测试通过：所有持仓状态均为 EXIT")


def test_rsi_strategy_info():
    """测试 RSIStrategy.info() 返回值为字典"""
    strategy = RSIStrategy(window_size=14, enter_long=70, exit_long=30,
                          enter_short=30, exit_short=70)
    
    # 获取策略信息
    info = strategy.info()
    
    # 验证返回值为字典
    assert isinstance(info, dict), "info() 返回值应该是字典"
    
    # 验证字典包含预期的键
    expected_keys = {'strategy_name', 'window_size', 'enter_long', 'exit_long', 
                    'enter_short', 'exit_short'}
    actual_keys = set(info.keys())
    assert actual_keys == expected_keys, \
        f"info() 返回的键 {actual_keys} 应该等于 {expected_keys}"
    
    # 验证策略名称
    assert info['strategy_name'] == 'RSI', f"策略名称应该为 'RSI'，实际为 {info['strategy_name']}"
    print("✓ 策略信息测试通过：info() 返回值包含所有预期的参数")


if __name__ == '__main__':
    print("=" * 60)
    print("开始测试 RSIStrategy 的 run() 函数返回值")
    print("=" * 60)
    
    try:
        test_rsi_strategy_run_return_type()
        test_rsi_strategy_run_return_length()
        test_rsi_strategy_run_return_values()
        test_rsi_strategy_no_parameters()
        test_rsi_strategy_info()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ 测试失败：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
