"""
LSTM 模型回测辅助模块 - 处理模型加载、推理和数据预处理

该模块提供：
    - LSTMModelLoader：加载训练好的 LSTM 模型
    - LSTMPredictor：处理滑动窗口推理
    - 与 Backtrader 的无缝集成
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.ml.lstm_model import LSTMModel, get_device
from src.utils.logger import logger


class LSTMModelLoader:
    """加载和管理训练好的 LSTM 模型及其配置。

    Attributes:
        model_dir: 模型检查点所在目录
        model: 加载的 LSTMModel 实例
        device: 推理设备 (cuda/mps/cpu)
    """

    def __init__(self, model_dir: str | Path) -> None:
        """初始化模型加载器。

        Args:
            model_dir: 包含最佳模型检查点的目录路径
        """
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "best_lstm_model.pt"
        self.device = get_device()
        self.model: Optional[LSTMModel] = None

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"模型检查点不存在: {self.model_path}\n"
                f"请先运行 python script/train_lstm.py 进行训练"
            )

    def load(self, input_size: int = 4) -> LSTMModel:
        """加载 LSTM 模型。

        Args:
            input_size: 输入特征数量（必须与训练时一致）

        Returns:
            已加载到指定设备的 LSTMModel 实例
        """
        # 构建模型
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
        )

        # 加载权重
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"✅ 模型加载成功: {self.model_path}")
        return self.model


class LSTMPredictor:
    """LSTM 推理引擎，处理滑动窗口和批量预测。

    特性：
        - 维护 lookback 窗口的历史数据
        - 支持实时特征计算（log returns）
        - 返回购买概率 [0, 1]
    """

    def __init__(
        self,
        model: LSTMModel,
        lookback: int = 60,
        feature_names: list[str] | None = None,
    ) -> None:
        """初始化预测器。

        Args:
            model: 已加载的 LSTMModel
            lookback: 滑动窗口大小
            feature_names: 特征名称列表（用于日志）
        """
        self.model = model
        self.lookback = lookback
        self.device = model.lstm.weight_ih_l0.device
        self.feature_count = model.lstm.input_size

        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(self.feature_count)
        ]
        self.feature_buffer = np.zeros((lookback, self.feature_count), dtype=np.float32)
        self.buffer_full = False

        # 归一化器状态（从训练脚本同步）
        self.scaler: Optional[MinMaxScaler] = None
        self._scaler_warned = False  # 避免重复警告

        # 预先分配 GPU tensor，避免每次推理时重新创建
        self._input_tensor = torch.zeros(
            (1, lookback, self.feature_count),
            dtype=torch.float32,
            device=self.device
        )

    def set_scaler(self, scaler: MinMaxScaler) -> None:
        """设置归一化器（必须与训练时使用的相同）。

        Args:
            scaler: 已在训练数据上 fit 过的 MinMaxScaler
        """
        self.scaler = scaler
        logger.info("✅ MinMaxScaler 已设置")

    def update_features(self, ffd_close: float, log_return: float,
                       volume: float, dollar_volume: float) -> None:
        """用最新的特征值更新滑动窗口缓冲。

        Args:
            ffd_close: 分数差分收盘价
            log_return: 对数收益率
            volume: 交易量
            dollar_volume: 美元成交量
        """
        raw_features = np.array([ffd_close, log_return, volume, dollar_volume], dtype=np.float32)

        # 应用归一化
        if self.scaler is not None:
            scaled_features = self.scaler.transform(raw_features.reshape(1, -1))[0]
        else:
            if not self._scaler_warned:
                logger.warning("⚠️ 未设置 scaler，使用未归一化的特征")
                self._scaler_warned = True
            scaled_features = raw_features

        # 移动窗口：删除最旧的行，添加最新的行（使用 roll 比 vstack 快）
        self.feature_buffer = np.roll(self.feature_buffer, -1, axis=0)
        self.feature_buffer[-1] = scaled_features

        if not self.buffer_full and np.all(self.feature_buffer != 0):
            self.buffer_full = True
            logger.info("✅ 特征缓冲已满，开始生成预测")

    def predict(self) -> float:
        """使用当前缓冲中的数据进行推理。

        Returns:
            购买概率，范围 [0, 1]。如果缓冲未满，返回 0.5（中立）
        """
        if not self.buffer_full:
            return 0.5  # 缓冲未满时保持中立

        # 直接将 numpy 数组复制到预分配的 GPU tensor（避免重复创建）
        self._input_tensor[0].copy_(torch.from_numpy(self.feature_buffer))

        # 推理（已在 __init__ 预分配 tensor，减少内存开销）
        with torch.no_grad():
            probs = self.model.predict_proba(self._input_tensor)  # (1, 1)
            prob = probs[0, 0].item()  # 避免 squeeze().cpu()

        return float(prob)


class DollarBarDataPreprocessor:
    """将 dollar-bar CSV 数据转换为 Backtrader 兼容格式，并计算 log returns。

    特性：
        - 读取 dollar-bar CSV（带 ffd_close 和标签）
        - 计算 log returns 以获得平稳特征
        - 输出 Backtrader 兼容格式
    """

    @staticmethod
    def preprocess(csv_path: str | Path) -> pd.DataFrame:
        """加载和预处理数据。

        Args:
            csv_path: dollar-bar CSV 文件路径

        Returns:
            处理过的 DataFrame，包含 OHLCV 和计算的特征
        """
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        logger.info(f"原始数据形状: {df.shape}")

        # 计算 log returns（平稳化原始价格）
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # 删除 NaN 行
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"删除了 {before - len(df)} 行 NaN → {len(df)} 行")

        # 准备 Backtrader 所需的列
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.set_index("datetime", inplace=True)

        # 保留 Backtrader 所需的列及计算的特征
        cols_to_keep = ["open", "high", "low", "close", "volume", "ffd_close", "log_return", "dollar_volume"]
        available_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[available_cols].astype("float64").copy()

        # 排序并移除重复的索引
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()

        logger.info(f"预处理后数据: {df.shape}")
        return df
