"""
Training script for the LSTM classifier on BTCUSDT Dollar-Bar data.

Usage:
    python script/train_lstm.py                          # uses defaults
    python script/train_lstm.py --epochs 100 --lr 5e-4   # override hyper-params

This script:
    1. Loads dollar-bar CSV data with factor columns and triple-barrier labels.
    2. Drops rows where 'label' is NaN or == 0 (label 0 is sparse/noisy).
    3. Creates log returns (stationary vs raw close) and selects features.
    4. Applies MinMaxScaling (fit only on train split).
    5. Performs a chronological train/val split (80/20).
    6. Constructs QuantDataset sliding-window sequences (lookback=60).
    7. Trains an LSTMModel via the Trainer class.
    8. Saves the best checkpoint and evaluation report.

Author: CryptoQuant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.lstm_model import LSTMModel, QuantDataset, Trainer, get_device


# ---------------------------------------------------------------------------
# Configuration Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "preprocess_data"
    / "factor"
    / "BTCUSDT"
    / "BTCUSDT_2025-01-01_2025-12-31_dollar_bars_4m_labeled.csv"
)

FEATURE_COLUMNS: list[str] = ["ffd_close", "log_return", "volume", "dollar_volume"]
LABEL_COLUMN: str = "label"
LOOKBACK: int = 60
TRAIN_RATIO: float = 0.8


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(
    data_path: str | Path,
    feature_cols: list[str],
    label_col: str,
    train_ratio: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Load CSV, clean, scale features, and split chronologically.

    Steps:
        1. Read CSV, parse datetime index.
        2. Drop rows where label is NaN or label == 0 (too sparse/noisy).
        3. Compute log returns to stabilize price scale.
        4. Extract feature matrix and label vector.
        5. Chronological train/val split (no shuffling).
        6. Fit MinMaxScaler on TRAIN set only, then transform both.

    Args:
        data_path: Path to the dollar-bar CSV file.
        feature_cols: List of column names to use as input features.
        label_col: Name of the target label column.
        train_ratio: Fraction of data used for training (rest = validation).

    Returns:
        (X_train_scaled, y_train, X_val_scaled, y_val, scaler)
    """
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    logger.info(f"Raw data shape: {df.shape}")

    # --- Drop NaN labels and label==0 (label 0 is sparse/noisy, hurts learning) ---
    before = len(df)
    df = df.dropna(subset=[label_col])
    df = df[df[label_col] != 0].copy()
    logger.info(
        f"Dropped {before - len(df)} rows with NaN/label==0 -> {len(df)} rows remaining"
    )

    # --- Log returns for stationarity (raw price levels are non-stationary) ---
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # --- Drop rows with NaN in engineered features or ffd_close ---
    required_cols = feature_cols + [label_col]
    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    logger.info(
        f"Dropped {before - len(df)} rows with NaN features -> {len(df)} rows remaining"
    )

    # --- Remap labels: -1 -> 0 (Down), +1 -> 1 (Up) ---
    df[label_col] = df[label_col].map({-1: 0, 1: 1}).astype(int)

    # --- Extract arrays ---
    features = df[feature_cols].values  # (N, num_features)
    labels = df[label_col].values       # (N,)

    # Validate labels contain only expected values
    unique_labels = set(labels.astype(int))
    assert unique_labels.issubset({0, 1}), f"Unexpected label values: {unique_labels}"
    logger.info(
        f"Label distribution: { {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))} }"
    )

    # --- Chronological split ---
    split_idx = int(len(features) * train_ratio)
    X_train_raw, X_val_raw = features[:split_idx], features[split_idx:]
    y_train, y_val = labels[:split_idx], labels[split_idx:]
    logger.info(f"Train samples: {len(X_train_raw)} | Val samples: {len(X_val_raw)}")

    # --- MinMaxScaling (fit on train only to avoid look-ahead bias) ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    logger.info("MinMaxScaler fitted on training set and applied to both splits")

    return X_train_scaled, y_train, X_val_scaled, y_val, scaler


# ---------------------------------------------------------------------------
# Positive-Class Weight (handle class imbalance)
# ---------------------------------------------------------------------------

def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Compute positive class weight for BCEWithLogitsLoss.

    pos_weight = (#neg / #pos) to upweight the minority class (Up).

    Args:
        labels: Binary label array with values in {0, 1}.

    Returns:
        Scalar tensor with pos_weight.
    """
    num_pos = float(np.sum(labels == 1))
    num_neg = float(np.sum(labels == 0))
    pos_weight = num_neg / max(num_pos, 1.0)
    logger.info(f"Positive class weight (neg/pos): {pos_weight:.4f}")
    return torch.tensor(pos_weight, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training hyper-parameters."""
    parser = argparse.ArgumentParser(
        description="Train LSTM classifier on BTCUSDT dollar-bar data"
    )
    parser.add_argument(
        "--data", type=str, default=str(DEFAULT_DATA_PATH),
        help="Path to the labelled dollar-bar CSV file."
    )
    parser.add_argument("--lookback", type=int, default=LOOKBACK, help="Sliding window size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden units.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument(
        "--save_dir", type=str, default=str(PROJECT_ROOT / "models"),
        help="Directory to save checkpoints."
    )
    parser.add_argument(
        "--train_ratio", type=float, default=TRAIN_RATIO,
        help="Fraction of data for training (rest = validation)."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point: load data -> build model -> train -> evaluate."""
    args = parse_args()
    device = get_device()

    # ---- 1. Load & preprocess ----
    X_train, y_train, X_val, y_val, scaler = load_and_preprocess(
        data_path=args.data,
        feature_cols=FEATURE_COLUMNS,
        label_col=LABEL_COLUMN,
        train_ratio=args.train_ratio,
    )

    # ---- 2. Build datasets ----
    train_ds = QuantDataset(features=X_train, labels=y_train, lookback=args.lookback)
    val_ds = QuantDataset(features=X_val, labels=y_val, lookback=args.lookback)
    logger.info(
        f"Datasets created - Train sequences: {len(train_ds)} | "
        f"Val sequences: {len(val_ds)} | "
        f"Lookback: {args.lookback} | Features: {X_train.shape[1]}"
    )

    # ---- 3. Compute positive class weight for BCE loss ----
    pos_weight = compute_pos_weight(y_train)

    # ---- 4. Build model ----
    num_features = X_train.shape[1]
    model = LSTMModel(
        input_size=num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built - Total params: {total_params:,} | Trainable: {trainable_params:,}")
    logger.info(f"\n{model}")

    # ---- 5. Train ----
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=device,
        save_dir=args.save_dir,
        pos_weight=pos_weight,
    )
    trainer.fit()

    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
    logger.info(f"Artifacts saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
