"""
LSTM Model, Dataset, and Trainer for Crypto Quantitative Trading.

This module provides:
    - LSTMModel: A multi-layer LSTM network for time-series classification.
    - QuantDataset: A PyTorch Dataset that creates sliding-window sequences.
    - Trainer: Orchestrates training, validation, checkpointing, and metrics logging.

Author: CryptoQuant
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from loguru import logger


# ---------------------------------------------------------------------------
# 1. Device Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Automatically select the best available accelerator.

    Priority: CUDA (NVIDIA GPU) -> MPS (Apple Silicon) -> CPU.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device


# ---------------------------------------------------------------------------
# 2. QuantDataset — Sliding-Window Sequence Dataset
# ---------------------------------------------------------------------------

class QuantDataset(Dataset):
    """PyTorch Dataset that generates fixed-length sliding-window sequences.

    Each sample is a tuple (X, y) where:
        - X: Tensor of shape (lookback, num_features)
        - y: Tensor scalar — binary label (0=Down, 1=Up) at the END of the window.

    Args:
        features: 2-D NumPy array of shape (T, num_features), already scaled.
        labels: 1-D NumPy array of shape (T,) with values in {0, 1}.
        lookback: Number of past time-steps used as input context.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        lookback: int = 60,
    ) -> None:
        super().__init__()
        assert len(features) == len(labels), (
            f"features ({len(features)}) and labels ({len(labels)}) length mismatch"
        )
        assert lookback < len(features), (
            f"lookback ({lookback}) must be < data length ({len(features)})"
        )

        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.lookback = lookback
        # Total available samples = len - lookback
        self.n_samples = len(features) - lookback

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single (sequence, label) pair.

        The sequence spans [idx, idx+lookback) and the label is at idx+lookback.
        """
        x = self.features[idx : idx + self.lookback]  # (lookback, num_features)
        y = float(self.labels[idx + self.lookback])

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. LSTMModel — Multi-Layer LSTM Classifier
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """Multi-layer LSTM followed by a binary classification head.

    Architecture:
        Input → LSTM (multi-layer, dropout) → LayerNorm → FC → ReLU → Dropout → FC (1 logit)

    Args:
        input_size: Number of input features per time-step.
        hidden_size: Number of LSTM hidden units.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between LSTM layers and in FC head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # --- LSTM Encoder ---
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # --- Classification Head ---
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, lookback, input_size).

        Returns:
            Logits tensor of shape (batch, 1).
        """
        # lstm_out: (batch, lookback, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the output of the LAST time-step
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Classification head
        out = self.layer_norm(last_hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, 1)
        return out

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities for the positive class (Up).

        Args:
            x: Input tensor of shape (batch, lookback, input_size).

        Returns:
            Probabilities in [0, 1] with shape (batch, 1).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# 4. Trainer — Training / Validation / Checkpointing
# ---------------------------------------------------------------------------

class Trainer:
    """Handles the full training lifecycle for the LSTM classifier.

    Responsibilities:
        - Build DataLoaders with chronological train/val split (no shuffle to preserve time order).
        - Run training loop with loss & accuracy logging.
        - Run validation loop after each epoch.
        - Save the best model checkpoint (by val AUC when available).
        - Produce a final classification report at the end.

    Args:
        model: An instance of LSTMModel.
        train_dataset: QuantDataset for training.
        val_dataset: QuantDataset for validation.
        lr: Learning rate.
        batch_size: Mini-batch size.
        num_epochs: Maximum training epochs.
        device: Torch device (auto-detected if None).
        save_dir: Directory for saving checkpoints and logs.
        pos_weight: Optional scalar tensor to re-weight positive class in BCEWithLogitsLoss.
    """

    def __init__(
        self,
        model: LSTMModel,
        train_dataset: QuantDataset,
        val_dataset: QuantDataset,
        lr: float = 1e-3,
        batch_size: int = 256,
        num_epochs: int = 50,
        device: Optional[torch.device] = None,
        save_dir: str | Path = "models",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- DataLoaders (no shuffle — time-series!) ---
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # --- Loss & Optimizer ---
        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )

        # --- Tracking ---
        self.best_val_acc: float = 0.0
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> tuple[float, float]:
        """Run one training epoch.

        Returns:
            (average_loss, accuracy) for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)  # (B, lookback, features)
            y_batch = y_batch.to(self.device)  # (B,)

            self.optimizer.zero_grad()
            logits = self.model(X_batch).squeeze(1)  # (B,)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping to stabilise LSTM training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(self) -> tuple[float, float, float]:
        """Run validation pass.

        Returns:
            (average_loss, accuracy, auc) for the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_probs: list[float] = []
        all_labels: list[int] = []

        for X_batch, y_batch in self.val_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch).squeeze(1)
            loss = self.criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)
            all_probs.extend(probs.detach().cpu().tolist())
            all_labels.extend(y_batch.detach().cpu().int().tolist())

        avg_loss = total_loss / total
        accuracy = correct / total
        auc = self._safe_auc(all_labels, all_probs)
        return avg_loss, accuracy, auc

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        """Save model weights and training metadata."""
        ckpt_path = self.save_dir / "best_lstm_model.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
            },
            ckpt_path,
        )
        logger.info(f"💾 Checkpoint saved → {ckpt_path}  (val_acc={val_acc:.4f})")

    def _save_history(self) -> None:
        """Persist training history to JSON for later analysis."""
        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"📊 Training history saved → {history_path}")

    # ------------------------------------------------------------------
    # Full Classification Report on Validation Set
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> str:
        """Generate sklearn classification report on the validation set.

        Returns:
            Formatted classification report string.
        """
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[float] = []

        for X_batch, y_batch in self.val_loader:
            X_batch = X_batch.to(self.device)
            logits = self.model(X_batch).squeeze(1)
            probs = torch.sigmoid(logits).cpu().tolist()
            preds = [1 if p >= 0.5 else 0 for p in probs]
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y_batch.int().tolist())

        # Map class indices back to semantic names
        target_names = ["Down (0)", "Up (1)"]
        report = classification_report(
            all_labels, all_preds, target_names=target_names, digits=4, zero_division=0
        )
        cm = confusion_matrix(all_labels, all_preds)
        auc = self._safe_auc(all_labels, all_probs)
        auc_line = f"AUC-ROC: {auc:.4f}" if not np.isnan(auc) else "AUC-ROC: NaN (single class)"
        return (
            f"\n{'='*60}\nClassification Report (Validation Set)\n{'='*60}\n"
            f"{report}\nConfusion Matrix:\n{cm}\n{auc_line}\n"
        )

    @staticmethod
    def _safe_auc(labels: list[int], probs: list[float]) -> float:
        """Compute AUC safely; returns NaN if only one class is present."""
        try:
            return float(roc_auc_score(labels, probs))
        except ValueError:
            return float("nan")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Execute the full training loop with validation and checkpointing."""
        logger.info(
            f"Starting training: {self.num_epochs} epochs | "
            f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)} | "
            f"Device: {self.device}"
        )

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc, val_auc = self._validate()

            # Step the LR scheduler based on validation loss
            self.scheduler.step(val_loss)

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch [{epoch:>3}/{self.num_epochs}]  "
                f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  Val AUC: {val_auc:.4f}  |  "
                f"LR: {current_lr:.2e}"
            )

            # Save best checkpoint
            metric = val_auc if not np.isnan(val_auc) else val_acc
            if metric > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

        # --- Post-training ---
        self._save_history()

        # Load best model and print final evaluation
        best_ckpt = self.save_dir / "best_lstm_model.pt"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(
                f"Loaded best model from epoch {ckpt['epoch']} "
                f"(val_acc={ckpt['val_acc']:.4f})"
            )

        report = self.evaluate()
        logger.info(report)
        logger.info("Training complete ✅")
