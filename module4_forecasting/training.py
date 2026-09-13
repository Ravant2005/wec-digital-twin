"""
training.py — Training loop for Tier-1 LSTM/GRU wave forecasters (Module 4.2).

Responsibilities
----------------
- PyTorch Dataset adapter wrapping ForecastDatasetV2
- DataLoader construction (train shuffled, val/test ordered)
- Weighted MSE loss across 4 target channels
- Training loop with validation
- Early stopping on validation loss
- Best-checkpoint saving
- Deterministic seeding (Python, NumPy, PyTorch, MPS/CUDA)
- Experiment configuration recording

Training split discipline
-------------------------
Training DataLoader shuffles WINDOWS (not raw time series).
Windows were already created entirely within 2010–2021 by ForecastDatasetV2.
Shuffling windows is safe because each window is self-contained.

Validation and test DataLoaders are NOT shuffled.
This preserves chronological order for diagnostic inspection.

Scaling
-------
The WaveScalerV2 is fitted on training data only (done in build_datasets_v2).
Features are scaled before entering the network.
Targets are scaled for loss computation.
Inverse transform to physical units is done in evaluation.py.

Missing data
------------
NaN replacement (0.0 after z-score) is performed inside model.forward().
The scaler preserves NaN during transform; the model handles them.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from module4_forecasting.dataset_v2 import ForecastDatasetV2, WaveScalerV2
from module4_forecasting.models import ModelConfig, _WaveForecasterBase, build_model


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Full experiment configuration (model + training hyperparameters).

    Parameters
    ----------
    model_type : str
        "lstm" or "gru".
    feature_mode : str
        "wave_only" or "wave_wind".
    n_features : int
        Number of input features (7 or 11).
    input_length : int
        Input sequence length.  Default 48.
    forecast_horizon : int
        Forecast horizon.  Default 48.
    hidden_size : int
        RNN hidden units.  Default 128.
    num_layers : int
        Stacked RNN layers.  Default 2.
    dropout : float
        Dropout between layers.  Default 0.1.
    batch_size : int
        Mini-batch size.  Default 128.
    learning_rate : float
        Adam learning rate.  Default 1e-3.
    max_epochs : int
        Maximum training epochs.  Default 50.
    patience : int
        Early-stopping patience (epochs without val improvement).  Default 7.
    seed : int
        Global random seed.  Default 42.
    channel_weights : list of float
        Per-target loss weights [Hs, Tp, sin_dir, cos_dir].  Default [1,1,1,1].
    experiment_name : str
        Used for results directory naming.
    """

    model_type:       str         = "lstm"
    feature_mode:     str         = "wave_only"
    n_features:       int         = 7
    input_length:     int         = 48
    forecast_horizon: int         = 48
    hidden_size:      int         = 128
    num_layers:       int         = 2
    dropout:          float       = 0.1
    batch_size:       int         = 128
    learning_rate:    float       = 1e-3
    max_epochs:       int         = 50
    patience:         int         = 7
    seed:             int         = 42
    channel_weights:  List[float] = None  # set in __post_init__
    experiment_name:  str         = "exp"

    def __post_init__(self) -> None:
        if self.channel_weights is None:
            self.channel_weights = [1.0, 1.0, 1.0, 1.0]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            n_features=self.n_features,
            forecast_horizon=self.forecast_horizon,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )


# ---------------------------------------------------------------------------
# PyTorch Dataset adapter
# ---------------------------------------------------------------------------

class _TorchDataset(Dataset):
    """
    Thin PyTorch Dataset wrapper around ForecastDatasetV2.

    Materialises all windows into contiguous float32 arrays once at
    construction time.  With ~105k windows × (48×7 + 48×4) = ~52M floats
    this is ~200 MB — acceptable for this dataset size.

    Scaling is applied here so the DataLoader workers see pre-scaled data.
    NaN values in X are preserved (replaced with 0.0 inside model.forward).
    """

    def __init__(self, ds: ForecastDatasetV2, scaler: WaveScalerV2,
                 label: str = "") -> None:
        super().__init__()
        N = len(ds)
        tag = f" [{label}]" if label else ""
        # Materialise all windows with progress reporting
        print(f"    Building X{tag}: {N:,} windows × {ds.config.input_length} steps × {ds.config.n_features} features...", end="", flush=True)
        X_all = ds.get_all_X()          # (N, L, F)  float64, may contain NaN
        print(" done.", flush=True)
        print(f"    Building y{tag}: {N:,} windows × {ds.config.forecast_horizon} steps × 4 targets...", end="", flush=True)
        y_all = ds.get_all_targets()    # (N, H, 4)  float64
        print(" done.", flush=True)

        # Scale features (NaN preserved) and targets
        print(f"    Scaling{tag}...", end="", flush=True)
        X_scaled = scaler.transform(X_all)
        y_scaled = scaler.transform_target(y_all)
        print(" done.", flush=True)

        self._X = torch.from_numpy(X_scaled.astype(np.float32))
        self._y = torch.from_numpy(y_scaled.astype(np.float32))

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y[idx]


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch (CPU + MPS/CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS does not expose a separate seed API; torch.manual_seed covers it


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return MPS if available, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss across 4 target channels.

    Loss = mean over (B, H) of sum_c [ w_c * (pred_c - true_c)^2 ]

    NaN in targets is handled by masking those positions out.
    In practice targets have no NaN (ERA5 true values are complete),
    but the mask is retained for robustness.
    """

    def __init__(self, weights: List[float]) -> None:
        super().__init__()
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: [B, H, 4]
        err2 = (pred - target) ** 2          # [B, H, 4]
        valid = ~torch.isnan(target)
        err2 = torch.where(valid, err2, torch.zeros_like(err2))
        weighted = err2 * self.weights       # broadcast over [B, H, 4]
        return weighted.mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Output of a completed training run."""
    train_losses:      List[float]
    val_losses:        List[float]
    best_epoch:        int
    best_val_loss:     float
    checkpoint_path:   str
    config:            TrainConfig
    duration_seconds:  float


def train(
    train_ds:   ForecastDatasetV2,
    val_ds:     ForecastDatasetV2,
    scaler:     WaveScalerV2,
    config:     TrainConfig,
    results_dir: str,
) -> TrainingResult:
    """
    Train a wave forecaster and save the best checkpoint.

    Parameters
    ----------
    train_ds, val_ds : ForecastDatasetV2
        Pre-built datasets from build_datasets_v2.
    scaler : WaveScalerV2
        Fitted on training data only.
    config : TrainConfig
        Full experiment configuration.
    results_dir : str
        Directory to save checkpoint and config.

    Returns
    -------
    TrainingResult
    """
    os.makedirs(results_dir, exist_ok=True)
    seed_everything(config.seed)
    device = get_device()

    # --- Build PyTorch datasets ---
    print(f"  Materialising datasets (train={len(train_ds):,}, val={len(val_ds):,})...", flush=True)
    t0 = time.time()
    torch_train = _TorchDataset(train_ds, scaler, label="train")
    torch_val   = _TorchDataset(val_ds,   scaler, label="val")
    print(f"  Materialised in {time.time()-t0:.1f}s", flush=True)

    # Training loader: shuffle=True (windows are self-contained within 2010-2021)
    # Val loader: shuffle=False (preserve chronological order)
    train_loader = DataLoader(
        torch_train, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        torch_val, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # --- Model ---
    model_cfg = config.to_model_config()
    model = build_model(config.model_type, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {config.model_type.upper()} | params={n_params:,} | device={device}", flush=True)

    # --- Optimiser and loss ---
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = WeightedMSELoss(config.channel_weights).to(device)

    # --- Training loop ---
    train_losses: List[float] = []
    val_losses:   List[float] = []
    best_val_loss = float("inf")
    best_epoch    = 0
    patience_ctr  = 0
    checkpoint_path = os.path.join(results_dir, "best_model.pt")

    t_start = time.time()

    for epoch in range(1, config.max_epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimiser.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(X_batch)
        train_loss = epoch_loss / len(torch_train)
        train_losses.append(train_loss)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * len(X_batch)
        val_loss = val_loss_sum / len(torch_val)
        val_losses.append(val_loss)

        elapsed = time.time() - t_start
        print(
            f"  Epoch {epoch:3d}/{config.max_epochs} | "
            f"train={train_loss:.5f} | val={val_loss:.5f} | "
            f"elapsed={elapsed:.0f}s",
            flush=True,
        )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_ctr  = 0
            # Save checkpoint
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "val_loss":       val_loss,
                "config":         config.to_dict(),
                "scaler_mean":    scaler.fitted_stats["mean"].tolist(),
                "scaler_std":     scaler.fitted_stats["std"].tolist(),
                "scaler_feature_mode": scaler.feature_mode,
            }, checkpoint_path)
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                print(f"  Early stopping at epoch {epoch} (patience={config.patience})", flush=True)
                break

    duration = time.time() - t_start
    print(f"  Best epoch: {best_epoch} | best val loss: {best_val_loss:.5f}", flush=True)
    print(f"  Training duration: {duration:.1f}s", flush=True)

    # Save config
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Save loss history
    history = {
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return TrainingResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        checkpoint_path=checkpoint_path,
        config=config,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[_WaveForecasterBase, TrainConfig, WaveScalerV2]:
    """
    Load a saved checkpoint and reconstruct model + scaler.

    Parameters
    ----------
    checkpoint_path : str
    device : torch.device or None
        If None, uses get_device().

    Returns
    -------
    model : loaded model in eval mode
    config : TrainConfig
    scaler : WaveScalerV2 with fitted stats restored
    """
    if device is None:
        device = get_device()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]

    config = TrainConfig(**cfg_dict)
    model_cfg = config.to_model_config()
    model = build_model(config.model_type, model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Restore scaler
    scaler = WaveScalerV2(feature_mode=ckpt["scaler_feature_mode"])
    n_feat = model_cfg.n_features
    mean = np.array(ckpt["scaler_mean"])
    std  = np.array(ckpt["scaler_std"])
    scaler._mean    = mean
    scaler._std     = std
    scaler._fitted  = True
    scaler._n_features = n_feat

    return model, config, scaler
