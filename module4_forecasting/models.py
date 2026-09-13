"""
models.py — Tier-1 LSTM/GRU wave forecasters (Module 4.2).

Architecture
------------
Both models use an encoder–decoder design:

  Encoder : stacked LSTM (or GRU) reads the full input sequence
            [B, input_length, n_features] → hidden state h_enc

  Decoder : stacked LSTM (or GRU) autoregressively generates
            forecast_horizon steps using h_enc as initial state.
            At each decoder step the input is the previous decoder
            output (4 targets), NOT the true future target.
            This guarantees no teacher-forcing leakage at inference.

Missing-data contract
---------------------
NaN values in X are replaced with 0.0 AFTER z-score normalisation
(so 0.0 ≈ the training mean) before entering the network.
Validity mask columns (binary 0/1) are passed through unchanged and
carry the missingness signal.  The network therefore sees:
  - continuous features: 0.0 where missing, scaled value where present
  - validity flags: 0.0 where missing, 1.0 where present

This replacement is performed inside forward() so the caller never
needs to pre-process NaNs.

Output
------
Raw network output shape: [B, forecast_horizon, 4]
  channel 0 : Hs   (normalised space)
  channel 1 : Tp   (normalised space)
  channel 2 : sin_direction  (no activation — free regression)
  channel 3 : cos_direction  (no activation — free regression)

Inverse-transform to physical units is done in evaluation.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Hyperparameter configuration shared by LSTM and GRU forecasters.

    Parameters
    ----------
    n_features : int
        Number of input features per timestep (7 for wave_only, 11 for wave_wind).
    forecast_horizon : int
        Number of future steps to predict.  Default 48.
    hidden_size : int
        Number of hidden units per RNN layer.  Default 128.
    num_layers : int
        Number of stacked RNN layers.  Default 2.
    dropout : float
        Dropout probability between RNN layers (applied when num_layers > 1).
        Default 0.1.
    """

    n_features:       int   = 7
    forecast_horizon: int   = 48
    hidden_size:      int   = 128
    num_layers:       int   = 2
    dropout:          float = 0.1

    def __post_init__(self) -> None:
        assert self.n_features       >= 1,  "n_features must be >= 1"
        assert self.forecast_horizon >= 1,  "forecast_horizon must be >= 1"
        assert self.hidden_size      >= 1,  "hidden_size must be >= 1"
        assert self.num_layers       >= 1,  "num_layers must be >= 1"
        assert 0.0 <= self.dropout   < 1.0, "dropout must be in [0, 1)"


# ---------------------------------------------------------------------------
# Internal base class
# ---------------------------------------------------------------------------

N_TARGETS = 4  # Hs, Tp, sin_direction, cos_direction


class _WaveForecasterBase(nn.Module):
    """
    Shared encoder–decoder skeleton for LSTM and GRU variants.

    Subclasses supply self._encoder and self._decoder (nn.LSTM or nn.GRU).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # Decoder input at each step: previous output (N_TARGETS)
        self._decoder_input_size = N_TARGETS
        # Projection from decoder hidden to output
        self._output_proj = nn.Linear(config.hidden_size, N_TARGETS)

    # Subclasses must define self._encoder and self._decoder

    def _encode(self, x: torch.Tensor):
        """
        Run encoder over input sequence.

        Parameters
        ----------
        x : Tensor [B, L, n_features]
            NaN values already replaced with 0.0.

        Returns
        -------
        hidden : encoder final hidden state (passed to decoder as h_0)
        """
        raise NotImplementedError

    def _decode_step(self, dec_input: torch.Tensor, hidden):
        """
        Run one decoder step.

        Parameters
        ----------
        dec_input : Tensor [B, 1, decoder_input_size]
        hidden    : decoder hidden state

        Returns
        -------
        out    : Tensor [B, 1, hidden_size]
        hidden : updated hidden state
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [B, input_length, n_features]
            May contain NaN (replaced with 0.0 internally).

        Returns
        -------
        Tensor [B, forecast_horizon, N_TARGETS]
            Predictions in normalised space.
        """
        # Replace NaN with 0.0 (training-mean proxy after z-score)
        x = torch.nan_to_num(x, nan=0.0)

        B = x.size(0)
        H = self.config.forecast_horizon

        # Encode
        hidden = self._encode(x)

        # Decode autoregressively — no future targets used
        # Initial decoder input: zeros (neutral start)
        dec_input = x.new_zeros(B, 1, self._decoder_input_size)
        outputs = []
        for _ in range(H):
            out, hidden = self._decode_step(dec_input, hidden)   # [B,1,hidden]
            pred = self._output_proj(out)                         # [B,1,4]
            outputs.append(pred)
            dec_input = pred.detach()  # feed prediction as next input (no teacher forcing)

        return torch.cat(outputs, dim=1)  # [B, H, 4]


# ---------------------------------------------------------------------------
# LSTM forecaster
# ---------------------------------------------------------------------------

class WaveForecasterLSTM(_WaveForecasterBase):
    """
    Stacked LSTM encoder–decoder wave forecaster.

    Input  : [B, input_length, n_features]
    Output : [B, forecast_horizon, 4]

    The encoder reads the full input window and produces a context
    (h_n, c_n).  The decoder generates each future step autoregressively
    using the previous prediction as input.  No future ground truth is
    ever used during inference.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        self._encoder = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self._decoder = nn.LSTM(
            input_size=self._decoder_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

    def _encode(self, x: torch.Tensor):
        _, (h_n, c_n) = self._encoder(x)
        return (h_n, c_n)

    def _decode_step(self, dec_input: torch.Tensor, hidden):
        out, hidden = self._decoder(dec_input, hidden)
        return out, hidden


# ---------------------------------------------------------------------------
# GRU forecaster
# ---------------------------------------------------------------------------

class WaveForecasterGRU(_WaveForecasterBase):
    """
    Stacked GRU encoder–decoder wave forecaster.

    Identical interface to WaveForecasterLSTM.
    GRU uses a single hidden state (no cell state).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        rnn_dropout = config.dropout if config.num_layers > 1 else 0.0
        self._encoder = nn.GRU(
            input_size=config.n_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self._decoder = nn.GRU(
            input_size=self._decoder_input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

    def _encode(self, x: torch.Tensor):
        _, h_n = self._encoder(x)
        return h_n

    def _decode_step(self, dec_input: torch.Tensor, hidden):
        out, hidden = self._decoder(dec_input, hidden)
        return out, hidden


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    model_type: Literal["lstm", "gru"],
    config: ModelConfig,
) -> _WaveForecasterBase:
    """
    Instantiate a forecaster by name.

    Parameters
    ----------
    model_type : "lstm" or "gru"
    config : ModelConfig

    Returns
    -------
    WaveForecasterLSTM or WaveForecasterGRU
    """
    if model_type == "lstm":
        return WaveForecasterLSTM(config)
    elif model_type == "gru":
        return WaveForecasterGRU(config)
    else:
        raise ValueError(f"model_type must be 'lstm' or 'gru', got '{model_type}'")
