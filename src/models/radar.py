"""Radar blockage model: per-frame 2D-CNN -> adaptive pool -> LSTM -> FC head.

Faithful to paper §III-E / ML_Proj_Vault/modalities/radar.md:
  * 3x Conv2D (each Conv -> BatchNorm -> ReLU), channels 8 -> 32 -> 64 -> 128, spatial downsampling;
  * adaptive average pooling per frame -> 128-d feature vector;
  * single-layer LSTM, hidden 64, take the final hidden state;
  * 2-layer FC head, ReLU, dropout p=0.3;
  * output = K horizon logits (t+1..t+5).

Input per window: (B, W, 8, 256, 64)  (see src/data/radar_features.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    # stride-2 conv halves H,W each block (256x64 -> 128x32 -> 64x16 -> 32x8)
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class RadarBlockageModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        horizon: int = 5,
        lstm_hidden: int = 64,
        fc_hidden: int = 64,
        dropout: float = 0.3,
        conv_channels: tuple[int, int, int] = (32, 64, 128),
    ):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.cnn = nn.Sequential(
            _conv_block(in_channels, c1),
            _conv_block(c1, c2),
            _conv_block(c2, c3),
            nn.AdaptiveAvgPool2d(1),         # -> (c3, 1, 1)
        )
        self.feat_dim = c3
        self.lstm = nn.LSTM(self.feat_dim, lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, 8, 256, 64) -> logits (B, horizon)."""
        B, W = x.shape[:2]
        x = x.flatten(0, 1)                  # (B*W, 8, 256, 64)
        f = self.cnn(x).flatten(1)           # (B*W, feat_dim)
        f = f.view(B, W, self.feat_dim)      # (B, W, feat_dim)
        out, _ = self.lstm(f)                # (B, W, hidden)
        return self.head(out[:, -1, :])      # (B, horizon) logits

    def param_groups(self, lr: float):
        return [{"params": self.parameters(), "lr": lr}]
