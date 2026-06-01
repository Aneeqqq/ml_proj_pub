"""Camera blockage model: ResNet-18 (per frame) -> LSTM -> FC head.

Faithful to paper §III-B / ML_Proj_Vault/modalities/camera.md:
  * ResNet-18, ImageNet-pretrained, final FC removed -> 512-d embedding per frame, shared
    across the W frames;
  * single-layer LSTM, hidden 128, take the final hidden state;
  * 2-layer FC head, ReLU, dropout p=0.4;
  * output = K horizon logits (t+1..t+5), one sigmoid per future step (multi-label).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class CameraBlockageModel(nn.Module):
    def __init__(
        self,
        horizon: int = 5,
        lstm_hidden: int = 128,
        fc_hidden: int = 128,
        dropout: float = 0.4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feat_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()              # drop the classification layer
        self.backbone = backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.lstm = nn.LSTM(self.feat_dim, lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, W, 3, H, H) -> logits (B, horizon)."""
        B, W = x.shape[:2]
        x = x.flatten(0, 1)                # (B*W, 3, H, H)
        feats = self.backbone(x)           # (B*W, 512)
        feats = feats.view(B, W, self.feat_dim)
        out, _ = self.lstm(feats)          # (B, W, hidden)
        h_last = out[:, -1, :]             # final time step (B, hidden)
        return self.head(h_last)           # (B, horizon) logits

    def param_groups(self, backbone_lr: float, head_lr: float):
        """Lower LR for the pretrained backbone, higher for LSTM+head."""
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        rest = list(self.lstm.parameters()) + list(self.head.parameters())
        groups = [{"params": rest, "lr": head_lr}]
        if bb:
            groups.append({"params": bb, "lr": backbone_lr})
        return groups
