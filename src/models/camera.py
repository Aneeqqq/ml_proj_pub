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
        self.freeze_backbone = freeze_backbone
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

    def train(self, mode: bool = True):
        """Keep a frozen backbone in eval mode so its BatchNorm running stats don't drift."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

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


class R2Plus1DBlockage(nn.Module):
    """R(2+1)D-18 video network (Kinetics-400 pretrained) for blockage prediction.

    Why (see vault plan/strategy-reassessment.md): the blockage cue is *motion* of vehicles toward
    the LOS corridor — invisible in any single frame. R(2+1)D's factorized 3D convolutions come
    pretrained on video, so spatiotemporal (motion) features are built in, unlike per-frame
    ResNet+LSTM where temporal reasoning must be learned from our small dataset.

    Input: (B, W, 3, H, H) with H=112 (native Kinetics resolution) -> logits (B, horizon).
    Normalization should use Kinetics stats (see KINETICS_MEAN/STD).
    """

    def __init__(self, horizon: int = 1, dropout: float = 0.3, pretrained: bool = True):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        net = r2plus1d_18(weights=weights)
        feat_dim = net.fc.in_features            # 512
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, horizon))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)             # (B,W,3,H,W) -> (B,3,T,H,W)
        return self.head(self.backbone(x))

    def param_groups(self, backbone_lr: float, head_lr: float):
        return [{"params": self.head.parameters(), "lr": head_lr},
                {"params": self.backbone.parameters(), "lr": backbone_lr}]


KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD = (0.22803, 0.22145, 0.216989)


def build_camera_model(mcfg: dict, smoke: bool = False) -> nn.Module:
    """Factory: arch = resnet18_lstm (paper-faithful) | r2plus1d_18 (video, Kinetics)."""
    arch = mcfg.get("arch", "resnet18_lstm")
    pretrained = mcfg.get("pretrained", True) and not smoke
    if arch == "r2plus1d_18":
        return R2Plus1DBlockage(horizon=mcfg["horizon"], dropout=mcfg.get("dropout", 0.3),
                                pretrained=pretrained)
    return CameraBlockageModel(
        horizon=mcfg["horizon"], lstm_hidden=mcfg["lstm_hidden"], fc_hidden=mcfg["fc_hidden"],
        dropout=mcfg["dropout"], pretrained=pretrained,
        freeze_backbone=mcfg.get("freeze_backbone", False),
    )
