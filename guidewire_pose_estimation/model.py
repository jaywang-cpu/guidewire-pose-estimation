"""
Model architectures for Guidewire Pose Estimation.

Includes:
  1. Direct regression model (ResNet backbone + dual FC heads)
  2. Heatmap-based localization model (ResNet encoder + deconv head + soft-argmax)
  3. Permutation-invariant loss with Hungarian matching for guidewire assignment
  4. Support for multiple backbones (ResNet-18/34/50, EfficientNet-B0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Optional, Tuple
from config import ModelConfig


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def _get_backbone_spatial(name: str, pretrained: bool, in_channels: int):
    """
    Returns (backbone_layers, feature_dim) where backbone_layers is an nn.Sequential
    that outputs a spatial feature map (B, C, H, W) — no GAP, no FC.
    For input_size=384, output is (B, 512, 12, 12) for ResNet-18/34.
    """
    weights_map = {
        "resnet18": ("ResNet18_Weights", models.resnet18, 512),
        "resnet34": ("ResNet34_Weights", models.resnet34, 512),
        "resnet50": ("ResNet50_Weights", models.resnet50, 2048),
    }

    if name in weights_map:
        weights_attr, model_fn, feature_dim = weights_map[name]
        if pretrained:
            weights = getattr(models, weights_attr).IMAGENET1K_V1
            resnet = model_fn(weights=weights)
        else:
            resnet = model_fn(weights=None)

        # Adapt first conv for single-channel input
        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=7, stride=2, padding=3, bias=False,
            )
            if pretrained:
                new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
            resnet.conv1 = new_conv

        # Extract layers up to layer4 (before avgpool/fc)
        backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        return backbone, feature_dim

    else:
        raise ValueError(f"Backbone {name} not supported for spatial mode")


def _build_resnet_encoder_with_skips(name: str, pretrained: bool, in_channels: int) -> Dict:
    """
    Build a ResNet encoder that returns intermediate feature maps for skip connections.
    Returns dict with 'stem', 'layer1' .. 'layer4' modules and their channel counts.
    """
    weights_map = {
        "resnet18": (models.resnet18, "ResNet18_Weights", [64, 64, 128, 256, 512]),
        "resnet34": (models.resnet34, "ResNet34_Weights", [64, 64, 128, 256, 512]),
        "resnet50": (models.resnet50, "ResNet50_Weights", [64, 256, 512, 1024, 2048]),
    }

    if name not in weights_map:
        raise ValueError(f"Heatmap model requires ResNet backbone, got {name}")

    model_fn, weights_name, channels = weights_map[name]

    if pretrained:
        weights = getattr(models, weights_name).IMAGENET1K_V1
        resnet = model_fn(weights=weights)
    else:
        resnet = model_fn(weights=None)

    if in_channels != 3:
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding, bias=False,
        )
        if pretrained:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv

    stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

    return {
        "stem": stem,
        "layer1": resnet.layer1,
        "layer2": resnet.layer2,
        "layer3": resnet.layer3,
        "layer4": resnet.layer4,
        "channels": channels,  # [stem, layer1, layer2, layer3, layer4]
    }


# ---------------------------------------------------------------------------
# 1. Direct Regression Model
# ---------------------------------------------------------------------------

class GuidewireRegressionModel(nn.Module):
    """
    Direct regression with spatial-aware position head:
        Input:  (B, 1, H, W) grayscale fluoroscopy
        Output: positions (B, 2, 2) in [0,1], directions (B, 2, 2) as unit (sin θ, cos θ)

    Position head uses conv layers on spatial feature maps + soft-argmax (preserves where).
    Direction head uses GAP (only needs what orientation).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone, feat_dim = _get_backbone_spatial(
            config.backbone, config.pretrained, config.in_channels
        )

        # Position head: conv on spatial features -> per-wire heatmap -> soft-argmax
        self.position_conv = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, config.num_guidewires, 1),
        )

        # Direction head: GAP + FC (orientation doesn't need precise spatial info)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.direction_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_guidewires * 2),
            nn.Tanh(),
        )

    def _soft_argmax(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Differentiable soft-argmax: (B, K, H, W) -> (B, K, 2) in [0, 1]."""
        B, K, H, W = heatmaps.shape
        weights = F.softmax(heatmaps.view(B, K, -1), dim=-1).view(B, K, H, W)

        device = heatmaps.device
        x_grid = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
        y_grid = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)

        x = (weights * x_grid).sum(dim=(-2, -1))
        y = (weights * y_grid).sum(dim=(-2, -1))

        return torch.stack([x, y], dim=-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)  # (B, C, 12, 12) for input 384

        # Position: spatial conv -> soft-argmax
        heatmaps = self.position_conv(feat)  # (B, K, 12, 12)
        pos = self._soft_argmax(heatmaps)    # (B, K, 2)

        # Direction: GAP -> FC
        pooled = self.gap(feat).flatten(1)   # (B, C)
        dir_raw = self.direction_head(pooled).view(-1, self.config.num_guidewires, 2)
        dir_norm = F.normalize(dir_raw, p=2, dim=-1)

        return {
            "positions": pos,
            "directions": dir_norm,
        }

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# 2. Heatmap-based Model
# ---------------------------------------------------------------------------

class _DecoderBlock(nn.Module):
    """Upsample + concat skip + conv block for the heatmap decoder."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch from rounding
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeatmapGuidewireModel(nn.Module):
    """
    Heatmap-based model with U-Net style skip connections:
        - ResNet encoder with intermediate feature maps
        - Decoder with skip connections for precise spatial localization
        - Soft-argmax extracts differentiable (x, y) from heatmaps
        - Separate FC head predicts direction from pooled features
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Build encoder with skip-connection access
        base = _build_resnet_encoder_with_skips(
            config.backbone, config.pretrained, config.in_channels
        )
        self.stem = base["stem"]
        self.layer1 = base["layer1"]
        self.layer2 = base["layer2"]
        self.layer3 = base["layer3"]
        self.layer4 = base["layer4"]
        ch = base["channels"]  # [stem_ch, l1_ch, l2_ch, l3_ch, l4_ch]

        # Decoder with skip connections (U-Net style)
        self.dec4 = _DecoderBlock(ch[4], ch[3], 256)   # layer4 + layer3
        self.dec3 = _DecoderBlock(256,    ch[2], 128)   # + layer2
        self.dec2 = _DecoderBlock(128,    ch[1], 64)    # + layer1

        # Final heatmap head
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, config.num_guidewires, 1),
        )

        # Direction head (from pooled encoder features)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.direction_head = nn.Sequential(
            nn.Linear(ch[4], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.num_guidewires * 2),
            nn.Tanh(),
        )

        self.heatmap_size = config.heatmap_size

    def _soft_argmax(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Differentiable soft-argmax: extract (x, y) normalized to [0, 1].
        Input: (B, K, H, W)  Output: (B, K, 2)
        """
        B, K, H, W = heatmaps.shape
        flat = heatmaps.view(B, K, -1)
        weights = F.softmax(flat, dim=-1).view(B, K, H, W)

        device = heatmaps.device
        y_grid = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, K, H, W)
        x_grid = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, K, H, W)

        x = (weights * x_grid).sum(dim=(-2, -1))
        y = (weights * y_grid).sum(dim=(-2, -1))

        return torch.stack([x, y], dim=-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encoder with intermediate features for skip connections
        s0 = self.stem(x)      # stride 4
        s1 = self.layer1(s0)   # stride 4
        s2 = self.layer2(s1)   # stride 8
        s3 = self.layer3(s2)   # stride 16
        s4 = self.layer4(s3)   # stride 32

        # Decoder with skips
        d4 = self.dec4(s4, s3)  # stride 16
        d3 = self.dec3(d4, s2)  # stride 8
        d2 = self.dec2(d3, s1)  # stride 4

        heatmaps = self.heatmap_conv(d2)
        heatmaps = F.interpolate(
            heatmaps, size=(self.heatmap_size, self.heatmap_size),
            mode="bilinear", align_corners=False
        )

        positions = self._soft_argmax(heatmaps)

        # Direction from pooled deepest features
        pooled = self.global_pool(s4).flatten(1)
        dir_raw = self.direction_head(pooled).view(-1, self.config.num_guidewires, 2)
        dir_norm = F.normalize(dir_raw, p=2, dim=-1)

        return {
            "positions": positions,
            "directions": dir_norm,
            "heatmaps": heatmaps,
        }

    def freeze_backbone(self):
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = True


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config: ModelConfig) -> nn.Module:
    """Build model based on configuration."""
    if config.use_heatmap_head:
        return HeatmapGuidewireModel(config)
    else:
        return GuidewireRegressionModel(config)


# ---------------------------------------------------------------------------
# Loss functions with Hungarian matching
# ---------------------------------------------------------------------------

class GuidewireLoss(nn.Module):
    """
    Permutation-invariant loss for guidewire pose estimation.

    Since the model has no inherent ordering of guidewire predictions,
    we compute the loss under all possible assignments of predicted wires
    to ground-truth wires and use the assignment that minimizes total cost
    (Hungarian matching). For K=2 wires, this is simply comparing both
    orderings and taking the minimum.

    Components:
        - Position loss: Smooth L1 / MSE / L1 on normalized (x, y)
        - Direction loss: 1 - cosine similarity on (sin θ, cos θ)
        - Optional heatmap loss: MSE between predicted and target Gaussian heatmaps
    """

    def __init__(
        self,
        position_weight: float = 1.0,
        direction_weight: float = 1.0,
        position_loss_fn: str = "smooth_l1",
        direction_loss_fn: str = "cosine",
        heatmap_weight: float = 0.5,
        heatmap_sigma: float = 3.0,
        heatmap_size: int = 64,
    ):
        super().__init__()
        self.position_weight = position_weight
        self.direction_weight = direction_weight
        self.heatmap_weight = heatmap_weight
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size

        if position_loss_fn == "smooth_l1":
            self.pos_loss_fn = nn.SmoothL1Loss(reduction="none")
        elif position_loss_fn == "mse":
            self.pos_loss_fn = nn.MSELoss(reduction="none")
        elif position_loss_fn == "l1":
            self.pos_loss_fn = nn.L1Loss(reduction="none")
        else:
            raise ValueError(f"Unknown position loss: {position_loss_fn}")

        self.direction_loss_type = direction_loss_fn

    def _direction_loss_unreduced(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-wire direction loss (unreduced). Returns (B, K)."""
        if self.direction_loss_type == "cosine":
            cos_sim = F.cosine_similarity(pred, target, dim=-1)  # (B, K)
            return 1 - cos_sim
        else:
            return ((pred - target) ** 2).sum(dim=-1)  # (B, K)

    def _compute_pairwise_cost(
        self,
        pred_pos: torch.Tensor,    # (B, K, 2)
        pred_dir: torch.Tensor,    # (B, K, 2)
        target_pos: torch.Tensor,  # (B, K, 2)
        target_dir: torch.Tensor,  # (B, K, 2)
    ) -> torch.Tensor:
        """
        For K=2, compute the total cost for both possible assignments:
          - Identity: pred[0]->gt[0], pred[1]->gt[1]
          - Swapped:  pred[0]->gt[1], pred[1]->gt[0]
        Returns (B, 2) costs for each assignment.
        """
        B = pred_pos.shape[0]

        # Assignment 1: identity
        pos_cost_id = self.pos_loss_fn(pred_pos, target_pos).sum(dim=-1).sum(dim=-1)  # (B,)
        dir_cost_id = self._direction_loss_unreduced(pred_dir, target_dir).sum(dim=-1)  # (B,)
        cost_id = self.position_weight * pos_cost_id + self.direction_weight * dir_cost_id

        # Assignment 2: swapped
        target_pos_swap = target_pos[:, [1, 0], :]
        target_dir_swap = target_dir[:, [1, 0], :]
        pos_cost_sw = self.pos_loss_fn(pred_pos, target_pos_swap).sum(dim=-1).sum(dim=-1)
        dir_cost_sw = self._direction_loss_unreduced(pred_dir, target_dir_swap).sum(dim=-1)
        cost_sw = self.position_weight * pos_cost_sw + self.direction_weight * dir_cost_sw

        return torch.stack([cost_id, cost_sw], dim=-1)  # (B, 2)

    def _generate_heatmap_targets(
        self, positions: torch.Tensor, size: int
    ) -> torch.Tensor:
        """Generate Gaussian heatmap targets. positions: (B, K, 2) in [0,1]."""
        B, K, _ = positions.shape
        device = positions.device

        y_grid = torch.linspace(0, 1, size, device=device)
        x_grid = torch.linspace(0, 1, size, device=device)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")

        px = positions[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        py = positions[:, :, 1].unsqueeze(-1).unsqueeze(-1)

        sigma = self.heatmap_sigma / size
        heatmaps = torch.exp(
            -((xx.unsqueeze(0).unsqueeze(0) - px) ** 2 +
              (yy.unsqueeze(0).unsqueeze(0) - py) ** 2) / (2 * sigma ** 2)
        )
        return heatmaps

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        pred_pos = predictions["positions"]    # (B, 2, 2)
        pred_dir = predictions["directions"]   # (B, 2, 2)
        tgt_pos = targets["positions"]         # (B, 2, 2)
        tgt_dir = targets["directions"]        # (B, 2, 2)

        # Direct loss — targets are pre-sorted by x-coordinate in the dataset,
        # so wire ordering is deterministic. No Hungarian matching needed.
        pos_loss = self.pos_loss_fn(pred_pos, tgt_pos).mean()
        dir_loss = self._direction_loss_unreduced(pred_dir, tgt_dir).mean()
        total = self.position_weight * pos_loss + self.direction_weight * dir_loss

        loss_dict = {
            "loss": total,
            "pos_loss": pos_loss,
            "dir_loss": dir_loss,
            "swap_rate": torch.tensor(0.0),  # no matching, kept for API compat
        }

        # Heatmap loss
        if "heatmaps" in predictions:
            target_heatmaps = self._generate_heatmap_targets(
                tgt_pos, self.heatmap_size
            )
            hm_loss = F.mse_loss(
                torch.sigmoid(predictions["heatmaps"]), target_heatmaps
            )
            loss_dict["heatmap_loss"] = hm_loss
            loss_dict["loss"] = total + self.heatmap_weight * hm_loss

        return loss_dict
