"""
Multi-Scale Feature Pyramid Module for LITE++

This module implements the Multi-Scale Feature Pyramid (MSFP) component that
extracts appearance features from multiple backbone layers and fuses them
into discriminative representations for multi-object tracking.

Key Contributions:
1. Multi-scale feature extraction from early/mid/late backbone layers
2. Three fusion strategies: concatenation, attention-weighted, channel-adaptive
3. Efficient spatial pooling with minimal computational overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Literal, Dict, Optional


class FeatureFusionModule(nn.Module):
    """
    Fuses features from multiple backbone layers into a unified representation.

    Supports three fusion strategies:
    - concat: Concatenate features and project via MLP
    - attention: Learned attention weights per layer (default)
    - adaptive: Channel-wise attention using Squeeze-and-Excitation

    Args:
        layer_channels: List of channel counts for each input layer
        output_dim: Dimension of the fused output features
        fusion_type: Fusion strategy to use
    """

    def __init__(
        self,
        layer_channels: List[int],
        output_dim: int = 128,
        fusion_type: Literal["concat", "attention", "adaptive"] = "attention",
    ):
        super().__init__()
        self.layer_channels = layer_channels
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.total_channels = sum(layer_channels)

        if fusion_type == "concat":
            self.projector = nn.Sequential(
                nn.Linear(self.total_channels, output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
            )

        elif fusion_type == "attention":
            self.layer_weights = nn.Parameter(
                torch.ones(len(layer_channels)) / len(layer_channels)
            )
            self.layer_projectors = nn.ModuleList(
                [nn.Linear(ch, output_dim) for ch in layer_channels]
            )
            self.final_norm = nn.LayerNorm(output_dim)

        elif fusion_type == "adaptive":
            # Squeeze-and-Excitation style channel attention
            self.channel_attention = nn.Sequential(
                nn.Linear(self.total_channels, self.total_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.total_channels // 4, self.total_channels),
                nn.Sigmoid(),
            )
            self.projector = nn.Sequential(
                nn.Linear(self.total_channels, output_dim),
                nn.LayerNorm(output_dim),
            )

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse features from multiple layers.

        Args:
            layer_features: List of tensors, each (N, C_i) where C_i is layer channels

        Returns:
            Fused features of shape (N, output_dim)
        """
        if self.fusion_type == "concat":
            concatenated = torch.cat(layer_features, dim=-1)
            return self.projector(concatenated)

        elif self.fusion_type == "attention":
            weights = F.softmax(self.layer_weights, dim=0)
            projected = [
                proj(feat) for proj, feat in zip(self.layer_projectors, layer_features)
            ]
            weighted_sum = sum(w * feat for w, feat in zip(weights, projected))
            return self.final_norm(weighted_sum)

        elif self.fusion_type == "adaptive":
            concatenated = torch.cat(layer_features, dim=-1)
            attention = self.channel_attention(concatenated)
            attended = concatenated * attention
            return self.projector(attended)

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return the learned attention weights (for attention fusion only)."""
        if self.fusion_type == "attention":
            return F.softmax(self.layer_weights, dim=0).detach()
        return None


class MultiScaleFeaturePyramid(nn.Module):
    """
    Multi-Scale Feature Pyramid for appearance feature extraction.

    Extracts features from multiple backbone layers of a YOLO detector
    and fuses them for improved object re-identification.

    This approach captures both fine-grained details (early layers)
    and semantic information (late layers).

    Args:
        layer_configs: Dict mapping layer names to channel counts
        fusion_type: Feature fusion strategy
        output_dim: Final embedding dimension
        spatial_pool: Spatial pooling method ("mean", "max", "adaptive")
    """

    # Default layer configurations for different YOLO versions
    YOLO_LAYER_CHANNELS = {
        "yolov8n": {"layer4": 32, "layer9": 64, "layer14": 128},
        "yolov8s": {"layer4": 64, "layer9": 128, "layer14": 256},
        "yolov8m": {"layer4": 96, "layer9": 192, "layer14": 288},
        "yolov8l": {"layer4": 128, "layer9": 256, "layer14": 384},
        "yolov8x": {"layer4": 160, "layer9": 320, "layer14": 480},
    }

    def __init__(
        self,
        layer_configs: Dict[str, int] = None,
        fusion_type: Literal["concat", "attention", "adaptive"] = "attention",
        output_dim: int = 128,
        spatial_pool: Literal["mean", "max", "adaptive"] = "mean",
        yolo_variant: str = "yolov8m",
    ):
        super().__init__()

        # Use provided config or default based on YOLO variant
        if layer_configs is None:
            layer_configs = self.YOLO_LAYER_CHANNELS.get(
                yolo_variant, self.YOLO_LAYER_CHANNELS["yolov8m"]
            )

        self.layer_configs = layer_configs
        self.layer_names = list(layer_configs.keys())
        self.layer_channels = list(layer_configs.values())
        self.spatial_pool = spatial_pool
        self.output_dim = output_dim

        # Feature fusion module
        self.fusion = FeatureFusionModule(
            layer_channels=self.layer_channels,
            output_dim=output_dim,
            fusion_type=fusion_type,
        )

        # Feature maps captured during forward pass
        self._feature_maps: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def _spatial_pool_fn(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Apply spatial pooling to reduce feature map to vector."""
        if self.spatial_pool == "mean":
            return torch.mean(feature_map, dim=(2, 3))
        elif self.spatial_pool == "max":
            return torch.amax(feature_map, dim=(2, 3))
        elif self.spatial_pool == "adaptive":
            mean_pool = torch.mean(feature_map, dim=(2, 3))
            max_pool = torch.amax(feature_map, dim=(2, 3))
            return (mean_pool + max_pool) / 2
        return torch.mean(feature_map, dim=(2, 3))

    def register_hooks(self, model) -> None:
        """
        Register forward hooks on the YOLO model to capture feature maps.

        Args:
            model: YOLO model instance (from ultralytics)
        """
        self._remove_hooks()
        self._feature_maps.clear()

        for layer_name in self.layer_names:
            layer_idx = int(layer_name.replace("layer", ""))

            def make_hook(name):
                def hook_fn(module, input, output):
                    self._feature_maps[name] = output
                return hook_fn

            if hasattr(model, "model") and hasattr(model.model, "model"):
                if layer_idx < len(model.model.model):
                    hook = model.model.model[layer_idx].register_forward_hook(
                        make_hook(layer_name)
                    )
                    self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def extract_roi_features(
        self,
        boxes: np.ndarray,
        image_size: tuple,
    ) -> torch.Tensor:
        """
        Extract ROI features from cached feature maps.

        Args:
            boxes: Detection boxes (N, 4+) [x1, y1, x2, y2, ...]
            image_size: Original image size (H, W)

        Returns:
            Fused features (N, output_dim)
        """
        if len(boxes) == 0:
            return torch.empty(0, self.output_dim)

        h, w = image_size
        device = next(iter(self._feature_maps.values())).device

        all_box_features = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                all_box_features.append(torch.zeros(self.output_dim, device=device))
                continue

            layer_features = []

            for layer_name in self.layer_names:
                feat_map = self._feature_maps[layer_name]
                _, c, fh, fw = feat_map.shape

                # Map box to feature map coordinates
                fx1 = int(x1 * fw / w)
                fy1 = int(y1 * fh / h)
                fx2 = int(x2 * fw / w) + 1
                fy2 = int(y2 * fh / h) + 1

                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(fw, fx2), min(fh, fy2)

                if fx2 <= fx1 or fy2 <= fy1:
                    roi_feat = feat_map.mean(dim=(2, 3))
                else:
                    roi = feat_map[:, :, fy1:fy2, fx1:fx2]
                    roi_feat = self._spatial_pool_fn(roi)

                layer_features.append(roi_feat.squeeze(0))

            # Fuse multi-layer features
            stacked = [f.unsqueeze(0) for f in layer_features]
            fused = self.fusion(stacked).squeeze(0)
            all_box_features.append(fused)

        features = torch.stack(all_box_features)
        features = F.normalize(features, p=2, dim=1)

        return features

    def forward(
        self, feature_maps: Dict[str, torch.Tensor], boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with provided feature maps.

        Args:
            feature_maps: Dict of layer name to feature tensor
            boxes: Detection boxes (N, 4)

        Returns:
            Fused features (N, output_dim)
        """
        self._feature_maps = feature_maps
        # Assume standard image size, should be provided in practice
        return self.extract_roi_features(boxes.cpu().numpy(), (640, 640))


def create_feature_pyramid(
    yolo_variant: str = "yolov8m",
    fusion_type: str = "attention",
    output_dim: int = 128,
    layers: List[str] = None,
) -> MultiScaleFeaturePyramid:
    """
    Factory function to create a MultiScaleFeaturePyramid.

    Args:
        yolo_variant: YOLO model variant (yolov8n, yolov8s, yolov8m, etc.)
        fusion_type: Feature fusion strategy
        output_dim: Output embedding dimension
        layers: Optional custom layer names

    Returns:
        Configured MultiScaleFeaturePyramid instance
    """
    if layers is None:
        layers = ["layer4", "layer9", "layer14"]

    default_channels = MultiScaleFeaturePyramid.YOLO_LAYER_CHANNELS.get(
        yolo_variant, MultiScaleFeaturePyramid.YOLO_LAYER_CHANNELS["yolov8m"]
    )

    layer_configs = {layer: default_channels.get(layer, 128) for layer in layers}

    return MultiScaleFeaturePyramid(
        layer_configs=layer_configs,
        fusion_type=fusion_type,
        output_dim=output_dim,
        yolo_variant=yolo_variant,
    )
