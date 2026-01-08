"""
LITE++: Multi-Layer Feature Extraction for Multi-Object Tracking

This module extends LITE with multi-scale feature pyramid fusion,
extracting and combining features from multiple backbone layers.

Author: AntVision AI Research
Target: ECCV 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Literal


class FeatureFusionModule(nn.Module):
    """
    Fuses features from multiple layers into a unified representation.

    Supports three fusion strategies:
    - concat: Concatenate and project via MLP
    - attention: Learned attention weights per layer
    - adaptive: Channel-wise attention (SE-style)
    """

    def __init__(
        self,
        layer_channels: List[int],
        output_dim: int = 128,
        fusion_type: Literal["concat", "attention", "adaptive"] = "attention"
    ):
        super().__init__()
        self.layer_channels = layer_channels
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.total_channels = sum(layer_channels)

        if fusion_type == "concat":
            # Simple concatenation + MLP projection
            self.projector = nn.Sequential(
                nn.Linear(self.total_channels, output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim)
            )

        elif fusion_type == "attention":
            # Learned attention weights for each layer
            self.layer_weights = nn.Parameter(torch.ones(len(layer_channels)) / len(layer_channels))
            # Project each layer to same dimension first
            self.layer_projectors = nn.ModuleList([
                nn.Linear(ch, output_dim) for ch in layer_channels
            ])
            self.final_norm = nn.LayerNorm(output_dim)

        elif fusion_type == "adaptive":
            # Channel-wise attention (Squeeze-and-Excitation style)
            self.channel_attention = nn.Sequential(
                nn.Linear(self.total_channels, self.total_channels // 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.total_channels // 4, self.total_channels),
                nn.Sigmoid()
            )
            self.projector = nn.Sequential(
                nn.Linear(self.total_channels, output_dim),
                nn.LayerNorm(output_dim)
            )

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_features: List of tensors, each of shape (num_boxes, layer_channels[i])

        Returns:
            Fused features of shape (num_boxes, output_dim)
        """
        if self.fusion_type == "concat":
            # Concatenate all features
            concatenated = torch.cat(layer_features, dim=-1)
            return self.projector(concatenated)

        elif self.fusion_type == "attention":
            # Project each layer and compute weighted sum
            weights = F.softmax(self.layer_weights, dim=0)
            projected = [proj(feat) for proj, feat in zip(self.layer_projectors, layer_features)]
            weighted_sum = sum(w * feat for w, feat in zip(weights, projected))
            return self.final_norm(weighted_sum)

        elif self.fusion_type == "adaptive":
            # Concatenate, apply channel attention, then project
            concatenated = torch.cat(layer_features, dim=-1)
            attention = self.channel_attention(concatenated)
            attended = concatenated * attention
            return self.projector(attended)


class LITEPlus:
    """
    LITE++: Multi-Layer Feature Extraction for Real-Time MOT

    Extends LITE by extracting appearance features from multiple YOLO backbone
    layers and fusing them for improved discriminability.

    Key innovations:
    1. Multi-scale feature extraction from early/mid/late layers
    2. Learnable feature fusion (attention-weighted or channel-wise)
    3. Efficient spatial pooling with configurable strategies

    Args:
        model: YOLO model instance
        layer_indices: List of layer names to extract from (e.g., ["layer4", "layer9", "layer14"])
        fusion_type: Feature fusion strategy ("concat", "attention", "adaptive")
        output_dim: Final embedding dimension
        spatial_pool: Spatial pooling method ("mean", "max", "adaptive")
        device: Computation device
    """

    # Default layer configurations for different YOLO versions
    LAYER_CONFIGS = {
        "yolov8": {
            "early": "layer4",    # 48 channels, h/2 x w/2
            "mid": "layer9",      # 96 channels, h/4 x w/4
            "late": "layer14",    # 192 channels, h/8 x w/8
        },
        "yolov11": {
            "early": "layer4",
            "mid": "layer9",
            "late": "layer14",
        }
    }

    # Channel counts for each layer (YOLOv8m)
    LAYER_CHANNELS = {
        "layer0": 32,
        "layer1": 64,
        "layer2": 64,
        "layer3": 128,
        "layer4": 48,   # After first conv
        "layer5": 96,
        "layer6": 96,
        "layer7": 96,
        "layer8": 192,
        "layer9": 96,
        "layer10": 192,
        "layer11": 192,
        "layer12": 192,
        "layer13": 384,
        "layer14": 192,
    }

    def __init__(
        self,
        model,
        layer_indices: List[str] = None,
        fusion_type: str = "attention",
        output_dim: int = 128,
        spatial_pool: str = "mean",
        imgsz: int = 1280,
        conf: float = 0.25,
        device: str = 'cuda:0'
    ):
        self.model = model
        self.layer_indices = layer_indices or ["layer4", "layer9", "layer14"]
        self.fusion_type = fusion_type
        self.output_dim = output_dim
        self.spatial_pool = spatial_pool
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

        # Get channel counts for selected layers
        self.layer_channels = [
            self.LAYER_CHANNELS.get(layer, 48) for layer in self.layer_indices
        ]

        # Initialize fusion module if using learned fusion
        if fusion_type in ["attention", "adaptive"]:
            self.fusion_module = FeatureFusionModule(
                layer_channels=self.layer_channels,
                output_dim=output_dim,
                fusion_type=fusion_type
            ).to(device)
            self.fusion_module.eval()
        else:
            self.fusion_module = None

    def _spatial_pool(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Apply spatial pooling to reduce feature map to vector."""
        if self.spatial_pool == "mean":
            return torch.mean(feature_map, dim=(1, 2))
        elif self.spatial_pool == "max":
            return torch.amax(feature_map, dim=(1, 2))
        elif self.spatial_pool == "adaptive":
            # Combination of mean and max
            mean_pool = torch.mean(feature_map, dim=(1, 2))
            max_pool = torch.amax(feature_map, dim=(1, 2))
            return (mean_pool + max_pool) / 2
        else:
            return torch.mean(feature_map, dim=(1, 2))

    def extract_appearance_features(self, image, boxes) -> np.ndarray:
        """
        Extract multi-scale appearance features for detected boxes.

        Args:
            image: Input image (np.ndarray, H x W x C)
            boxes: Bounding boxes (np.ndarray, N x 6) - [x1, y1, x2, y2, conf, cls]

        Returns:
            features: Appearance features (np.ndarray, N x output_dim)
        """
        if len(boxes) == 0:
            return np.array([])

        org_h, org_w = image.shape[:2]

        # Extract feature maps from multiple layers
        # Note: This requires the modified ultralytics that supports multi-layer extraction
        # For now, we extract each layer separately and cache the image encoding

        all_layer_features = []

        for layer_idx, layer_name in enumerate(self.layer_indices):
            # Get feature map for this layer
            results = self.model.predict(
                image,
                classes=[0],
                verbose=False,
                imgsz=self.imgsz,
                appearance_feature_layer=layer_name,
                conf=self.conf,
                return_feature_map=True
            )

            feature_map = results[0].appearance_feature_map  # (C, H, W)
            h_map, w_map = feature_map.shape[1:]

            # Extract features for each box
            box_features = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                # Scale box coordinates to feature map resolution
                fx1 = max(0, int(x1 * w_map / org_w) - 1)
                fx2 = min(w_map, int(x2 * w_map / org_w) + 1)
                fy1 = max(0, int(y1 * h_map / org_h) - 1)
                fy2 = min(h_map, int(y2 * h_map / org_h) + 1)

                # Crop feature map for this box
                cropped = feature_map[:, fy1:fy2, fx1:fx2]  # (C, h, w)

                # Apply spatial pooling
                if cropped.numel() > 0:
                    pooled = self._spatial_pool(cropped)  # (C,)
                else:
                    pooled = torch.zeros(feature_map.shape[0], device=feature_map.device)

                box_features.append(pooled)

            # Stack features for all boxes from this layer
            layer_features = torch.stack(box_features, dim=0)  # (N, C)
            all_layer_features.append(layer_features)

        # Fuse features from all layers
        if self.fusion_type == "concat" and self.fusion_module is None:
            # Simple concatenation without learned projection
            fused = torch.cat(all_layer_features, dim=-1)
        else:
            # Use fusion module
            with torch.no_grad():
                fused = self.fusion_module(all_layer_features)

        # L2 normalize
        fused = F.normalize(fused, p=2, dim=-1)

        return fused.cpu().numpy()

    def extract_appearance_features_efficient(self, image, boxes, cached_features=None):
        """
        Efficient extraction when feature maps are already cached.

        This method is useful when processing the same frame with different boxes
        (e.g., during tracking updates).

        Args:
            image: Input image
            boxes: Bounding boxes
            cached_features: Pre-computed feature maps dict {layer_name: feature_map}

        Returns:
            features: Appearance features
        """
        if len(boxes) == 0:
            return np.array([])

        org_h, org_w = image.shape[:2]

        # Use cached features if available, otherwise extract
        if cached_features is None:
            cached_features = {}
            for layer_name in self.layer_indices:
                results = self.model.predict(
                    image,
                    classes=[0],
                    verbose=False,
                    imgsz=self.imgsz,
                    appearance_feature_layer=layer_name,
                    conf=self.conf,
                    return_feature_map=True
                )
                cached_features[layer_name] = results[0].appearance_feature_map

        all_layer_features = []

        for layer_name in self.layer_indices:
            feature_map = cached_features[layer_name]
            h_map, w_map = feature_map.shape[1:]

            box_features = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                fx1 = max(0, int(x1 * w_map / org_w) - 1)
                fx2 = min(w_map, int(x2 * w_map / org_w) + 1)
                fy1 = max(0, int(y1 * h_map / org_h) - 1)
                fy2 = min(h_map, int(y2 * h_map / org_h) + 1)

                cropped = feature_map[:, fy1:fy2, fx1:fx2]

                if cropped.numel() > 0:
                    pooled = self._spatial_pool(cropped)
                else:
                    pooled = torch.zeros(feature_map.shape[0], device=feature_map.device)

                box_features.append(pooled)

            layer_features = torch.stack(box_features, dim=0)
            all_layer_features.append(layer_features)

        if self.fusion_type == "concat" and self.fusion_module is None:
            fused = torch.cat(all_layer_features, dim=-1)
        else:
            with torch.no_grad():
                fused = self.fusion_module(all_layer_features)

        fused = F.normalize(fused, p=2, dim=-1)

        return fused.cpu().numpy(), cached_features


class LITEPlusPCA(LITEPlus):
    """
    LITE++ with PCA-based feature fusion.

    Uses Principal Component Analysis to reduce dimensionality of
    concatenated multi-layer features. More efficient than learned
    fusion but requires offline PCA fitting.
    """

    def __init__(
        self,
        model,
        layer_indices: List[str] = None,
        output_dim: int = 128,
        pca_model_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            layer_indices=layer_indices,
            fusion_type="concat",
            output_dim=output_dim,
            **kwargs
        )

        self.pca_model = None
        self.pca_mean = None
        self.pca_components = None

        if pca_model_path is not None:
            self.load_pca(pca_model_path)

    def fit_pca(self, features: np.ndarray, save_path: Optional[str] = None):
        """
        Fit PCA on a collection of features.

        Args:
            features: Feature matrix (N_samples, total_channels)
            save_path: Optional path to save PCA model
        """
        from sklearn.decomposition import PCA

        self.pca_model = PCA(n_components=self.output_dim)
        self.pca_model.fit(features)

        self.pca_mean = torch.tensor(self.pca_model.mean_, dtype=torch.float32).to(self.device)
        self.pca_components = torch.tensor(
            self.pca_model.components_.T, dtype=torch.float32
        ).to(self.device)

        if save_path is not None:
            np.savez(
                save_path,
                mean=self.pca_model.mean_,
                components=self.pca_model.components_,
                explained_variance=self.pca_model.explained_variance_ratio_
            )

        print(f"PCA fitted. Explained variance: {sum(self.pca_model.explained_variance_ratio_):.3f}")

    def load_pca(self, path: str):
        """Load pre-fitted PCA model."""
        data = np.load(path)
        self.pca_mean = torch.tensor(data['mean'], dtype=torch.float32).to(self.device)
        self.pca_components = torch.tensor(data['components'].T, dtype=torch.float32).to(self.device)

    def _apply_pca(self, features: torch.Tensor) -> torch.Tensor:
        """Apply PCA transformation."""
        if self.pca_mean is None:
            raise RuntimeError("PCA model not fitted. Call fit_pca() first.")

        centered = features - self.pca_mean
        projected = torch.mm(centered, self.pca_components)
        return projected

    def extract_appearance_features(self, image, boxes) -> np.ndarray:
        """Extract features with PCA reduction."""
        if len(boxes) == 0:
            return np.array([])

        # Get concatenated features from parent class
        # Temporarily disable fusion module
        orig_fusion = self.fusion_module
        self.fusion_module = None

        org_h, org_w = image.shape[:2]
        all_layer_features = []

        for layer_name in self.layer_indices:
            results = self.model.predict(
                image,
                classes=[0],
                verbose=False,
                imgsz=self.imgsz,
                appearance_feature_layer=layer_name,
                conf=self.conf,
                return_feature_map=True
            )

            feature_map = results[0].appearance_feature_map
            h_map, w_map = feature_map.shape[1:]

            box_features = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])

                fx1 = max(0, int(x1 * w_map / org_w) - 1)
                fx2 = min(w_map, int(x2 * w_map / org_w) + 1)
                fy1 = max(0, int(y1 * h_map / org_h) - 1)
                fy2 = min(h_map, int(y2 * h_map / org_h) + 1)

                cropped = feature_map[:, fy1:fy2, fx1:fx2]

                if cropped.numel() > 0:
                    pooled = self._spatial_pool(cropped)
                else:
                    pooled = torch.zeros(feature_map.shape[0], device=feature_map.device)

                box_features.append(pooled)

            layer_features = torch.stack(box_features, dim=0)
            all_layer_features.append(layer_features)

        # Concatenate all layer features
        concatenated = torch.cat(all_layer_features, dim=-1)

        # Apply PCA
        with torch.no_grad():
            projected = self._apply_pca(concatenated)

        # L2 normalize
        projected = F.normalize(projected, p=2, dim=-1)

        self.fusion_module = orig_fusion

        return projected.cpu().numpy()


# Factory function for easy instantiation
def create_lite_plus(
    model,
    variant: str = "attention",
    layers: List[str] = None,
    output_dim: int = 128,
    **kwargs
) -> LITEPlus:
    """
    Factory function to create LITE++ instances.

    Args:
        model: YOLO model
        variant: "attention", "adaptive", "concat", or "pca"
        layers: List of layer names (default: ["layer4", "layer9", "layer14"])
        output_dim: Embedding dimension
        **kwargs: Additional arguments

    Returns:
        LITEPlus instance
    """
    layers = layers or ["layer4", "layer9", "layer14"]

    if variant == "pca":
        return LITEPlusPCA(
            model=model,
            layer_indices=layers,
            output_dim=output_dim,
            **kwargs
        )
    else:
        return LITEPlus(
            model=model,
            layer_indices=layers,
            fusion_type=variant,
            output_dim=output_dim,
            **kwargs
        )
