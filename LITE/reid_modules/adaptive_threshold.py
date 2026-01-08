"""
Adaptive Threshold Learning Module for LITE++

This module learns scene-aware confidence thresholds that adapt to different
environments (crowded scenes, varying lighting, different domains).

Key insight: Optimal threshold varies significantly:
- MOT17 (medium density): ~0.25
- MOT20 (high density): ~0.05
- Traffic scenes: ~0.30
- Retail scenes: ~0.15

The module predicts per-scene thresholds based on global scene features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Literal


class SceneEncoder(nn.Module):
    """
    Encodes scene-level features for threshold prediction.

    Extracts global context from backbone features to characterize:
    - Scene density (crowded vs sparse)
    - Lighting conditions
    - Domain characteristics
    """

    def __init__(
        self,
        input_channels: int = 192,
        hidden_dim: int = 128,
        output_dim: int = 64
    ):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.encoder = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )

        # Also extract spatial statistics
        self.spatial_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # 4x4 spatial grid
            nn.Flatten(),
            nn.Linear(input_channels * 16, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) backbone feature map

        Returns:
            scene_encoding: (B, 2*output_dim) scene representation
        """
        # Global average pooling
        global_feat = self.global_pool(features).flatten(1)  # (B, C)
        global_encoding = self.encoder(global_feat)  # (B, output_dim)

        # Spatial statistics
        spatial_encoding = self.spatial_encoder(features)  # (B, output_dim)

        # Concatenate global and spatial features
        scene_encoding = torch.cat([global_encoding, spatial_encoding], dim=1)

        return scene_encoding


class AdaptiveThresholdModule(nn.Module):
    """
    Learns to predict optimal confidence thresholds based on scene characteristics.

    Key features:
    - Scene-aware threshold prediction
    - Bounded output in valid range [min_threshold, max_threshold]
    - Differentiable for end-to-end training
    - Supports multiple threshold types (detection, association, age)
    """

    def __init__(
        self,
        input_channels: int = 192,
        hidden_dim: int = 128,
        min_threshold: float = 0.01,
        max_threshold: float = 0.5,
        default_threshold: float = 0.25,
        num_thresholds: int = 1,
        threshold_names: Optional[List[str]] = None
    ):
        """
        Args:
            input_channels: Number of channels in backbone feature map
            hidden_dim: Hidden dimension for scene encoder
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            default_threshold: Default threshold for initialization
            num_thresholds: Number of thresholds to predict (e.g., det, assoc, age)
            threshold_names: Names for each threshold
        """
        super().__init__()

        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.default_threshold = default_threshold
        self.num_thresholds = num_thresholds
        self.threshold_names = threshold_names or [f'threshold_{i}' for i in range(num_thresholds)]

        # Scene encoder
        self.scene_encoder = SceneEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim // 2
        )

        # Threshold predictor head
        self.threshold_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_thresholds)
        )

        # Initialize to predict default threshold
        self._init_to_default()

    def _init_to_default(self):
        """Initialize to predict default threshold."""
        # Compute the logit that gives default threshold after sigmoid
        target_sigmoid = (self.default_threshold - self.min_threshold) / (
            self.max_threshold - self.min_threshold
        )
        target_logit = np.log(target_sigmoid / (1 - target_sigmoid + 1e-8))

        # Initialize final layer bias
        with torch.no_grad():
            self.threshold_predictor[-1].bias.fill_(target_logit)
            self.threshold_predictor[-1].weight.fill_(0.0)

    def forward(
        self,
        features: torch.Tensor,
        return_encoding: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict scene-adaptive thresholds.

        Args:
            features: (B, C, H, W) backbone feature map
            return_encoding: Whether to return scene encoding

        Returns:
            thresholds: (B, num_thresholds) predicted thresholds
            scene_encoding: Optional scene encoding
        """
        # Encode scene
        scene_encoding = self.scene_encoder(features)  # (B, hidden_dim)

        # Predict thresholds (in logit space)
        logits = self.threshold_predictor(scene_encoding)  # (B, num_thresholds)

        # Apply sigmoid and scale to valid range
        thresholds = torch.sigmoid(logits)
        thresholds = self.min_threshold + (self.max_threshold - self.min_threshold) * thresholds

        if return_encoding:
            return thresholds, scene_encoding
        return thresholds, None

    def get_thresholds_dict(
        self,
        features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get thresholds as a dictionary with named entries.

        Args:
            features: (1, C, H, W) single image features

        Returns:
            Dictionary mapping threshold names to values
        """
        thresholds, _ = self.forward(features)
        thresholds = thresholds.squeeze(0).cpu().numpy()

        return {
            name: float(thresh)
            for name, thresh in zip(self.threshold_names, thresholds)
        }


class AdaptiveThresholdLoss(nn.Module):
    """
    Loss function for training adaptive threshold module.

    Supports multiple training strategies:
    1. Supervised: Learn from optimal thresholds found via grid search
    2. REINFORCE: Learn from tracking performance feedback
    3. Gumbel-Softmax: Differentiable threshold selection
    """

    def __init__(
        self,
        loss_type: Literal['supervised', 'reinforce', 'gumbel'] = 'supervised',
        temperature: float = 1.0,
        entropy_weight: float = 0.01
    ):
        super().__init__()

        self.loss_type = loss_type
        self.temperature = temperature
        self.entropy_weight = entropy_weight

    def supervised_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Supervised loss: MSE between predicted and optimal thresholds.

        Args:
            predicted: (B, num_thresholds) predicted thresholds
            target: (B, num_thresholds) target optimal thresholds
        """
        return F.mse_loss(predicted, target)

    def reinforce_loss(
        self,
        predicted: torch.Tensor,
        rewards: torch.Tensor,
        baseline: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        REINFORCE loss for learning from tracking performance.

        Args:
            predicted: (B, num_thresholds) predicted thresholds
            rewards: (B,) tracking performance (e.g., HOTA)
            baseline: Optional baseline for variance reduction
        """
        if baseline is None:
            baseline = rewards.mean()

        advantage = rewards - baseline

        # Log probability of the predicted threshold
        # Using a Gaussian distribution centered at the prediction
        log_prob = -0.5 * (predicted ** 2).sum(dim=1)

        # REINFORCE gradient
        loss = -(advantage.detach() * log_prob).mean()

        return loss

    def gumbel_loss(
        self,
        logits: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Gumbel-Softmax based loss for differentiable threshold selection.

        Args:
            logits: (B, num_bins) logits over threshold bins
            rewards: (B, num_bins) expected reward for each bin
        """
        # Gumbel-Softmax sampling
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        soft_samples = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)

        # Expected reward
        expected_reward = (soft_samples * rewards).sum(dim=-1)

        # Maximize expected reward (minimize negative)
        loss = -expected_reward.mean()

        # Add entropy regularization for exploration
        entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        loss = loss - self.entropy_weight * entropy.mean()

        return loss

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss based on selected strategy.
        """
        if self.loss_type == 'supervised':
            return self.supervised_loss(predicted, target)
        elif self.loss_type == 'reinforce':
            return self.reinforce_loss(predicted, target, **kwargs)
        elif self.loss_type == 'gumbel':
            return self.gumbel_loss(predicted, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class DensityEstimator(nn.Module):
    """
    Estimates scene density for threshold guidance.

    High density scenes (crowded) need lower thresholds to detect
    partially occluded objects.
    """

    def __init__(self, input_channels: int = 192):
        super().__init__()

        self.density_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus()  # Positive density estimate
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict scene density.

        Returns:
            density: (B, 1) estimated number of objects per unit area
        """
        return self.density_head(features)


class MultiThresholdPredictor(nn.Module):
    """
    Predicts multiple tracking-related thresholds:
    - Detection confidence threshold
    - Association (matching) threshold
    - Track age threshold (frames before deletion)

    Each threshold is adapted based on scene characteristics.
    """

    def __init__(
        self,
        input_channels: int = 192,
        hidden_dim: int = 128
    ):
        super().__init__()

        # Shared scene encoder
        self.scene_encoder = SceneEncoder(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim // 2
        )

        # Detection threshold predictor
        self.det_threshold = AdaptiveThresholdModule(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            min_threshold=0.01,
            max_threshold=0.5,
            default_threshold=0.25,
            num_thresholds=1,
            threshold_names=['detection']
        )

        # Association threshold predictor (cosine distance)
        self.assoc_threshold = AdaptiveThresholdModule(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            min_threshold=0.1,
            max_threshold=0.5,
            default_threshold=0.3,
            num_thresholds=1,
            threshold_names=['association']
        )

        # Max age predictor (in frames)
        self.age_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Positive output
        )

        # Density estimator for guidance
        self.density_estimator = DensityEstimator(input_channels)

    def forward(
        self,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict all tracking thresholds.

        Args:
            features: (B, C, H, W) backbone features

        Returns:
            Dictionary with all threshold predictions
        """
        # Shared scene encoding
        scene_encoding = self.scene_encoder(features)

        # Individual threshold predictions
        det_thresh, _ = self.det_threshold(features)
        assoc_thresh, _ = self.assoc_threshold(features)

        # Max age prediction (scaled to reasonable range 15-90 frames)
        age_raw = self.age_predictor(scene_encoding)
        max_age = 15 + 75 * torch.sigmoid(age_raw)

        # Density estimation
        density = self.density_estimator(features)

        return {
            'detection_threshold': det_thresh.squeeze(-1),
            'association_threshold': assoc_thresh.squeeze(-1),
            'max_age': max_age.squeeze(-1),
            'density': density.squeeze(-1)
        }


class AdaptiveThresholdIntegration:
    """
    Integration helper for using adaptive thresholds with LITE++.

    Wraps the threshold module for easy use in tracking pipeline.
    """

    def __init__(
        self,
        model,  # YOLO model
        threshold_module: Optional[AdaptiveThresholdModule] = None,
        device: str = 'cuda:0',
        feature_layer: str = 'layer14'
    ):
        self.model = model
        self.device = device
        self.feature_layer = feature_layer

        # Layer channel mapping
        self.layer_channels = {
            'layer4': 48,
            'layer9': 96,
            'layer14': 192,
            'layer17': 384,
            'layer20': 576
        }

        # Initialize threshold module if not provided
        if threshold_module is None:
            channels = self.layer_channels.get(feature_layer, 192)
            self.threshold_module = AdaptiveThresholdModule(
                input_channels=channels,
                hidden_dim=128,
                min_threshold=0.01,
                max_threshold=0.5,
                default_threshold=0.25
            ).to(device)
        else:
            self.threshold_module = threshold_module.to(device)

        self.threshold_module.eval()

        # Feature extraction hook
        self._features = None
        self._register_hook()

    def _register_hook(self):
        """Register forward hook to extract features."""
        layer_idx = int(self.feature_layer.replace('layer', ''))

        def hook_fn(module, input, output):
            self._features = output

        # Register hook on the model's backbone
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            self.model.model.model[layer_idx].register_forward_hook(hook_fn)

    def get_adaptive_threshold(self, image: np.ndarray) -> float:
        """
        Get adaptive threshold for a given image.

        Args:
            image: Input image (BGR, HWC)

        Returns:
            Optimal confidence threshold for this scene
        """
        # Run model to get features (without NMS for speed)
        with torch.no_grad():
            _ = self.model.predict(image, verbose=False)

            if self._features is not None:
                features = self._features.unsqueeze(0) if self._features.dim() == 3 else self._features
                threshold, _ = self.threshold_module(features)
                return threshold.item()

        return self.threshold_module.default_threshold

    def get_all_thresholds(self, image: np.ndarray) -> Dict[str, float]:
        """
        Get all adaptive thresholds for a given image.

        Returns dictionary with detection, association, and age thresholds.
        """
        if not isinstance(self.threshold_module, MultiThresholdPredictor):
            return {'detection': self.get_adaptive_threshold(image)}

        with torch.no_grad():
            _ = self.model.predict(image, verbose=False)

            if self._features is not None:
                features = self._features.unsqueeze(0) if self._features.dim() == 3 else self._features
                return self.threshold_module(features)

        return {'detection': self.threshold_module.default_threshold}


def create_adaptive_threshold_module(
    input_channels: int = 192,
    variant: Literal['single', 'multi'] = 'single',
    pretrained_path: Optional[str] = None,
    device: str = 'cuda:0'
) -> nn.Module:
    """
    Factory function to create adaptive threshold module.

    Args:
        input_channels: Number of channels from backbone feature layer
        variant: 'single' for detection threshold only, 'multi' for all thresholds
        pretrained_path: Path to pretrained weights
        device: Device to place module on

    Returns:
        Configured threshold prediction module
    """
    if variant == 'single':
        module = AdaptiveThresholdModule(
            input_channels=input_channels,
            hidden_dim=128,
            min_threshold=0.01,
            max_threshold=0.5,
            default_threshold=0.25,
            num_thresholds=1,
            threshold_names=['detection']
        )
    elif variant == 'multi':
        module = MultiThresholdPredictor(
            input_channels=input_channels,
            hidden_dim=128
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Load pretrained weights if provided
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        module.load_state_dict(state_dict)

    return module.to(device)
