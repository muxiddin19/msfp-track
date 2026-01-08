"""
LITE++ Unified Module

Combines all LITE++ innovations in a single unified interface:
1. Multi-layer feature extraction (MSFP)
2. Adaptive feature fusion (attention, concat, adaptive)
3. Scene-adaptive confidence thresholds
4. Domain-aware processing (optional)

This is the primary module for ECCV 2026 submission.

Usage:
    from reid_modules import LITEPlusPlusUnified, create_lite_plus_plus

    # Create unified LITE++ module
    reid_model = create_lite_plus_plus(
        model=yolo_model,
        fusion_type='attention',
        adaptive_threshold=True,
        device='cuda:0'
    )

    # Extract features with adaptive threshold
    features, threshold = reid_model.extract_with_adaptive_threshold(image, boxes)
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Literal, Union
from pathlib import Path

from .lite_plus import FeatureFusionModule, LITEPlus
from .adaptive_threshold import (
    AdaptiveThresholdModule,
    MultiThresholdPredictor,
    create_adaptive_threshold_module
)


class LITEPlusPlusUnified(nn.Module):
    """
    Unified LITE++ module combining multi-layer features and adaptive thresholds.

    Key innovations:
    1. Multi-Scale Feature Pyramid (MSFP): Extracts from layers 4, 9, 14
    2. Adaptive Feature Fusion: Learned attention weights across scales
    3. Scene-Adaptive Thresholds: Predicts optimal confidence per scene
    4. Optional domain adaptation support
    """

    # Default layer configurations (will be auto-detected if possible)
    # These are fallback values for YOLOv8m
    LAYER_CONFIGS = {
        'yolov8': {
            'layer4': {'idx': 4, 'channels': 48},
            'layer9': {'idx': 9, 'channels': 96},
            'layer14': {'idx': 14, 'channels': 192},
            'layer17': {'idx': 17, 'channels': 384},
            'layer20': {'idx': 20, 'channels': 576},
        },
        'yolov8n': {  # YOLOv8 nano
            'layer4': {'idx': 4, 'channels': 64},
            'layer9': {'idx': 9, 'channels': 256},
            'layer14': {'idx': 14, 'channels': 192},
            'layer17': {'idx': 17, 'channels': 192},
            'layer20': {'idx': 20, 'channels': 384},
        },
        'yolov8s': {  # YOLOv8 small
            'layer4': {'idx': 4, 'channels': 64},
            'layer9': {'idx': 9, 'channels': 256},
            'layer14': {'idx': 14, 'channels': 256},
            'layer17': {'idx': 17, 'channels': 256},
            'layer20': {'idx': 20, 'channels': 512},
        },
        'yolov8m': {  # YOLOv8 medium
            'layer4': {'idx': 4, 'channels': 96},
            'layer9': {'idx': 9, 'channels': 384},
            'layer14': {'idx': 14, 'channels': 288},
            'layer17': {'idx': 17, 'channels': 288},
            'layer20': {'idx': 20, 'channels': 576},
        },
        'yolov11': {  # YOLOv11 may have different structure
            'layer4': {'idx': 4, 'channels': 64},
            'layer9': {'idx': 9, 'channels': 128},
            'layer14': {'idx': 14, 'channels': 256},
        },
        'auto': {}  # Will be auto-detected
    }

    @staticmethod
    def _detect_layer_channels(model, layer_indices: List[str], device: str = 'cpu') -> Dict[str, Dict]:
        """Auto-detect channel sizes by running a forward pass with hooks."""
        detected_config = {}
        feature_shapes = {}

        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    feature_shapes[name] = output.shape
                elif isinstance(output, tuple) and len(output) > 0:
                    if isinstance(output[0], torch.Tensor):
                        feature_shapes[name] = output[0].shape
            return hook_fn

        # Register hooks for requested layers
        hooks = []
        for layer_name in layer_indices:
            layer_idx = int(layer_name.replace('layer', ''))
            if hasattr(model, 'model') and hasattr(model.model, 'model'):
                if layer_idx < len(model.model.model):
                    hook = model.model.model[layer_idx].register_forward_hook(make_hook(layer_name))
                    hooks.append(hook)

        # Run forward pass with dummy input
        try:
            dummy_input = torch.randn(1, 3, 640, 640).to(device)
            with torch.no_grad():
                model.model.to(device)
                _ = model.model(dummy_input)
        except Exception:
            pass

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Build configuration from detected shapes
        for layer_name in layer_indices:
            layer_idx = int(layer_name.replace('layer', ''))
            if layer_name in feature_shapes:
                channels = feature_shapes[layer_name][1]
                detected_config[layer_name] = {'idx': layer_idx, 'channels': channels}
            else:
                # Fallback to default
                detected_config[layer_name] = {'idx': layer_idx, 'channels': 128}

        return detected_config

    def __init__(
        self,
        model,
        layer_indices: List[str] = None,
        fusion_type: Literal['concat', 'attention', 'adaptive'] = 'attention',
        output_dim: int = 128,
        enable_adaptive_threshold: bool = True,
        threshold_module_path: Optional[str] = None,
        device: str = 'cuda:0',
        yolo_version: str = 'yolov8'
    ):
        """
        Args:
            model: YOLO detection model
            layer_indices: Which layers to extract features from
            fusion_type: How to fuse multi-scale features
            output_dim: Dimension of output feature vectors
            enable_adaptive_threshold: Whether to use adaptive thresholds
            threshold_module_path: Path to pretrained threshold module
            device: Device to run on
            yolo_version: YOLO version for correct layer mapping
        """
        super().__init__()

        self.model = model
        self.device = device
        self.yolo_version = yolo_version
        self.enable_adaptive_threshold = enable_adaptive_threshold

        # Default layers for multi-scale extraction
        if layer_indices is None:
            layer_indices = ['layer4', 'layer9', 'layer14']
        self.layer_indices = layer_indices

        # Get layer configurations - auto-detect if version not found or set to 'auto'
        if yolo_version == 'auto' or yolo_version not in self.LAYER_CONFIGS:
            # Auto-detect channel sizes from the model
            self.layer_config = self._detect_layer_channels(model, layer_indices, device)
        else:
            self.layer_config = self.LAYER_CONFIGS.get(yolo_version, self.LAYER_CONFIGS['yolov8'])

        # Get channel counts for selected layers
        self.layer_channels = []
        for layer_name in layer_indices:
            if layer_name in self.layer_config:
                self.layer_channels.append(self.layer_config[layer_name]['channels'])
            else:
                # Try auto-detection for missing layer
                detected = self._detect_layer_channels(model, [layer_name], device)
                if layer_name in detected:
                    self.layer_config[layer_name] = detected[layer_name]
                    self.layer_channels.append(detected[layer_name]['channels'])
                else:
                    raise ValueError(f"Unknown layer: {layer_name}")

        # Feature fusion module
        self.fusion_module = FeatureFusionModule(
            layer_channels=self.layer_channels,
            output_dim=output_dim,
            fusion_type=fusion_type
        ).to(device)

        # Adaptive threshold module
        if enable_adaptive_threshold:
            # Use features from the deepest layer for threshold prediction
            deepest_layer = layer_indices[-1]
            threshold_channels = self.layer_config[deepest_layer]['channels']

            self.threshold_module = create_adaptive_threshold_module(
                input_channels=threshold_channels,
                variant='single',
                pretrained_path=threshold_module_path,
                device=device
            )
        else:
            self.threshold_module = None

        # Register hooks for feature extraction
        self._multi_layer_features = {}
        self._register_hooks()

        # Output dimension
        self.output_dim = output_dim

    def _register_hooks(self):
        """Register forward hooks to capture multi-layer features."""
        self._hook_handles = []

        for layer_name in self.layer_indices:
            layer_idx = self.layer_config[layer_name]['idx']

            def make_hook(name):
                def hook_fn(module, input, output):
                    self._multi_layer_features[name] = output
                return hook_fn

            # Register hook on the backbone
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
                handle = self.model.model.model[layer_idx].register_forward_hook(
                    make_hook(layer_name)
                )
                self._hook_handles.append(handle)

    def _clear_features(self):
        """Clear cached features."""
        self._multi_layer_features.clear()

    def extract_appearance_features(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> np.ndarray:
        """
        Extract multi-scale fused appearance features for detected boxes.

        Args:
            image: Input image (BGR, HWC)
            boxes: Detection boxes (N, 6) [x1, y1, x2, y2, conf, cls]

        Returns:
            features: (N, output_dim) appearance feature vectors
        """
        if len(boxes) == 0:
            return np.array([])

        # Clear previous features
        self._clear_features()

        # Run detection to populate feature maps
        with torch.no_grad():
            _ = self.model.predict(image, verbose=False)

        # Check if we captured all required layers
        if len(self._multi_layer_features) != len(self.layer_indices):
            print(f"Warning: Expected {len(self.layer_indices)} layers, "
                  f"got {len(self._multi_layer_features)}")
            return np.zeros((len(boxes), self.output_dim))

        # Get image dimensions for coordinate mapping
        h, w = image.shape[:2]

        # Collect features from each layer for each box
        all_box_features = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                all_box_features.append(torch.zeros(self.output_dim, device=self.device))
                continue

            # Extract features from each layer
            layer_features = []

            for layer_name in self.layer_indices:
                feat_map = self._multi_layer_features[layer_name]

                # Move feature map to the correct device
                feat_map = feat_map.to(self.device)

                # Get feature map dimensions
                _, c, fh, fw = feat_map.shape

                # Map box coordinates to feature map coordinates
                fx1 = int(x1 * fw / w)
                fy1 = int(y1 * fh / h)
                fx2 = int(x2 * fw / w) + 1
                fy2 = int(y2 * fh / h) + 1

                # Clamp to feature map bounds
                fx1, fy1 = max(0, fx1), max(0, fy1)
                fx2, fy2 = min(fw, fx2), min(fh, fy2)

                if fx2 <= fx1 or fy2 <= fy1:
                    # Use global average if box is too small
                    roi_feat = feat_map.mean(dim=(2, 3))  # (1, C)
                else:
                    # Extract ROI features with average pooling
                    roi = feat_map[:, :, fy1:fy2, fx1:fx2]
                    roi_feat = roi.mean(dim=(2, 3))  # (1, C)

                layer_features.append(roi_feat)

            # Fuse multi-layer features
            fused = self.fusion_module(layer_features)  # (1, output_dim)
            all_box_features.append(fused.squeeze(0))

        # Stack all box features
        features = torch.stack(all_box_features)  # (N, output_dim)

        # L2 normalize
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.detach().cpu().numpy()

    def extract_with_adaptive_threshold(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Extract features and predict optimal confidence threshold.

        Args:
            image: Input image
            boxes: Detection boxes

        Returns:
            features: Appearance feature vectors
            threshold: Predicted optimal confidence threshold
        """
        features = self.extract_appearance_features(image, boxes)

        threshold = 0.25  # Default
        if self.enable_adaptive_threshold and self.threshold_module is not None:
            # Get features from deepest layer for threshold prediction
            deepest_layer = self.layer_indices[-1]
            if deepest_layer in self._multi_layer_features:
                with torch.no_grad():
                    feat_map = self._multi_layer_features[deepest_layer]
                    thresh_pred, _ = self.threshold_module(feat_map)
                    threshold = thresh_pred.item()

        return features, threshold

    def get_adaptive_threshold(self, image: np.ndarray) -> float:
        """
        Get adaptive threshold for an image without extracting box features.
        """
        self._clear_features()

        with torch.no_grad():
            _ = self.model.predict(image, verbose=False)

        if not self.enable_adaptive_threshold or self.threshold_module is None:
            return 0.25

        deepest_layer = self.layer_indices[-1]
        if deepest_layer in self._multi_layer_features:
            feat_map = self._multi_layer_features[deepest_layer]
            thresh_pred, _ = self.threshold_module(feat_map)
            return thresh_pred.item()

        return 0.25

    def get_scene_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """
        Get comprehensive scene analysis including threshold and other metrics.
        """
        self._clear_features()

        with torch.no_grad():
            _ = self.model.predict(image, verbose=False)

        analysis = {
            'adaptive_threshold': 0.25,
            'estimated_density': 0.0,
        }

        if self.enable_adaptive_threshold and self.threshold_module is not None:
            deepest_layer = self.layer_indices[-1]
            if deepest_layer in self._multi_layer_features:
                feat_map = self._multi_layer_features[deepest_layer]

                if isinstance(self.threshold_module, MultiThresholdPredictor):
                    results = self.threshold_module(feat_map)
                    analysis['adaptive_threshold'] = results['detection_threshold'].item()
                    analysis['association_threshold'] = results['association_threshold'].item()
                    analysis['max_age'] = results['max_age'].item()
                    analysis['estimated_density'] = results['density'].item()
                else:
                    thresh_pred, _ = self.threshold_module(feat_map)
                    analysis['adaptive_threshold'] = thresh_pred.item()

        return analysis

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training.

        Args:
            image: (B, C, H, W) batch of images
            boxes: List of (N, 4) boxes per image

        Returns:
            features: Extracted features
            thresholds: Predicted thresholds (if enabled)
        """
        raise NotImplementedError(
            "Use extract_appearance_features() for inference. "
            "Training forward pass not yet implemented."
        )


class LITEPlusPlusTracker:
    """
    Complete LITE++ tracking wrapper that integrates with DeepSORT/ByteTrack.

    Provides a drop-in replacement for standard ReID modules with:
    - Multi-scale features
    - Adaptive thresholds
    - Optional domain-specific processing
    """

    def __init__(
        self,
        model,
        tracker_type: Literal['deepsort', 'bytetrack', 'ocsort'] = 'deepsort',
        fusion_type: str = 'attention',
        enable_adaptive_threshold: bool = True,
        threshold_module_path: Optional[str] = None,
        device: str = 'cuda:0'
    ):
        self.model = model
        self.tracker_type = tracker_type
        self.device = device

        # Initialize LITE++ module
        self.reid_module = LITEPlusPlusUnified(
            model=model,
            fusion_type=fusion_type,
            enable_adaptive_threshold=enable_adaptive_threshold,
            threshold_module_path=threshold_module_path,
            device=device
        )

        # Current adaptive threshold
        self.current_threshold = 0.25

    def extract_appearance_features(
        self,
        image: np.ndarray,
        boxes: np.ndarray
    ) -> np.ndarray:
        """Extract features (standard interface)."""
        features, threshold = self.reid_module.extract_with_adaptive_threshold(
            image, boxes
        )
        self.current_threshold = threshold
        return features

    def get_current_threshold(self) -> float:
        """Get the most recently predicted threshold."""
        return self.current_threshold

    def update_threshold(self, image: np.ndarray):
        """Update threshold based on current scene."""
        self.current_threshold = self.reid_module.get_adaptive_threshold(image)


def create_lite_plus_plus(
    model,
    fusion_type: Literal['concat', 'attention', 'adaptive'] = 'attention',
    layers: Optional[List[str]] = None,
    output_dim: int = 128,
    enable_adaptive_threshold: bool = True,
    threshold_module_path: Optional[str] = None,
    device: str = 'cuda:0',
    return_tracker: bool = False,
    tracker_type: str = 'deepsort'
) -> Union[LITEPlusPlusUnified, LITEPlusPlusTracker]:
    """
    Factory function to create LITE++ module.

    Args:
        model: YOLO detection model
        fusion_type: Feature fusion strategy
        layers: Backbone layers to use
        output_dim: Output feature dimension
        enable_adaptive_threshold: Use adaptive thresholds
        threshold_module_path: Path to pretrained threshold module
        device: Computing device
        return_tracker: Return full tracker wrapper
        tracker_type: Type of tracker if return_tracker=True

    Returns:
        LITE++ module or tracker wrapper
    """
    if layers is None:
        layers = ['layer4', 'layer9', 'layer14']

    lite_plus_plus = LITEPlusPlusUnified(
        model=model,
        layer_indices=layers,
        fusion_type=fusion_type,
        output_dim=output_dim,
        enable_adaptive_threshold=enable_adaptive_threshold,
        threshold_module_path=threshold_module_path,
        device=device
    )

    if return_tracker:
        return LITEPlusPlusTracker(
            model=model,
            tracker_type=tracker_type,
            fusion_type=fusion_type,
            enable_adaptive_threshold=enable_adaptive_threshold,
            threshold_module_path=threshold_module_path,
            device=device
        )

    return lite_plus_plus


# Convenience aliases
LITEPlusPlus = LITEPlusPlusUnified
