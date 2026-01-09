"""
LITE++: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time MOT

This package implements LITE++, an efficient approach to multi-object tracking
that extracts rich appearance features from multiple backbone layers and
predicts scene-adaptive confidence thresholds.

Key Components:
- Multi-Scale Feature Pyramid (MSFP): Extracts features from multiple YOLO layers
- Feature Fusion Module: Attention-weighted fusion of multi-scale features
- Adaptive Threshold Learning (ATL): Scene-aware confidence threshold prediction

For research inquiries, please contact the authors.
"""

__version__ = "1.0.0"
__author__ = "AntVision AI Research"

from .models.feature_pyramid import FeatureFusionModule, MultiScaleFeaturePyramid
from .models.adaptive_threshold import (
    AdaptiveThresholdModule,
    SceneEncoder,
    MultiThresholdPredictor,
)
from .models.litepp import LITEPlusPlus, create_litepp
from .trackers.litepp_tracker import LITEPlusPlusTracker

__all__ = [
    "FeatureFusionModule",
    "MultiScaleFeaturePyramid",
    "AdaptiveThresholdModule",
    "SceneEncoder",
    "MultiThresholdPredictor",
    "LITEPlusPlus",
    "LITEPlusPlusTracker",
    "create_litepp",
]
