"""
MSFP-Track: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time MOT

This package implements MSFP-Track, an efficient approach to multi-object tracking
that extracts rich appearance features from multiple backbone layers using RoIAlign
and instance-adaptive attention, and predicts scene-adaptive confidence thresholds.

Key Components:
- Multi-Scale Feature Pyramid (MSFP): Extracts features from YOLOv8 layers 4, 9, 14
- Feature Fusion Module: Instance-adaptive attention-weighted fusion
- Adaptive Threshold Learning (ATL): Scene-aware threshold prediction with EMA smoothing

Paper: ECCV 2026 (Under Review)
Code: https://anonymous.4open.science/r/msfp-track
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
