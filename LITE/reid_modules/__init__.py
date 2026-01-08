# Core LITE modules (always available)
from .lite import LITE
from .lite_plus import LITEPlus, LITEPlusPCA, create_lite_plus, FeatureFusionModule
from .adaptive_threshold import (
    AdaptiveThresholdModule,
    MultiThresholdPredictor,
    AdaptiveThresholdLoss,
    AdaptiveThresholdIntegration,
    create_adaptive_threshold_module
)
from .lite_plus_unified import (
    LITEPlusPlusUnified,
    LITEPlusPlusTracker,
    LITEPlusPlus,
    create_lite_plus_plus
)

# Optional modules with external dependencies
# These are lazily imported to avoid import errors when dependencies are missing
def _lazy_import(name):
    """Lazy import helper for optional modules."""
    import importlib
    try:
        module = importlib.import_module(f'.{name}', package='reid_modules')
        return getattr(module, name.title().replace('_', ''))
    except (ImportError, AttributeError):
        return None

# Try to import modules with optional dependencies
try:
    from .strongsort import StrongSORT
except ImportError:
    StrongSORT = None  # fastreid not available

try:
    from .deepsort import DeepSORT
except ImportError:
    DeepSORT = None

try:
    from .osnet import OSNet
except ImportError:
    OSNet = None

try:
    from .gfn import GFN
except ImportError:
    GFN = None

__all__ = [
    # Core LITE modules (always available)
    "LITE",
    # LITE+ (multi-layer)
    "LITEPlus",
    "LITEPlusPCA",
    "create_lite_plus",
    "FeatureFusionModule",
    # LITE++ (unified with adaptive thresholds)
    "LITEPlusPlusUnified",
    "LITEPlusPlusTracker",
    "LITEPlusPlus",
    "create_lite_plus_plus",
    # Adaptive threshold module
    "AdaptiveThresholdModule",
    "MultiThresholdPredictor",
    "AdaptiveThresholdLoss",
    "AdaptiveThresholdIntegration",
    "create_adaptive_threshold_module",
    # Optional modules (may be None if dependencies missing)
    "StrongSORT",
    "DeepSORT",
    "OSNet",
    "GFN",
]