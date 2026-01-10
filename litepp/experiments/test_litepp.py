"""
Test LITE++ Package Installation and Basic Functionality
"""

import sys
import numpy as np

print("Testing LITE++ package...")
print(f"Python version: {sys.version}")

# Test imports
try:
    from litepp import create_litepp, LITEPlusPlus
    from litepp.models import FeatureFusionModule, AdaptiveThresholdModule
    from litepp.trackers import LITEPlusPlusTracker
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Test FeatureFusionModule
try:
    import torch

    # Test fusion module
    layer_channels = [64, 256, 192]
    fusion = FeatureFusionModule(layer_channels, output_dim=128, fusion_type='attention')

    # Create dummy features
    batch_size = 4
    features = [torch.randn(batch_size, ch) for ch in layer_channels]

    # Forward pass
    output = fusion(features)
    assert output.shape == (batch_size, 128), f"Expected shape (4, 128), got {output.shape}"
    print(f"[OK] FeatureFusionModule: input shapes {[f.shape for f in features]} -> output {output.shape}")

    # Test attention weights
    weights = fusion.get_attention_weights()
    if weights is not None:
        print(f"[OK] Attention weights: {weights.numpy()}")

except Exception as e:
    print(f"[ERROR] FeatureFusionModule test failed: {e}")

# Test AdaptiveThresholdModule
try:
    threshold_module = AdaptiveThresholdModule(
        input_channels=192,
        hidden_dim=128,
        min_threshold=0.01,
        max_threshold=0.50,
        default_threshold=0.25
    )

    # Create dummy feature map
    feature_map = torch.randn(1, 192, 20, 20)

    # Forward pass
    threshold, encoding = threshold_module(feature_map, return_encoding=True)
    print(f"[OK] AdaptiveThresholdModule: predicted threshold = {threshold.item():.4f}")

except Exception as e:
    print(f"[ERROR] AdaptiveThresholdModule test failed: {e}")

# Test with YOLO model (if available)
try:
    from ultralytics import YOLO

    print("\nTesting with YOLOv8...")
    model = YOLO('yolov8n.pt')  # Use nano for quick test

    # Create LITE++ module
    litepp = create_litepp(
        model=model,
        fusion_type='attention',
        enable_adaptive_threshold=True,
        device='cpu'  # Use CPU for testing
    )

    print(f"[OK] LITE++ module created")
    print(f"     - Layers: {litepp.layer_indices}")
    print(f"     - Channels: {litepp.layer_channels}")
    print(f"     - Output dim: {litepp.output_dim}")

    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_boxes = np.array([[100, 100, 200, 300, 0.9, 0]])

    features, threshold = litepp.extract_with_adaptive_threshold(dummy_image, dummy_boxes)
    print(f"[OK] Feature extraction: {features.shape}, threshold: {threshold:.4f}")

except ImportError:
    print("[SKIP] ultralytics not available for YOLO test")
except Exception as e:
    print(f"[ERROR] YOLO integration test failed: {e}")

print("\n" + "="*50)
print("LITE++ package test completed!")
print("="*50)
