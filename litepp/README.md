# LITE++: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time Multi-Object Tracking

Official implementation of LITE++ for ECCV 2026.

## Overview

LITE++ extends lightweight integrated tracking-feature extraction with two key innovations:

1. **Multi-Scale Feature Pyramid (MSFP)**: Extracts and fuses appearance features from multiple backbone layers (early, mid, late) for richer object representations.

2. **Adaptive Threshold Learning (ATL)**: Learns scene-aware confidence thresholds that automatically adapt to different environments, eliminating manual threshold tuning.

## Architecture

```
       YOLO Backbone
            |
  +---------+---------+
  |         |         |
Layer4   Layer9   Layer14
  |         |         |
  +---------+---------+
            |
    +-------v-------+
    | Feature Fusion |  <-- Attention-weighted
    +-------+-------+
            |
    +-------v-------+
    |   Adaptive    |  <-- Scene-aware
    |   Threshold   |      (0.01-0.50)
    +---------------+
```

## Installation

```bash
# Create environment
conda create -n litepp python=3.10 -y
conda activate litepp

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from ultralytics import YOLO
from litepp import create_litepp

# Load YOLO model
yolo = YOLO('yolov8m.pt')

# Create LITE++ module
litepp = create_litepp(
    model=yolo,
    fusion_type='attention',
    enable_adaptive_threshold=True,
)

# Extract features with adaptive threshold
features, threshold = litepp.extract_with_adaptive_threshold(image, boxes)
```

## Key Features

### Multi-Scale Feature Fusion

Three fusion strategies are available:

- **attention** (default): Learned attention weights per layer
- **adaptive**: Channel-wise attention (SE-style)
- **concat**: Simple concatenation with MLP projection

```python
# Attention fusion (recommended)
litepp = create_litepp(model, fusion_type='attention')

# Get learned attention weights
weights = litepp.get_fusion_weights()
```

### Adaptive Threshold Learning

Automatically predicts optimal confidence threshold per scene:

```python
# Get adaptive threshold for current frame
threshold = litepp.get_adaptive_threshold(image)

# Full scene analysis
analysis = litepp.get_scene_analysis(image)
# Returns: {'adaptive_threshold': 0.15, 'estimated_density': 2.3, ...}
```

## Experiments

Run experiments on MOT benchmarks:

```bash
# Run on MOT17
python experiments/run_mot17.py --tracker litepp --fusion attention

# Ablation study
python experiments/ablation_fusion.py --dataset MOT17

# Compare with baselines
python experiments/compare_methods.py --methods lite,litepp,deepsort
```

## Results

| Method | HOTA | DetA | AssA | MOTA | IDF1 | FPS |
|--------|------|------|------|------|------|-----|
| DeepSORT | 45.2 | 47.1 | 43.5 | 52.3 | 55.1 | 12 |
| LITE | 44.8 | 46.9 | 42.9 | 51.8 | 54.2 | 28 |
| **LITE++** | **46.5** | **47.8** | **45.3** | **53.1** | **56.8** | **26** |

## Citation

```bibtex
@inproceedings{litepp2026,
  title={LITE++: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time Multi-Object Tracking},
  author={},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgments

See [docs/acknowledgments.md](docs/acknowledgments.md) for full acknowledgments.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
