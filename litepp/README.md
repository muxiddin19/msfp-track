# MSFP-Track: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time Multi-Object Tracking

Official implementation of MSFP-Track for ECCV 2026.

## Overview

MSFP-Track extends lightweight tracking-feature extraction with two key innovations:

1. **Multi-Scale Feature Pyramid (MSFP)**: Extracts and fuses appearance features from multiple backbone layers (Layer 4, 9, 14) using RoIAlign and instance-adaptive attention for richer object representations.

2. **Adaptive Threshold Learning (ATL)**: Learns scene-aware confidence thresholds with EMA temporal smoothing that automatically adapt to different environments, eliminating manual threshold tuning.

## Architecture

```
       YOLOv8/v11 Backbone
            |
  +---------+---------+
  |         |         |
Layer4   Layer9   Layer14
(64ch)   (256ch)  (192ch)
  |         |         |
  +----RoIAlign-------+
            |
    +-------v-------+
    | Instance-Adaptive |  <-- Attention-weighted fusion
    |     Attention     |
    +-------+-------+
            |
    +-------v-------+
    |   Adaptive    |  <-- Scene-aware threshold
    |   Threshold   |      (0.01-0.50) + EMA
    +---------------+
```

## Installation

```bash
# Create environment
conda create -n msfptrack python=3.10 -y
conda activate msfptrack

# Install from root directory
cd lite
pip install -e .

# Or install dependencies only
pip install -r litepp/requirements.txt
```

## Quick Start

```python
from ultralytics import YOLO
from litepp import create_litepp

# Load YOLO model
yolo = YOLO('yolov8m.pt')

# Create MSFP-Track module
msfp_track = create_litepp(
    model=yolo,
    fusion_type='attention',
    enable_adaptive_threshold=True,
)

# Extract features with adaptive threshold
features, threshold = msfp_track.extract_with_adaptive_threshold(image, boxes)
```

## Key Features

### Multi-Scale Feature Fusion

Three fusion strategies are available:

- **attention** (default): Instance-adaptive attention weights per detection
- **adaptive**: Channel-wise attention (SE-style)
- **concat**: Simple concatenation with MLP projection

```python
# Attention fusion (recommended)
msfp = create_litepp(model, fusion_type='attention')

# Get learned attention weights
weights = msfp.get_fusion_weights()
# Returns: tensor([0.35, 0.40, 0.25]) for [Layer4, Layer9, Layer14]
```

### Adaptive Threshold Learning

Automatically predicts optimal confidence threshold per scene with temporal smoothing:

```python
# Get adaptive threshold for current frame
threshold = msfp.get_adaptive_threshold(image)

# Threshold is EMA-smoothed for stability
# tau_t = 0.9 * tau_{t-1} + 0.1 * tau_raw
```

## Results

### MOT17 Test Set (Public Detections)

| Method | HOTA | DetA | AssA | IDF1 | IDSW | FPS |
|--------|------|------|------|------|------|-----|
| DeepSORT | 45.6 | 45.8 | 45.5 | 57.1 | 2008 | 13.7 |
| ByteTrack | 54.8 | 57.9 | 51.9 | 66.3 | 2196 | 29.7 |
| LITE (baseline) | 61.1 | 61.5 | 60.8 | 73.2 | 1876 | 28.3 |
| **MSFP-Track** | **63.2** | **63.4** | **63.0** | **75.8** | **1512** | 26.1 |

### ReID Feature Quality

| Method | AUC | Pos-Neg Gap |
|--------|-----|-------------|
| Single Layer | 0.941 | 0.069 |
| MSFP (concat) | 0.959 | 0.107 |
| MSFP (attention) | 0.962 | 0.112 |

## Experiments

Run experiments on MOT benchmarks:

```bash
# Run on MOT17
python litepp/experiments/run_mot17.py --tracker msfptrack --fusion attention

# Ablation study on fusion strategies
python litepp/experiments/ablation_fusion.py --dataset MOT17

# Generate paper figures
python litepp/experiments/generate_paper_visualizations.py
```

## Citation

```bibtex
@inproceedings{msfptrack2026,
  title={MSFP-Track: Multi-Scale Feature Pyramid with Adaptive Thresholds for Real-Time Multi-Object Tracking},
  author={Anonymous},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgments

This work builds upon:
- [LITE](https://arxiv.org/abs/2409.04187) paradigm for lightweight feature extraction
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [DeepSORT](https://github.com/nwojke/deep_sort) for tracking framework

See [docs/acknowledgments.md](docs/acknowledgments.md) for full acknowledgments.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
