# LITE++: Multi-Scale Feature Fusion with Adaptive Thresholds for Real-Time Multi-Object Tracking

This repository contains the official implementation of **LITE++**, an extension of the LITE paradigm for real-time multi-object tracking with multi-scale feature fusion and adaptive threshold learning.

## Key Features

- **Multi-Scale Feature Pyramid Fusion (MSFP)**: Extracts and combines features from multiple YOLO backbone layers (layer 4, 9, 14) for richer appearance representations.

- **Adaptive Threshold Learning (ATL)**: Scene-aware confidence threshold prediction that eliminates manual threshold tuning across different datasets.

- **Three Fusion Strategies**: Concatenation, attention-weighted, and channel-adaptive (SE-style) fusion with comprehensive ablation analysis.

- **Real-Time Performance**: Maintains real-time tracking speed while improving association accuracy.

## Architecture

```
       YOLOv8/v11 Backbone
              │
  ┌───────────┼───────────┐
  │           │           │
Layer 4   Layer 9     Layer 14
(64ch)    (256ch)     (192ch)
  │           │           │
  └───────────┼───────────┘
              │
      ┌───────▼───────┐
      │FeatureFusion  │ ← concat/attention/adaptive
      │   Module      │
      └───────┬───────┘
              │
      ┌───────▼───────┐
      │   Adaptive    │ ← Scene-aware threshold
      │   Threshold   │   (0.01-0.50)
      └───────────────┘
```

## Installation

```bash
# Create conda environment
conda create -n lite_pp python=3.10 -y
conda activate lite_pp

# Install dependencies
cd LITE
pip install -r requirements.txt

# Install ultralytics
pip install ultralytics
```

## Quick Start

```python
from ultralytics import YOLO
from reid_modules import create_lite_plus_plus

# Load YOLO model
model = YOLO('yolov8m.pt')

# Create LITE++ module
reid_model = create_lite_plus_plus(
    model=model,
    fusion_type='attention',           # or 'concat', 'adaptive'
    layers=['layer4', 'layer9', 'layer14'],
    output_dim=128,
    enable_adaptive_threshold=True,
    device='cuda:0'
)

# Extract features with adaptive threshold
features, threshold = reid_model.extract_with_adaptive_threshold(image, boxes)
```

## Running Experiments

```bash
# Compare LITE variants
python experiments/compare_lite_variants.py --dataset MOT17 --seq_name MOT17-02-FRCNN

# Run full ECCV 2026 experiments
python experiments/run_eccv_experiments.py --experiment all --dataset MOT17

# Train adaptive threshold module
python experiments/train_adaptive_threshold.py --extract_features --dataset MOT17
```

## Available Modules

| Module | Description |
|--------|-------------|
| `reid_modules/lite.py` | Original LITE (single layer) |
| `reid_modules/lite_plus.py` | LITE+ (multi-layer fusion) |
| `reid_modules/adaptive_threshold.py` | Adaptive threshold learning |
| `reid_modules/lite_plus_unified.py` | LITE++ (unified module) |

## Results

### ReID Feature Quality (ROC-AUC)

| Method | AUC | Pos-Neg Gap |
|--------|-----|-------------|
| LITE (layer14) | 0.996 | 0.028 |
| LITE+ (attention) | **0.997** | 0.054 |
| LITE+ (adaptive) | 0.993 | **0.059** |

### Tracking Performance (MOT17)

| Method | HOTA | AssA | FPS |
|--------|------|------|-----|
| DeepSORT | 43.7 | 42.5 | 13.7 |
| LITE:DeepSORT | 43.0 | 41.9 | 28.3 |
| **LITE++** (ours) | **44.5** | **43.6** | 26.1 |

## Citation

```bibtex
@inproceedings{lite_plus_plus_eccv2026,
    title={LITE++: Multi-Scale Feature Fusion with Adaptive Thresholds for Real-Time Multi-Object Tracking},
    author={Anonymous},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2026}
}

@inproceedings{lite_iconip2024,
    title={LITE: A Paradigm Shift in Multi-Object Tracking},
    author={Mukhiddinov, Mukhriddin and Cho, Jinsoo},
    booktitle={International Conference on Neural Information Processing (ICONIP)},
    year={2024}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgements

This work builds upon the [LITE](https://arxiv.org/abs/2409.04187) paradigm and uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.
