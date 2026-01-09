# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LITE++ is an extension of the LITE (Lightweight Integrated Tracking-Feature Extraction) paradigm for real-time multi-object tracking. It integrates ReID feature extraction directly into the object detection pipeline, extracting appearance features from intermediate layers of YOLO detectors.

**Key Innovations:**
- Multi-Scale Feature Pyramid (MSFP): Features from multiple backbone layers
- Adaptive Threshold Learning (ATL): Scene-aware confidence thresholds
- Three fusion strategies: concat, attention, adaptive

**Target Venue:** ECCV 2026

## Repository Structure

```
lite/
├── litepp/                      # LITE++ source code (main package)
│   ├── __init__.py              # Package exports
│   ├── models/
│   │   ├── feature_pyramid.py   # Multi-Scale Feature Pyramid
│   │   ├── adaptive_threshold.py # Adaptive Threshold Learning
│   │   └── litepp.py            # Unified LITE++ module
│   ├── trackers/
│   │   └── litepp_tracker.py    # Tracker integration (DeepSORT etc.)
│   ├── utils/
│   │   ├── evaluation.py        # ReID & tracking metrics
│   │   └── visualization.py     # Plotting utilities
│   ├── experiments/
│   │   ├── run_mot17.py         # MOT17 experiments
│   │   └── ablation_fusion.py   # Fusion ablation study
│   ├── configs/
│   │   └── default.yaml         # Default configuration
│   ├── scripts/
│   │   └── run_experiments.sh   # Experiment runner
│   └── docs/
│       └── acknowledgments.md   # Citations for LITE, DeepSORT, etc.
├── eccv2026/                    # Paper materials (LaTeX)
├── cli/                         # Claude Code agents and commands
├── lite.pdf                     # Original LITE paper (reference)
├── prime.pdf                    # Evaluation framework paper
└── RESEARCH_PLAN_ECCV2026.md    # Research plan
```

## Build & Run Commands

### Environment Setup
```bash
# Create environment
conda create -n litepp python=3.10 -y
conda activate litepp

# Install LITE++ as package
cd litepp
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Quick Start
```python
from ultralytics import YOLO
from litepp import create_litepp

# Load YOLO model
model = YOLO('yolov8m.pt')

# Create LITE++ module
litepp = create_litepp(
    model=model,
    fusion_type='attention',
    enable_adaptive_threshold=True,
    device='cuda:0'
)

# Extract features with adaptive threshold
features, threshold = litepp.extract_with_adaptive_threshold(image, boxes)
```

### Using the Tracker Wrapper
```python
from litepp.trackers import LITEPlusPlusTracker

tracker = LITEPlusPlusTracker(
    model=yolo_model,
    fusion_type='attention',
    enable_adaptive_threshold=True,
)

# Standard ReID interface
features = tracker.extract_appearance_features(image, boxes)
threshold = tracker.get_current_threshold()
```

### Running Experiments
```bash
cd litepp

# Run MOT17 experiments
python experiments/run_mot17.py --fusion_type attention --adaptive_threshold

# Fusion ablation study
python experiments/ablation_fusion.py --dataset MOT17

# Run all experiments
bash scripts/run_experiments.sh
```

## Architecture

### LITE++ Pipeline
```
       YOLO Backbone
            │
  ┌─────────┼─────────┐
  │         │         │
Layer4   Layer9   Layer14
(early)  (mid)    (late)
  │         │         │
  └─────────┼─────────┘
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

### Key Components

#### 1. Multi-Scale Feature Pyramid (litepp/models/feature_pyramid.py)
Extracts features from multiple backbone layers:
- `layer4`: Fine-grained details (early features)
- `layer9`: Medium-level features
- `layer14`: Semantic features (late features)

#### 2. Feature Fusion Module
Three strategies in `FeatureFusionModule`:
- **attention** (default): Learned attention weights per layer
- **adaptive**: SE-style channel-wise fusion
- **concat**: Concatenation + MLP projection

#### 3. Adaptive Threshold Learning (litepp/models/adaptive_threshold.py)
- `AdaptiveThresholdModule`: Predicts optimal detection threshold per scene
- `MultiThresholdPredictor`: Predicts detection, association, and max_age
- Eliminates manual tuning (0.25 for MOT17 vs 0.05 for MOT20)

#### 4. Unified Module (litepp/models/litepp.py)
`LITEPlusPlus` class combines all components:
```python
from litepp import LITEPlusPlus, create_litepp

# Factory function (recommended)
litepp = create_litepp(
    model=yolo_model,
    fusion_type='attention',
    layers=['layer4', 'layer9', 'layer14'],
    output_dim=128,
    enable_adaptive_threshold=True,
)

# Get scene analysis
analysis = litepp.get_scene_analysis(image)
# Returns: {'adaptive_threshold': 0.15, 'estimated_density': 2.3, ...}

# Get fusion attention weights
weights = litepp.get_fusion_weights()
```

## Datasets

Supported: MOT17, MOT20, DanceTrack

Dataset structure:
```
datasets/
├── MOT17/train/MOT17-02-FRCNN/
│   ├── img1/
│   ├── gt/gt.txt
│   └── det/det.txt
```

## Evaluation

Uses HOTA (Higher Order Tracking Accuracy) as primary metric:
```
HOTA = sqrt(DetA × AssA)
```

Where:
- DetA: Detection accuracy
- AssA: Association accuracy

## Key Configuration (litepp/configs/default.yaml)

```yaml
features:
  layers: [layer4, layer9, layer14]
  fusion_type: attention
  output_dim: 128

adaptive_threshold:
  enabled: true
  min_threshold: 0.01
  max_threshold: 0.50

tracking:
  max_age: 30
  max_cosine_distance: 0.3
```

## CLI Agents

The `cli/` folder contains Claude Code configurations:
- `agents/`: AI agent configurations
- `commands/`: Command definitions
- `skills/`: Skill definitions

## Research Plan

See `RESEARCH_PLAN_ECCV2026.md` for:
- Technical approach details
- Ablation study design
- Experimental methodology
- Paper outline

## Paper Materials

The `eccv2026/` folder contains LaTeX files for the ECCV 2026 submission.

## Acknowledgments

This work builds upon:
- LITE paradigm (ICONIP 2024)
- DeepSORT tracker
- Ultralytics YOLOv8

See `litepp/docs/acknowledgments.md` for full citations.
