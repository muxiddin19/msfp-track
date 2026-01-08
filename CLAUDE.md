# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LITE (Lightweight Integrated Tracking-Feature Extraction) is a multi-object tracking (MOT) research project that integrates ReID (Re-Identification) feature extraction directly into the object detection pipeline. The core innovation eliminates the need for separate ReID model inference by extracting appearance features from intermediate layers of YOLO detectors (YOLOv8/v11).

**Key Papers:**
- LITE: A Paradigm Shift in Multi-Object Tracking (ICONIP 2024, arXiv:2409.04187)
- Practical Evaluation Framework for Real-Time Multi-Object Tracking (IEEE Access 2025)

## Repository Structure

```
lite/
├── LITE/                    # Main source code
│   ├── track.py            # Main tracking execution script
│   ├── run.py              # Script runner for experiments
│   ├── opts.py             # Command-line options and configurations
│   ├── app.py              # Streamlit web UI for video tracking comparison
│   ├── reid.py             # ReID evaluation module
│   ├── solutions.py        # High-level solutions (counting, heatmaps, parking)
│   ├── deep_sort/          # DeepSORT tracker implementation
│   │   ├── tracker.py      # Main Tracker class with matching logic
│   │   ├── kalman_filter.py
│   │   ├── nn_matching.py  # Nearest neighbor matching
│   │   └── linear_assignment.py
│   ├── reid_modules/       # ReID feature extraction implementations
│   │   ├── lite.py         # LITE paradigm (extracts features from YOLO layers)
│   │   ├── deepsort.py     # Original DeepSORT ReID
│   │   └── strongsort.py   # StrongSORT ReID
│   └── scripts/            # Experiment scripts
│       └── run_experiment.sh
├── cli/                    # Claude Code agents, commands, and skills
│   ├── agents/             # AI agent configurations
│   ├── commands/           # Command definitions (e.g., ultra-think)
│   └── skills/             # Skill definitions
├── lite.pdf                # LITE paper
└── prime.pdf               # Extended evaluation framework paper
```

## Build & Run Commands

### Environment Setup
```bash
cd LITE
python3.10 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt

# Clone required dependencies
git clone https://github.com/Jumabek/ultralytics.git
git clone https://github.com/humblebeeintel/yolo_tracking.git
git clone https://github.com/humblebeeintel/TrackEval
bash scripts/setup_fastreid.sh
```

### Running Experiments
```bash
# Run tracking experiment with specific tracker
bash scripts/run_experiment.sh -s train

# Or use Python directly
python run.py \
    --dataset MOT17 \
    --split train \
    --tracker_name LITEDeepSORT \
    --input_resolution 1280 \
    --min_confidence 0.25 \
    --yolo_model yolov8m \
    --appearance_feature_layer layer14

# Run ReID evaluation
python reid.py --dataset MOT17 --seq_name MOT17-02-FRCNN --split train --tracker LITEDeepSORT --save
```

### Available Trackers
- **Pure Motion**: `SORT`, `OCSORT`, `Bytetrack`
- **Motion + ReID**: `DeepSORT`, `StrongSORT`, `DeepOCSORT`, `BoTSORT`
- **LITE Variants**: `LITEDeepSORT`, `LITEStrongSORT`, `LITEDeepOCSORT`, `LITEBoTSORT`

### Demo & Web UI
```bash
# Basic tracking demo
python demo.py --source demo/VIRAT_S_010204_07_000942_000989.mp4

# Streamlit web UI for tracker comparison
streamlit run app.py

# Solutions (object counting, heatmaps)
python solutions.py --source videos/shortened_enterance.mp4 --solution object_counter heatmap
```

## Architecture

### LITE Paradigm
The LITE paradigm extracts appearance features from intermediate convolutional layers of YOLO during detection inference, eliminating separate ReID model overhead:

1. **Feature Map Extraction**: During YOLO inference, extract feature maps from early layers (default: `layer14` with 48 channels)
2. **Bounding Box Mapping**: Map detected bounding boxes to the downscaled feature map resolution
3. **Feature Cropping**: Crop regions from the feature map corresponding to detections
4. **Spatial Reduction**: Average across spatial dimensions to get fixed-size embeddings (dim=48)

### Tracker Architecture
```
Detection (YOLO) → ReID Features → Kalman Filter → Cost Matrix → Hungarian Matching → Track Management
                        ↑
              LITE extracts from here
```

### Key Configuration Options (opts.py)
- `--tracker_name`: Tracker selection (affects matching strategy and ReID)
- `--appearance_feature_layer`: YOLO layer for LITE feature extraction (e.g., "layer14")
- `--input_resolution`: Detection input size (default: 1280)
- `--min_confidence`: Detection confidence threshold (default: 0.25)
- `--max_cosine_distance`: Appearance matching threshold (default: 0.7)
- `--max_age`: Frames before track deletion (default: 30)

### StrongSORT Enhancements
When `--tracker_name StrongSORT` is used, these are auto-enabled:
- `--BoT`: BoT (Bag of Tricks) ReID model
- `--NSA`: NSA Kalman filter
- `--EMA`: Exponential Moving Average for features
- `--MC`: Motion + appearance cost matching
- `--woC`: Vanilla matching (no cascade)

## Datasets
Supported: MOT17, MOT20, KITTI, PersonPath22, VIRAT-S, DanceTrack

Dataset structure:
```
LITE/datasets/
├── MOT/train/MOT17-02-FRCNN/
│   ├── img1/
│   ├── gt/gt.txt
│   └── det/det.txt
```

## Evaluation
Uses HOTA (Higher Order Tracking Accuracy) as primary metric, balancing detection accuracy (DetA) and association accuracy (AssA):
```
HOTA = sqrt(DetA × AssA)
```

## CLI Agents (for research workflows)
The `cli/` folder contains Claude Code agent configurations:
- `research-orchestrator.md`: Coordinates multi-phase research projects
- `academic-researcher.md`: Academic paper analysis
- `code-reviewer.md`: Code review workflows
- `ultra-think.md`: Enhanced reasoning mode for complex problems

## Key Implementation Details

### LITE Feature Extraction (reid_modules/lite.py)
- Uses `model.predict(..., return_feature_map=True)` to get intermediate features
- Default layer produces 48-channel feature maps at h/2 × w/2 resolution
- L2 normalization applied after spatial averaging

### Tracker Matching (deep_sort/tracker.py)
- SORT: IoU-only matching
- DeepSORT/LITE variants: Cascade matching with appearance + IoU
- Confirmed tracks use appearance; unconfirmed use IoU

### Performance Insights from Papers
- Optimal confidence thresholds vary by dataset (0.25 for MOT17, 0.05 for MOT20)
- LITE achieves 2-10x speedup over original ReID-based trackers
- Small HOTA differences (1-2%) are not conclusive for ranking trackers

## LITE++ (ECCV 2026 Research)

LITE++ extends the LITE paradigm with four key innovations:

### 1. Multi-Scale Feature Pyramid (reid_modules/lite_plus.py)
Extracts and fuses features from multiple backbone layers for richer representations:
- `layer4`: 48 channels, high spatial resolution (fine details)
- `layer9`: 96 channels, medium resolution
- `layer14`: 192 channels, semantic features (original LITE)

### 2. Feature Fusion Strategies
Three fusion approaches in `FeatureFusionModule`:
- **concat**: Simple concatenation + MLP projection
- **attention**: Learned attention weights per layer (default)
- **adaptive**: SE-style channel-wise fusion

```python
from reid_modules import create_lite_plus, create_lite_plus_plus

# LITE+ (multi-layer only)
reid_model = create_lite_plus(model, variant='attention', layers=['layer4', 'layer9', 'layer14'])

# LITE++ (multi-layer + adaptive thresholds)
reid_model = create_lite_plus_plus(model, fusion_type='attention', enable_adaptive_threshold=True)
```

### 3. Adaptive Threshold Learning (reid_modules/adaptive_threshold.py)
Scene-aware confidence threshold prediction:
- `AdaptiveThresholdModule`: Predicts optimal detection threshold per scene
- `MultiThresholdPredictor`: Predicts detection, association, and max_age thresholds
- Eliminates manual threshold tuning (0.25 for MOT17 vs 0.05 for MOT20)

### 4. Unified LITE++ Module (reid_modules/lite_plus_unified.py)
Complete integration:
```python
from reid_modules import LITEPlusPlusUnified, create_lite_plus_plus

reid_model = create_lite_plus_plus(
    model=yolo_model,
    fusion_type='attention',
    layers=['layer4', 'layer9', 'layer14'],
    enable_adaptive_threshold=True,
    device='cuda:0'
)

# Extract features with adaptive threshold
features, threshold = reid_model.extract_with_adaptive_threshold(image, boxes)
```

### LITE++ Experiments
```bash
# Compare LITE vs LITE+ vs LITE++
python experiments/compare_lite_variants.py --dataset MOT17 --seq_name MOT17-02-FRCNN

# Run full ECCV 2026 experiments
python experiments/run_eccv_experiments.py --experiment all --dataset MOT17

# Train adaptive threshold module
python experiments/train_adaptive_threshold.py --dataset MOT17 --extract_features
```

### Research Plan
See `RESEARCH_PLAN_ECCV2026.md` for:
- Technical approach details
- Ablation study design
- Experimental methodology
- Paper outline

### Available Modules
```
reid_modules/
├── lite.py              # Original LITE (single layer)
├── lite_plus.py         # LITE+ (multi-layer fusion)
├── adaptive_threshold.py # Adaptive threshold learning
├── lite_plus_unified.py # LITE++ (unified module)
├── strongsort.py        # StrongSORT ReID
├── deepsort.py          # DeepSORT ReID
├── osnet.py             # OSNet ReID
└── gfn.py               # GFN detector
```
