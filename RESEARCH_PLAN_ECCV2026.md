# ECCV 2026 Research Plan: LITE++

## Adaptive Multi-Scale Tracking with Learned Feature Fusion

**Target Conference**: ECCV 2026 (Expected deadline: ~March 2026)
**Conference Dates**: September 8-13, 2026

---

## 1. Executive Summary

We propose **LITE++**, an advancement of the LITE paradigm that addresses four key limitations:

1. **Single-layer feature limitation** → Multi-scale feature pyramid fusion
2. **Manual threshold tuning** → Learned adaptive confidence thresholds
3. **Domain-agnostic design** → Domain-aware feature adaptation
4. **CNN-only architecture** → Transformer-compatible design (RT-DETR, YOLO-World)

**Novelty Statement**: LITE++ is the first unified framework that combines hierarchical feature fusion, adaptive threshold learning, and domain adaptation for real-time multi-object tracking, while extending the paradigm to transformer-based detectors.

---

## 2. Current SOTA Analysis

### 2.1 Transformer-Based MOT (2024-2025)
| Method | Venue | HOTA | Key Innovation |
|--------|-------|------|----------------|
| [MATR](https://arxiv.org/html/2509.21715v1) | 2025 | 59.0 mAssocA | Motion-aware transformer |
| [CO-MOT](https://proceedings.iclr.cc/paper_files/paper/2025/file/8428da31da191712130ce8cce265691a-Paper-Conference.pdf) | ICLR 2025 | 56.2 | One-to-set matching with shadow queries |
| MeMOTR | ICCV 2023 | 56.7 | Long-term memory augmentation |
| [MASA](https://github.com/luanshiyinyang/awesome-multiple-object-tracking) | CVPR 2024 | - | Matching anything by segmenting |

### 2.2 Tracking-by-Detection Methods
| Method | Speed (FPS) | HOTA | ReID Cost |
|--------|-------------|------|-----------|
| LITE:DeepSORT (ours) | 28.3 | 43.0 | Zero |
| DeepSORT | 13.7 | 43.7 | High |
| StrongSORT | 5.1 | 41.7 | Very High |
| ByteTrack | 29.7 | 43.8 | None |

### 2.3 Research Gap
- **No existing work** combines multi-scale features with adaptive thresholds
- **Transformer-based trackers** are slow (~10 FPS) for real-time applications
- **Domain adaptation** in MOT is under-explored for practical deployment

---

## 3. Technical Approach

### 3.1 Multi-Scale Feature Pyramid Fusion (MSFP)

```
                    ┌─────────────────┐
                    │   YOLOv8/v11    │
                    │   Backbone      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
   │ Layer 4  │         │ Layer 9  │         │ Layer 14│
   │ (48 ch)  │         │ (96 ch)  │         │ (192 ch)│
   │ h/2×w/2  │         │ h/4×w/4  │         │ h/8×w/8 │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                    │                    │
        └──────────┬─────────┴──────────┬────────┘
                   │                    │
            ┌──────▼──────┐      ┌──────▼──────┐
            │ PCA/Learned │      │   Adaptive  │
            │  Selection  │      │   Weights   │
            └──────┬──────┘      └──────┬──────┘
                   │                    │
                   └─────────┬──────────┘
                             │
                    ┌────────▼────────┐
                    │  Fused Feature  │
                    │   (d=128/256)   │
                    └─────────────────┘
```

**Implementation Strategy**:
1. Extract features from 3 backbone layers (early/mid/late)
2. Resize all to common resolution via bilinear interpolation
3. Apply learned attention weights OR PCA for dimensionality reduction
4. Concatenate + MLP projection to final embedding

### 3.2 Adaptive Threshold Learning (ATL)

**Problem**: Optimal confidence threshold varies by dataset (0.25 for MOT17, 0.05 for MOT20)

**Solution**: Scene-aware threshold predictor

```python
class AdaptiveThresholdModule(nn.Module):
    def __init__(self, feature_dim=256):
        self.scene_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, scene_features):
        # Output: confidence threshold in [0.05, 0.5]
        base_threshold = 0.05
        range_threshold = 0.45
        return base_threshold + range_threshold * self.scene_encoder(scene_features)
```

**Training**: Optimize threshold jointly with tracking loss via REINFORCE or Gumbel-Softmax

### 3.3 Domain-Aware Feature Adaptation (DAFA)

**Approach**: Lightweight domain classifier + feature normalization

```
┌─────────────┐     ┌─────────────┐
│  Traffic    │     │   Retail    │
│  Domain     │     │   Domain    │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
        ┌────────▼────────┐
        │ Domain Classifier│
        │   (auxiliary)    │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │ Domain-Specific │
        │ Normalization   │
        └─────────────────┘
```

**Key Insight**: Different domains have different:
- Object scales (traffic: large vehicles, retail: full-body humans)
- Occlusion patterns (traffic: linear, retail: random)
- Motion characteristics (traffic: predictable, retail: erratic)

### 3.4 Transformer-Compatible Design

**Extending LITE to RT-DETR and YOLO-World**:

```python
class TransformerLITE:
    """Extract features from transformer encoder/decoder"""

    def __init__(self, model, layer_indices=[3, 6, 9]):
        self.model = model
        self.layer_indices = layer_indices  # Transformer layer indices

    def extract_features(self, image, boxes):
        # Hook into transformer encoder layers
        encoder_features = []
        for idx in self.layer_indices:
            feat = self.model.encoder.layers[idx].output
            encoder_features.append(feat)

        # Multi-head attention pooling for each box
        return self.attention_pool(encoder_features, boxes)
```

---

## 4. Experimental Design

### 4.1 Datasets

| Dataset | Purpose | Sequences | Density |
|---------|---------|-----------|---------|
| MOT17 | Main benchmark | 7 train / 7 test | Medium |
| MOT20 | Crowded scenes | 4 train / 4 test | High |
| DanceTrack | Association challenge | 40 train | High motion |
| PersonPath22 | Diverse scenarios | 98 test | Various |
| **AntVision-Traffic** | Domain-specific (new) | TBD | Traffic |
| **AntVision-Retail** | Domain-specific (new) | TBD | Retail |

### 4.2 Ablation Studies

1. **Feature Fusion Ablation**
   - Single layer (baseline) vs. 2-layer vs. 3-layer
   - PCA vs. learned selection vs. attention weights

2. **Threshold Learning Ablation**
   - Fixed threshold vs. scene-adaptive
   - Different threshold ranges

3. **Domain Adaptation Ablation**
   - No adaptation vs. normalization vs. full DAFA

4. **Architecture Ablation**
   - YOLOv8 vs. YOLOv11 vs. RT-DETR vs. YOLO-World

### 4.3 Comparison Methods

| Category | Methods |
|----------|---------|
| Motion-only | SORT, OC-SORT, ByteTrack |
| ReID-based | DeepSORT, StrongSORT, BoTSORT, Deep OC-SORT |
| LITE family | LITE:DeepSORT, LITE:BoTSORT |
| Transformer | MOTR, TrackFormer, CO-MOT, MATR |

### 4.4 Metrics

- **HOTA** (primary): Balances detection and association
- **AssA**: Association accuracy
- **DetA**: Detection accuracy
- **IDF1**: Identity F1 score
- **FPS**: Processing speed (holistic pipeline)

---

## 5. Implementation Plan

### Phase 1: Foundation (Weeks 1-4)
- [ ] Implement multi-layer feature extraction hooks for YOLOv8
- [ ] Create feature fusion module (concat + MLP)
- [ ] Benchmark baseline on MOT17

### Phase 2: Adaptive Threshold (Weeks 5-8)
- [ ] Implement scene encoder network
- [ ] Design differentiable threshold selection
- [ ] Train and evaluate on MOT17/MOT20

### Phase 3: Multi-Scale Fusion (Weeks 9-12)
- [ ] Implement PCA-based feature selection
- [ ] Implement learned attention weights
- [ ] Compare fusion strategies

### Phase 4: Domain Adaptation (Weeks 13-16)
- [ ] Collect/annotate AntVision domain data
- [ ] Implement domain classifier
- [ ] Train domain-adaptive model

### Phase 5: Transformer Extension (Weeks 17-20)
- [ ] Port LITE to RT-DETR
- [ ] Port LITE to YOLO-World
- [ ] Benchmark transformer variants

### Phase 6: Paper Writing (Weeks 21-24)
- [ ] Write methodology section
- [ ] Create figures and tables
- [ ] Write introduction and related work
- [ ] Revise and polish

---

## 6. Expected Contributions

1. **LITE++**: A unified framework advancing real-time MOT with:
   - Multi-scale feature pyramid fusion
   - Scene-adaptive confidence thresholds
   - Domain-aware feature adaptation
   - Transformer-compatible design

2. **Theoretical Analysis**: Understanding of feature discriminability across layers

3. **Practical Impact**:
   - Deployment-ready for AntVision (traffic/retail/security)
   - Edge device optimization (Jetson Orin)

4. **Open Source**: Full code release with pretrained models

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Multi-scale fusion adds latency | Use efficient attention; benchmark early |
| Threshold learning unstable | Start with supervised pretraining |
| Domain data insufficient | Augment with synthetic data |
| Transformer LITE too slow | Focus on efficient variants (RT-DETR Lite) |

---

## 8. Paper Outline

```
1. Introduction (1 page)
   - MOT importance and challenges
   - LITE paradigm recap
   - Limitations and our contributions

2. Related Work (1 page)
   - Tracking-by-detection methods
   - ReID in MOT
   - Transformer-based tracking
   - Domain adaptation in vision

3. Method (3 pages)
   - 3.1 Overview of LITE++
   - 3.2 Multi-Scale Feature Pyramid Fusion
   - 3.3 Adaptive Threshold Learning
   - 3.4 Domain-Aware Feature Adaptation
   - 3.5 Transformer-Compatible Design

4. Experiments (3 pages)
   - 4.1 Setup and implementation details
   - 4.2 Comparison with SOTA
   - 4.3 Ablation studies
   - 4.4 Domain-specific evaluation
   - 4.5 Edge device deployment

5. Conclusion (0.5 page)

References (1.5 pages)
```

---

## 9. Timeline

```
2025 Q3 (Jul-Sep): Phase 1-2 (Foundation + Threshold)
2025 Q4 (Oct-Dec): Phase 3-4 (Fusion + Domain)
2026 Q1 (Jan-Feb): Phase 5 (Transformer)
2026 Feb-Mar: Phase 6 (Writing + Submission)
```

---

## 10. Resources Needed

- **Compute**: 4x A100 GPUs for training (or cloud equivalent)
- **Data**: AntVision domain videos (traffic, retail, security)
- **Personnel**: 2-3 researchers for implementation and experiments

---

## Next Steps

1. **Immediate**: Set up codebase structure for LITE++
2. **Week 1**: Implement multi-layer feature extraction
3. **Week 2**: Create evaluation pipeline with TrackEval
4. **Week 3**: Run baseline experiments

---

*Document created for ECCV 2026 submission planning*
*Project: LITE++ | Team: AntVision AI Research*
