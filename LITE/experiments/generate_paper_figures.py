"""
Generate Publication-Quality Figures for ECCV 2026 Paper

This script generates:
1. ROC curves comparing LITE variants
2. Score distribution plots
3. t-SNE visualization of feature embeddings
4. Speed comparison bar charts

Usage:
    python experiments/generate_paper_figures.py \
        --dataset MOT17 \
        --output_dir ../eccv2026/figures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import cv2
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

from ultralytics import YOLO
from reid_modules.lite_plus_unified import LITEPlusPlusUnified


# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme for paper
COLORS = {
    'LITE': '#1f77b4',           # Blue
    'LITE+_concat': '#2ca02c',    # Green
    'LITE+_attention': '#ff7f0e', # Orange
    'LITE++_full': '#d62728',     # Red
}

LABELS = {
    'LITE': 'LITE (single-layer)',
    'LITE+_concat': 'LITE+ (concat)',
    'LITE+_attention': 'LITE+ (attention)',
    'LITE++_full': 'LITE++ (ours)',
}


def load_ground_truth(gt_path: str) -> Dict[int, List[Dict]]:
    """Load ground truth annotations grouped by frame."""
    gt_by_frame = defaultdict(list)

    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])

            # Filter: only consider visible pedestrians
            if len(parts) > 7:
                consider = int(parts[6])
                cls = int(parts[7])
                visibility = float(parts[8]) if len(parts) > 8 else 1.0
                if consider == 0 or cls != 1 or visibility < 0.25:
                    continue

            gt_by_frame[frame_id].append({
                'track_id': track_id,
                'bbox': [x, y, x + w, y + h, 1.0, 0],
            })

    return dict(gt_by_frame)


def get_variant_config(variant: str) -> Dict:
    """Get configuration for specified variant."""
    configs = {
        'LITE': {
            'layers': ['layer14'],
            'fusion_type': 'concat',
            'output_dim': 64,
            'adaptive_threshold': False,
        },
        'LITE+_concat': {
            'layers': ['layer4', 'layer9', 'layer14'],
            'fusion_type': 'concat',
            'output_dim': 128,
            'adaptive_threshold': False,
        },
        'LITE+_attention': {
            'layers': ['layer4', 'layer9', 'layer14'],
            'fusion_type': 'attention',
            'output_dim': 128,
            'adaptive_threshold': False,
        },
        'LITE++_full': {
            'layers': ['layer4', 'layer9', 'layer14'],
            'fusion_type': 'attention',
            'output_dim': 128,
            'adaptive_threshold': True,
        },
    }
    return configs[variant]


def create_reid_model(yolo_model_name: str, variant: str, device: str = 'cpu'):
    """Create ReID model for specified variant with fresh YOLO model."""
    # Load fresh YOLO model to avoid state issues
    model = YOLO(f'{yolo_model_name}.pt')

    cfg = get_variant_config(variant)
    return LITEPlusPlusUnified(
        model=model,
        layer_indices=cfg['layers'],
        fusion_type=cfg['fusion_type'],
        output_dim=cfg['output_dim'],
        enable_adaptive_threshold=cfg['adaptive_threshold'],
        device=device,
        yolo_version='auto'
    )


def extract_features(
    reid_model,
    seq_dir: str,
    gt_by_frame: Dict,
    max_frames: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for ground truth boxes."""
    all_features = []
    all_track_ids = []

    img_dir = os.path.join(seq_dir, 'img1')
    frame_ids = sorted(gt_by_frame.keys())[:max_frames]

    for frame_id in frame_ids:
        img_path = os.path.join(img_dir, f'{frame_id:06d}.jpg')
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)
        annotations = gt_by_frame[frame_id]
        if len(annotations) == 0:
            continue

        boxes = np.array([ann['bbox'] for ann in annotations])
        track_ids = [ann['track_id'] for ann in annotations]

        features = reid_model.extract_appearance_features(image, boxes)

        if len(features) > 0:
            all_features.append(features)
            all_track_ids.extend(track_ids)

    if len(all_features) == 0:
        return np.array([]), np.array([])

    return np.vstack(all_features), np.array(all_track_ids)


def compute_roc_data(features: np.ndarray, track_ids: np.ndarray) -> Dict:
    """Compute ROC curve data and metrics."""
    if len(features) < 10:
        return {'auc': 0.5, 'fpr': [0, 1], 'tpr': [0, 1], 'pos_scores': [], 'neg_scores': []}

    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = features_norm @ features_norm.T

    # Create positive/negative masks
    n = len(track_ids)
    pos_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if track_ids[i] == track_ids[j]:
                pos_mask[i, j] = True

    neg_mask = ~pos_mask
    np.fill_diagonal(neg_mask, False)

    pos_scores = similarity_matrix[pos_mask]
    neg_scores = similarity_matrix[neg_mask]

    # Subsample negatives
    if len(neg_scores) > len(pos_scores) * 10:
        np.random.seed(42)
        neg_scores = np.random.choice(neg_scores, size=len(pos_scores) * 10, replace=False)

    if len(pos_scores) == 0:
        return {'auc': 0.5, 'fpr': [0, 1], 'tpr': [0, 1], 'pos_scores': [], 'neg_scores': neg_scores}

    # Compute ROC
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return {
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'pos_scores': pos_scores,
        'neg_scores': neg_scores,
        'pos_mean': float(np.mean(pos_scores)),
        'neg_mean': float(np.mean(neg_scores)),
        'pos_std': float(np.std(pos_scores)),
        'neg_std': float(np.std(neg_scores)),
    }


def measure_speed(reid_model, image: np.ndarray, boxes: np.ndarray, n_runs: int = 30) -> float:
    """Measure feature extraction speed in ms."""
    # Warmup
    for _ in range(5):
        _ = reid_model.extract_appearance_features(image, boxes)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = reid_model.extract_appearance_features(image, boxes)
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000  # ms


def plot_roc_curves(results: Dict, output_path: str):
    """Generate publication-quality ROC curve figure."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot variants in order
    variants = ['LITE', 'LITE+_concat', 'LITE+_attention', 'LITE++_full']
    linestyles = ['-', '--', '-.', '-']
    linewidths = [1.5, 1.5, 1.5, 2.5]

    for variant, ls, lw in zip(variants, linestyles, linewidths):
        if variant not in results:
            continue
        data = results[variant]
        label = f"{LABELS[variant]} (AUC={data['auc']:.3f})"
        ax.plot(data['fpr'], data['tpr'],
                color=COLORS[variant], linestyle=ls, linewidth=lw, label=label)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, alpha=0.5, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved ROC curves to {output_path}")


def plot_score_distributions(results: Dict, output_path: str):
    """Generate score distribution comparison figure."""
    variants = ['LITE', 'LITE+_attention', 'LITE++_full']
    available = [v for v in variants if v in results and len(results[v]['pos_scores']) > 0]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 3.5))
    if len(available) == 1:
        axes = [axes]

    for ax, variant in zip(axes, available):
        data = results[variant]
        pos_scores = data['pos_scores']
        neg_scores = data['neg_scores']

        # Histogram
        bins = np.linspace(0, 1, 40)
        ax.hist(neg_scores, bins=bins, alpha=0.6, color='#d62728', label='Different ID', density=True)
        ax.hist(pos_scores, bins=bins, alpha=0.6, color='#2ca02c', label='Same ID', density=True)

        # Mean lines
        ax.axvline(data['neg_mean'], color='#d62728', linestyle='--', linewidth=2)
        ax.axvline(data['pos_mean'], color='#2ca02c', linestyle='--', linewidth=2)

        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title(LABELS[variant])
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved score distributions to {output_path}")


def plot_gap_comparison(results: Dict, output_path: str):
    """Generate bar chart comparing positive-negative gaps."""
    variants = ['LITE', 'LITE+_concat', 'LITE+_attention', 'LITE++_full']
    available = [v for v in variants if v in results]

    gaps = [results[v]['pos_mean'] - results[v]['neg_mean'] for v in available]
    aucs = [results[v]['auc'] for v in available]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(available))
    colors = [COLORS[v] for v in available]
    labels = [LABELS[v].split('(')[0].strip() for v in available]

    # Gap comparison
    bars1 = ax1.bar(x, gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Positive-Negative Gap')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.set_ylim([0, max(gaps) * 1.2])
    ax1.set_title('(a) Identity Discrimination Gap')

    # Add values on bars
    for bar, gap in zip(bars1, gaps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{gap:.3f}', ha='center', va='bottom', fontsize=10)

    # AUC comparison
    bars2 = ax2.bar(x, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('ROC-AUC')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylim([0.85, 1.0])
    ax2.set_title('(b) Re-ID Discriminability')

    # Add values on bars
    for bar, auc_val in zip(bars2, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{auc_val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved gap comparison to {output_path}")


def plot_speed_comparison(speed_results: Dict, output_path: str):
    """Generate speed comparison figure."""
    variants = ['LITE', 'LITE+_concat', 'LITE+_attention', 'LITE++_full']
    available = [v for v in variants if v in speed_results]

    speeds = [speed_results[v] for v in available]
    colors = [COLORS[v] for v in available]
    labels = [LABELS[v].split('(')[0].strip() for v in available]

    fig, ax = plt.subplots(figsize=(7, 4))

    x = np.arange(len(available))
    bars = ax.bar(x, speeds, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_ylabel('Time per Frame (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_title('Feature Extraction Speed Comparison')

    # Baseline reference line
    baseline = speeds[0]
    ax.axhline(baseline, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(len(available) - 0.5, baseline + 0.2, 'LITE baseline', fontsize=9, color='gray')

    # Add values and overhead
    for i, (bar, speed) in enumerate(zip(bars, speeds)):
        overhead = ((speed - baseline) / baseline) * 100 if i > 0 else 0
        if i == 0:
            label = f'{speed:.1f}ms'
        else:
            label = f'{speed:.1f}ms\n(+{overhead:.0f}%)' if overhead > 0 else f'{speed:.1f}ms'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                label, ha='center', va='bottom', fontsize=9)

    ax.set_ylim([0, max(speeds) * 1.3])

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved speed comparison to {output_path}")


def plot_tsne(features: np.ndarray, track_ids: np.ndarray, variant: str, output_path: str):
    """Generate t-SNE visualization."""
    if len(features) < 30:
        print(f"Skipping t-SNE for {variant}: not enough features")
        return

    # Subsample if needed
    if len(features) > 500:
        np.random.seed(42)
        idx = np.random.choice(len(features), 500, replace=False)
        features = features[idx]
        track_ids = track_ids[idx]

    # Run t-SNE
    print(f"Computing t-SNE for {variant}...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) // 4), random_state=42)
    embedded = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    unique_ids = np.unique(track_ids)
    n_tracks = min(20, len(unique_ids))  # Limit to 20 tracks for clarity

    cmap = plt.cm.tab20(np.linspace(0, 1, n_tracks))

    for i, tid in enumerate(unique_ids[:n_tracks]):
        mask = track_ids == tid
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                  c=[cmap[i]], s=30, alpha=0.7, label=f'ID {tid}')

    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f't-SNE Feature Visualization: {LABELS[variant]}')

    if n_tracks <= 10:
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.savefig(output_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved t-SNE to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--dataset', type=str, default='MOT17')
    parser.add_argument('--seq_name', type=str, default='MOT17-02-FRCNN')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_frames', type=int, default=150)
    parser.add_argument('--yolo_model', type=str, default='yolov8n')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str, default='../eccv2026/figures')
    args = parser.parse_args()

    # Setup paths
    seq_dir = f'datasets/{args.dataset}/{args.split}/{args.seq_name}'
    gt_path = os.path.join(seq_dir, 'gt/gt.txt')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(seq_dir):
        print(f"Error: Sequence directory not found: {seq_dir}")
        return

    # Load ground truth
    print(f"Loading ground truth...")
    gt_by_frame = load_ground_truth(gt_path)
    print(f"Loaded {len(gt_by_frame)} frames")

    # Get sample image for speed testing
    first_frame = min(gt_by_frame.keys())
    sample_image = cv2.imread(os.path.join(seq_dir, 'img1', f'{first_frame:06d}.jpg'))
    sample_boxes = np.array([ann['bbox'] for ann in gt_by_frame[first_frame]])

    # Run experiments for each variant
    variants = ['LITE', 'LITE+_concat', 'LITE+_attention', 'LITE++_full']
    results = {}
    features_dict = {}
    speed_results = {}

    for variant in variants:
        print(f"\n{'='*50}")
        print(f"Processing: {LABELS[variant]}")
        print('='*50)

        # Create fresh model for each variant to avoid state issues
        print(f"Loading YOLO model: {args.yolo_model}")
        reid_model = create_reid_model(args.yolo_model, variant, args.device)

        # Extract features
        print("Extracting features...")
        features, track_ids = extract_features(
            reid_model, seq_dir, gt_by_frame, args.max_frames
        )

        if len(features) == 0:
            print(f"No features extracted for {variant}")
            continue

        print(f"Extracted {len(features)} features from {len(np.unique(track_ids))} tracks")

        # Compute ROC data
        print("Computing metrics...")
        roc_data = compute_roc_data(features, track_ids)
        results[variant] = roc_data
        features_dict[variant] = (features, track_ids)

        # Measure speed
        print("Measuring speed...")
        speed_results[variant] = measure_speed(reid_model, sample_image, sample_boxes)

        print(f"  AUC: {roc_data['auc']:.4f}")
        print(f"  Gap: {roc_data['pos_mean'] - roc_data['neg_mean']:.4f}")
        print(f"  Speed: {speed_results[variant]:.2f} ms")

    # Generate figures
    print("\n" + "="*50)
    print("Generating figures...")
    print("="*50)

    # 1. ROC curves
    plot_roc_curves(results, str(output_dir / 'roc_curves.pdf'))

    # 2. Score distributions
    plot_score_distributions(results, str(output_dir / 'score_distributions.pdf'))

    # 3. Gap and AUC comparison
    plot_gap_comparison(results, str(output_dir / 'metric_comparison.pdf'))

    # 4. Speed comparison
    plot_speed_comparison(speed_results, str(output_dir / 'speed_comparison.pdf'))

    # 5. t-SNE for best variant
    if 'LITE++_full' in features_dict:
        features, track_ids = features_dict['LITE++_full']
        plot_tsne(features, track_ids, 'LITE++_full', str(output_dir / 'tsne_litepp.pdf'))

    # Print summary
    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"Figures saved to: {output_dir}")
    print("\nGenerated files:")
    for f in output_dir.glob('*'):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
