"""
LITE vs LITE++ Comparison Script

This script compares the discriminative power of:
1. LITE (single-layer): Original LITE with layer14
2. LITE++ (multi-layer): Multi-scale fusion with layer4, layer9, layer14

Metrics:
- Feature distinctiveness (positive vs negative match scores)
- ROC-AUC for ReID capability
- Computational overhead (FPS impact)

Usage:
    python experiments/compare_lite_variants.py \
        --dataset MOT17 \
        --seq_name MOT17-02-FRCNN \
        --split train
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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

from ultralytics import YOLO
from reid_modules import LITE, create_lite_plus


def load_ground_truth(gt_path: str) -> Dict[int, List[Tuple]]:
    """Load ground truth annotations grouped by frame."""
    gt_by_frame = defaultdict(list)

    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            # Filter: only consider visible pedestrians (class 1, visibility > 0.25)
            if len(parts) > 7:
                consider = int(parts[6])
                cls = int(parts[7])
                visibility = float(parts[8]) if len(parts) > 8 else 1.0
                if consider == 0 or cls != 1 or visibility < 0.25:
                    continue

            gt_by_frame[frame_id].append({
                'track_id': track_id,
                'bbox': [x, y, x + w, y + h, 1.0, 0],  # [x1, y1, x2, y2, conf, cls]
            })

    return dict(gt_by_frame)


def extract_features_for_sequence(
    model,
    reid_model,
    seq_dir: str,
    gt_by_frame: Dict,
    max_frames: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for ground truth boxes across frames.

    Returns:
        features: (N, D) feature matrix
        track_ids: (N,) track IDs for each feature
    """
    all_features = []
    all_track_ids = []

    img_dir = os.path.join(seq_dir, 'img1')
    frame_ids = sorted(gt_by_frame.keys())[:max_frames]

    for frame_id in frame_ids:
        # Load image
        img_path = os.path.join(img_dir, f'{frame_id:06d}.jpg')
        if not os.path.exists(img_path):
            continue

        image = cv2.imread(img_path)

        # Get boxes for this frame
        annotations = gt_by_frame[frame_id]
        if len(annotations) == 0:
            continue

        boxes = np.array([ann['bbox'] for ann in annotations])
        track_ids = [ann['track_id'] for ann in annotations]

        # Extract features
        features = reid_model.extract_appearance_features(image, boxes)

        if len(features) > 0:
            all_features.append(features)
            all_track_ids.extend(track_ids)

    if len(all_features) == 0:
        return np.array([]), np.array([])

    return np.vstack(all_features), np.array(all_track_ids)


def compute_reid_metrics(
    features: np.ndarray,
    track_ids: np.ndarray
) -> Dict[str, float]:
    """
    Compute ReID metrics: positive/negative match scores and ROC-AUC.
    """
    if len(features) < 10:
        return {'auc': 0.0, 'pos_mean': 0.0, 'neg_mean': 0.0}

    # Compute pairwise cosine similarities
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    similarity_matrix = features_norm @ features_norm.T

    # Create mask for positive pairs (same track_id)
    n = len(track_ids)
    pos_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if track_ids[i] == track_ids[j]:
                pos_mask[i, j] = True

    neg_mask = ~pos_mask
    np.fill_diagonal(neg_mask, False)

    # Extract positive and negative scores
    pos_scores = similarity_matrix[pos_mask]
    neg_scores = similarity_matrix[neg_mask]

    # Subsample if too many negatives
    if len(neg_scores) > len(pos_scores) * 10:
        neg_scores = np.random.choice(neg_scores, size=len(pos_scores) * 10, replace=False)

    if len(pos_scores) == 0:
        return {'auc': 0.5, 'pos_mean': 0.0, 'neg_mean': float(np.mean(neg_scores))}

    # Compute ROC-AUC
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return {
        'auc': roc_auc,
        'pos_mean': float(np.mean(pos_scores)),
        'neg_mean': float(np.mean(neg_scores)),
        'pos_std': float(np.std(pos_scores)),
        'neg_std': float(np.std(neg_scores)),
        'fpr': fpr,
        'tpr': tpr,
    }


def measure_speed(
    model,
    reid_model,
    image: np.ndarray,
    boxes: np.ndarray,
    n_runs: int = 50
) -> float:
    """Measure feature extraction speed."""
    # Warmup
    for _ in range(5):
        _ = reid_model.extract_appearance_features(image, boxes)

    # Timed runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for _ in range(n_runs):
        _ = reid_model.extract_appearance_features(image, boxes)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000  # ms per extraction


def visualize_comparison(
    results: Dict[str, Dict],
    features_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: str
):
    """Create comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'LITE': 'blue', 'LITE++_attention': 'red', 'LITE++_concat': 'green', 'LITE++_adaptive': 'orange'}

    for name, metrics in results.items():
        if 'fpr' in metrics:
            color = colors.get(name, 'gray')
            ax.plot(metrics['fpr'], metrics['tpr'],
                   label=f"{name} (AUC={metrics['auc']:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: LITE vs LITE++ Variants', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150)
    plt.close()

    # 2. Score Distributions
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, results.items()):
        pos_mean, pos_std = metrics['pos_mean'], metrics.get('pos_std', 0)
        neg_mean, neg_std = metrics['neg_mean'], metrics.get('neg_std', 0)

        ax.bar(['Positive', 'Negative'], [pos_mean, neg_mean],
               yerr=[pos_std, neg_std], capsize=5, color=['green', 'red'], alpha=0.7)
        ax.set_title(f'{name}', fontsize=12)
        ax.set_ylabel('Cosine Similarity')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'), dpi=150)
    plt.close()

    # 3. t-SNE Visualization (for first variant)
    for name, (features, track_ids) in features_dict.items():
        if len(features) < 50:
            continue

        # Subsample if too many
        if len(features) > 500:
            idx = np.random.choice(len(features), 500, replace=False)
            features_sub = features[idx]
            track_ids_sub = track_ids[idx]
        else:
            features_sub = features
            track_ids_sub = track_ids

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedded = tsne.fit_transform(features_sub)

        fig, ax = plt.subplots(figsize=(10, 8))
        unique_ids = np.unique(track_ids_sub)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_ids)))

        for idx, tid in enumerate(unique_ids[:20]):  # Limit to 20 tracks
            mask = track_ids_sub == tid
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                      c=[colors[idx]], label=f'ID {tid}', alpha=0.7)

        ax.set_title(f't-SNE Visualization: {name}', fontsize=14)
        ax.legend(loc='best', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tsne_{name.replace("+", "plus")}.png'), dpi=150)
        plt.close()

    # 4. Summary Table
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("LITE vs LITE++ Comparison Summary\n")
        f.write("=" * 60 + "\n\n")

        for name, metrics in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  ROC-AUC: {metrics['auc']:.4f}\n")
            f.write(f"  Positive Mean: {metrics['pos_mean']:.4f}\n")
            f.write(f"  Negative Mean: {metrics['neg_mean']:.4f}\n")
            f.write(f"  Gap (Pos-Neg): {metrics['pos_mean'] - metrics['neg_mean']:.4f}\n")
            if 'speed_ms' in metrics:
                f.write(f"  Speed: {metrics['speed_ms']:.2f} ms\n")

    print(f"\nResults saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare LITE vs LITE++ variants')
    parser.add_argument('--dataset', type=str, default='MOT17')
    parser.add_argument('--seq_name', type=str, default='MOT17-02-FRCNN')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--max_frames', type=int, default=100)
    parser.add_argument('--yolo_model', type=str, default='yolov8m')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='results/lite_comparison')
    args = parser.parse_args()

    # Setup paths
    seq_dir = f'datasets/{args.dataset}/{args.split}/{args.seq_name}'
    gt_path = os.path.join(seq_dir, 'gt/gt.txt')

    if not os.path.exists(seq_dir):
        print(f"Error: Sequence directory not found: {seq_dir}")
        print("Please ensure datasets are properly set up.")
        return

    # Load model
    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(f'{args.yolo_model}.pt')
    model.to(args.device)

    # Load ground truth
    print(f"Loading ground truth from: {gt_path}")
    gt_by_frame = load_ground_truth(gt_path)
    print(f"Loaded {len(gt_by_frame)} frames with annotations")

    # Initialize ReID models
    print("\nInitializing ReID models...")

    models_to_test = {
        'LITE': LITE(
            model=model,
            appearance_feature_layer='layer14',
            device=args.device
        ),
        'LITE++_attention': create_lite_plus(
            model=model,
            variant='attention',
            layers=['layer4', 'layer9', 'layer14'],
            output_dim=128,
            device=args.device
        ),
        'LITE++_concat': create_lite_plus(
            model=model,
            variant='concat',
            layers=['layer4', 'layer9', 'layer14'],
            output_dim=128,
            device=args.device
        ),
        'LITE++_adaptive': create_lite_plus(
            model=model,
            variant='adaptive',
            layers=['layer4', 'layer9', 'layer14'],
            output_dim=128,
            device=args.device
        ),
    }

    # Run comparison
    results = {}
    features_dict = {}

    for name, reid_model in models_to_test.items():
        print(f"\n{'=' * 40}")
        print(f"Testing: {name}")
        print('=' * 40)

        # Extract features
        print("  Extracting features...")
        features, track_ids = extract_features_for_sequence(
            model, reid_model, seq_dir, gt_by_frame, args.max_frames
        )

        if len(features) == 0:
            print(f"  No features extracted for {name}")
            continue

        print(f"  Extracted {len(features)} features from {len(np.unique(track_ids))} tracks")

        # Compute metrics
        print("  Computing ReID metrics...")
        metrics = compute_reid_metrics(features, track_ids)

        # Measure speed
        print("  Measuring speed...")
        sample_image = cv2.imread(os.path.join(seq_dir, 'img1', f'{min(gt_by_frame.keys()):06d}.jpg'))
        sample_boxes = np.array([ann['bbox'] for ann in list(gt_by_frame.values())[0]])
        metrics['speed_ms'] = measure_speed(model, reid_model, sample_image, sample_boxes)

        results[name] = metrics
        features_dict[name] = (features, track_ids)

        # Print results
        print(f"\n  Results for {name}:")
        print(f"    ROC-AUC: {metrics['auc']:.4f}")
        print(f"    Positive Mean: {metrics['pos_mean']:.4f}")
        print(f"    Negative Mean: {metrics['neg_mean']:.4f}")
        print(f"    Speed: {metrics['speed_ms']:.2f} ms")

    # Generate visualizations
    print("\n" + "=" * 40)
    print("Generating visualizations...")
    output_dir = os.path.join(args.output_dir, args.seq_name)
    visualize_comparison(results, features_dict, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        gap = metrics['pos_mean'] - metrics['neg_mean']
        print(f"{name:25s} | AUC: {metrics['auc']:.3f} | Gap: {gap:.3f} | Speed: {metrics['speed_ms']:.1f}ms")


if __name__ == '__main__':
    main()
