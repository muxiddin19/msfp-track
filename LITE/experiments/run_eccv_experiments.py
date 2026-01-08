"""
ECCV 2026 Experiment Runner: LITE++ Comprehensive Evaluation

This script runs all experiments needed for the ECCV 2026 submission:
1. Baseline comparison (LITE vs LITE+ vs LITE++)
2. Fusion strategy ablation (concat vs attention vs adaptive)
3. Layer combination ablation (single vs dual vs triple layer)
4. Adaptive threshold evaluation
5. Speed benchmarking

Usage:
    python experiments/run_eccv_experiments.py \
        --experiment all \
        --dataset MOT17 \
        --output_dir results/eccv2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

import torch
import cv2
from tqdm import tqdm

from ultralytics import YOLO
from reid_modules import (
    LITE,
    LITEPlus,
    create_lite_plus,
    LITEPlusPlusUnified,
    create_lite_plus_plus
)


class ExperimentConfig:
    """Configuration for ECCV 2026 experiments."""

    # Dataset configurations
    DATASETS = {
        'MOT17': {
            'train': [
                'MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN',
                'MOT17-09-FRCNN', 'MOT17-10-FRCNN', 'MOT17-11-FRCNN', 'MOT17-13-FRCNN'
            ],
            'test': [
                'MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN',
                'MOT17-07-FRCNN', 'MOT17-08-FRCNN', 'MOT17-12-FRCNN', 'MOT17-14-FRCNN'
            ]
        },
        'MOT20': {
            'train': ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05'],
            'test': ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']
        },
        'DanceTrack': {
            'train': [],  # Add DanceTrack sequences
            'test': []
        }
    }

    # Model variants to test
    VARIANTS = {
        'LITE': {
            'type': 'single_layer',
            'layer': 'layer14',
            'description': 'Original LITE (single layer)'
        },
        'LITE+_concat': {
            'type': 'multi_layer',
            'fusion': 'concat',
            'layers': ['layer4', 'layer9', 'layer14'],
            'description': 'LITE+ with concatenation fusion'
        },
        'LITE+_attention': {
            'type': 'multi_layer',
            'fusion': 'attention',
            'layers': ['layer4', 'layer9', 'layer14'],
            'description': 'LITE+ with attention fusion'
        },
        'LITE+_adaptive': {
            'type': 'multi_layer',
            'fusion': 'adaptive',
            'layers': ['layer4', 'layer9', 'layer14'],
            'description': 'LITE+ with adaptive (SE-style) fusion'
        },
        'LITE++_full': {
            'type': 'unified',
            'fusion': 'attention',
            'layers': ['layer4', 'layer9', 'layer14'],
            'adaptive_threshold': True,
            'description': 'LITE++ with attention fusion + adaptive thresholds'
        }
    }

    # Ablation: layer combinations
    LAYER_ABLATIONS = [
        ['layer14'],  # Single (original)
        ['layer9', 'layer14'],  # Dual (mid + late)
        ['layer4', 'layer14'],  # Dual (early + late)
        ['layer4', 'layer9', 'layer14'],  # Triple (all)
        ['layer4', 'layer9', 'layer14', 'layer17'],  # Quad
    ]

    # Threshold ablation
    THRESHOLD_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def run_tracking_experiment(
    seq_dir: str,
    reid_variant: str,
    yolo_model: str = 'yolov8m',
    device: str = 'cuda:0',
    confidence: float = 0.25,
    output_file: Optional[str] = None
) -> Dict:
    """
    Run tracking on a single sequence with specified ReID variant.

    Returns dictionary with timing and tracking results.
    """
    results = {
        'sequence': os.path.basename(seq_dir),
        'variant': reid_variant,
        'confidence': confidence,
        'frames': 0,
        'detections': 0,
        'tracks': 0,
        'feature_time_ms': 0,
        'total_time_s': 0,
    }

    try:
        # Load model
        model = YOLO(f'{yolo_model}.pt')
        model.to(device)

        # Initialize ReID module based on variant
        variant_config = ExperimentConfig.VARIANTS.get(reid_variant)
        if variant_config is None:
            raise ValueError(f"Unknown variant: {reid_variant}")

        if variant_config['type'] == 'single_layer':
            reid_model = LITE(
                model=model,
                appearance_feature_layer=variant_config['layer'],
                device=device
            )
        elif variant_config['type'] == 'multi_layer':
            reid_model = create_lite_plus(
                model=model,
                variant=variant_config['fusion'],
                layers=variant_config['layers'],
                device=device
            )
        elif variant_config['type'] == 'unified':
            reid_model = create_lite_plus_plus(
                model=model,
                fusion_type=variant_config['fusion'],
                layers=variant_config['layers'],
                enable_adaptive_threshold=variant_config.get('adaptive_threshold', False),
                device=device
            )

        # Get frame list
        img_dir = os.path.join(seq_dir, 'img1')
        frames = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

        # Run tracking
        feature_times = []
        total_detections = 0

        start_time = time.perf_counter()

        for frame_file in tqdm(frames, desc=f'{os.path.basename(seq_dir)}', leave=False):
            img_path = os.path.join(img_dir, frame_file)
            image = cv2.imread(img_path)

            # Run detection
            yolo_results = model.predict(
                image,
                classes=[0],  # Person class
                conf=confidence,
                verbose=False
            )

            boxes = yolo_results[0].boxes.data.cpu().numpy()
            total_detections += len(boxes)

            # Extract features
            if len(boxes) > 0:
                feat_start = time.perf_counter()
                features = reid_model.extract_appearance_features(image, boxes)
                feat_end = time.perf_counter()
                feature_times.append((feat_end - feat_start) * 1000)

        end_time = time.perf_counter()

        results['frames'] = len(frames)
        results['detections'] = total_detections
        results['feature_time_ms'] = np.mean(feature_times) if feature_times else 0
        results['total_time_s'] = end_time - start_time
        results['fps'] = len(frames) / (end_time - start_time)

    except Exception as e:
        results['error'] = str(e)

    return results


def run_reid_quality_experiment(
    seq_dir: str,
    gt_path: str,
    reid_variant: str,
    yolo_model: str = 'yolov8m',
    device: str = 'cuda:0',
    max_frames: int = 100
) -> Dict:
    """
    Evaluate ReID quality by computing match scores on ground truth boxes.

    Returns dictionary with ROC-AUC, positive/negative scores.
    """
    from sklearn.metrics import roc_curve, auc

    results = {
        'sequence': os.path.basename(seq_dir),
        'variant': reid_variant,
    }

    try:
        # Load model
        model = YOLO(f'{yolo_model}.pt')
        model.to(device)

        # Initialize ReID module
        variant_config = ExperimentConfig.VARIANTS.get(reid_variant)

        if variant_config['type'] == 'single_layer':
            reid_model = LITE(model=model, appearance_feature_layer=variant_config['layer'], device=device)
        elif variant_config['type'] == 'multi_layer':
            reid_model = create_lite_plus(model=model, variant=variant_config['fusion'],
                                          layers=variant_config['layers'], device=device)
        elif variant_config['type'] == 'unified':
            reid_model = create_lite_plus_plus(model=model, fusion_type=variant_config['fusion'],
                                               layers=variant_config['layers'], device=device)

        # Load ground truth
        gt_by_frame = defaultdict(list)
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])

                if len(parts) > 7:
                    consider = int(parts[6])
                    cls = int(parts[7])
                    visibility = float(parts[8]) if len(parts) > 8 else 1.0
                    if consider == 0 or cls != 1 or visibility < 0.25:
                        continue

                gt_by_frame[frame_id].append({
                    'track_id': track_id,
                    'bbox': [x, y, x + w, y + h, 1.0, 0]
                })

        # Extract features
        all_features = []
        all_track_ids = []

        img_dir = os.path.join(seq_dir, 'img1')
        frame_ids = sorted(gt_by_frame.keys())[:max_frames]

        for frame_id in tqdm(frame_ids, desc='Extracting features', leave=False):
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
            results['error'] = 'No features extracted'
            return results

        features = np.vstack(all_features)
        track_ids = np.array(all_track_ids)

        # Compute similarity matrix
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        similarity = features_norm @ features_norm.T

        # Separate positive and negative pairs
        n = len(track_ids)
        pos_scores, neg_scores = [], []

        for i in range(n):
            for j in range(i + 1, n):
                if track_ids[i] == track_ids[j]:
                    pos_scores.append(similarity[i, j])
                else:
                    neg_scores.append(similarity[i, j])

        # Subsample negatives
        if len(neg_scores) > len(pos_scores) * 10:
            neg_scores = np.random.choice(neg_scores, size=len(pos_scores) * 10, replace=False)

        if len(pos_scores) > 0:
            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
            y_scores = np.concatenate([pos_scores, neg_scores])

            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            results['auc'] = roc_auc
            results['pos_mean'] = float(np.mean(pos_scores))
            results['neg_mean'] = float(np.mean(neg_scores))
            results['pos_std'] = float(np.std(pos_scores))
            results['neg_std'] = float(np.std(neg_scores))
            results['gap'] = results['pos_mean'] - results['neg_mean']
        else:
            results['auc'] = 0.5
            results['pos_mean'] = 0.0
            results['neg_mean'] = float(np.mean(neg_scores)) if neg_scores else 0.0

    except Exception as e:
        results['error'] = str(e)

    return results


def run_baseline_comparison(
    dataset: str,
    split: str,
    output_dir: str,
    yolo_model: str = 'yolov8m',
    device: str = 'cuda:0'
):
    """Run baseline comparison: LITE vs LITE+ vs LITE++"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline Comparison")
    print("=" * 60)

    results = defaultdict(list)
    sequences = ExperimentConfig.DATASETS.get(dataset, {}).get(split, [])

    for seq_name in sequences:
        seq_dir = f'datasets/{dataset}/{split}/{seq_name}'
        gt_path = os.path.join(seq_dir, 'gt/gt.txt')

        if not os.path.exists(seq_dir):
            print(f"Skipping {seq_name}: directory not found")
            continue

        print(f"\nProcessing: {seq_name}")

        for variant in ['LITE', 'LITE+_attention', 'LITE++_full']:
            res = run_reid_quality_experiment(
                seq_dir, gt_path, variant, yolo_model, device
            )
            results[variant].append(res)
            print(f"  {variant}: AUC={res.get('auc', 0):.4f}, Gap={res.get('gap', 0):.4f}")

    # Save results
    output_path = os.path.join(output_dir, 'baseline_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(dict(results), f, indent=2)

    # Print summary
    print("\n" + "-" * 60)
    print("SUMMARY: Baseline Comparison")
    print("-" * 60)
    print(f"{'Variant':<20} | {'Avg AUC':>10} | {'Avg Gap':>10}")
    print("-" * 60)

    for variant, res_list in results.items():
        aucs = [r.get('auc', 0) for r in res_list if 'error' not in r]
        gaps = [r.get('gap', 0) for r in res_list if 'error' not in r]
        avg_auc = np.mean(aucs) if aucs else 0
        avg_gap = np.mean(gaps) if gaps else 0
        print(f"{variant:<20} | {avg_auc:>10.4f} | {avg_gap:>10.4f}")

    return results


def run_fusion_ablation(
    dataset: str,
    split: str,
    output_dir: str,
    yolo_model: str = 'yolov8m',
    device: str = 'cuda:0'
):
    """Run fusion strategy ablation: concat vs attention vs adaptive"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Fusion Strategy Ablation")
    print("=" * 60)

    results = defaultdict(list)
    sequences = ExperimentConfig.DATASETS.get(dataset, {}).get(split, [])[:3]  # Subset for speed

    for seq_name in sequences:
        seq_dir = f'datasets/{dataset}/{split}/{seq_name}'
        gt_path = os.path.join(seq_dir, 'gt/gt.txt')

        if not os.path.exists(seq_dir):
            continue

        print(f"\nProcessing: {seq_name}")

        for variant in ['LITE+_concat', 'LITE+_attention', 'LITE+_adaptive']:
            res = run_reid_quality_experiment(
                seq_dir, gt_path, variant, yolo_model, device
            )
            results[variant].append(res)
            print(f"  {variant}: AUC={res.get('auc', 0):.4f}")

    output_path = os.path.join(output_dir, 'fusion_ablation.json')
    with open(output_path, 'w') as f:
        json.dump(dict(results), f, indent=2)

    return results


def run_layer_ablation(
    dataset: str,
    split: str,
    output_dir: str,
    yolo_model: str = 'yolov8m',
    device: str = 'cuda:0'
):
    """Run layer combination ablation"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Layer Combination Ablation")
    print("=" * 60)

    results = {}
    sequences = ExperimentConfig.DATASETS.get(dataset, {}).get(split, [])[:2]  # Subset

    # Load model once
    model = YOLO(f'{yolo_model}.pt')
    model.to(device)

    for layers in ExperimentConfig.LAYER_ABLATIONS:
        layer_key = '+'.join(layers)
        print(f"\nTesting layers: {layer_key}")

        layer_results = []

        for seq_name in sequences:
            seq_dir = f'datasets/{dataset}/{split}/{seq_name}'
            gt_path = os.path.join(seq_dir, 'gt/gt.txt')

            if not os.path.exists(seq_dir):
                continue

            try:
                if len(layers) == 1:
                    reid_model = LITE(model=model, appearance_feature_layer=layers[0], device=device)
                else:
                    reid_model = create_lite_plus(model=model, variant='attention',
                                                  layers=layers, device=device)

                # Quick evaluation (subset of frames)
                res = run_reid_quality_experiment(
                    seq_dir, gt_path, 'custom', yolo_model, device, max_frames=50
                )
                res['layers'] = layers
                layer_results.append(res)

            except Exception as e:
                print(f"  Error: {e}")

        results[layer_key] = layer_results

    output_path = os.path.join(output_dir, 'layer_ablation.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_speed_benchmark(
    dataset: str,
    split: str,
    output_dir: str,
    yolo_model: str = 'yolov8m',
    device: str = 'cuda:0',
    n_warmup: int = 10,
    n_runs: int = 100
):
    """Benchmark speed of different variants"""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Speed Benchmark")
    print("=" * 60)

    # Get a sample sequence
    sequences = ExperimentConfig.DATASETS.get(dataset, {}).get(split, [])
    if not sequences:
        print("No sequences available")
        return {}

    seq_name = sequences[0]
    seq_dir = f'datasets/{dataset}/{split}/{seq_name}'
    img_dir = os.path.join(seq_dir, 'img1')
    frames = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])[:50]

    # Load sample image and get sample boxes
    sample_img = cv2.imread(os.path.join(img_dir, frames[0]))

    model = YOLO(f'{yolo_model}.pt')
    model.to(device)

    yolo_results = model.predict(sample_img, classes=[0], conf=0.25, verbose=False)
    sample_boxes = yolo_results[0].boxes.data.cpu().numpy()

    print(f"Sample boxes: {len(sample_boxes)}")

    results = {}

    for variant_name, variant_config in ExperimentConfig.VARIANTS.items():
        print(f"\nBenchmarking: {variant_name}")

        try:
            # Create model
            if variant_config['type'] == 'single_layer':
                reid_model = LITE(model=model, appearance_feature_layer=variant_config['layer'], device=device)
            elif variant_config['type'] == 'multi_layer':
                reid_model = create_lite_plus(model=model, variant=variant_config['fusion'],
                                              layers=variant_config['layers'], device=device)
            elif variant_config['type'] == 'unified':
                reid_model = create_lite_plus_plus(model=model, fusion_type=variant_config['fusion'],
                                                   layers=variant_config['layers'],
                                                   enable_adaptive_threshold=variant_config.get('adaptive_threshold', False),
                                                   device=device)

            # Warmup
            for _ in range(n_warmup):
                _ = reid_model.extract_appearance_features(sample_img, sample_boxes)

            # Timed runs
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            times = []
            for _ in tqdm(range(n_runs), desc='Timing', leave=False):
                start = time.perf_counter()
                _ = reid_model.extract_appearance_features(sample_img, sample_boxes)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)

            results[variant_name] = {
                'mean_ms': float(np.mean(times)),
                'std_ms': float(np.std(times)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times)),
                'n_boxes': len(sample_boxes)
            }

            print(f"  Time: {np.mean(times):.2f} +/- {np.std(times):.2f} ms")

        except Exception as e:
            print(f"  Error: {e}")
            results[variant_name] = {'error': str(e)}

    output_path = os.path.join(output_dir, 'speed_benchmark.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "-" * 60)
    print("SUMMARY: Speed Benchmark")
    print("-" * 60)
    print(f"{'Variant':<25} | {'Mean (ms)':>10} | {'Std (ms)':>10}")
    print("-" * 60)

    for variant, res in results.items():
        if 'error' not in res:
            print(f"{variant:<25} | {res['mean_ms']:>10.2f} | {res['std_ms']:>10.2f}")

    return results


def generate_paper_tables(output_dir: str):
    """Generate LaTeX tables for the paper."""
    print("\n" + "=" * 60)
    print("Generating Paper Tables")
    print("=" * 60)

    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)

    # Load results
    results_files = {
        'baseline': os.path.join(output_dir, 'baseline_comparison.json'),
        'fusion': os.path.join(output_dir, 'fusion_ablation.json'),
        'speed': os.path.join(output_dir, 'speed_benchmark.json'),
    }

    # Generate baseline comparison table
    if os.path.exists(results_files['baseline']):
        with open(results_files['baseline'], 'r') as f:
            baseline = json.load(f)

        table = r"""
\begin{table}[t]
\centering
\caption{Comparison of LITE variants on MOT17. AUC measures re-identification capability.}
\label{tab:baseline}
\begin{tabular}{lcccc}
\toprule
Method & AUC $\uparrow$ & Pos. Mean & Neg. Mean & Gap $\uparrow$ \\
\midrule
"""
        for variant, res_list in baseline.items():
            aucs = [r.get('auc', 0) for r in res_list if 'error' not in r]
            pos = [r.get('pos_mean', 0) for r in res_list if 'error' not in r]
            neg = [r.get('neg_mean', 0) for r in res_list if 'error' not in r]
            gaps = [r.get('gap', 0) for r in res_list if 'error' not in r]

            table += f"{variant} & {np.mean(aucs):.3f} & {np.mean(pos):.3f} & {np.mean(neg):.3f} & {np.mean(gaps):.3f} \\\\\n"

        table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        with open(os.path.join(tables_dir, 'baseline_table.tex'), 'w') as f:
            f.write(table)

    # Generate speed table
    if os.path.exists(results_files['speed']):
        with open(results_files['speed'], 'r') as f:
            speed = json.load(f)

        table = r"""
\begin{table}[t]
\centering
\caption{Speed comparison of LITE variants (ms per frame).}
\label{tab:speed}
\begin{tabular}{lcc}
\toprule
Method & Mean (ms) & Std (ms) \\
\midrule
"""
        for variant, res in speed.items():
            if 'error' not in res:
                table += f"{variant} & {res['mean_ms']:.2f} & {res['std_ms']:.2f} \\\\\n"

        table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
        with open(os.path.join(tables_dir, 'speed_table.tex'), 'w') as f:
            f.write(table)

    print(f"Tables saved to: {tables_dir}")


def main():
    parser = argparse.ArgumentParser(description='ECCV 2026 Experiment Runner')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['all', 'baseline', 'fusion', 'layer', 'speed', 'tables'])
    parser.add_argument('--dataset', type=str, default='MOT17')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--yolo_model', type=str, default='yolov8m')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='results/eccv2026')
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.dataset}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nECCV 2026 Experiment Runner")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"Device: {args.device}")

    # Save config
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Run experiments
    if args.experiment in ['all', 'baseline']:
        run_baseline_comparison(args.dataset, args.split, output_dir, args.yolo_model, args.device)

    if args.experiment in ['all', 'fusion']:
        run_fusion_ablation(args.dataset, args.split, output_dir, args.yolo_model, args.device)

    if args.experiment in ['all', 'layer']:
        run_layer_ablation(args.dataset, args.split, output_dir, args.yolo_model, args.device)

    if args.experiment in ['all', 'speed']:
        run_speed_benchmark(args.dataset, args.split, output_dir, args.yolo_model, args.device)

    if args.experiment in ['all', 'tables']:
        generate_paper_tables(output_dir)

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
