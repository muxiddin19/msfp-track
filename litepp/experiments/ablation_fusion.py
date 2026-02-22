"""
Ablation Study: Feature Fusion Strategies

Compares different feature fusion strategies:
- concat: Simple concatenation + MLP
- attention: Learned attention weights per layer
- adaptive: Channel-wise SE-style attention
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List
import time

import numpy as np
import torch
from tqdm import tqdm
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Fusion Strategy Ablation")

    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="yolov8m.pt",
        help="YOLO model weights",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/MOT17",
        help="Dataset root",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN"],
        help="Sequences to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ablation_fusion",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device",
    )

    return parser.parse_args()


def evaluate_fusion_strategy(
    fusion_type: str,
    yolo,
    data_root: str,
    sequences: List[str],
    device: str,
) -> Dict[str, float]:
    """Evaluate a single fusion strategy."""
    from litepp import create_litepp

    # Create LITE++ with specified fusion
    litepp = create_litepp(
        model=yolo,
        fusion_type=fusion_type,
        enable_adaptive_threshold=False,  # Isolate fusion effect
        device=device,
    )

    results = {
        "fusion_type": fusion_type,
        "extraction_times": [],
        "feature_dims": [],
    }

    for seq_name in sequences:
        seq_path = Path(data_root) / "train" / seq_name
        img_dir = seq_path / "img1"
        det_file = seq_path / "det" / "det.txt"

        if not seq_path.exists():
            continue

        # Load detections
        detections = np.loadtxt(det_file, delimiter=",")

        # Sample frames
        img_files = sorted(img_dir.glob("*.jpg"))[:100]

        for img_path in tqdm(img_files, desc=f"{fusion_type}/{seq_name}", leave=False):
            import cv2

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            frame_id = int(img_path.stem)
            frame_dets = detections[detections[:, 0] == frame_id]

            if len(frame_dets) < 2:
                continue

            boxes = frame_dets[:, 2:6].copy()
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            # Time feature extraction
            start = time.time()
            features = litepp.extract_appearance_features(image, boxes)
            elapsed = time.time() - start

            results["extraction_times"].append(elapsed)
            if len(features) > 0:
                results["feature_dims"].append(features.shape[1])

    # Compute statistics
    results["avg_time_ms"] = np.mean(results["extraction_times"]) * 1000
    results["std_time_ms"] = np.std(results["extraction_times"]) * 1000
    results["fps"] = 1000 / results["avg_time_ms"]

    # Get attention weights if available
    if fusion_type == "attention":
        weights = litepp.get_fusion_weights()
        if weights is not None:
            results["attention_weights"] = weights.cpu().numpy().tolist()

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("Feature Fusion Ablation Study")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")

    print(f"Loading YOLO model: {args.yolo_weights}")
    yolo = YOLO(args.yolo_weights)

    # Test each fusion strategy
    fusion_types = ["concat", "attention", "adaptive"]
    all_results = []

    for fusion_type in fusion_types:
        print(f"\nEvaluating: {fusion_type}")
        results = evaluate_fusion_strategy(
            fusion_type=fusion_type,
            yolo=yolo,
            data_root=args.data_root,
            sequences=args.sequences,
            device=args.device,
        )
        all_results.append(results)

        print(f"  Avg time: {results['avg_time_ms']:.2f} ms")
        print(f"  FPS: {results['fps']:.1f}")

        if "attention_weights" in results:
            print(f"  Attention weights: {results['attention_weights']}")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Fusion':<12} {'Time (ms)':<12} {'FPS':<8}")
    print("-" * 32)

    for r in all_results:
        print(f"{r['fusion_type']:<12} {r['avg_time_ms']:<12.2f} {r['fps']:<8.1f}")

    # Save results
    import json

    results_file = output_dir / "ablation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
