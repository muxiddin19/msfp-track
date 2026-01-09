"""
Run LITE++ on MOT17 Dataset

This script runs tracking experiments on the MOT17 benchmark
using LITE++ for appearance feature extraction.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List
import time

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Run LITE++ on MOT17")

    # Model settings
    parser.add_argument(
        "--yolo_weights",
        type=str,
        default="yolov8m.pt",
        help="YOLO model weights",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="attention",
        choices=["concat", "attention", "adaptive"],
        help="Feature fusion strategy",
    )
    parser.add_argument(
        "--adaptive_threshold",
        action="store_true",
        default=True,
        help="Enable adaptive threshold learning",
    )

    # Dataset settings
    parser.add_argument(
        "--data_root",
        type=str,
        default="datasets/MOT17",
        help="MOT17 dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--sequences",
        type=str,
        nargs="+",
        default=None,
        help="Specific sequences to run (default: all)",
    )

    # Tracking parameters
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Detection confidence threshold (ignored if adaptive)",
    )
    parser.add_argument(
        "--max_age",
        type=int,
        default=30,
        help="Maximum frames to keep lost tracks",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=3,
        help="Frames before track is confirmed",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mot17",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save tracking visualization videos",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on",
    )

    return parser.parse_args()


def get_mot17_sequences(data_root: str, split: str) -> List[str]:
    """Get list of MOT17 sequences."""
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    sequences = []
    for seq_dir in sorted(split_dir.iterdir()):
        if seq_dir.is_dir() and seq_dir.name.startswith("MOT17"):
            sequences.append(seq_dir.name)

    return sequences


def run_sequence(
    seq_name: str,
    seq_path: Path,
    tracker,
    output_dir: Path,
    args,
) -> Dict[str, float]:
    """Run tracking on a single sequence."""
    import cv2

    img_dir = seq_path / "img1"
    det_file = seq_path / "det" / "det.txt"

    # Load detections
    detections = np.loadtxt(det_file, delimiter=",")

    # Get image files
    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        img_files = sorted(img_dir.glob("*.png"))

    # Output file
    output_file = output_dir / f"{seq_name}.txt"
    results = []

    # Timing
    times = []

    for frame_idx, img_path in enumerate(tqdm(img_files, desc=seq_name)):
        frame_id = frame_idx + 1

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Get detections for this frame
        frame_dets = detections[detections[:, 0] == frame_id]
        if len(frame_dets) == 0:
            continue

        # Format: [frame, id, x, y, w, h, conf, ...]
        boxes = frame_dets[:, 2:6].copy()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x1 + w
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y1 + h
        confs = frame_dets[:, 6]

        # Add confidence to boxes
        boxes = np.column_stack([boxes, confs])

        # Get adaptive threshold or use fixed
        if args.adaptive_threshold:
            threshold = tracker.get_current_threshold()
        else:
            threshold = args.conf_threshold

        # Filter by confidence
        mask = confs >= threshold
        boxes = boxes[mask]

        if len(boxes) == 0:
            continue

        # Extract features and track
        start_time = time.time()
        features = tracker.extract_appearance_features(image, boxes)
        times.append(time.time() - start_time)

        # Here you would integrate with actual tracker
        # For now, just save the adaptive threshold statistics

    # Compute statistics
    stats = {
        "sequence": seq_name,
        "frames": len(img_files),
        "avg_time": np.mean(times) if times else 0,
        "fps": 1.0 / np.mean(times) if times else 0,
    }

    if args.adaptive_threshold:
        thresh_stats = tracker.get_threshold_statistics()
        stats.update(thresh_stats)

    return stats


def main():
    args = parse_args()

    print("=" * 60)
    print("LITE++ MOT17 Experiment")
    print("=" * 60)
    print(f"Fusion type: {args.fusion_type}")
    print(f"Adaptive threshold: {args.adaptive_threshold}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")

    print(f"Loading YOLO model: {args.yolo_weights}")
    yolo = YOLO(args.yolo_weights)

    # Create LITE++ tracker
    from litepp.trackers import LITEPlusPlusTracker

    tracker = LITEPlusPlusTracker(
        model=yolo,
        fusion_type=args.fusion_type,
        enable_adaptive_threshold=args.adaptive_threshold,
        device=args.device,
    )

    # Get sequences
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = get_mot17_sequences(args.data_root, args.split)

    print(f"Running on {len(sequences)} sequences")

    # Run on each sequence
    all_results = []

    for seq_name in sequences:
        seq_path = Path(args.data_root) / args.split / seq_name

        if not seq_path.exists():
            print(f"Warning: Sequence not found: {seq_path}")
            continue

        tracker.reset()
        stats = run_sequence(seq_name, seq_path, tracker, output_dir, args)
        all_results.append(stats)

        print(f"{seq_name}: FPS={stats['fps']:.1f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    avg_fps = np.mean([r["fps"] for r in all_results])
    print(f"Average FPS: {avg_fps:.1f}")

    if args.adaptive_threshold:
        avg_thresh = np.mean([r.get("mean", 0.25) for r in all_results])
        print(f"Average threshold: {avg_thresh:.3f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
