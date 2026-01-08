"""
Training Script for Adaptive Threshold Module

This script trains the scene-aware threshold predictor using:
1. Grid search to find optimal thresholds per sequence
2. Supervised learning to train the predictor

Usage:
    python experiments/train_adaptive_threshold.py \
        --dataset MOT17 \
        --split train \
        --output_dir checkpoints/adaptive_threshold
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import subprocess
import tempfile

from ultralytics import YOLO
from reid_modules import (
    AdaptiveThresholdModule,
    MultiThresholdPredictor,
    AdaptiveThresholdLoss,
    create_adaptive_threshold_module
)


class ThresholdDataset(Dataset):
    """
    Dataset for training adaptive threshold module.

    Each sample contains:
    - Scene features from backbone
    - Optimal threshold found via grid search
    """

    def __init__(
        self,
        features_dir: str,
        threshold_labels: Dict[str, float]
    ):
        self.features_dir = Path(features_dir)
        self.threshold_labels = threshold_labels

        # List all feature files
        self.samples = []
        for seq_name, threshold in threshold_labels.items():
            seq_features_dir = self.features_dir / seq_name
            if seq_features_dir.exists():
                for feat_file in seq_features_dir.glob('*.npy'):
                    self.samples.append({
                        'feature_path': feat_file,
                        'threshold': threshold,
                        'seq_name': seq_name
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load feature
        features = np.load(sample['feature_path'])
        features = torch.from_numpy(features).float()

        # Get threshold
        threshold = torch.tensor([sample['threshold']], dtype=torch.float32)

        return features, threshold


def extract_scene_features(
    model,
    seq_dir: str,
    output_dir: str,
    feature_layer: str = 'layer14',
    sample_interval: int = 10,
    device: str = 'cuda:0'
):
    """
    Extract scene features for all frames in a sequence.

    Args:
        model: YOLO model
        seq_dir: Path to sequence directory
        output_dir: Directory to save features
        feature_layer: Which backbone layer to use
        sample_interval: Sample every N frames
        device: Device to use
    """
    os.makedirs(output_dir, exist_ok=True)

    img_dir = os.path.join(seq_dir, 'img1')
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    layer_idx = int(feature_layer.replace('layer', ''))

    # Register hook to capture features
    features_captured = []

    def hook_fn(module, input, output):
        features_captured.append(output.detach().cpu())

    # Find and register hook
    hook_handle = None
    if hasattr(model, 'model') and hasattr(model.model, 'model'):
        hook_handle = model.model.model[layer_idx].register_forward_hook(hook_fn)

    try:
        for i, frame_file in enumerate(tqdm(frame_files, desc='Extracting features')):
            if i % sample_interval != 0:
                continue

            img_path = os.path.join(img_dir, frame_file)
            image = cv2.imread(img_path)

            # Clear previous features
            features_captured.clear()

            # Run model
            with torch.no_grad():
                _ = model.predict(image, verbose=False)

            # Save features
            if features_captured:
                feat = features_captured[0].numpy()
                frame_idx = os.path.splitext(frame_file)[0]
                np.save(os.path.join(output_dir, f'{frame_idx}.npy'), feat)

    finally:
        if hook_handle:
            hook_handle.remove()


def grid_search_threshold(
    seq_dir: str,
    model_path: str,
    thresholds: List[float] = None,
    tracker_name: str = 'LITE:DeepSORT',
    device: str = 'cuda:0'
) -> Tuple[float, Dict[float, float]]:
    """
    Find optimal threshold for a sequence via grid search.

    Returns:
        best_threshold: Optimal threshold
        results: Dictionary mapping threshold -> HOTA score
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    results = {}
    seq_name = os.path.basename(seq_dir)

    for thresh in tqdm(thresholds, desc=f'Grid search for {seq_name}'):
        # Run tracker with this threshold
        # This is a simplified version - in practice, use proper evaluation
        try:
            cmd = [
                'python', 'track.py',
                '--yolo_model', model_path.replace('.pt', ''),
                '--tracker_name', tracker_name,
                '--dataset', 'MOT17',
                '--split', 'train',
                '--sequences', seq_name,
                '--min_confidence', str(thresh),
                '--device', device
            ]

            # Run tracking
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse HOTA from output (simplified)
            # In practice, run TrackEval to get accurate HOTA
            hota = 0.0  # Placeholder
            results[thresh] = hota

        except Exception as e:
            print(f"Error with threshold {thresh}: {e}")
            results[thresh] = 0.0

    # Find best threshold
    best_threshold = max(results.keys(), key=lambda x: results[x])

    return best_threshold, results


def create_pseudo_labels(
    dataset: str = 'MOT17',
    split: str = 'train'
) -> Dict[str, float]:
    """
    Create pseudo-labels for threshold training.

    Based on empirical observations:
    - MOT17: 0.25 works well for most sequences
    - MOT20 (crowded): 0.05-0.10 works better
    - Sparse scenes: 0.30-0.40 works better
    """
    # Heuristic-based pseudo labels
    # These should be refined via grid search
    pseudo_labels = {
        # MOT17 sequences - medium density
        'MOT17-02-FRCNN': 0.25,  # Static camera, medium density
        'MOT17-04-FRCNN': 0.20,  # Crowded outdoor
        'MOT17-05-FRCNN': 0.30,  # Sparse, moving camera
        'MOT17-09-FRCNN': 0.25,  # Indoor, medium
        'MOT17-10-FRCNN': 0.25,  # Night, medium
        'MOT17-11-FRCNN': 0.30,  # Indoor, sparse
        'MOT17-13-FRCNN': 0.20,  # Very crowded

        # MOT20 sequences - high density
        'MOT20-01': 0.10,
        'MOT20-02': 0.08,
        'MOT20-03': 0.05,
        'MOT20-05': 0.07,
    }

    return pseudo_labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions, _ = model(features)

        # Compute loss
        loss = criterion(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            predictions, _ = model(features)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            total_mae += torch.abs(predictions - targets).mean().item()

    n = len(dataloader)
    return total_loss / n, total_mae / n


def main():
    parser = argparse.ArgumentParser(description='Train adaptive threshold module')
    parser.add_argument('--dataset', type=str, default='MOT17')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--yolo_model', type=str, default='yolov8m')
    parser.add_argument('--feature_layer', type=str, default='layer14')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='checkpoints/adaptive_threshold')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract features before training')
    parser.add_argument('--grid_search', action='store_true',
                       help='Run grid search to find optimal thresholds')
    args = parser.parse_args()

    # Setup directories
    os.makedirs(args.output_dir, exist_ok=True)
    features_dir = os.path.join(args.output_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)

    # Load YOLO model
    print(f"Loading YOLO model: {args.yolo_model}")
    model = YOLO(f'{args.yolo_model}.pt')
    model.to(args.device)

    # Get sequence list
    from opts import data as dataset_info
    sequences = dataset_info.get(args.dataset, {}).get(args.split, [])

    if not sequences:
        print(f"No sequences found for {args.dataset}/{args.split}")
        return

    print(f"Found {len(sequences)} sequences")

    # Step 1: Extract features (if requested)
    if args.extract_features:
        print("\n" + "=" * 50)
        print("Step 1: Extracting scene features")
        print("=" * 50)

        for seq_name in sequences:
            seq_dir = f'datasets/{args.dataset}/{args.split}/{seq_name}'
            seq_features_dir = os.path.join(features_dir, seq_name)

            if os.path.exists(seq_features_dir) and len(os.listdir(seq_features_dir)) > 0:
                print(f"Features already exist for {seq_name}, skipping...")
                continue

            print(f"\nExtracting features for {seq_name}")
            extract_scene_features(
                model, seq_dir, seq_features_dir,
                feature_layer=args.feature_layer,
                device=args.device
            )

    # Step 2: Create or load threshold labels
    print("\n" + "=" * 50)
    print("Step 2: Creating threshold labels")
    print("=" * 50)

    labels_path = os.path.join(args.output_dir, 'threshold_labels.json')

    if args.grid_search:
        print("Running grid search for optimal thresholds...")
        threshold_labels = {}
        for seq_name in sequences:
            seq_dir = f'datasets/{args.dataset}/{args.split}/{seq_name}'
            best_thresh, _ = grid_search_threshold(
                seq_dir, f'{args.yolo_model}.pt', device=args.device
            )
            threshold_labels[seq_name] = best_thresh
            print(f"  {seq_name}: {best_thresh:.3f}")

        # Save labels
        with open(labels_path, 'w') as f:
            json.dump(threshold_labels, f, indent=2)
    elif os.path.exists(labels_path):
        print(f"Loading labels from {labels_path}")
        with open(labels_path, 'r') as f:
            threshold_labels = json.load(f)
    else:
        print("Using pseudo-labels based on heuristics")
        threshold_labels = create_pseudo_labels(args.dataset, args.split)
        with open(labels_path, 'w') as f:
            json.dump(threshold_labels, f, indent=2)

    print(f"Threshold labels: {threshold_labels}")

    # Step 3: Create dataset and dataloader
    print("\n" + "=" * 50)
    print("Step 3: Creating dataset")
    print("=" * 50)

    dataset = ThresholdDataset(features_dir, threshold_labels)
    print(f"Dataset size: {len(dataset)} samples")

    if len(dataset) == 0:
        print("No samples found! Please extract features first with --extract_features")
        return

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Step 4: Initialize model
    print("\n" + "=" * 50)
    print("Step 4: Initializing adaptive threshold module")
    print("=" * 50)

    layer_channels = {
        'layer4': 48, 'layer9': 96, 'layer14': 192,
        'layer17': 384, 'layer20': 576
    }
    input_channels = layer_channels.get(args.feature_layer, 192)

    threshold_model = create_adaptive_threshold_module(
        input_channels=input_channels,
        variant='single',
        device=args.device
    )

    criterion = AdaptiveThresholdLoss(loss_type='supervised')
    optimizer = optim.Adam(threshold_model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Step 5: Train
    print("\n" + "=" * 50)
    print("Step 5: Training")
    print("=" * 50)

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            threshold_model, train_loader, criterion, optimizer, args.device
        )

        val_loss, val_mae = validate(
            threshold_model, val_loader, criterion, args.device
        )

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                threshold_model.state_dict(),
                os.path.join(args.output_dir, 'best_model.pth')
            )

    # Save final model
    torch.save(
        threshold_model.state_dict(),
        os.path.join(args.output_dir, 'final_model.pth')
    )

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
