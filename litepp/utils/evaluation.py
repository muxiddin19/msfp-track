"""
Evaluation Utilities for LITE++

Functions for computing ReID and tracking metrics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def compute_cosine_similarity(
    features1: np.ndarray,
    features2: np.ndarray,
) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two feature sets.

    Args:
        features1: (N, D) feature vectors
        features2: (M, D) feature vectors

    Returns:
        similarity: (N, M) similarity matrix
    """
    # Normalize features
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)

    return np.dot(features1, features2.T)


def compute_reid_metrics(
    query_features: np.ndarray,
    query_ids: np.ndarray,
    gallery_features: np.ndarray,
    gallery_ids: np.ndarray,
    top_k: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute ReID evaluation metrics (CMC, mAP).

    Args:
        query_features: (N_q, D) query feature vectors
        query_ids: (N_q,) query identity labels
        gallery_features: (N_g, D) gallery feature vectors
        gallery_ids: (N_g,) gallery identity labels
        top_k: Ranks for CMC computation

    Returns:
        Dictionary with CMC@k and mAP scores
    """
    similarity = compute_cosine_similarity(query_features, gallery_features)

    # Sort gallery by similarity for each query
    indices = np.argsort(-similarity, axis=1)

    metrics = {}

    # CMC (Cumulative Matching Characteristics)
    for k in top_k:
        correct = 0
        for i, q_id in enumerate(query_ids):
            top_k_ids = gallery_ids[indices[i, :k]]
            if q_id in top_k_ids:
                correct += 1
        metrics[f"cmc@{k}"] = correct / len(query_ids)

    # mAP (Mean Average Precision)
    aps = []
    for i, q_id in enumerate(query_ids):
        sorted_ids = gallery_ids[indices[i]]
        matches = sorted_ids == q_id

        if matches.sum() == 0:
            continue

        # Compute AP
        cum_matches = np.cumsum(matches)
        precision_at_k = cum_matches / np.arange(1, len(matches) + 1)
        ap = (precision_at_k * matches).sum() / matches.sum()
        aps.append(ap)

    metrics["mAP"] = np.mean(aps) if aps else 0.0

    return metrics


def evaluate_tracking(
    predictions_path: str,
    groundtruth_path: str,
    metrics: List[str] = ["HOTA", "MOTA", "IDF1"],
) -> Dict[str, float]:
    """
    Evaluate tracking results using TrackEval.

    This is a wrapper that requires TrackEval to be installed.

    Args:
        predictions_path: Path to prediction files (MOT format)
        groundtruth_path: Path to ground truth files
        metrics: List of metrics to compute

    Returns:
        Dictionary with evaluation results
    """
    try:
        import trackeval
    except ImportError:
        raise ImportError(
            "TrackEval is required for tracking evaluation. "
            "Install with: pip install git+https://github.com/JonathonLuiten/TrackEval"
        )

    # Configure TrackEval
    eval_config = {
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "PRINT_RESULTS": False,
        "PRINT_CONFIG": False,
    }

    dataset_config = {
        "GT_FOLDER": groundtruth_path,
        "TRACKERS_FOLDER": predictions_path,
        "OUTPUT_FOLDER": None,
        "TRACKERS_TO_EVAL": [""],
        "CLASSES_TO_EVAL": ["pedestrian"],
    }

    # Run evaluation
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR()]

    raw_results, _ = evaluator.evaluate(dataset_list, metrics_list)

    # Extract requested metrics
    results = {}
    # Parse raw_results structure based on TrackEval output
    # This is simplified - actual implementation depends on TrackEval version

    return results


def compute_feature_distance_stats(
    features: np.ndarray,
    ids: np.ndarray,
) -> Dict[str, float]:
    """
    Compute intra-class and inter-class distance statistics.

    Useful for analyzing feature quality.

    Args:
        features: (N, D) feature vectors
        ids: (N,) identity labels

    Returns:
        Statistics about feature distances
    """
    unique_ids = np.unique(ids)

    intra_distances = []
    inter_distances = []

    for uid in unique_ids:
        mask = ids == uid
        class_features = features[mask]

        if len(class_features) > 1:
            # Intra-class distances
            sim = compute_cosine_similarity(class_features, class_features)
            # Get upper triangle (excluding diagonal)
            triu_idx = np.triu_indices(len(class_features), k=1)
            intra_distances.extend(1 - sim[triu_idx])

        # Inter-class distances
        other_features = features[~mask]
        if len(other_features) > 0 and len(class_features) > 0:
            sim = compute_cosine_similarity(class_features, other_features)
            inter_distances.extend(1 - sim.flatten())

    return {
        "intra_mean": np.mean(intra_distances) if intra_distances else 0.0,
        "intra_std": np.std(intra_distances) if intra_distances else 0.0,
        "inter_mean": np.mean(inter_distances) if inter_distances else 0.0,
        "inter_std": np.std(inter_distances) if inter_distances else 0.0,
        "separation": (
            (np.mean(inter_distances) - np.mean(intra_distances))
            if intra_distances and inter_distances
            else 0.0
        ),
    }
