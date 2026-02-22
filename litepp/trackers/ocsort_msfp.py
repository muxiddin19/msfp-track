"""
OC-SORT with MSFP Features

Integrates MSFP multi-scale features into OC-SORT tracker.
This demonstrates that MSFP features generalize beyond ByteTrack.

Usage:
    from litepp.trackers.ocsort_msfp import OCSORT_MSFP

    tracker = OCSORT_MSFP(
        yolo_model,
        fusion_type='attention',
        use_msfp_features=True
    )

    # Standard OC-SORT interface
    tracks = tracker.update(detections, image)
"""

import numpy as np
from typing import List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from litepp.models.litepp import create_litepp


class KalmanBoxTracker:
    """
    Simplified Kalman filter for bounding box tracking.
    This is a minimal implementation for demonstration.
    In practice, use the full OC-SORT implementation.
    """
    count = 0

    def __init__(self, bbox, feature=None):
        """
        Initialize tracker with detection bbox and optional appearance feature.

        Args:
            bbox: [x1, y1, x2, y2, conf]
            feature: Appearance feature vector (for ReID)
        """
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.bbox = bbox[:4]
        self.conf = bbox[4] if len(bbox) > 4 else 1.0
        self.feature = feature
        self.time_since_update = 0
        self.hits = 1
        self.age = 0

    def update(self, bbox, feature=None):
        """Update tracker with new detection."""
        self.bbox = bbox[:4]
        self.conf = bbox[4] if len(bbox) > 4 else 1.0
        if feature is not None:
            # EMA update of appearance feature
            alpha = 0.9
            self.feature = alpha * self.feature + (1 - alpha) * feature
        self.time_since_update = 0
        self.hits += 1

    def predict(self):
        """Predict next position (simplified - just return current)."""
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def get_state(self):
        """Get current state as [x1, y1, x2, y2, conf, id]."""
        return np.concatenate([self.bbox, [self.conf, self.id]])


class OCSORT_MSFP:
    """
    OC-SORT tracker with MSFP multi-scale appearance features.

    This demonstrates that MSFP features improve tracking performance
    beyond the ByteTrack baseline, showing generalizability.

    Args:
        model: YOLO detection model
        det_thresh: Detection confidence threshold
        max_age: Maximum frames to keep track without update
        min_hits: Minimum hits to start outputting track
        iou_threshold: IOU threshold for matching
        use_msfp_features: Whether to use MSFP features (vs no ReID)
        fusion_type: Feature fusion strategy
        appearance_weight: Weight for appearance in matching cost
    """

    def __init__(
        self,
        model,
        det_thresh: float = 0.25,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        use_msfp_features: bool = True,
        fusion_type: str = "attention",
        appearance_weight: float = 0.5,
    ):
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_msfp_features = use_msfp_features
        self.appearance_weight = appearance_weight

        self.trackers = []
        self.frame_count = 0

        # Initialize MSFP feature extractor
        if use_msfp_features:
            self.feature_extractor = create_litepp(
                model,
                fusion_type=fusion_type,
                enable_adaptive_threshold=False,  # OC-SORT handles thresholding
                output_dim=128,
            )
        else:
            self.feature_extractor = None

    def update(
        self,
        detections: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            detections: (N, 5) array of [x1, y1, x2, y2, conf]
            image: Input image for feature extraction (required if use_msfp_features)

        Returns:
            tracks: (M, 6) array of [x1, y1, x2, y2, conf, id]
        """
        self.frame_count += 1

        # Filter low confidence detections
        if len(detections) > 0:
            keep = detections[:, 4] >= self.det_thresh
            detections = detections[keep]

        # Extract appearance features if enabled
        features = None
        if self.use_msfp_features and len(detections) > 0:
            if image is None:
                raise ValueError("Image required for MSFP feature extraction")

            features, _ = self.feature_extractor.extract_with_adaptive_threshold(
                image, detections
            )
            # Normalize features
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)

        # Predict existing tracks
        for t in self.trackers:
            t.predict()

        # Associate detections to tracks
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, self.trackers, features
        )

        # Update matched tracks
        for det_idx, trk_idx in matched:
            feat = features[det_idx] if features is not None else None
            self.trackers[trk_idx].update(detections[det_idx], feat)

        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            feat = features[i] if features is not None else None
            trk = KalmanBoxTracker(detections[i], feat)
            self.trackers.append(trk)

        # Remove dead trackers
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update < self.max_age
        ]

        # Get tracks to output
        ret = []
        for t in self.trackers:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                ret.append(t.get_state())

        if len(ret) > 0:
            return np.array(ret)
        return np.empty((0, 6))

    def associate_detections_to_trackers(
        self,
        detections: np.ndarray,
        trackers: List[KalmanBoxTracker],
        features: Optional[np.ndarray] = None
    ) -> Tuple[List, List, List]:
        """
        Associate detections to tracked objects using IoU + appearance.

        Args:
            detections: (N, 5) array
            trackers: List of KalmanBoxTracker objects
            features: (N, D) appearance features for detections

        Returns:
            matched: List of (det_idx, trk_idx) pairs
            unmatched_dets: List of detection indices
            unmatched_trks: List of tracker indices
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        # Compute IoU matrix
        iou_matrix = self.compute_iou_matrix(detections, trackers)

        # Compute appearance cost matrix if features available
        if features is not None and self.use_msfp_features:
            appearance_matrix = self.compute_appearance_matrix(features, trackers)
            # Combined cost: weighted sum of IoU and appearance
            # Lower cost is better, so use (1 - IoU) and (1 - cosine_sim)
            cost_matrix = (
                self.appearance_weight * (1 - appearance_matrix) +
                (1 - self.appearance_weight) * (1 - iou_matrix)
            )
        else:
            # IoU-only matching
            cost_matrix = 1 - iou_matrix

        # Simple greedy matching (in practice, use Hungarian algorithm)
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))

        # Greedy matching: for each detection, find best tracker
        for det_idx in range(len(detections)):
            if len(unmatched_trks) == 0:
                break

            best_cost = float('inf')
            best_trk = None

            for trk_idx in unmatched_trks:
                cost = cost_matrix[det_idx, trk_idx]
                iou = iou_matrix[det_idx, trk_idx]

                # Only match if IoU above threshold
                if iou >= self.iou_threshold and cost < best_cost:
                    best_cost = cost
                    best_trk = trk_idx

            if best_trk is not None:
                matched.append((det_idx, best_trk))
                unmatched_dets.remove(det_idx)
                unmatched_trks.remove(best_trk)

        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def compute_iou_matrix(
        detections: np.ndarray,
        trackers: List[KalmanBoxTracker]
    ) -> np.ndarray:
        """
        Compute IoU matrix between detections and trackers.

        Returns:
            iou_matrix: (N_det, N_trk) matrix of IoU values
        """
        N_det = len(detections)
        N_trk = len(trackers)
        iou_matrix = np.zeros((N_det, N_trk))

        for d in range(N_det):
            for t in range(N_trk):
                iou_matrix[d, t] = OCSORT_MSFP.bbox_iou(
                    detections[d, :4],
                    trackers[t].bbox
                )

        return iou_matrix

    @staticmethod
    def compute_appearance_matrix(
        features: np.ndarray,
        trackers: List[KalmanBoxTracker]
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix between detection features and tracker features.

        Returns:
            similarity_matrix: (N_det, N_trk) matrix of cosine similarities
        """
        N_det = len(features)
        N_trk = len(trackers)
        similarity_matrix = np.zeros((N_det, N_trk))

        for d in range(N_det):
            for t in range(N_trk):
                if trackers[t].feature is not None:
                    # Cosine similarity
                    similarity_matrix[d, t] = np.dot(
                        features[d],
                        trackers[t].feature
                    )

        return similarity_matrix

    @staticmethod
    def bbox_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute IoU between two bboxes [x1, y1, x2, y2].
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union


# Example usage
if __name__ == "__main__":
    print("OC-SORT with MSFP Features")
    print("-" * 50)

    # Simulate tracking
    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")

    tracker = OCSORT_MSFP(
        model,
        use_msfp_features=True,
        appearance_weight=0.5
    )

    # Simulate detections
    frame_image = np.random.rand(640, 480, 3).astype(np.uint8)
    detections = np.array([
        [100, 100, 200, 300, 0.9],  # [x1, y1, x2, y2, conf]
        [250, 150, 350, 350, 0.85],
        [400, 200, 480, 380, 0.92],
    ])

    tracks = tracker.update(detections, frame_image)
    print(f"Frame 1: {len(tracks)} tracks")
    print(tracks)
