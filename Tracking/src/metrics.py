"""Evaluation metrics.

For the ball, we use the standard TrackNet ε-threshold metric (a prediction
is correct if it falls within ε pixels of GT). For players, MOTA is the right
choice but it's not implemented here — we just provide a stub with a pointer
to motmetrics.
"""

from pathlib import Path
import numpy as np
from scipy.spatial import distance

def ball_threshold_metrics(pred: list, gt: list, epsilon: float = 10) -> dict:
    """Standard TrackNet-style metric.

    pred, gt: list of (x, y) or None, aligned frame-by-frame.

    Returns: dict with TP / FP / FN / TN / precision / recall / accuracy / F1
    and the mean pixel distance among matched detections.
    """
    assert len(pred) == len(gt), f"length mismatch: {len(pred)} vs {len(gt)}"
    TP = FP = FN = TN = 0
    dists: list[float] = []
    for p, g in zip(pred, gt):
        if g is None and p is None:
            TN += 1
        elif g is None and p is not None:
            FP += 1
        elif g is not None and p is None:
            FN += 1
        else:
            d = ((p[0] - g[0]) ** 2 + (p[1] - g[1]) ** 2) ** 0.5
            dists.append(d)
            if d <= epsilon:
                TP += 1
            else:
                FP += 1
                FN += 1

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": prec, "recall": rec, "accuracy": acc, "F1": f1,
        "mean_dist_on_matched": float(np.mean(dists)) if dists else None,
        "epsilon_px": epsilon,
    }


def load_tracknet_gt(csv_path: str, n_frames: int) -> list:
    """Load TrackNet-format GT into a frame-aligned list of (x, y) or None.

    Expected columns: file_name, visibility, x-coordinate, y-coordinate.
    Frame index is parsed from the file_name stem if it's numeric.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    gt: list = [None] * n_frames
    for _, row in df.iterrows():
        stem = Path(str(row["file_name"])).stem
        if not stem.replace(".", "").isdigit():
            continue
        idx = int(stem)
        if idx >= n_frames:
            continue
        if int(row.get("visibility", 1)) == 0:
            continue
        x, y = float(row["x-coordinate"]), float(row["y-coordinate"])
        if np.isnan(x) or np.isnan(y):
            continue
        gt[idx] = (x, y)
    return gt


def is_point_in_image(x, y, input_width=1280, input_height=720):
    """From TennisCourtDetector utils.py: Checks if a point is within frame bounds."""
    if x is not None and y is not None:
        return (x >= 0) and (x <= input_width) and (y >= 0) and (y <= input_height)
    return False

def court_keypoint_metrics(pred_kps, gt_kps, max_dist=7, width=1280, height=720):
    """
    Adapted from TennisCourtDetector test.py.
    Calculates TP, FP, FN, TN based on a pixel distance threshold for 14 keypoints.
    """
    assert len(pred_kps) == 14 and len(gt_kps) == 14, "Must provide exactly 14 keypoints."
    
    tp = fp = fn = tn = 0
    dists = []

    for point_pred, point_gt in zip(pred_kps, gt_kps):
        x_pred, y_pred = point_pred
        x_gt, y_gt = point_gt

        pred_in_img = is_point_in_image(x_pred, y_pred, width, height)
        gt_in_img = is_point_in_image(x_gt, y_gt, width, height)

        if pred_in_img and gt_in_img:
            dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
            dists.append(dst)
            if dst < max_dist:
                tp += 1
            else:
                fp += 1
        elif pred_in_img and not gt_in_img:
            fp += 1
        elif not pred_in_img and gt_in_img:
            fn += 1
        elif not pred_in_img and not gt_in_img:
            tn += 1

    return tp, fp, fn, tn, dists
