"""Evaluation metrics.

For the ball, we use the standard TrackNet ε-threshold metric (a prediction
is correct if it falls within ε pixels of GT). For players, MOTA is the right
choice but it's not implemented here — we just provide a stub with a pointer
to motmetrics.
"""

from pathlib import Path

import numpy as np


def ball_threshold_metrics(pred: list, gt: list, epsilon: float = 10) -> dict:
    """Standard TrackNet-style metric.

    pred, gt: list of (x, y) or None, aligned frame-by-frame.

    Frames where gt is None are treated as unannotated and skipped entirely
    (neither FP nor TN). This matches sparse annotation schemes like
    RacketVision where only clearly visible frames are labeled.

    Returns: dict with TP / FP / FN / TN / precision / recall / accuracy / F1
    and the mean pixel distance among matched detections.
    """
    assert len(pred) == len(gt), f"length mismatch: {len(pred)} vs {len(gt)}"
    TP = FP = FN = TN = 0
    dists: list[float] = []
    for p, g in zip(pred, gt):
        if g is None:
            continue  # unannotated frame — skip entirely
        elif p is None:
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
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
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
