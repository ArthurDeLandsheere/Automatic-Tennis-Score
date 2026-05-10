"""
Evaluate YOLOv8 player detection and spatial filtering on an image dataset.
Integrates CourtDetector to strictly filter detections outside the court polygon.

Usage:
    python -m scripts.evaluate_detection \
        --dataset data/tennis/ground_truth/tennistracker/train \
        --apply-filter
"""

import argparse
import sys
import os
import glob
from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm.auto import tqdm
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.court import CourtDetector
from src.players import build_court_polygon

def bb_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0.0
    return iou

def load_yolo_labels(txt_path, img_w, img_h):
    """Convert YOLO format (class, cx, cy, w, h) to pixel [x1, y1, x2, y2]."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    
    PLAYER_CLASSES = [0, 1] 

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                if class_id in PLAYER_CLASSES:
                    cx, cy, w, h = map(float, parts[1:5])
                    x1 = (cx - w / 2) * img_w
                    y1 = (cy - h / 2) * img_h
                    x2 = (cx + w / 2) * img_w
                    y2 = (cy + h / 2) * img_h
                    boxes.append([x1, y1, x2, y2])
    return boxes

def apply_spatial_filter(detections, frame_h, court_polygon=None):
    """
    Single-frame spatial filtering mimicking src.players logic.
    1. Court Polygon Filter: Drops detections whose centers are outside the polygon.
    2. Vertical split & Size filter: Keeps 1 largest player in top half, 1 in bottom.
    """
    filtered = []
    half_y = frame_h / 2.0

    if court_polygon is not None:
        in_court = []
        for det in detections:
            cx = (det[0] + det[2]) / 2.0
            cy = (det[1] + det[3]) / 2.0
            if court_polygon.contains(Point(cx, cy)):
                in_court.append(det)
        detections = in_court

    detections.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)

    top_half = [b for b in detections if (b[1] + b[3])/2.0 < half_y]
    bot_half = [b for b in detections if (b[1] + b[3])/2.0 >= half_y]

    if top_half: filtered.append(top_half[0])
    if bot_half: filtered.append(bot_half[0])

    return filtered

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to split folder (e.g. dataset/train)")
    parser.add_argument("--yolo-weights", default="checkpoints/yolov8m.pt")
    parser.add_argument("--court-weights", default="checkpoints/court/model_tennis_court_det.pt")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--apply-filter", action="store_true", help="Apply the court polygon and spatial filter")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading YOLOv8...")
    yolo = YOLO(args.yolo_weights)
    
    print("Loading Court Detector...")
    court_detector = CourtDetector(model_path=args.court_weights, device=device)
    
    img_dir = os.path.join(args.dataset, "images")
    lbl_dir = os.path.join(args.dataset, "labels")
    
    image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    print(f"Found {len(image_paths)} images to evaluate.")

    TP, FP, FN = 0, 0, 0

    for img_path in tqdm(image_paths, desc="Evaluating Images"):
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]


        filename = os.path.basename(img_path)
        txt_path = os.path.join(lbl_dir, filename.replace(".jpg", ".txt"))
        gt_boxes = load_yolo_labels(txt_path, img_w, img_h)

        court_poly = None
        if args.apply_filter:
            # Detect 14 keypoints
            _, keypoints = court_detector.detect(img)
            
            valid_kps = [[float(x), float(y)] for x, y in keypoints if x is not None and y is not None]
            

            if len(valid_kps) >= 4:
                court_poly = build_court_polygon([valid_kps], img_w, img_h)


        results = yolo.predict(img, classes=[0], conf=0.3, verbose=False)[0]
        pred_boxes = []
        if results.boxes is not None:
            for box in results.boxes.xyxy.cpu().numpy():
                pred_boxes.append(box.tolist())

        if args.apply_filter:
            pred_boxes = apply_spatial_filter(pred_boxes, img_h, court_polygon=court_poly)


        matched_gt = set()
        matched_pred = set()

        for p_idx, p_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt: continue
                iou = bb_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_iou >= args.iou_threshold:
                TP += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
            else:
                FP += 1  # Prediction doesn't match any GT (Ghost detection)

        FN += (len(gt_boxes) - len(matched_gt))  # Missed players

    # Calculate Metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print("DETECTION METRICS (IoU > {:.2f})".format(args.iou_threshold))
    print("Filter applied: {}".format("YES (Court Polygon + Zone)" if args.apply_filter else "NO (Raw YOLO)"))
    print("="*40)
    print(f"True Positives (TP)  : {TP}")
    print(f"False Positives (FP) : {FP} (Ghost detections / Crowd)")
    print(f"False Negatives (FN) : {FN} (Missed players)")
    print("-" * 40)
    print(f"Precision            : {precision:.3f}")
    print(f"Recall               : {recall:.3f}")
    print(f"F1-Score             : {f1:.3f}")
    print("="*40)

if __name__ == "__main__":
    main()

# [MADE BY CLAUDE (Anthropic)]