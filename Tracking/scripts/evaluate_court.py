import json
import argparse
import numpy as np
import cv2
from pathlib import Path
import sys
from scipy.spatial import distance
import torch
from src.court_reference import CourtReference
from src.court_homography import get_trans_matrix


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.court import CourtDetector

def is_point_in_image(x, y, input_width=1280, input_height=720):
    """Checks if a point is valid and within frame bounds."""
    if x is not None and y is not None:
        return (x >= 0) and (x <= input_width) and (y >= 0) and (y <= input_height)
    return False

def court_keypoint_metrics(pred_kps, gt_kps, max_dist=7, width=1280, height=720):
    """
    Calculates TP, FP, FN, TN based on a 7-pixel distance threshold for 14 keypoints.
    Matches the original TennisCourtDetector validation logic.
    """
    assert len(pred_kps) == 14 and len(gt_kps) == 14, "Must provide exactly 14 keypoints."
    
    tp = fp = fn = tn = 0
    dists = []

    for point_pred, point_gt in zip(pred_kps, gt_kps):
        x_pred, y_pred = point_pred[0], point_pred[1]
        x_gt, y_gt = point_gt[0], point_gt[1]

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate court 14-keypoint detection.")
    parser.add_argument("--val-json", required=True, help="Path to the ground truth val.json")
    parser.add_argument("--images-dir", required=True, help="Path to the directory containing images")
    parser.add_argument("--model-path", required=True, help="Path to the CourtDetector .pt weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    court_ref = CourtReference()
    refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

    # Load ground truth data
    with open(args.val_json, 'r') as f:
        gt_data = json.load(f)

    # Initialize the detector
    print(f"Loading CourtDetector on {args.device}...")
    detector = CourtDetector(model_path=args.model_path, device=args.device)

    total_tp = total_fp = total_fn = total_tn = 0
    total_tp_b = total_fp_b = total_fn_b = total_tn_b = 0
    all_dists = []
    all_dists_b= []
    missing_images = 0

    print(f"Starting evaluation on {len(gt_data)} frames...")

    for i, item in enumerate(gt_data):
        img_id = item["id"]
        gt_kps = item["kps"]  # List of 14 [x, y] coordinates
        
        img_path = Path(args.images_dir) / f"{img_id}.png"
        if not img_path.exists():
            missing_images += 1
            continue
            
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        _, pred_kps = detector.detect(image)
        
        # Calculate metrics for this frame
        tp, fp, fn, tn, dists = court_keypoint_metrics(pred_kps, gt_kps, width=w, height=h)
    
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        all_dists.extend(dists)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(gt_data)} images...")

    eps = 1e-15
    precision = total_tp / (total_tp + total_fp + eps)
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + eps)
    mean_dist = np.mean(all_dists) if all_dists else 0.0
    median_dist = np.median(all_dists) if all_dists else 0.0

    print("\n" + "="*40)
    print("=== Court Evaluation Results ===")
    print("="*40)
    print(f"Images Evaluated:   {len(gt_data) - missing_images}")
    if missing_images > 0:
        print(f"Missing Images:     {missing_images} (Skipped)")
    print(f"Total Keypoints:    {total_tp + total_fp + total_fn + total_tn}")
    print(f"True Positives:     {total_tp}")
    print(f"False Positives:    {total_fp}")
    print(f"False Negatives:    {total_fn}")
    print("-" * 40)
    print(f"Precision:          {precision:.4f}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Mean Pixel Error:   {mean_dist:.2f} px")
    print(f"Median Pixel Error: {median_dist:.2f} px")
    print("="*40)

if __name__ == "__main__":
    main()

# [MADE BY CLAUDE (Anthropic)]