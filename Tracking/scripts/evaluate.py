"""Evaluate ball-tracking output against a TrackNet-format CSV.

Usage:
    python -m scripts.evaluate \
        --tracks outputs/tracks/match1.json \
        --gt data/tennis/ground_truth/match1.csv \
        --epsilon 4 7 10
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.io_utils import load_tracks
from src.metrics import ball_threshold_metrics, load_tracknet_gt


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ball tracking against GT.")
    parser.add_argument("--tracks", required=True, help="Path to tracks JSON")
    parser.add_argument("--gt", required=True, help="Path to TrackNet-format CSV")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[4, 7, 10],
                        help="Pixel thresholds to evaluate at (TrackNet uses 4)")
    parser.add_argument("--use-raw", action="store_true",
                        help="Evaluate the raw (un-smoothed) detections "
                             "(by default we evaluate the smoothed track)")
    parser.add_argument("--save-json", default=None,
                        help="Optional path to dump the metrics dict")
    args = parser.parse_args()

    tracks = load_tracks(args.tracks)
    n = tracks["n_frames"]

    if args.use_raw:
        print("Evaluating RAW detections (pre-smoothing)")
        # Raw = positions where 'interpolated' is False; the rest are None
        pred = []
        for f in tracks["frames"]:
            if f["ball"] is None or f["ball"]["interpolated"]:
                pred.append(None)
            else:
                pred.append((f["ball"]["x"], f["ball"]["y"]))
    else:
        print("Evaluating SMOOTHED track (default)")
        pred = [
            (f["ball"]["x"], f["ball"]["y"]) if f["ball"] is not None else None
            for f in tracks["frames"]
        ]

    gt = load_tracknet_gt(args.gt, n)
    n_gt = sum(1 for g in gt if g is not None)
    print(f"GT has ball in {n_gt} / {n} frames ({100 * n_gt / n:.1f}%)")

    all_metrics = {}
    for eps in args.epsilon:
        m = ball_threshold_metrics(pred, gt, epsilon=eps)
        all_metrics[f"epsilon_{int(eps)}"] = m
        print(f"\nε = {eps:.0f} px")
        print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
        print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
              f"accuracy={m['accuracy']:.3f}  F1={m['F1']:.3f}")
        if m["mean_dist_on_matched"] is not None:
            print(f"  mean pixel error on matched: {m['mean_dist_on_matched']:.2f}")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved metrics → {args.save_json}")


if __name__ == "__main__":
    main()
