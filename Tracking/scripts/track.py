"""Run the full tracking pipeline on a video.

Usage:
    python -m scripts.track \
        --video data/tennis/videos/match1.mp4 \
        --output outputs/tracks/match1.json \
        --yolo-weights checkpoints/yolov8m.pt \
        --tracknet-weights checkpoints/tracknet/model_best.pt
"""

import argparse
import sys
from pathlib import Path

import cv2
import torch

# Make `src` importable when run from anywhere
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ball import load_ball_model, smooth_ball_track, track_ball
from src.court import CourtDetector
from src.io_utils import get_video_info, save_tracks
from src.players import (
    filter_and_label_players,
    player_count_stats,
    select_main_player_ids,
    track_players,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tracking on one tennis video.")
    parser.add_argument("--video", required=True, help="Path to the input mp4")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--yolo-weights", default="checkpoints/yolov8m.pt",
                        help="YOLO weights (auto-downloaded by ultralytics if missing)")
    parser.add_argument("--tracknet-weights", default="checkpoints/tracknet/model_best.pt",
                        help="TrackNet weights from yastrebksv/TrackNet")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto-detected if omitted)")
    parser.add_argument("--ball-chunk", type=int, default=16,
                        help="GPU batch size for TrackNet")
    parser.add_argument("--ball-conf", type=float, default=0.5,
                        help="Min heatmap peak (0-1) to accept a ball detection")
    parser.add_argument("--no-court", action="store_true",
                        help="Skip court detection (faster but loses geometric filter)")

    parser.add_argument("--original-video", default=None,
                        help="Original video path to store in JSON (overrides the processed video path)")
    parser.add_argument("--original-width",  type=int, default=None)
    parser.add_argument("--original-height", type=int, default=None)

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Video info & sanity ────────────────────────────────────────────────
    info = get_video_info(args.video)
    print(f"Video: {args.video}")
    print(f"  frames={info['n_frames']}  fps={info['fps']:.2f}  "
          f"resolution={info['width']}x{info['height']}  duration={info['duration_s']:.1f}s")
    if (info["width"], info["height"]) != (1280, 720):
        print(f"  WARNING: resolution is {info['width']}x{info['height']}, not 1280x720.")
        print("  TrackNet was trained on 720p — accuracy will degrade.")
        print(f"  Resize first: ffmpeg -i in.mp4 -vf scale=1280:720 out.mp4")

    # ── Players ────────────────────────────────────────────────────────────
    print("\n[1/3] Player tracking (YOLOv8 + ByteTrack)")
    raw_player_tracks = track_players(
        args.video,
        yolo_weights=args.yolo_weights,
        n_frames_total=info["n_frames"],
    )
    main_ids = select_main_player_ids(raw_player_tracks, top_k=2)
    labeled_players = filter_and_label_players(raw_player_tracks, main_ids)
    stats = player_count_stats(labeled_players)
    print(f"  2 players: {stats['both']}  /  1 player: {stats['one']}  /  0 players: {stats['none']}")

    # ── Court ──────────────────────────────────────────────────────────────
    court_polygon = None
    if not args.no_court:
        print("\n[2/3] Court detection (trying up to 5 frames)")
        cap = cv2.VideoCapture(str(args.video))
        detector = CourtDetector()
        candidate_frames = [
            info["n_frames"] // 6,
            info["n_frames"] // 4,
            info["n_frames"] // 2,
            3 * info["n_frames"] // 4,
            5 * info["n_frames"] // 6,
        ]
        for attempt, frame_idx in enumerate(candidate_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()
            if detector.detect(frame) is not None:
                poly = detector.get_court_polygon()
                if poly is not None:
                    court_polygon = poly
                    print(f"  Court detected on attempt {attempt+1} (frame {frame_idx})")
                    print(f"  Best configuration: {detector.best_conf}  |  Score: {detector.court_score:.0f}")
                    print(f"  Court polygon: {len(poly)} vertices")
                    break
            print(f"  Attempt {attempt+1} (frame {frame_idx}) failed, trying next...")
        else:
            print("  Court detection failed on all candidate frames — proceeding without polygon.")
        cap.release()
    else:
        print("\n[2/3] Skipping court detection (--no-court)")

    # ── Ball ───────────────────────────────────────────────────────────────
    print("\n[3/3] Ball tracking (TrackNet)")
    if not Path(args.tracknet_weights).exists():
        raise FileNotFoundError(
            f"TrackNet weights not found at {args.tracknet_weights}. "
            "Download model_best.pt from the yastrebksv/TrackNet README link."
        )
    ball_model = load_ball_model(args.tracknet_weights, device=device)
    ball_positions_raw, ball_confidence = track_ball(
        args.video, ball_model, device=device,
        chunk=args.ball_chunk, conf_threshold=args.ball_conf,
    )
    n_detected = sum(1 for p in ball_positions_raw if p is not None)
    print(f"  Detected: {n_detected} / {len(ball_positions_raw)} "
          f"({100 * n_detected / len(ball_positions_raw):.1f}%)")

    ball_positions_smooth = smooth_ball_track(ball_positions_raw)
    n_after = sum(1 for p in ball_positions_smooth if p is not None)
    print(f"  After smoothing: {n_after} / {len(ball_positions_smooth)} "
          f"({100 * n_after / len(ball_positions_smooth):.1f}%) "
          f"— recovered {n_after - n_detected} via interpolation")

    # ── Save ───────────────────────────────────────────────────────────────
    video_path_to_save = args.original_video or args.video
    if args.original_width and args.original_height:
        info["width"]  = args.original_width
        info["height"] = args.original_height

    save_tracks(
        output_path=args.output,
        video_path=video_path_to_save,
        video_info=info,
        main_player_ids=main_ids,
        labeled_players=labeled_players,
        ball_positions_raw=ball_positions_raw,
        ball_positions_smooth=ball_positions_smooth,
        ball_confidence=ball_confidence,
        court_polygon=court_polygon.tolist() if court_polygon is not None else None,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
