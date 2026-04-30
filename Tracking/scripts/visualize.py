"""Generate annotated outputs from a tracks JSON.

Usage:
    python -m scripts.visualize \
        --tracks outputs/tracks/match1.json \
        --out-dir outputs/videos/match1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.io_utils import load_tracks
from src.viz import make_annotated_video, plot_trajectory, show_sample_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise a tracking output.")
    parser.add_argument("--tracks", required=True, help="Path to tracks JSON (from scripts/track.py)")
    parser.add_argument("--video", default=None,
                        help="Path to source video (defaults to the one in the JSON)")
    parser.add_argument("--out-dir", required=True, help="Directory for output files")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip annotated video (just sample frames + trajectory)")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="How many sample frames to render in the grid")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_tracks(args.tracks)
    video_path = args.video or tracks["video_path"]
    print(f"Visualising {video_path}")

    # Reconstruct frame-aligned lists from the JSON
    n = tracks["n_frames"]
    labeled_players = [f["players"] for f in tracks["frames"]]
    ball_positions = [
        (f["ball"]["x"], f["ball"]["y"]) if f["ball"] is not None else None
        for f in tracks["frames"]
    ]
    court_polygon = (
        np.array(tracks["court_polygon"], dtype=np.int32)
        if tracks.get("court_polygon") is not None else None
    )

    # 1) Sample frames grid
    show_sample_frames(
        video_path, labeled_players, ball_positions=ball_positions,
        n=args.n_samples, title=f"Tracking — {Path(video_path).name}",
        save_path=str(out_dir / "sample_frames.png"),
    )

    # 2) Trajectory plot
    plot_trajectory(
        video_path, ball_positions, save_path=str(out_dir / "trajectory.png"),
    )

    # 3) Full annotated video
    if not args.no_video:
        make_annotated_video(
            video_path,
            out_path=str(out_dir / "annotated.mp4"),
            labeled_players=labeled_players,
            ball_positions=ball_positions,
            court_polygon=court_polygon,
        )

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
