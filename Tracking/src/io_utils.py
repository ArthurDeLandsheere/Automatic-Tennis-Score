"""I/O utilities: video metadata, JSON serialisation of tracking results.

The JSON schema is intentionally simple so the score-prediction stage (which
will consume both this output and the action-spotting predictions) doesn't
need any extra parsing logic.
"""

import json
from pathlib import Path
from typing import Any

import cv2


def get_video_info(video_path: str) -> dict[str, Any]:
    """Return basic video metadata: fps, frame count, resolution."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    info = {
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration_s"] = info["n_frames"] / info["fps"] if info["fps"] else 0.0
    return info


def save_tracks(
    output_path: str,
    video_path: str,
    video_info: dict,
    main_player_ids: list[int],
    labeled_players: list[list[dict]],
    ball_positions_raw: list[tuple | None],
    ball_positions_smooth: list[tuple | None],
    ball_confidence: list[float],
    court_keypoints_per_frame: list | None = None,
    in_play_flags: list[bool] | None = None,
) -> None:
    """Dump the full tracking output to JSON.

    The schema:
      - Top-level: video metadata + main player IDs
      - frames[i] is one entry per video frame, aligned by index
      - ball.interpolated tells the score logic which detections were filled in
        by the smoother (so it can down-weight them if it wants to)

    Court keypoints (14 per frame, when detected):
        0/1  baseline_top left/right          (FAR baseline corners)
        2/3  baseline_bottom left/right       (NEAR baseline corners)
        4/5  left_inner_line top/bottom       (left singles sideline)
        6/7  right_inner_line top/bottom      (right singles sideline)
        8/9  top_inner_line left/right        (FAR service-box corners)
       10/11 bottom_inner_line left/right     (NEAR service-box corners)
       12/13 middle_line top/bottom           (net centre points)
    """
    n = video_info["n_frames"]
    assert len(labeled_players) == n, f"player tracks length mismatch: {len(labeled_players)} vs {n}"
    assert len(ball_positions_smooth) == n, f"ball tracks length mismatch: {len(ball_positions_smooth)} vs {n}"

    frames = []
    for i in range(n):
        p_smooth = ball_positions_smooth[i]
        p_raw = ball_positions_raw[i]
        ball_entry = None
        if p_smooth is not None:
            ball_entry = {
                "x": float(p_smooth[0]),
                "y": float(p_smooth[1]),
                "conf": float(ball_confidence[i]),
                "interpolated": p_raw is None,
            }
        frames.append({
            "frame_idx": i,
            "players": labeled_players[i],
            "ball": ball_entry,
            "in_play":   in_play_flags[i] if in_play_flags is not None else True,
            "court_keypoints": court_keypoints_per_frame[i] if court_keypoints_per_frame is not None else []
        })

    payload = {
        "video": str(Path(video_path).name),
        "video_path": str(video_path),
        "fps": video_info["fps"],
        "width": video_info["width"],
        "height": video_info["height"],
        "n_frames": n,
        "main_player_ids": main_player_ids,
        "frames": frames,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f)
    print(f"Wrote tracks → {output_path}")


def load_tracks(path: str) -> dict:
    """Load a tracks JSON file produced by save_tracks."""
    with open(path) as f:
        return json.load(f)