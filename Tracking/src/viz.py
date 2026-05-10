"""Drawing & visualisation utilities."""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

_COLOR_BY_SIDE = {
    "top":     (0, 180, 255),    # orange in BGR
    "bottom":  (255, 120, 0),    # blue in BGR
    "unknown": (180, 180, 180),
}
# Color per action label (BGR)
_COLOR_BY_ACTION = {
    "far_court_serve":    (0, 200, 255),   # yellow
    "near_court_serve":   (0, 165, 255),   # orange
    "far_court_bounce":   (255, 100, 0),   # blue
    "near_court_bounce":  (255, 50, 50),   # darker blue
    "far_court_swing":    (0, 200, 0),     # green
    "near_court_swing":   (0, 140, 0),     # darker green
}
_DEFAULT_ACTION_COLOR = (200, 200, 200)
_EVENT_WINDOW = 5  # frames before/after event to show banner


def draw_players(frame: np.ndarray, frame_dets: list[dict]) -> np.ndarray:
    vis = frame.copy()
    for d in frame_dets:
        x1, y1, x2, y2 = map(int, d["bbox"])
        side = d.get("side", "unknown")
        color = _COLOR_BY_SIDE[side]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"id={d['id']} {side} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis


def draw_ball(frame: np.ndarray, pos, trail: Optional[list] = None) -> np.ndarray:
    vis = frame.copy()
    if trail:
        for i, p in enumerate(trail):
            if p is None:
                continue
            alpha = (i + 1) / len(trail)
            cv2.circle(vis, (int(p[0]), int(p[1])), 3,
                       (0, int(255 * alpha), int(255 * (1 - alpha))), -1)
    if pos is not None:
        cv2.circle(vis, (int(pos[0]), int(pos[1])), 8, (0, 255, 255), 2)
        cv2.circle(vis, (int(pos[0]), int(pos[1])), 2, (0, 255, 255), -1)
    return vis


def draw_court(frame: np.ndarray, court_keypoints: Optional[list]) -> np.ndarray:
    if not court_keypoints:
        return frame
    vis = frame.copy()
    for kp in court_keypoints:
        if kp is None:
            continue
        x, y = kp
        cv2.circle(vis, (int(x), int(y)), 6, (0, 0, 255), -1)
    return vis


def _draw_static_event_dashboard(frame: np.ndarray, current_states: dict) -> np.ndarray:
    """Draw a static dashboard of all events, highlighted based on state."""
    h, w = frame.shape[:2]
    labels = list(_COLOR_BY_ACTION.keys())
    num_labels = len(labels)

    banner_h = 40
    box_w = w // num_labels

    # Shorter names for better display
    display_names = {
        "far_court_serve": "Far Serve",
        "near_court_serve": "Near Serve",
        "far_court_bounce": "Far Bounce",
        "near_court_bounce": "Near Bounce",
        "far_court_swing": "Far Swing",
        "near_court_swing": "Near Swing",
    }

    for i, label in enumerate(labels):
        state = current_states.get(label, 0)
        
        x1 = i * box_w
        # Ensure the last box reaches the exact edge of the frame
        x2 = (i + 1) * box_w if i < num_labels - 1 else w 
        y1, y2 = 0, banner_h

        base_color = _COLOR_BY_ACTION.get(label, _DEFAULT_ACTION_COLOR)

        # Set opacity based on state (2: happening, 1: approaching, 0: inactive)
        if state == 2:
            alpha = 0.85
        elif state == 1:
            alpha = 0.40
        else:
            alpha = 0.15

        # Extract the region of interest and blend the color
        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (x2 - x1, y2 - y1), base_color, -1)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

        # Draw box border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Text properties - LARGER AND BOLDER
        text = display_names.get(label, label)
        font_scale = 0.6  # Increased from 0.45
        
        # Calculate text size to center it properly
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2

        # Pure white for approaching/happening, light gray for inactive
        txt_color = (255, 255, 255) if state > 0 else (200, 200, 200)
        
        # Bolder text depending on state
        thickness = 3 if state == 2 else (2 if state == 1 else 1)

        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, thickness)

    return frame


def draw_all(
    frame: np.ndarray,
    player_dets: list[dict],
    ball_pos,
    ball_trail: Optional[list] = None,
    court_keypoints: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = draw_court(frame, court_keypoints)
    vis = draw_players(vis, player_dets)
    vis = draw_ball(vis, ball_pos, trail=ball_trail)
    return vis


def show_sample_frames(
    video_path: str,
    per_frame_dets: list,
    ball_positions: Optional[list] = None,
    court_keypoints_per_frame=None,
    n: int = 4,
    title: str = "",
    save_path: Optional[str] = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(total * 0.1, total * 0.9, n).astype(int)

    rows = (n + 1) // 2
    cols = min(n, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        ball_pos = ball_positions[idx] if ball_positions is not None else None
        trail = (ball_positions[max(0, idx - 15):idx + 1]
                 if ball_positions is not None else None)
        court_kps = court_keypoints_per_frame[idx] if court_keypoints_per_frame is not None else None
        vis = draw_all(frame, per_frame_dets[idx], ball_pos, ball_trail=trail, court_keypoints=court_kps)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {idx}")
        ax.axis("off")
    cap.release()
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Wrote sample frames → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_trajectory(
    video_path: str,
    ball_positions: list,
    save_path: Optional[str] = None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    _, bg = cap.read()
    cap.release()

    valid = [(i, p[0], p[1]) for i, p in enumerate(ball_positions) if p is not None]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), alpha=0.4)
    if valid:
        idx_arr, xa, ya = zip(*valid)
        sc = ax.scatter(xa, ya, c=idx_arr, cmap="viridis", s=8)
        plt.colorbar(sc, ax=ax, label="frame index")
    ax.set_title(f"Ball trajectory ({len(valid)} detections)")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Wrote trajectory → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def make_annotated_video(
    video_path: str,
    out_path: str,
    labeled_players: list,
    ball_positions: list,
    court_keypoints_per_frame: Optional[list] = None,
    action_events: Optional[list] = None,
    trail_len: int = 15,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    # Build a per-frame event state lookup
    # States -> 0: inactive, 1: approaching/leaving, 2: happening
    event_states = defaultdict(lambda: {k: 0 for k in _COLOR_BY_ACTION.keys()})
    
    if action_events:
        # Define the windows in frames based on video FPS
        approach_window = int(fps * 1.0)  # 1.0 seconds before and after
        happen_window = int(fps * 0.4)    # 0.4 seconds before and after
        
        for event in action_events:
            f = event['frame']
            label = event['label']
            
            # 1. Mark approaching/leaving window
            start_app = max(0, f - approach_window)
            end_app = min(n - 1, f + approach_window)
            for i in range(start_app, end_app + 1):
                if event_states[i][label] < 1:  # Don't overwrite state 2
                    event_states[i][label] = 1
                    
            # 2. Mark the exact happening moment with an expanded window
            start_hap = max(0, f - happen_window)
            end_hap = min(n - 1, f + happen_window)
            for i in range(start_hap, end_hap + 1):
                event_states[i][label] = 2

    for idx in tqdm(range(n), desc="Writing video"):
        ret, frame = cap.read()
        if not ret:
            break
        trail = ball_positions[max(0, idx - trail_len):idx + 1]
        court_kps = court_keypoints_per_frame[idx] if court_keypoints_per_frame is not None else None
        
        # Draw court, players, and ball
        vis = draw_all(frame, labeled_players[idx], ball_positions[idx],
                    ball_trail=trail, court_keypoints=court_kps)

        # Draw the static event dashboard
        vis = _draw_static_event_dashboard(vis, event_states[idx])

        # Shift the frame counter text down slightly so it doesn't overlap the new 40px banner
        txt = f"frame {idx}/{n}  |  ball={'yes' if ball_positions[idx] else 'no'}"
        cv2.putText(vis, txt, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(vis)

    cap.release()
    out.release()
    print(f"Wrote annotated video → {out_path}")
    print("Tip: re-encode to H.264 for inline players: "
          f"ffmpeg -y -i {out_path} -vcodec libx264 -crf 23 {out_path.replace('.mp4', '_h264.mp4')}")