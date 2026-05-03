"""Drawing & visualisation utilities."""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


_COLOR_BY_SIDE = {
    "top":     (0, 180, 255),    # orange in BGR
    "bottom":  (255, 120, 0),    # blue in BGR
    "unknown": (180, 180, 180),
}


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


def draw_court(frame: np.ndarray, court_polygon: Optional[np.ndarray]) -> np.ndarray:
    if court_polygon is None:
        return frame
    vis = frame.copy()
    cv2.polylines(vis, [court_polygon.astype(np.int32)], True, (0, 0, 255), 2)
    return vis


def draw_all(
    frame: np.ndarray,
    player_dets: list[dict],
    ball_pos,
    ball_trail: Optional[list] = None,
    court_polygon: Optional[np.ndarray] = None,
) -> np.ndarray:
    vis = draw_court(frame, court_polygon)
    vis = draw_players(vis, player_dets)
    vis = draw_ball(vis, ball_pos, trail=ball_trail)
    return vis


def show_sample_frames(
    video_path: str,
    per_frame_dets: list,
    ball_positions: Optional[list] = None,
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
        vis = draw_all(frame, per_frame_dets[idx], ball_pos, ball_trail=trail)
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
    court_polygons: Optional[list] = None,
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

    for idx in tqdm(range(n), desc="Writing video"):
        ret, frame = cap.read()
        if not ret:
            break
        trail = ball_positions[max(0, idx - trail_len):idx + 1]
        court_polygon = court_polygons[idx] if court_polygons is not None else None
        vis = draw_all(frame, labeled_players[idx], ball_positions[idx],
                    ball_trail=trail, court_polygon=court_polygon)
        txt = f"frame {idx}/{n}  |  ball={'yes' if ball_positions[idx] else 'no'}"
        cv2.putText(vis, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(vis)

    cap.release()
    out.release()
    print(f"Wrote annotated video → {out_path}")
    print("Tip: re-encode to H.264 for inline players: "
          f"ffmpeg -y -i {out_path} -vcodec libx264 -crf 23 {out_path.replace('.mp4', '_h264.mp4')}")
