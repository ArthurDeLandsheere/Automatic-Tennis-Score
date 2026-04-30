"""Ball tracking with TrackNet.

Two functions matter:
  - track_ball: runs the model on a video using a 3-frame sliding window and a
    GPU batch flush. Memory is bounded by `chunk` regardless of video length.
  - smooth_ball_track: linear interp + velocity-based outlier rejection.
"""

from typing import Optional

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from .tracknet_model import BallTrackerNet


# Defaults that match TrackNet V1 (yastrebksv repo)
TRACKNET_H, TRACKNET_W = 360, 640
DEFAULT_BALL_CHUNK = 16
DEFAULT_BALL_CONF_THRESHOLD = 0.5
DEFAULT_MAX_PX_PER_FRAME = 100   # ~160 km/h upper bound at 720p / 30fps
DEFAULT_MAX_GAP_TO_INTERP = 6


def load_ball_model(weights_path: str, device: str = "cuda") -> BallTrackerNet:
    """Load BallTrackerNet weights. Tolerant to a few common state_dict layouts."""
    model = BallTrackerNet()
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_ball_model] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[load_ball_model] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    return model.to(device).eval()


def _preprocess_for_tracknet(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.resize(frame_bgr, (TRACKNET_W, TRACKNET_H))


def _extract_ball_pos(
    heatmap: torch.Tensor, h_orig: int, w_orig: int, conf_threshold: float
) -> tuple[Optional[tuple[float, float]], float]:
    """Find the peak in a TrackNet output and rescale to original-image coords.

    Handles the multiple shapes different TrackNet variants emit.
    """
    hm = heatmap.detach().cpu().numpy()

    if hm.ndim == 2 and hm.shape[0] == 256:
        # (256, H*W)
        flat = hm.argmax(axis=0)
        gray = flat.reshape(TRACKNET_H, TRACKNET_W).astype(np.uint8)
    elif hm.ndim == 3 and hm.shape[0] == 256:
        gray = hm.argmax(axis=0).astype(np.uint8)
    elif hm.ndim == 3 and hm.shape[0] == 1:
        gray = (hm[0] * 255).clip(0, 255).astype(np.uint8)
    elif hm.ndim == 2:
        gray = (hm * 255).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unexpected TrackNet output shape: {hm.shape}")

    if gray.max() == 0:
        return None, 0.0

    _, _, _, max_loc = cv2.minMaxLoc(gray)
    peak_val = float(gray[max_loc[1], max_loc[0]]) / 255.0
    if peak_val < conf_threshold:
        return None, peak_val

    x = max_loc[0] * (w_orig / TRACKNET_W)
    y = max_loc[1] * (h_orig / TRACKNET_H)
    return (float(x), float(y)), peak_val


def track_ball(
    video_path: str,
    model: BallTrackerNet,
    device: str = "cuda",
    chunk: int = DEFAULT_BALL_CHUNK,
    conf_threshold: float = DEFAULT_BALL_CONF_THRESHOLD,
) -> tuple[list[Optional[tuple[float, float]]], list[float]]:
    """Run TrackNet over a video using a 3-frame rolling buffer + batched flush.

    The first 2 frames are returned as None (the buffer needs 3 frames to
    produce its first prediction). All other frames get a (x, y) or None.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ball_positions: list[Optional[tuple[float, float]]] = [None] * total
    ball_confidence: list[float] = [0.0] * total

    buffer_frames: list[np.ndarray] = []
    pending_inputs: list[np.ndarray] = []
    pending_target_idx: list[int] = []
    shape_logged = [False]   # closure-mutable flag

    def flush() -> None:
        if not pending_inputs:
            return
        batch = torch.from_numpy(np.stack(pending_inputs)).float().to(device) / 255.0
        with torch.no_grad():
            out = model(batch)
        if not shape_logged[0]:
            print(f"  [debug] TrackNet output shape per batch: {tuple(out.shape)}")
            shape_logged[0] = True
        for i, t_idx in enumerate(pending_target_idx):
            pos, conf = _extract_ball_pos(out[i], h_orig, w_orig, conf_threshold)
            ball_positions[t_idx] = pos
            ball_confidence[t_idx] = conf
        pending_inputs.clear()
        pending_target_idx.clear()

    pbar = tqdm(total=total, desc="TrackNet (batched)")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        prep = _preprocess_for_tracknet(frame)
        buffer_frames.append(prep)
        if len(buffer_frames) > 3:
            buffer_frames.pop(0)
        if len(buffer_frames) == 3:
            stacked = np.concatenate(
                [buffer_frames[2], buffer_frames[1], buffer_frames[0]], axis=2
            )
            stacked = stacked.transpose(2, 0, 1)  # (9, H, W)
            pending_inputs.append(stacked)
            pending_target_idx.append(frame_idx)
            if len(pending_inputs) >= chunk:
                flush()
        frame_idx += 1
        pbar.update(1)
    flush()
    pbar.close()
    cap.release()
    return ball_positions, ball_confidence


def smooth_ball_track(
    positions: list,
    max_gap: int = DEFAULT_MAX_GAP_TO_INTERP,
    max_velocity: float = DEFAULT_MAX_PX_PER_FRAME,
) -> list:
    """Velocity-based outlier rejection + linear interp across short gaps.

    Returns a *new* list; the input is not modified.
    """
    pos = list(positions)

    # 1) Outlier rejection (velocity-based, bidirectional)
    for t in range(1, len(pos) - 1):
        if pos[t] is None:
            continue
        prev_p = next((pos[k] for k in range(t - 1, -1, -1) if pos[k] is not None), None)
        next_p = next((pos[k] for k in range(t + 1, len(pos)) if pos[k] is not None), None)
        if prev_p is None or next_p is None:
            continue
        dx_prev = pos[t][0] - prev_p[0]
        dy_prev = pos[t][1] - prev_p[1]
        if (dx_prev ** 2 + dy_prev ** 2) ** 0.5 > max_velocity:
            dx_next = next_p[0] - pos[t][0]
            dy_next = next_p[1] - pos[t][1]
            if (dx_next ** 2 + dy_next ** 2) ** 0.5 > max_velocity:
                pos[t] = None

    # 2) Linear interp across short gaps
    i = 0
    while i < len(pos):
        if pos[i] is not None:
            i += 1
            continue
        j = i
        while j < len(pos) and pos[j] is None:
            j += 1
        gap = j - i
        if 0 < i and j < len(pos) and gap <= max_gap:
            x0, y0 = pos[i - 1]
            x1, y1 = pos[j]
            for k in range(gap):
                frac = (k + 1) / (gap + 1)
                pos[i + k] = (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))
        i = j
    return pos
