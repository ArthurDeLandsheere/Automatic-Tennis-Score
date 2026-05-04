"""Court detection using TennisCourtDetector (yastrebksv).

Replaces the classical Hough-line CourtDetector with a deep learning
keypoint model (TrackNet architecture, 15-channel output).

Public interface is unchanged:
    tracker = CourtTracker(model_path=..., device=...)
    polygon, in_play = tracker.update(frame)   # frame is 1280x720
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.court_tracknet import BallTrackerNet
from src.court_postprocess import postprocess, refine_kps
from src.court_homography import get_trans_matrix


# ---------------------------------------------------------------------------
# Unchanged helpers
# ---------------------------------------------------------------------------

def frame_has_court(frame: np.ndarray, threshold: int = 3000) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    return int(np.sum(lines > 0)) > threshold


# ---------------------------------------------------------------------------
# Deep-learning court detector
# ---------------------------------------------------------------------------

# The model was trained at this resolution
_MODEL_W = 640
_MODEL_H = 360

# Keypoints that refine_kps should skip (net-related points, unreliable)
_SKIP_REFINE = {8, 9, 12}


class CourtDetector:
    """
    Per-frame court keypoint detector backed by TennisCourtDetector.

    Inputs:  BGR frame at 1280×720
    Outputs: homography matrix (3×3, court-ref → frame) or None
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self._scale_x = None  # set on first frame
        self._scale_y = None

        model = BallTrackerNet(out_channels=15)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        self.model = model.to(device)

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference on one BGR frame (any resolution).
        Returns a 3×3 homography matrix (court-ref → frame coords) or None.
        """
        h_orig, w_orig = frame.shape[:2]

        # Resize to model input resolution
        img = cv2.resize(frame, (_MODEL_W, _MODEL_H))
        inp = (img.astype(np.float32) / 255.)
        inp = torch.tensor(np.rollaxis(inp, 2, 0)).unsqueeze(0)

        with torch.no_grad():
            out = self.model(inp.float().to(self.device))[0]
        pred = torch.sigmoid(out).cpu().numpy()

        # Scale factors to bring keypoints back to original frame resolution
        sx = w_orig / _MODEL_W
        sy = h_orig / _MODEL_H

        # Extract keypoints from heatmaps
        points = []
        for kps_num in range(14):
            heatmap = (pred[kps_num] * 255).astype(np.uint8)
            x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)

            if x_pred is not None and y_pred is not None:
                # Scale up to original frame resolution
                x_scaled = float(x_pred)
                y_scaled = float(y_pred)

                margin = 20  # pixels — same crop margin refine_kps uses internally
                if kps_num not in _SKIP_REFINE and \
                        margin <= int(x_scaled) < w_orig - margin and \
                        margin <= int(y_scaled) < h_orig - margin:
                    x_scaled, y_scaled = refine_kps(frame, int(y_scaled), int(x_scaled))

                points.append((x_scaled, y_scaled))
            else:
                points.append((None, None))

        matrix = get_trans_matrix(points)
        return matrix, points  # 3×3 or None


# ---------------------------------------------------------------------------
# CourtTracker — same public interface as before
# ---------------------------------------------------------------------------

class CourtTracker:
    """
    Tracks court keypoints across video frames using per-frame DL inference.

    Each frame is detected independently — no homography propagation needed.
    Frames where frame_has_court() is False are skipped and return empty keypoints.

    Usage:
        tracker = CourtTracker(model_path=..., device=...)
        keypoints, in_play = tracker.update(frame)   # frame is BGR numpy array

    Keypoint index mapping (14 points):
        0: baseline_top[0]        (top-left baseline corner)
        1: baseline_top[1]        (top-right baseline corner)
        2: baseline_bottom[0]     (bottom-left baseline corner)
        3: baseline_bottom[1]     (bottom-right baseline corner)
        4: left_inner_line[0]     (top-left singles sideline)
        5: left_inner_line[1]     (bottom-left singles sideline)
        6: right_inner_line[0]    (top-right singles sideline)
        7: right_inner_line[1]    (bottom-right singles sideline)
        8: top_inner_line[0]      (top-left service box corner)
        9: top_inner_line[1]      (top-right service box corner)
       10: bottom_inner_line[0]   (bottom-left service box corner)
       11: bottom_inner_line[1]   (bottom-right service box corner)
       12: middle_line[0]         (net centre, top half)
       13: middle_line[1]         (net centre, bottom half)

    The "top" half of the court (indices 0, 1, 4, 6, 8, 9, 12) is the FAR
    side; the "bottom" half (indices 2, 3, 5, 7, 10, 11, 13) is the NEAR side.
    The net runs between kp[12] and kp[13].
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self._detector = CourtDetector(model_path=model_path, device=device)
        self._last_keypoints: list = []

    def update(self, frame: np.ndarray) -> tuple[list, bool]:
        """
        Process one frame.
        Returns (keypoints, in_play).
          - keypoints: list of 14 (x, y) pairs, or [] if not detected
          - in_play:   True when court lines are visible in this frame
        """
        in_play = frame_has_court(frame)

        if not in_play:
            return [], False

        _matrix, keypoints = self._detector.detect(frame)
        if keypoints:
            self._last_keypoints = keypoints

        return self._last_keypoints, True