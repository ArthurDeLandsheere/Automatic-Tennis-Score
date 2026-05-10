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

def frame_has_court(frame: np.ndarray, threshold: int = 1000) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    if(int(np.sum(lines > 0)) <= threshold):
        print(int(np.sum(lines > 0)))
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


def project_keypoints(matrix) -> list[tuple[float, float]] | None:
    """Project the 14 canonical court keypoints into frame coords."""
    if matrix is None:
        return None
    proj = cv2.perspectiveTransform(refer_kps, matrix).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in proj]

class CourtTracker:
    """
    Per-frame court tracker with homography-based projection and
    shot-cut-aware persistence.

    Behaviour:
      - Each frame is detected fresh.
      - If the model's 14 raw points fit a court homography (residual
        below threshold), we project the canonical keypoints through
        it — guaranteeing geometric consistency.
      - If the current frame fails, we fall back to the last good
        homography (assumes the broadcast camera barely moves within a
        shot — true for tennis).
      - When a shot cut is detected, the cached homography is dropped
        so we don't carry the previous camera's geometry into the new
        one.
    """

    def __init__(self, model_path: str, device: str = "cpu",
                 shot_cut_threshold: float = 18.0):
        self._detector = CourtDetector(model_path=model_path, device=device)
        self._last_matrix = None
        self._prev_gray_small = None
        self._cut_thresh = shot_cut_threshold

    def _is_shot_cut(self, frame: np.ndarray) -> bool:
        """Cheap hard-cut detector: mean abs-diff on a 32×18 grayscale thumbnail."""
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 18))
        if self._prev_gray_small is None:
            self._prev_gray_small = small
            return False
        diff = float(np.abs(small.astype(np.int16) - self._prev_gray_small.astype(np.int16)).mean())
        self._prev_gray_small = small
        return diff > self._cut_thresh

    def update(self, frame: np.ndarray) -> tuple[list, bool]:
        if self._is_shot_cut(frame):
            self._last_matrix = None

        in_play = frame_has_court(frame)
        if not in_play:
            # print("testa")
            return [], False

        matrix, raw_points = self._detector.detect(frame)
        if matrix is not None:
            # print("testb")
            self._last_matrix = matrix

        # if self._last_matrix is None:
            # print("testc")


        if self._last_matrix is not None:
            # print("testd")
            from src.court_homography import project_keypoints
            return project_keypoints(self._last_matrix), True

        # Tier 2: no homography has fit yet this shot. Fall back to raw model
        # output if it's reasonably populated (>= 8/14). Score side's _kps_have
        # guard handles missing indices, and the visualizer's edge-drawing
        # version handles Nones too.
        valid = sum(1 for p in raw_points if p[0] is not None)
        if valid >= 8:
            # print("teste")
            return [(p[0], p[1]) if p[0] is not None else (None, None) for p in raw_points], True

        # print("testf")
        return [], True


