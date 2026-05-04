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
from src.court_homography import get_trans_matrix, refer_kps
from src.court_reference import CourtReference


# ---------------------------------------------------------------------------
# Unchanged helpers
# ---------------------------------------------------------------------------

def point_in_court(point_xy: tuple[float, float], court_polygon: np.ndarray) -> bool:
    """True if the point (x, y) is inside the polygon."""
    if court_polygon is None:
        return True
    result = cv2.pointPolygonTest(
        court_polygon, (float(point_xy[0]), float(point_xy[1])), False
    )
    return result >= 0


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
    Tracks the court polygon across video frames using per-frame DL inference.

    Compared to the previous optical-flow version, each frame is detected
    independently — no homography propagation, no revalidation needed.
    Frames where frame_has_court() is False are skipped and return None.

    Usage:
        tracker = CourtTracker(model_path=..., device=...)
        polygon, in_play = tracker.update(frame)   # frame is BGR numpy array
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self._detector = CourtDetector(model_path=model_path, device=device)
        self._court_ref = CourtReference()
        self._court_ref.build_court_reference()
        self._last_matrix: Optional[np.ndarray] = None
        self._last_keypoints: list = []

    def update(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], bool]:
        """
        Process one frame.
        Returns (court_polygon, in_play).
          - court_polygon: Nx2 int32 array, or None if not detected
          - in_play:       True when court lines are visible in this frame
        """
        in_play = frame_has_court(frame)

        if not in_play:
            return None, [], False

        matrix, keypoints = self._detector.detect(frame)
        if matrix is not None:
            self._last_matrix = matrix
            self._last_keypoints = keypoints

        if self._last_matrix is None:
            return None, [], True

        polygon = self._polygon_from_matrix(self._last_matrix, frame.shape)
        return polygon, self._last_keypoints, True

    def get_current_polygon(self) -> Optional[np.ndarray]:
        """Return polygon from the last successful detection without updating state."""
        if self._last_matrix is None:
            return None
        return self._polygon_from_matrix(self._last_matrix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _polygon_from_matrix(
        self,
        matrix: np.ndarray,
        shape: Optional[tuple] = None,
    ) -> Optional[np.ndarray]:
        """Warp the court reference silhouette and extract its contour polygon."""
        ref_court = self._court_ref.court  # binary reference image

        if shape is not None:
            h, w = shape[:2]
        else:
            h, w = ref_court.shape[:2]

        warped = cv2.warpPerspective(ref_court, matrix, (w, h))
        warped = (warped > 0).astype(np.uint8)

        contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx.reshape(-1, 2)