"""Court detection.

Copié-collé quasiment de tennis-tracking pour le court detection
"""

from itertools import combinations
from typing import Optional

import cv2
import numpy as np

# Sera utile pour la logique après
def point_in_court(point_xy: tuple[float, float], court_polygon: np.ndarray) -> bool:
    """True if the point (given by x,y) is inside the polygon """
    if court_polygon is None:
        return True
    in_court = cv2.pointPolygonTest(court_polygon, (float(point_xy[0]), float(point_xy[1])), False)
    return in_court >= 0


def detect_court_polygon(frame: np.ndarray, debug: bool = False):
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=W * 0.15, maxLineGap=20)
    if lines is None:
        return (None, frame.copy()) if debug else None

    pts = []
    for line in lines[:, 0]:
        pts.append((line[0], line[1]))
        pts.append((line[2], line[3]))
    hull = cv2.convexHull(np.array(pts).astype(np.int32))

    if debug:
        vis = frame.copy()
        for line in lines[:, 0]:
            cv2.line(vis, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 1)
        cv2.polylines(vis, [hull], True, (0, 0, 255), 3)
        return hull, vis
    return hull


def _line_intersection(line1, line2):
    """Return (x, y) intersection of two lines given as (pt1, pt2). Pure numpy."""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        raise ValueError("Lines are parallel")
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _sort_intersection_points(pts):
    """Sort 4 points top-left → top-right → bottom-left → bottom-right."""
    y_sorted = sorted(pts, key=lambda p: p[1])
    top = sorted(y_sorted[:2], key=lambda p: p[0])
    bottom = sorted(y_sorted[2:], key=lambda p: p[0])
    return top + bottom

def frame_has_court(frame: np.ndarray, threshold: int = 3000) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use HSV white mask instead of raw brightness — more robust to skin/clothing
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    # Require long horizontal runs (court lines are wide-frame features)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    lines = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    return int(np.sum(lines > 0)) > threshold


class CourtReference:
    """
    Court reference model
    """
    def __init__(self):
        self.baseline_top = ((286, 561), (1379, 561))
        self.baseline_bottom = ((286, 2935), (1379, 2935))
        self.net = ((286, 1748), (1379, 1748))
        self.left_court_line = ((286, 561), (286, 2935))
        self.right_court_line = ((1379, 561), (1379, 2935))
        self.left_inner_line = ((423, 561), (423, 2935))
        self.right_inner_line = ((1242, 561), (1242, 2935))
        self.middle_line = ((832, 1110), (832, 2386))
        self.top_inner_line = ((423, 1110), (1242, 1110))
        self.bottom_inner_line = ((423, 2386), (1242, 2386))
        self.top_extra_part = (832.5, 580)
        self.bottom_extra_part = (832.5, 2910)

        self.court_conf = {1: [*self.baseline_top, *self.baseline_bottom],
                           2: [self.left_inner_line[0], self.right_inner_line[0], self.left_inner_line[1],
                               self.right_inner_line[1]],
                           3: [self.left_inner_line[0], self.right_court_line[0], self.left_inner_line[1],
                               self.right_court_line[1]],
                           4: [self.left_court_line[0], self.right_inner_line[0], self.left_court_line[1],
                               self.right_inner_line[1]],
                           5: [*self.top_inner_line, *self.bottom_inner_line],
                           6: [*self.top_inner_line, self.left_inner_line[1], self.right_inner_line[1]],
                           7: [self.left_inner_line[0], self.right_inner_line[0], *self.bottom_inner_line],
                           8: [self.right_inner_line[0], self.right_court_line[0], self.right_inner_line[1],
                               self.right_court_line[1]],
                           9: [self.left_court_line[0], self.left_inner_line[0], self.left_court_line[1],
                               self.left_inner_line[1]],
                           10: [self.top_inner_line[0], self.middle_line[0], self.bottom_inner_line[0],
                                self.middle_line[1]],
                           11: [self.middle_line[0], self.top_inner_line[1], self.middle_line[1],
                                self.bottom_inner_line[1]],
                           12: [*self.bottom_inner_line, self.left_inner_line[1], self.right_inner_line[1]]}

        self.line_width = 1
        self.court_width = 1117
        self.court_height = 2408
        self.top_bottom_border = 549
        self.right_left_border = 274
        self.court_total_width = self.court_width + self.right_left_border * 2
        self.court_total_height = self.court_height + self.top_bottom_border * 2
        self.court = self._build_court_reference()

    def _build_court_reference(self):
        h = self.court_height + 2 * self.top_bottom_border
        w = self.court_width + 2 * self.right_left_border
        court = np.zeros((h, w), dtype=np.uint8)
        for endpoints in [self.baseline_top, self.baseline_bottom,
                          self.top_inner_line, self.bottom_inner_line,
                          self.left_court_line, self.right_court_line,
                          self.left_inner_line, self.right_inner_line,
                          self.middle_line]:
            cv2.line(court, endpoints[0], endpoints[1], 1, self.line_width)
        return cv2.dilate(court, np.ones((5, 5), dtype=np.uint8))

    def get_important_lines(self):
        """
        Returns all lines of the court
        """
        lines = [*self.baseline_top, *self.baseline_bottom, *self.net, *self.left_court_line, *self.right_court_line,
                 *self.left_inner_line, *self.right_inner_line, *self.middle_line,
                 *self.top_inner_line, *self.bottom_inner_line]
        return lines


class CourtDetector:
    """Detect the tennis court via Hough lines + best-fit homography."""

    def __init__(self):
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        self.court_reference = CourtReference()
        self.v_width = self.v_height = 0
        self.frame = self.gray = None
        self.court_warp_matrix: list = []
        self.game_warp_matrix: list = []
        self.court_score = 0.0
        self.best_conf: Optional[int] = None

    def detect(self, frame: np.ndarray):
        self.frame = frame
        self.v_height, self.v_width = frame.shape[:2]
        self.gray = self._threshold(frame)
        filtered = self._filter_pixels(self.gray)

        horizontal_lines, vertical_lines = self._detect_lines(filtered)
        if not horizontal_lines or not vertical_lines:
            return None

        court_warp, game_warp, self.court_score = self._find_homography(
            horizontal_lines, vertical_lines
        )
        if court_warp is None:
            return None

        self.court_warp_matrix.append(court_warp)
        self.game_warp_matrix.append(game_warp)
        return self.find_lines_location()

    def get_warped_court(self) -> np.ndarray:
        court = cv2.warpPerspective(
            self.court_reference.court,
            self.court_warp_matrix[-1],
            self.frame.shape[1::-1],
        )
        court[court > 0] = 1
        return court

    def get_court_polygon(self) -> Optional[np.ndarray]:
        """Return the warped court silhouette as an Nx2 int polygon, or None."""
        if not self.court_warp_matrix:
            return None
        mask = self.get_warped_court().astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea).reshape(-1, 2)

    def find_lines_location(self) -> np.ndarray:
        pts = np.array(
            self.court_reference.get_important_lines(), dtype=np.float32
        ).reshape((-1, 1, 2))
        return cv2.perspectiveTransform(pts, self.court_warp_matrix[-1]).reshape(-1)

    def _threshold(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.threshold(gray, self.colour_threshold, 255, cv2.THRESH_BINARY)[1]

    def _filter_pixels(self, gray):
        tau = self.dist_tau
        thr = self.intensity_threshold
        g = gray.astype(np.int16)
        bright_v = (
            (g[tau:-tau, tau:-tau] - g[tau + tau:, tau:-tau] > thr) &
            (g[tau:-tau, tau:-tau] - g[:-(tau * 2), tau:-tau] > thr)
        )
        bright_h = (
            (g[tau:-tau, tau:-tau] - g[tau:-tau, tau + tau:] > thr) &
            (g[tau:-tau, tau:-tau] - g[tau:-tau, :-(tau * 2)] > thr)
        )
        keep = bright_v | bright_h
        out = np.zeros_like(gray)
        out[tau:-tau, tau:-tau] = np.where(
            gray[tau:-tau, tau:-tau] > 0, keep.astype(np.uint8) * 255, 0
        )
        return out

    def _detect_lines(self, gray):
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80, minLineLength=100, maxLineGap=20)
        if lines is None:
            return [], []
        lines = np.squeeze(lines)
        if lines.ndim == 1:
            lines = lines[np.newaxis]
        horizontal, vertical = self._classify_lines(lines)
        horizontal, vertical = self._merge_lines(horizontal, vertical)
        return horizontal, vertical

    def _classify_lines(self, lines):
        horizontal, vertical = [], []
        highest_y = np.inf
        lowest_y = 0
        for line in lines:
            x1, y1, x2, y2 = line
            if abs(x1 - x2) > 2 * abs(y1 - y2):
                horizontal.append(line)
            else:
                vertical.append(line)
                highest_y = min(highest_y, y1, y2)
                lowest_y = max(lowest_y, y1, y2)
        h = lowest_y - highest_y
        lo = lowest_y + h / 15
        hi = highest_y - h * 2 / 15
        clean_h = [ln for ln in horizontal if lo > ln[1] > hi and lo > ln[3] > hi]
        return clean_h, vertical

    def _merge_lines(self, horizontal_lines, vertical_lines):
        horizontal_lines = sorted(horizontal_lines, key=lambda l: l[0])
        mask = [True] * len(horizontal_lines)
        new_h = []
        for i, line in enumerate(horizontal_lines):
            if not mask[i]:
                continue
            for j, s_line in enumerate(horizontal_lines[i + 1:]):
                if mask[i + j + 1] and abs(int(s_line[1]) - int(line[3])) < 10:
                    pts = sorted(
                        [(line[0], line[1]), (line[2], line[3]),
                         (s_line[0], s_line[1]), (s_line[2], s_line[3])],
                        key=lambda p: p[0],
                    )
                    line = np.array([*pts[0], *pts[-1]])
                    mask[i + j + 1] = False
            new_h.append(line)

        vertical_lines = sorted(vertical_lines, key=lambda l: l[1])
        xl, yl = 0, int(self.v_height * 6 / 7)
        xr, yr = self.v_width, int(self.v_height * 6 / 7)
        mask = [True] * len(vertical_lines)
        new_v = []
        for i, line in enumerate(vertical_lines):
            if not mask[i]:
                continue
            for j, s_line in enumerate(vertical_lines[i + 1:]):
                if mask[i + j + 1]:
                    try:
                        xi, _ = _line_intersection(
                            ((line[0], line[1]), (line[2], line[3])),
                            ((xl, yl), (xr, yr)),
                        )
                        xj, _ = _line_intersection(
                            ((s_line[0], s_line[1]), (s_line[2], s_line[3])),
                            ((xl, yl), (xr, yr)),
                        )
                    except ValueError:
                        continue
                    if abs(float(xi) - float(xj)) < 10:
                        pts = sorted(
                            [(line[0], line[1]), (line[2], line[3]),
                             (s_line[0], s_line[1]), (s_line[2], s_line[3])],
                            key=lambda p: p[1],
                        )
                        line = np.array([*pts[0], *pts[-1]])
                        mask[i + j + 1] = False
            new_v.append(line)

        return new_h, new_v

    def _find_homography(self, horizontal_lines, vertical_lines):
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None

        for h_pair in combinations(horizontal_lines, 2):
            for v_pair in combinations(vertical_lines, 2):
                h1, h2 = h_pair
                v1, v2 = v_pair
                try:
                    i1 = _line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v1[:2]), tuple(v1[2:])))
                    i2 = _line_intersection((tuple(h1[:2]), tuple(h1[2:])), (tuple(v2[:2]), tuple(v2[2:])))
                    i3 = _line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v1[:2]), tuple(v1[2:])))
                    i4 = _line_intersection((tuple(h2[:2]), tuple(h2[2:])), (tuple(v2[:2]), tuple(v2[2:])))
                except (ValueError, ZeroDivisionError):
                    continue
                intersections = _sort_intersection_points([i1, i2, i3, i4])

                for conf_id, configuration in self.court_reference.court_conf.items():
                    matrix, _ = cv2.findHomography(
                        np.float32(configuration), np.float32(intersections), method=0,
                    )
                    if matrix is None:
                        continue
                    inv_matrix = cv2.invert(matrix)[1]
                    score = self._get_confi_score(matrix)
                    if score > max_score:
                        max_score = score
                        max_mat = matrix
                        max_inv_mat = inv_matrix
                        self.best_conf = conf_id

        return max_mat, max_inv_mat, max_score

    def _get_confi_score(self, matrix):
        court = cv2.warpPerspective(self.court_reference.court, matrix, self.frame.shape[1::-1])
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        return np.sum(correct) - 0.5 * np.sum(wrong)

# ------------------------------------------------------------------
# Written by Claude
# ------------------------------------------------------------------
class CourtTracker:
    """
    Tracks the court polygon across video frames.

    Strategy:
      1. Scan frames from the start until frame_has_court() is True,
         then run CourtDetector for the initial polygon.
      2. For each subsequent frame, estimate camera motion via sparse
         optical flow and warp the court homography forward.
      3. Periodically revalidate the warped homography using the detector's
         scoring function. If score drops below threshold, attempt re-detection.

    Usage:
        tracker = CourtTracker()
        cap = cv2.VideoCapture(video_path)
        for frame_idx in range(n_frames):
            ret, frame = cap.read()
            polygon, in_play = tracker.update(frame)
            # polygon: np.ndarray (Nx2) or None
            # in_play: bool
    """

    def __init__(
        self,
        revalidate_every: int = 100,       # frames between homography score checks
        revalidate_threshold: float = 0.5, # fraction of initial score below which we re-detect
        max_corners: int = 200,            # optical flow feature points
        of_quality: float = 0.01,
        of_min_dist: float = 10.0,
    ):
        self.revalidate_every = revalidate_every
        self.revalidate_threshold = revalidate_threshold
        self.max_corners = max_corners
        self.of_quality = of_quality
        self.of_min_dist = of_min_dist

        self._detector = CourtDetector()
        self._warp_matrix: Optional[np.ndarray] = None   # current H (court→frame)
        self._initial_score: float = 0.0
        self._baseline_score: float = 0.0
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count: int = 0
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> tuple[Optional[np.ndarray], bool]:
        """
        Process one frame.
        Returns (court_polygon, in_play).
          - court_polygon: Nx2 int32 array or None if not detected yet
          - in_play:       True when court lines are visible in this frame
        """
        in_play = frame_has_court(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self._initialized:
            if in_play:
                polygon = self._initial_detect(frame, gray)
            else:
                polygon = None
        else:
            if in_play:
                self._update_homography(gray)
                if self._frame_count % self.revalidate_every == 0:
                    self._revalidate(frame)
                polygon = self._polygon_from_warp(frame)
                self._prev_gray = gray
            else:
                # Camera is not showing the court — keep last known matrix
                # but don't update optical flow (no court features to track)
                polygon = None

        self._frame_count += 1
        return polygon, in_play

    def get_current_polygon(self) -> Optional[np.ndarray]:
        """Return polygon from the current warp matrix without updating state."""
        if self._warp_matrix is None:
            return None
        return self._polygon_from_warp_matrix(self._warp_matrix)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_detect(self, frame: np.ndarray, gray: np.ndarray) -> Optional[np.ndarray]:
        result = self._detector.detect(frame)
        if result is None or not self._detector.court_warp_matrix:
            return None
        self._warp_matrix = self._detector.court_warp_matrix[-1].copy()
        self._initial_score = self._detector.court_score
        self._baseline_score = self._detector.court_score
        self._initialized = True
        print(f"  [CourtTracker] Initial detection succeeded at frame {self._frame_count} "
              f"(score={self._initial_score:.0f}, conf={self._detector.best_conf})")
        self._prev_gray = gray
        return self._polygon_from_warp(frame)

    def _update_homography(self, gray: np.ndarray) -> None:
        """Estimate frame-to-frame camera motion and warp H forward."""
        if self._prev_gray is None or self._warp_matrix is None:
            return

        # Build a mask of where the court lines currently are
        h, w = gray.shape
        court_mask = cv2.warpPerspective(
            self._detector.court_reference.court,
            self._warp_matrix, (w, h)
        )
        court_mask = (court_mask > 0).astype(np.uint8)
        # Dilate to give some slack around the lines
        court_mask = cv2.dilate(court_mask, np.ones((15, 15), np.uint8))

        # Find good features to track in the previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.of_quality,
            minDistance=self.of_min_dist,
            mask=court_mask,   # only track features on court lines
        )
        if prev_pts is None or len(prev_pts) < 4:
            return

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, prev_pts, None
        )
        if curr_pts is None:
            return

        good_prev = prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]
        if len(good_prev) < 4:
            return

        # Estimate the 2D rigid motion (translation + rotation, no shear)
        # as an affine transform. For mostly pan cameras this is sufficient.
        M, inliers = cv2.estimateAffinePartial2D(
            good_prev, good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if M is None:
            return

        M3 = np.eye(3, dtype=np.float64)
        M3[:2, :] = M
        candidate = M3 @ self._warp_matrix

        # Reject updates that move the court polygon too far — likely a bad OF estimate
        prev_poly = self._polygon_from_warp_matrix(self._warp_matrix, (h, w))
        cand_poly = self._polygon_from_warp_matrix(candidate, (h, w))
        if prev_poly is not None and cand_poly is not None:
            shift = np.linalg.norm(cand_poly.mean(axis=0) - prev_poly.mean(axis=0))
            if shift > 30:  # pixels — if the centroid jumps more than 30px, distrust it
                return
        self._warp_matrix = candidate

    def _revalidate(self, frame: np.ndarray) -> None:
        """Check current warp matrix quality; re-detect if score dropped too much."""
        if self._warp_matrix is None:
            return
        self._detector.frame = frame
        self._detector.gray = self._detector._threshold(frame)
        score = self._detector._get_confi_score(self._warp_matrix)
        if score < self._baseline_score * self.revalidate_threshold:
            ratio = score / self._initial_score if self._initial_score > 0 else 0.0
            print(f"  [CourtTracker] Score dropped to {ratio:.2f} of initial — re-detecting at frame {self._frame_count}")
            result = self._detector.detect(frame)
            if result is not None and self._detector.court_warp_matrix:
                self._warp_matrix = self._detector.court_warp_matrix[-1].copy()
                self._initial_score = self._detector.court_score
                self._baseline_score = self._detector.court_score
                print(f"  [CourtTracker] Re-detection succeeded (score={self._initial_score:.0f})")
            else:
                print(f"  [CourtTracker] Re-detection failed — keeping current matrix")

    def _polygon_from_warp(self, frame: np.ndarray) -> Optional[np.ndarray]:
        return self._polygon_from_warp_matrix(self._warp_matrix, frame.shape)

    def _polygon_from_warp_matrix(
        self,
        matrix: np.ndarray,
        shape: Optional[tuple] = None,
    ) -> Optional[np.ndarray]:
        if matrix is None:
            return None
        ref_court = self._detector.court_reference.court
        h = ref_court.shape[0] if shape is None else shape[0]
        w = ref_court.shape[1] if shape is None else shape[1]
        warped = cv2.warpPerspective(ref_court, matrix, (w, h))
        warped = (warped > 0).astype(np.uint8)

        contours, _ = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx.reshape(-1, 2)