from src.court_reference import CourtReference
import numpy as np
import cv2
from scipy.spatial import distance
import logging
log = logging.getLogger(__name__)

court_ref = CourtReference()
refer_kps = np.array(court_ref.key_points, dtype=np.float32).reshape((-1, 1, 2))

court_conf_ind = {}
for i in range(len(court_ref.court_conf)):
    conf = court_ref.court_conf[i+1]
    inds = []
    for j in range(4):
        inds.append(court_ref.key_points.index(conf[j]))
    court_conf_ind[i+1] = inds

def get_trans_matrix(points, max_residual: float = 500.0):
    best_matrix = None
    matrix_trans = None
    best_residual = np.inf
    for conf_ind in range(1, 13):
        conf = court_ref.court_conf[conf_ind]

        inds = court_conf_ind[conf_ind]
        inters = [points[inds[0]], points[inds[1]], points[inds[2]], points[inds[3]]]
        if any(None in x for x in inters):
            continue
        matrix, _ = cv2.findHomography(np.float32(conf), np.float32(inters), method=0)
        if matrix is None:
            continue
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix).reshape(-1, 2)
        dists = [
            distance.euclidean(points[i], trans_kps[i])
            for i in range(12)
            if i not in inds and points[i][0] is not None
        ]
        if not dists:
            continue
        residual = float(np.median(dists))
        if residual < best_residual:
            best_matrix = matrix
            best_residual = residual

    if best_matrix is None:
        log.info("get_trans_matrix: no 4-point config had all 4 points detected")
        return None

    log.info(f"get_trans_matrix: best residual = {best_residual:.1f}px "
             f"(threshold = {max_residual:.1f}px) "
             f"-> {'accept' if best_residual <= max_residual else 'reject'}")
    if best_residual > max_residual:
        return None
    return best_matrix 

def project_keypoints(matrix) -> list[tuple[float, float]] | None:
    """Project the 14 canonical court keypoints into frame coords."""
    if matrix is None:
        return None
    proj = cv2.perspectiveTransform(refer_kps, matrix).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in proj]

