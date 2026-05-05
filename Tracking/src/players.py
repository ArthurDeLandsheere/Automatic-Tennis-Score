"""Player detection & tracking: YOLOv8 (COCO person) + ByteTrack.

We track everyone the detector finds (including ball kids, umpire, line judges)
and then post-filter to the 2 main players using a presence × median-area score.
"""

from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLO
import math
from shapely.geometry import Point, Polygon


def track_players(
    video_path: str,
    yolo_weights: str = "yolov8m.pt",
    conf: float = 0.3,
    iou: float = 0.5,
    n_frames_total: Optional[int] = None,
) -> list[list[dict]]:
    """Run YOLOv8 + ByteTrack on a video and return per-frame detections.

    Returns: list of length n_frames, each entry is a list of dicts with
    keys: id (int), bbox ([x1,y1,x2,y2] floats), conf (float).
    """
    yolo = YOLO(yolo_weights)
    tracks_per_frame: list[list[dict]] = []

    # stream=True iterates per-frame instead of materialising all results;
    # crucial for long videos.
    results_gen = yolo.track(
        source=str(video_path),
        classes=[0],          # COCO `person`
        persist=True,
        tracker="bytetrack_tennis.yaml",
        conf=conf,
        iou=iou,
        stream=True,
        verbose=False,
    )

    for result in tqdm(results_gen, total=n_frames_total, desc="YOLOv8+ByteTrack"):
        frame_dets = []

        # Temp debug
        n_raw = len(result.boxes) if result.boxes is not None else 0
        n_tracked = len(result.boxes.id) if (result.boxes is not None and result.boxes.id is not None) else 0
        if n_raw != n_tracked:
            print(f"frame: {n_raw} detections, {n_tracked} with ID")

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().numpy()
            bboxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for id_, bbox, c in zip(ids, bboxes, confs):
                frame_dets.append({
                    "id": int(id_),
                    "bbox": bbox.tolist(),
                    "conf": float(c),
                })
        tracks_per_frame.append(frame_dets)

    # Padding in the case of length mismatch
    if n_frames_total is not None and len(tracks_per_frame) < n_frames_total:
        missing = n_frames_total - len(tracks_per_frame)
        print(f"[warn] YOLO generator stopped {missing} frames early — padding with empty frames")
        tracks_per_frame.extend([[] for _ in range(missing)])

    return tracks_per_frame

def build_court_polygon(
    court_keypoints_per_frame: list[list[list[float]]],
    frame_w: int,
    frame_h: int,
    margin_tb: float = 0.15,
    margin_lr: float = 0.01,
) -> Optional[Polygon]:
    """Build a convex hull polygon from all court keypoints across frames,
    expanded asymmetrically: more on top/bottom (to include players behind
    the baselines) and less on left/right (to exclude the chairman's chair).

    margin_tb: vertical expansion as a fraction of frame height
    margin_lr: horizontal expansion as a fraction of frame width
    """
    all_pts = []
    for kps in court_keypoints_per_frame:
        all_pts.extend(kps)

    if len(all_pts) < 4:
        return None

    pts = np.array(all_pts, dtype=float)
    hull = Polygon(pts).convex_hull

    # Compute centroid of the hull — we scale outward from there
    cx = hull.centroid.x
    cy = hull.centroid.y

    expand_x = margin_lr * frame_w
    expand_y = margin_tb * frame_h

    # Expand each hull vertex outward from the centroid, independently
    # per axis — this gives asymmetric margins without distorting the shape
    expanded_coords = []
    for x, y in hull.exterior.coords:
        dx = x - cx
        dy = y - cy
        # Scale dx/dy so that the vertex moves outward by expand_x/expand_y
        # We use the sign of dx/dy to determine outward direction
        new_x = cx + dx + (expand_x if dx >= 0 else -expand_x)
        new_y = cy + dy + (expand_y if dy >= 0 else -expand_y)
        expanded_coords.append((new_x, new_y))

    return Polygon(expanded_coords)


def select_main_player_ids(
    tracks_per_frame: list[list[dict]],
    top_k: int = 2,
    court_polygon: Optional[Polygon] = None,
    frame_h: Optional[int] = None,
    verbose: bool = True,
) -> list[int]:
    """Pick the top_k IDs using presence score, with court-zone filtering
    and a vertical-zone constraint (one ID per half of the frame).

    Steps:
      1. If a court_polygon is provided, discard detections whose bbox center
         falls outside it — this eliminates the chairman, umpire, etc.
      2. Score each remaining ID by frames_seen only (area is a tiebreaker,
         not the primary signal, so partial-frame players aren't penalised).
      3. Apply a vertical-zone constraint: pick the highest-scoring ID whose
         median center-y is in the top half, and the highest-scoring ID in the
         bottom half. This guarantees one player per side regardless of score.
    """
    id_frame_count: Counter = Counter()
    id_areas: defaultdict = defaultdict(list)
    id_center_ys: defaultdict = defaultdict(list)

    for frame in tracks_per_frame:
        for d in frame:
            x1, y1, x2, y2 = d["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Court-zone filter: skip if center is outside the polygon
            if court_polygon is not None and not court_polygon.contains(Point(cx, cy)):
                continue

            id_frame_count[d["id"]] += 1
            id_areas[d["id"]].append((x2 - x1) * (y2 - y1))
            id_center_ys[d["id"]].append(cy)

    if not id_frame_count:
        if verbose:
            print("  [warn] No detections survived court-zone filter — falling back to unfiltered.")
        # Fallback: rebuild without filter
        for frame in tracks_per_frame:
            for d in frame:
                x1, y1, x2, y2 = d["bbox"]
                id_frame_count[d["id"]] += 1
                id_areas[d["id"]].append((x2 - x1) * (y2 - y1))
                id_center_ys[d["id"]].append((y1 + y2) / 2.0)

    scored = []
    for pid, cnt in id_frame_count.items():
        median_area = float(np.median(id_areas[pid]))
        median_cy = float(np.median(id_center_ys[pid]))
        # Primary: frames seen. Secondary: log area as tiebreaker only.
        score = cnt * 1000 + math.log1p(median_area)
        scored.append((pid, cnt, median_area, median_cy, score))
    scored.sort(key=lambda t: -t[4])

    if verbose:
        print("Top candidates (id, frames_seen, median_area, median_cy, score):")
        for row in scored[:8]:
            print(f"  id={row[0]:>3}  frames={row[1]:>4}  area={row[2]:>8.0f}  "
                  f"cy={row[3]:>6.1f}  score={row[4]:>12.1f}")

    # Vertical-zone constraint
    half = (frame_h / 2.0) if frame_h is not None else None

    if half is not None and len(scored) >= 2:
        top_half_ids = [t for t in scored if t[3] < half]
        bot_half_ids = [t for t in scored if t[3] >= half]

        top_pick = top_half_ids[0][0] if top_half_ids else scored[0][0]
        bot_pick = next(
            (t[0] for t in bot_half_ids if t[0] != top_pick),
            scored[1][0],
        )
        main_ids = [top_pick, bot_pick]
    else:
        main_ids = [t[0] for t in scored[:top_k]]

    if verbose:
        for pid in main_ids:
            row = next(t for t in scored if t[0] == pid)
            half_label = "top-half" if (half and row[3] < half) else "bot-half"
            print(f"  KEPT id={pid} ({half_label})")
        print(f"Main player IDs: {main_ids}")

    return main_ids

def select_main_player_ids_segmented(
    tracks_per_frame: list[list[dict]],
    court_keypoints_per_frame: Optional[list[list[list[float]]]] = None,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
    min_segment_frames: int = 15,
    verbose: bool = True,
) -> list[tuple[int, int, set[int]]]:
    """Run select_main_player_ids independently per detected segment,
    building a per-segment court polygon from that segment's keypoints only.

    Returns a list of (start, end, {id1, id2}) tuples — one per segment.
    """
    segments = detect_cuts(tracks_per_frame, min_segment_frames=min_segment_frames)

    if verbose:
        print(f"  Detected {len(segments)} segment(s) after cut detection.")

    results = []
    for seg_start, seg_end in segments:
        seg_tracks = tracks_per_frame[seg_start:seg_end]

        # Build a polygon specific to this segment's camera angle
        seg_polygon = None
        if court_keypoints_per_frame is not None and frame_w and frame_h:
            seg_keypoints = court_keypoints_per_frame[seg_start:seg_end]
            seg_polygon = build_court_polygon(
                seg_keypoints,
                frame_w=frame_w,
                frame_h=frame_h,
            )
            if verbose:
                if seg_polygon is not None:
                    print(f"\n  Segment [{seg_start}–{seg_end}] ({seg_end - seg_start} frames) "
                          f"— polygon bounds: ({seg_polygon.bounds[0]:.0f}, {seg_polygon.bounds[1]:.0f}, "
                          f"{seg_polygon.bounds[2]:.0f}, {seg_polygon.bounds[3]:.0f})")
                else:
                    print(f"\n  Segment [{seg_start}–{seg_end}] ({seg_end - seg_start} frames) "
                          f"— no polygon (not enough keypoints), skipping court filter.")
        elif verbose:
            print(f"\n  Segment [{seg_start}–{seg_end}] ({seg_end - seg_start} frames)")

        ids = select_main_player_ids(
            seg_tracks,
            top_k=2,
            court_polygon=seg_polygon,
            frame_h=frame_h,
            verbose=verbose,
        )
        results.append((seg_start, seg_end, set(ids)))

    return results


def filter_and_label_players(
    tracks_per_frame: list[list[dict]],
    segments: list[tuple[int, int, set[int]]],
) -> list[list[dict]]:
    """Keep only the main IDs for each segment and label top/bottom.

    segments is a list of (start, end, {id1, id2}) tuples as returned by
    select_main_player_ids_segmented. Frames that fall outside all segments
    (e.g. very short inter-cut clips) are left empty.
    """
    # Build a per-frame lookup: frame_idx -> set of valid IDs
    valid_ids_per_frame: list[Optional[set[int]]] = [None] * len(tracks_per_frame)
    for seg_start, seg_end, ids in segments:
        for i in range(seg_start, seg_end):
            valid_ids_per_frame[i] = ids

    out = []
    for i, frame in enumerate(tracks_per_frame):
        valid = valid_ids_per_frame[i]
        if valid is None:
            out.append([])
            continue

        kept = [dict(d) for d in frame if d["id"] in valid]
        if len(kept) == 2:
            kept.sort(key=lambda d: d["bbox"][3])  # by y2 (feet)
            kept[0]["side"] = "top"
            kept[1]["side"] = "bottom"
        elif len(kept) == 1:
            kept[0]["side"] = "unknown"
        out.append(kept)

    return out

def detect_cuts(
    tracks_per_frame: list[list[dict]],
    min_segment_frames: int = 15,
) -> list[tuple[int, int]]:
    """Detect hard cuts by looking for frames where all tracked IDs are new.

    A cut is declared when there is zero ID overlap between the previous
    non-empty frame and the current non-empty frame. Returns a list of
    (start, end) frame index pairs (end is exclusive) for each segment.
    """
    segments = []
    seg_start = 0
    prev_ids: set[int] = set()

    for i, frame in enumerate(tracks_per_frame):
        if not frame:
            continue
        curr_ids = {d["id"] for d in frame}
        if prev_ids and curr_ids.isdisjoint(prev_ids):
            # Zero overlap → hard cut
            if i - seg_start >= min_segment_frames:
                segments.append((seg_start, i))
            seg_start = i
        prev_ids = curr_ids

    # Last segment
    if len(tracks_per_frame) - seg_start >= min_segment_frames:
        segments.append((seg_start, len(tracks_per_frame)))

    return segments

def player_count_stats(labeled_players: list[list[dict]]) -> dict:
    """Quick stats: how often did we see 2 / 1 / 0 players?"""
    n_both = sum(1 for f in labeled_players if len(f) == 2)
    n_one = sum(1 for f in labeled_players if len(f) == 1)
    n_none = sum(1 for f in labeled_players if len(f) == 0)
    return {"both": n_both, "one": n_one, "none": n_none, "total": len(labeled_players)}
