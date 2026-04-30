"""Player detection & tracking: YOLOv8 (COCO person) + ByteTrack.

We track everyone the detector finds (including ball kids, umpire, line judges)
and then post-filter to the 2 main players using a presence × median-area score.
"""

from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from tqdm.auto import tqdm
from ultralytics import YOLO


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
        tracker="bytetrack.yaml",
        conf=conf,
        iou=iou,
        stream=True,
        verbose=False,
    )

    for result in tqdm(results_gen, total=n_frames_total, desc="YOLOv8+ByteTrack"):
        frame_dets = []
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

    return tracks_per_frame


def select_main_player_ids(
    tracks_per_frame: list[list[dict]],
    top_k: int = 2,
    verbose: bool = True,
) -> list[int]:
    """Pick the top_k IDs by (frames_seen × median bbox area).

    The intuition: real players are on screen most of the time and they
    occupy more pixels than ball kids or the umpire. Multiplying handles the
    edge cases where one of the two heuristics fails on its own.
    """
    id_frame_count: Counter = Counter()
    id_areas: defaultdict = defaultdict(list)
    for frame in tracks_per_frame:
        for d in frame:
            id_frame_count[d["id"]] += 1
            x1, y1, x2, y2 = d["bbox"]
            id_areas[d["id"]].append((x2 - x1) * (y2 - y1))

    scored = []
    for pid, cnt in id_frame_count.items():
        median_area = float(np.median(id_areas[pid]))
        scored.append((pid, cnt, median_area, cnt * median_area))
    scored.sort(key=lambda t: -t[3])
    main_ids = [t[0] for t in scored[:top_k]]

    if verbose:
        print("Top candidates (id, frames_seen, median_area, score):")
        for row in scored[:6]:
            marker = "  <-- KEPT" if row[0] in main_ids else ""
            print(f"  id={row[0]:>3}  frames={row[1]:>4}  area={row[2]:>8.0f}  score={row[3]:>10.0f}{marker}")
        print(f"Main player IDs: {main_ids}")

    return main_ids


def filter_and_label_players(
    tracks_per_frame: list[list[dict]],
    main_ids: list[int],
) -> list[list[dict]]:
    """Keep only main_ids and add a per-frame 'side' field ('top'/'bottom').

    Side is computed each frame from the y-coordinate of the bbox bottom edge
    (≈ feet). This handles change-of-ends correctly: the ID stays with the
    player but `side` flips when they physically swap.
    """
    out = []
    for frame in tracks_per_frame:
        kept = [dict(d) for d in frame if d["id"] in main_ids]
        if len(kept) == 2:
            kept.sort(key=lambda d: d["bbox"][3])  # by y2 (feet)
            kept[0]["side"] = "top"
            kept[1]["side"] = "bottom"
        elif len(kept) == 1:
            kept[0]["side"] = "unknown"
        out.append(kept)
    return out


def player_count_stats(labeled_players: list[list[dict]]) -> dict:
    """Quick stats: how often did we see 2 / 1 / 0 players?"""
    n_both = sum(1 for f in labeled_players if len(f) == 2)
    n_one = sum(1 for f in labeled_players if len(f) == 1)
    n_none = sum(1 for f in labeled_players if len(f) == 0)
    return {"both": n_both, "one": n_one, "none": n_none, "total": len(labeled_players)}
