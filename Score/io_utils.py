"""
I/O helpers for the score-prediction stage.

Loads:
  - Tracking JSON  (produced by Tracking/scripts/track.py)
  - Action-spotting predictions JSON (produced by Action-Spotting inference)

and merges them into a single list of per-frame dicts indexed by frame_idx.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_tracking(path: str | Path) -> dict[str, Any]:
    """
    Load a tracking JSON and return it as-is.

    Expected top-level keys (from Tracking/src/io_utils.py schema):
        video, video_path, fps, width, height, n_frames,
        main_player_ids, court_polygon, frames

    Each frame in ``frames`` has:
        frame_idx, players (list of 2 dicts), ball (dict or null)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tracking JSON not found: {path}")
    with open(path) as f:
        data = json.load(f)
    log.info("Loaded tracking: %s  (%d frames, fps=%.3f)",
             data.get("video"), data.get("n_frames"), data.get("fps", 0))
    return data


def load_predictions(path: str | Path, video_name: str | None = None) -> list[dict[str, Any]]:
    """
    Load action-spotting predictions and return them as a sorted list of events.

    The action-spotting repo writes predictions in the format::

        [
          {
            "video": "match1",
            "fps": 25.0,
            "num_frames": 604,
            "num_events": 20,
            "events": [
              {"frame": 106, "label": "far_court_serve",   "comment": "serve"},
              {"frame": 115, "label": "near_court_bounce", "comment": ""},
              ...
            ]
          },
          ...
        ]

    If the file contains multiple videos, ``video_name`` (stem, no extension)
    is used to select the right entry.  If only one entry is present it is
    used directly.

    Returns a list of event dicts sorted by frame, each with keys:
        frame (int), label (str), comment (str)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions JSON not found: {path}")
    with open(path) as f:
        raw = json.load(f)

    entries: list[dict] = raw if isinstance(raw, list) else [raw]

    entry: dict | None = None
    if len(entries) == 1:
        entry = entries[0]
    elif video_name is not None:
        stem = Path(video_name).stem
        for e in entries:
            if Path(e.get("video", "")).stem == stem:
                entry = e
                break
        if entry is None:
            raise ValueError(
                f"Video '{video_name}' not found in predictions file "
                f"(available: {[e.get('video') for e in entries]})"
            )
    else:
        raise ValueError(
            "Predictions file contains multiple videos — pass video_name to select one."
        )

    events: list[dict] = sorted(entry.get("events", []), key=lambda e: e["frame"])
    log.info("Loaded predictions: %s  (%d events)", path.name, len(events))
    return events


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_frames(
    tracking: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach action-spotting events to the tracking frames they fall on.

    ``events`` is the sorted list returned by ``load_predictions``.
    Each event is attached to the tracking frame whose ``frame_idx`` matches
    the event's ``frame`` field exactly (action-spotting frame numbers are
    integer frame indices, same coordinate system as the tracking JSON).

    Returns a list of merged dicts (one per tracking frame), sorted by
    frame_idx::

        {
          "frame_idx": int,
          "players":   list[dict],       # from tracking
          "ball":      dict | None,      # from tracking
          "events":    list[dict],       # action-spotting events on this frame
                                         # each: {frame, label, comment}
        }
    """
    events_by_frame: dict[int, list[dict]] = {}
    for ev in events:
        f = int(ev["frame"])
        events_by_frame.setdefault(f, []).append(ev)

    merged = []
    for frame in tracking["frames"]:
        fidx = frame["frame_idx"]
        merged.append({
            "frame_idx": fidx,
            "players":   frame.get("players", []),
            "ball":      frame.get("ball"),
            "events":    events_by_frame.get(fidx, []),
        })
    merged.sort(key=lambda d: d["frame_idx"])
    return merged


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_score_output(result: dict[str, Any], path: str | Path) -> None:
    """
    Write the final score result dict to a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Score output saved to %s", path)