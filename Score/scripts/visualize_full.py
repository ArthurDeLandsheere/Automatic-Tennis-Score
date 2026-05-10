#!/usr/bin/env python3
"""
Render an annotated video showing tracking + action-spotting events + score.

Inspired by:
  - Action-Spotting/visualize_predictions.py  (event overlay style, colors, score boxes)
  - Tracking/scripts/visualize.py             (overall structure, tracking visualization)
"""

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Visual config
# ---------------------------------------------------------------------------
EVENT_WINDOW_SEC = 0.5      # how long event labels stay on screen (each side of center frame)
BALL_TRAIL_LEN = 8

# Event colors lifted from visualize_predictions.py (BGR)
LABEL_COLORS = {
    'far_court_serve':   (255, 160,   0),
    'near_court_serve':  (255, 120,   0),
    'far_court_swing':   ( 50, 120, 255),
    'near_court_swing':  ( 50, 180, 255),
    'far_court_bounce':  ( 50, 200,  80),
    'near_court_bounce': ( 50, 150,  50),
}
DEFAULT_EVENT_COLOR = (160, 160, 160)

COLOR_PLAYER_NEAR = (0, 255,   0)
COLOR_PLAYER_FAR  = (0, 165, 255)
COLOR_BALL        = (0,   0, 255)
COLOR_BALL_INTERP = (255, 200,  0)
COLOR_COURT       = (255, 255,  0)

COLOR_HUD_BG = (0, 0, 0)
COLOR_HUD_FG = (255, 255, 255)

POINT_LABELS = {0: "0", 1: "15", 2: "30", 3: "40"}


# ---------------------------------------------------------------------------
# Loaders / lookups
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)

def build_state_lookup(score):
    """frame_idx -> {'in_point': bool, 'serve_number': int, 'server_side': str}."""
    return {s["frame_idx"]: s for s in score.get("frame_states", [])}


def build_tracking_lookup(tracking):
    return {f["frame_idx"]: f for f in tracking["frames"]}


def build_event_label_map(predictions, fps, window_sec):
    """
    Frame -> [(label, score, is_exact_center_frame), ...].
    Adapted from visualize_predictions.py:build_frame_label_map.
    """
    entry = predictions[0] if isinstance(predictions, list) else predictions
    events = entry.get("events", [])
    window_frames = int(window_sec * fps)
    frame_map = defaultdict(list)
    for ev in events:
        label = ev["label"]
        score = ev.get("score")             # may be missing depending on AS version
        center = int(ev["frame"])
        for f in range(center - window_frames, center + window_frames + 1):
            frame_map[f].append((label, score, center == f))
    return frame_map


def build_score_lookups(score):
    """Returns (completed_points_sorted_by_end, all_points_sorted_by_start)."""
    points = sorted(score.get("points", []), key=lambda p: p["start_frame"])
    completed = sorted((p["end_frame"], p["score_after"]) for p in points)
    return completed, points

def build_bounce_lookup(score):
    return {b["frame_idx"]: b for b in score.get("bounces", [])}


def get_score_at(completed, frame_idx):
    state = {"points": {"far": 0, "near": 0},
             "games":  {"far": 0, "near": 0},
             "sets":   {"far": 0, "near": 0}}
    for end_frame, sa in completed:
        if end_frame <= frame_idx:
            state = sa
        else:
            break
    return state


def get_active_point(points, frame_idx):
    for p in points:
        if p["start_frame"] <= frame_idx <= p["end_frame"]:
            return p
    return None

def get_last_completed_point(points, frame_idx):
    last = None
    for p in points:
        if p["end_frame"] <= frame_idx:
            last = p
        else:
            break
    return last


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# def draw_court(frame, polygon):
#     if not polygon:
#         return
#     pts = [(int(x), int(y)) for x, y in polygon]
#     for i in range(len(pts)):
#         cv2.line(frame, pts[i], pts[(i + 1) % len(pts)], COLOR_COURT, 2)

COURT_EDGES = [
    (0, 1),   # doubles baseline (FAR)
    (2, 3),   # doubles baseline (NEAR)
    (0, 2),   # left  doubles sideline
    (1, 3),   # right doubles sideline
    (4, 5),   # left  singles sideline
    (6, 7),   # right singles sideline
    (8, 9),   # FAR  service line
    (10, 11), # NEAR service line
    (12, 13), # center service line
]

def draw_court(frame, keypoints):
    """Draw the court as named edges between the 14 keypoints."""
    if not keypoints or len(keypoints) < 14:
        return
    pts = [None if kp is None else (int(kp[0]), int(kp[1])) for kp in keypoints]
    for i, j in COURT_EDGES:
        if pts[i] is not None and pts[j] is not None:
            cv2.line(frame, pts[i], pts[j], COLOR_COURT, 2)


def draw_players(frame, players):
    for p in players:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        side = p.get("side", "unknown")
        color = COLOR_PLAYER_NEAR if side == "bottom" else COLOR_PLAYER_FAR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"id={p.get('id')} {side}",
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_ball(frame, trail):
    n = len(trail)
    if n == 0:
        return
    for i, (x, y, interp) in enumerate(trail):
        radius = max(2, int(2 + 4 * (i + 1) / n))
        color = COLOR_BALL_INTERP if interp else COLOR_BALL
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)


def draw_event_panel(frame, labels_at_frame, y_start=180):
    """
    Stacked translucent label boxes on the left side.
    Adapted from visualize_predictions.py:draw_overlay.
    """
    if not labels_at_frame:
        return

    y_offset = y_start
    for label, score, is_exact in labels_at_frame:
        color = LABEL_COLORS.get(label, DEFAULT_EVENT_COLOR)
        text = f"{label}  {score:.2f}" if score is not None else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        pad = 8
        x1, y1 = 15, y_offset - th - pad
        x2, y2 = 15 + tw + pad * 2, y_offset + pad

        # Translucent fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        # Border (thicker on the exact center frame of the event)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_exact else 1)
        cv2.putText(frame, text, (15 + pad, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += th + pad * 2 + 8


def draw_hud(frame, score_state, state, active_point, frame_idx, n_frames, bounce_lookup, last_completed=None):
    s_f, s_n = score_state["sets"]["far"],  score_state["sets"]["near"]
    g_f, g_n = score_state["games"]["far"], score_state["games"]["near"]
    p_f = POINT_LABELS.get(score_state["points"]["far"],  str(score_state["points"]["far"]))
    p_n = POINT_LABELS.get(score_state["points"]["near"], str(score_state["points"]["near"]))

    lines = [
        f"FAR   sets {s_f}  games {g_f}  pts {p_f}",
        f"NEAR  sets {s_n}  games {g_n}  pts {p_n}",

    ]
    if state and state.get("in_point"):
        srv = state.get("server_side", "?").upper()
        n   = state.get("serve_number", 1)
        ordinal = {1: "1st", 2: "2nd"}.get(n, f"{n}th")
        lines.append(f"State: IN POINT  |  Serve: {ordinal} ({srv})")
    else:
        # Between points: next serve is always a 1st serve.
        # server_side is carried from the last point (or initial value).
        srv = "?"
        if state:
            srv = state.get("server_side", "?").upper()
        elif active_point:
            srv = active_point.get("server_side", "?").upper()
        lines.append(f"State: BETWEEN POINTS  |  Next: 1st ({srv})")

    if state and state.get("pending_point_end"):
        pending_frame = state.get("pending_point_end_frame")
        pending_reason = state.get("pending_point_end_reason", "?")
        lines.append(f"Pending end: {pending_reason} @ {pending_frame}")

    if active_point:
        srv = active_point.get("server_side", "?").upper()
        n = active_point.get("serve_number", 1)
        lines.append(f"Serving: {srv} ({n}{'st' if n == 1 else 'nd'})")
    if last_completed:
        lines.append(f"Last pt: {last_completed.get('winner_side','?').upper()} "
                     f"({last_completed.get('reason','?')})")
    lines.append(f"Frame {frame_idx}/{n_frames}")

    last_bounce = None
    bo = bounce_lookup
    for f in sorted(bo.keys()):
        if f <= frame_idx:
            last_bounce = bo[f]
        else:
            break
    if last_bounce:
        verdict = "OUT" if last_bounce["is_out"] else "IN"
        lines.append(f"Last bounce: {verdict} on {last_bounce['side']} ({last_bounce['context']}) @ {last_bounce['frame_idx']}")

    x0, y0, line_h, pad = 10, 10, 26, 6
    box_w = 320
    box_h = line_h * len(lines) + pad * 2
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), COLOR_HUD_BG, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), COLOR_HUD_FG, 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (x0 + pad, y0 + pad + (i + 1) * line_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD_FG, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",       required=True)
    ap.add_argument("--tracking",    required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--score",       required=True)
    ap.add_argument("--output",      required=True)
    args = ap.parse_args()

    tracking    = load_json(args.tracking)
    predictions = load_json(args.predictions)
    score       = load_json(args.score)

    track_lookup           = build_tracking_lookup(tracking)
    completed, all_points  = build_score_lookups(score)
    top_level_court        = tracking.get("court_keypoints")

    state_lookup = build_state_lookup(score)
    bounce_lookup = build_bounce_lookup(score)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps      = cap.get(cv2.CAP_PROP_FPS)
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    event_label_map = build_event_label_map(predictions, fps, EVENT_WINDOW_SEC)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output,
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    ball_trail = deque(maxlen=BALL_TRAIL_LEN)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        track_data = track_lookup.get(frame_idx, {})

        # Tracking layer (court → players → ball)
        court = track_data.get("court_keypoints", top_level_court)
        draw_court(frame, court)
        draw_players(frame, track_data.get("players", []))

        ball = track_data.get("ball")
        if ball:
            ball_trail.append((ball["x"], ball["y"], ball.get("interpolated", False)))
        draw_ball(frame, ball_trail)

        # Score HUD (top-left)
        score_state = get_score_at(completed, frame_idx)
        state = state_lookup.get(frame_idx, {})
        active_pt   = get_active_point(all_points, frame_idx)
        completed_pt = get_last_completed_point(all_points, frame_idx)
        draw_hud(frame, score_state, state, active_pt, frame_idx, n_frames, bounce_lookup, completed_pt)

        # Action events panel (left, below HUD)
        draw_event_panel(frame, event_label_map.get(frame_idx, []))

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  ... {frame_idx}/{n_frames} frames")

    cap.release()
    writer.release()
    print(f"Wrote {frame_idx} annotated frames to {args.output}")


if __name__ == "__main__":
    main()