"""
eval_score.py
Evaluate a tennis-pipeline score.json against a manually annotated GT CSV.

Usage:
    python eval_score.py --pred score.json --gt ground_truth.csv [--fps 25]

What this script measures
-------------------------
We split the evaluation into three independent layers, because errors at
different stages of the pipeline (segmentation, winner attribution, score
state) compound multiplicatively, and a single end-of-match correctness
number would mostly tell us how long the clip is.

  1. Rally segmentation:
        Did we detect that a rally happened, regardless of who won?
        Reported as precision / recall / F1 across several start-frame
        tolerances.

        We match by start_frame proximity rather than interval IoU
        because predicted end_frame in this pipeline is "the frame of
        the next serve" for most rallies (point closure is triggered
        when a new serve event is detected, not when the rally actually
        ends), making IoU artificially low even when scoring was
        otherwise correct. start_frame is set at the serve event itself
        and is the reliable signal for alignment.

  2. Winner attribution on matched rallies:
        Given that a predicted rally was correctly matched to a GT one,
        did we correctly identify the winner? Confusion matrix uses
        "server wins" as the positive class.

  3. Score-state correctness at rally-start checkpoints:
        At the moment each new point begins, is the on-screen score
        correct? This is the only moment a viewer actually reads the
        scoreboard, so we evaluate state agreement only there rather
        than continuously through frames (which would be polluted by
        the same end_frame issue noted in (1)).

Lets and excluded annotations are filtered out of all metrics.
"""

import argparse
import csv
import json
from typing import Iterable


# ─────────────────────────────────────────────────────────────────────────────
# Matching: start-frame proximity, greedy
# ─────────────────────────────────────────────────────────────────────────────

def match_points_by_start(pred, gt, max_offset: int = 30):
    """
    Match predicted to ground-truth rallies by start-frame proximity.

    A predicted rally i and a GT rally j are paired if
        |pred[i].start_frame - gt[j].start_frame| <= max_offset
    Pairs are picked greedily, smallest distance first, with each
    rally on either side used at most once.

    Returns
    -------
    matches      : list of (pred_idx, gt_idx, frame_distance)
    unmatched_p  : indices of predicted rallies with no GT counterpart
                   (these are false positives — phantom rallies)
    unmatched_g  : indices of GT rallies with no predicted counterpart
                   (these are false negatives — missed rallies)
    """
    candidates = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            d = abs(p["start_frame"] - g["start_frame"])
            if d <= max_offset:
                candidates.append((d, i, j))
    candidates.sort()  # smallest distance first

    matches, used_p, used_g = [], set(), set()
    for d, i, j in candidates:
        if i in used_p or j in used_g:
            continue
        matches.append((i, j, d))
        used_p.add(i)
        used_g.add(j)

    unmatched_p = [i for i in range(len(pred)) if i not in used_p]
    unmatched_g = [j for j in range(len(gt))   if j not in used_g]
    return matches, unmatched_p, unmatched_g


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_pred(path: str) -> list[dict]:
    """Load score.json and return its 'points' list."""
    with open(path) as f:
        return json.load(f)["points"]


def load_gt(path: str) -> list[dict]:
    """
    Load ground_truth.csv and return a normalized list of GT rallies.
    Lets and rallies marked 'exclude' are filtered out.

    Expected CSV columns:
      rally_idx, start_frame, end_frame, winner_side, server_side,
      serve_number, outcome, notes
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        outcome = (r.get("outcome") or "").strip().lower()
        if outcome in {"let", "exclude"}:
            continue
        out.append({
            "start_frame":  int(r["start_frame"]),
            "end_frame":    int(r["end_frame"]),
            "winner_side":  r["winner_side"].strip().lower(),
            "server_side":  r["server_side"].strip().lower(),
            "serve_number": int(r["serve_number"]),
            "outcome":      outcome,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Score-state replay
# ─────────────────────────────────────────────────────────────────────────────

def state_at_frame(points: list[dict], frame_idx: int) -> tuple:
    """
    Return (sets_far, sets_near, games_far, games_near, points_far, points_near)
    as of the given frame, by walking the points list.

    A point is considered to have updated the state once frame_idx has passed
    its end_frame. Points whose end_frame is beyond frame_idx haven't happened
    yet from the perspective of an observer at frame_idx.

    Note: the "points within current game" component (last two values) only
    reflects the simple counter in `score_after`. If you want exact 0/15/30/40
    semantics with deuce/advantage, use the TennisScore-driven version of
    `gt_score_after` below.
    """
    s = {
        "sets":   {"far": 0, "near": 0},
        "games":  {"far": 0, "near": 0},
        "points": {"far": 0, "near": 0},
    }
    for p in points:
        if p["end_frame"] > frame_idx:
            break
        s = p.get("score_after", s)
    return (
        s["sets"]["far"],   s["sets"]["near"],
        s["games"]["far"],  s["games"]["near"],
        s["points"]["far"], s["points"]["near"],
    )


def gt_score_after(gt_points: list[dict]) -> list[dict]:
    """
    Attach a `score_after` field to each GT rally, so state_at_frame() can
    be applied to GT just like to predictions.

    Naive scorer: incremental points counter, no deuce/advantage tracking,
    no game/set transitions. Sufficient for the rally-start checkpoint
    metric *if* you only compare the sets/games columns. For full-state
    parity (including 0/15/30/40), replace this function's body with:

        from score import TennisScore
        def gt_score_after(gt_points):
            ts = TennisScore()
            out = []
            for g in gt_points:
                ts.server_side = g["server_side"]
                ts.point_start_frame = g["start_frame"]
                ts.point_won(g["winner_side"], g["end_frame"],
                             reason=g.get("outcome", "gt"))
                out.append({**g, "score_after": {
                    "sets":   dict(ts.sets),
                    "games":  dict(ts.games),
                    "points": dict(ts.points),
                }})
            return out

    The same TennisScore drives both predicted and GT states then, so any
    disagreement reflects rally misattribution rather than scorer mismatch.
    """
    s = {
        "sets":   {"far": 0, "near": 0},
        "games":  {"far": 0, "near": 0},
        "points": {"far": 0, "near": 0},
    }
    out = []
    for g in gt_points:
        s = {k: dict(v) for k, v in s.items()}
        s["points"][g["winner_side"]] += 1
        out.append({**g, "score_after": s})
    return out


def state_at_checkpoints(pred_points, gt_points) -> dict | None:
    """
    For each GT rally with start frame S, compare predicted and GT state
    at frame S-1 — that is, the state right before this rally begins,
    which equals the cumulative state of all PRIOR rallies.

    This sidesteps the predicted end_frame issue entirely: we never ask
    "is the predicted score correct *during* the rally", only "is it
    correct at the moment a viewer would read the scoreboard between
    points".
    """
    n = len(gt_points)
    if n == 0:
        return None

    full = sets_ok = games_ok = 0
    for g in gt_points:
        f = g["start_frame"] - 1
        ps = state_at_frame(pred_points, f)
        gs = state_at_frame(gt_points, f)
        if ps == gs:
            full += 1
        if ps[:2] == gs[:2]:
            sets_ok += 1
        if ps[:4] == gs[:4]:
            games_ok += 1

    return {
        "n":     n,
        "full":  full / n,
        "sets":  sets_ok / n,
        "games": games_ok / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def report_segmentation(pred, gt, fps: float, offsets: Iterable[int]):
    """Rally segmentation P/R/F1 across several start-frame tolerances."""
    print("─── Rally segmentation (start-frame matching) ───")
    print(f"{'Δframes':>8} {'Δseconds':>9} "
          f"{'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
    for off in offsets:
        m, up, ug = match_points_by_start(pred, gt, off)
        tp, fp, fn = len(m), len(up), len(ug)
        p, r, f = _prf(tp, fp, fn)
        print(f"{off:>8} {off/fps:>9.2f} "
              f"{tp:>4} {fp:>4} {fn:>4} "
              f"{p:>7.3f} {r:>7.3f} {f:>7.3f}")


def report_attribution(matches, pred, gt):
    """Winner accuracy + server-wins confusion matrix on matched rallies."""
    if not matches:
        print("No matched rallies — skipping attribution metrics.")
        return

    correct = sum(
        1 for i, j, _ in matches
        if pred[i]["winner_side"] == gt[j]["winner_side"]
    )
    print(f"\n─── Winner attribution on matched rallies ───")
    print(f"Winner accuracy: {correct}/{len(matches)} "
          f"= {100*correct/len(matches):.1f}%")

    tp = fp = fn = tn = 0
    for i, j, _ in matches:
        gt_serv   = (gt[j]["winner_side"]   == gt[j]["server_side"])
        pred_serv = (pred[i]["winner_side"] == pred[i]["server_side"])
        if   gt_serv     and pred_serv:     tp += 1
        elif gt_serv     and not pred_serv: fn += 1
        elif not gt_serv and pred_serv:     fp += 1
        else:                               tn += 1
    print(f"\nConfusion matrix (positive class = server wins the rally):")
    print(f"             pred:server  pred:receiver")
    print(f"gt:server    {tp:>10}   {fn:>12}")
    print(f"gt:receiver  {fp:>10}   {tn:>12}")

    sn_correct = sum(
        1 for i, j, _ in matches
        if pred[i].get("serve_number") == gt[j]["serve_number"]
    )
    print(f"\nServe-number (1st/2nd) accuracy on matched: "
          f"{sn_correct}/{len(matches)} "
          f"= {100*sn_correct/len(matches):.1f}%")


def report_state(pred, gt):
    """Score-state agreement at rally-start checkpoints."""
    print(f"\n─── Score state at rally-start checkpoints ───")
    cp = state_at_checkpoints(pred, gt)
    if cp is None:
        print("No GT rallies — skipping state metric.")
        return
    print(f"Checkpoints (one per GT rally start): {cp['n']}")
    print(f"Sets correct:     {100*cp['sets']:5.1f}%")
    print(f"Games correct:    {100*cp['games']:5.1f}%")
    print(f"Full state right: {100*cp['full']:5.1f}%")


def report_errors(matches, unmatched_p, unmatched_g, pred, gt, max_show: int = 20):
    """Inspectable error lists for manual investigation."""
    print(f"\n─── Errors for inspection ───")

    if unmatched_p:
        print(f"\nPhantom predicted rallies "
              f"(predicted but no GT match) — {len(unmatched_p)}:")
        for i in unmatched_p[:max_show]:
            p = pred[i]
            print(f"  pred[{i}]: f{p['start_frame']}-{p['end_frame']}  "
                  f"winner={p['winner_side']:>4}  "
                  f"reason={p.get('reason','?')}")
        if len(unmatched_p) > max_show:
            print(f"  ... and {len(unmatched_p) - max_show} more")

    if unmatched_g:
        print(f"\nMissed ground-truth rallies "
              f"(GT but no predicted match) — {len(unmatched_g)}:")
        for j in unmatched_g[:max_show]:
            g = gt[j]
            print(f"  gt[{j}]:   f{g['start_frame']}-{g['end_frame']}  "
                  f"winner={g['winner_side']:>4}  "
                  f"outcome={g['outcome']}")
        if len(unmatched_g) > max_show:
            print(f"  ... and {len(unmatched_g) - max_show} more")

    bad = [
        (i, j) for i, j, _ in matches
        if pred[i]["winner_side"] != gt[j]["winner_side"]
    ]
    if bad:
        print(f"\nWrong winner on matched rallies — {len(bad)}:")
        for i, j in bad[:max_show]:
            p, g = pred[i], gt[j]
            print(f"  pred[{i}] f{p['start_frame']}-{p['end_frame']}  "
                  f"won={p['winner_side']:>4} (gt={g['winner_side']:>4})  "
                  f"reason={p.get('reason','?')}")
        if len(bad) > max_show:
            print(f"  ... and {len(bad) - max_show} more")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True,
                    help="Path to score.json from the pipeline.")
    ap.add_argument("--gt", required=True,
                    help="Path to ground_truth.csv produced by manual annotation.")
    ap.add_argument("--fps", type=float, default=None,
                    help="Frames per second of the clip; if omitted, read "
                         "from score.json's 'fps' field.")
    ap.add_argument("--match-offset", type=int, default=25,
                    help="Start-frame tolerance (frames) for the primary "
                         "downstream metrics. Default 25 frames "
                         "(~1s at 25fps).")
    args = ap.parse_args()

    # Load fps from pred if not supplied
    fps = args.fps
    if fps is None:
        with open(args.pred) as f:
            fps = float(json.load(f).get("fps", 25.0))

    pred = load_pred(args.pred)
    gt   = gt_score_after(load_gt(args.gt))

    print(f"Predicted rallies: {len(pred)}")
    print(f"GT rallies:        {len(gt)}")
    print(f"FPS:               {fps}\n")

    # 1. Rally segmentation across multiple tolerances
    report_segmentation(
        pred, gt, fps,
        offsets=[int(0.5 * fps), int(1.0 * fps), int(2.0 * fps)],
    )

    # 2. Winner attribution at the chosen primary tolerance
    matches, unmatched_p, unmatched_g = match_points_by_start(
        pred, gt, args.match_offset
    )
    print(f"\n(Downstream metrics use --match-offset={args.match_offset} "
          f"frames ≈ {args.match_offset/fps:.2f}s.)")
    report_attribution(matches, pred, gt)

    # 3. Score-state at rally-start checkpoints
    report_state(pred, gt)

    # 4. Inspectable error lists
    report_errors(matches, unmatched_p, unmatched_g, pred, gt)


if __name__ == "__main__":
    main()