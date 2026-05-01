"""
scripts/score.py — entry point for score prediction.

Usage
-----
Single video:
    python -m scripts.score \\
        --tracking  outputs/tracks/match1.json \\
        --preds     ../Action-Spotting/checkpoints/tennis_rny008gsm_gru_rgb/match1_preds.json \\
        --output    outputs/scores/match1_score.json

Batch (all pairs found under --tracks-dir / --preds-dir):
    python -m scripts.score \\
        --tracks-dir  outputs/tracks/ \\
        --preds-dir   ../Action-Spotting/checkpoints/tennis_rny008gsm_gru_rgb/ \\
        --output-dir  outputs/scores/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make sure the repo root is on sys.path when invoked with python -m
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import load_tracking, load_predictions, merge_frames, save_score_output
from src.score import ScoreComputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_one(tracking_path: Path, preds_path: Path, output_path: Path) -> None:
    log.info("=== %s ===", tracking_path.stem)

    tracking = load_tracking(tracking_path)
    predictions = load_predictions(preds_path)
    frames = merge_frames(tracking, predictions)

    log.info("Merged %d frames  (tracking=%d, preds=%d)",
             len(frames), tracking["n_frames"], len(predictions))

    computer = ScoreComputer(tracking)
    result = computer.run(frames)
    result.setdefault("video", tracking.get("video", tracking_path.stem))

    save_score_output(result, output_path)
    log.info("Done → %s", output_path)


def _match_pairs(
    tracks_dir: Path,
    preds_dir: Path,
) -> list[tuple[Path, Path]]:
    """
    Pair tracking JSONs with prediction JSONs by stem name.

    Tracking:    tracks_dir/<name>.json
    Predictions: preds_dir/<name>_preds.json  OR  preds_dir/<name>.json
    """
    pairs = []
    for track_file in sorted(tracks_dir.glob("*.json")):
        stem = track_file.stem
        # Try both naming conventions used by the action-spotting repo
        candidates = [
            preds_dir / f"{stem}_preds.json",
            preds_dir / f"{stem}.json",
        ]
        pred_file = next((c for c in candidates if c.exists()), None)
        if pred_file is None:
            log.warning("No predictions found for %s — skipping", stem)
            continue
        pairs.append((track_file, pred_file))
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute tennis scores from tracking + action-spotting outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single-video mode
    single = p.add_argument_group("single-video mode")
    single.add_argument("--tracking", type=Path, help="Path to tracking JSON")
    single.add_argument("--preds",    type=Path, help="Path to action-spotting predictions JSON")
    single.add_argument("--output",   type=Path, help="Output JSON path")

    # Batch mode
    batch = p.add_argument_group("batch mode")
    batch.add_argument("--tracks-dir",  type=Path, help="Directory of tracking JSONs")
    batch.add_argument("--preds-dir",   type=Path, help="Directory of predictions JSONs")
    batch.add_argument("--output-dir",  type=Path, help="Directory for output JSONs")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- single-video mode ------------------------------------------------
    if args.tracking or args.preds:
        if not args.tracking or not args.preds or not args.output:
            log.error("Single-video mode requires --tracking, --preds, and --output.")
            sys.exit(1)
        _run_one(args.tracking, args.preds, args.output)
        return

    # ---- batch mode -------------------------------------------------------
    if args.tracks_dir or args.preds_dir:
        if not args.tracks_dir or not args.preds_dir or not args.output_dir:
            log.error("Batch mode requires --tracks-dir, --preds-dir, and --output-dir.")
            sys.exit(1)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pairs = _match_pairs(args.tracks_dir, args.preds_dir)
        if not pairs:
            log.error("No matching pairs found — nothing to do.")
            sys.exit(1)
        for track_file, pred_file in pairs:
            out = args.output_dir / f"{track_file.stem}_score.json"
            try:
                _run_one(track_file, pred_file, out)
            except Exception as exc:          # keep going on partial failures
                log.error("Failed on %s: %s", track_file.stem, exc)
        return

    # ---- nothing provided -------------------------------------------------
    log.error("Provide either single-video or batch arguments.  Use --help for details.")
    sys.exit(1)


if __name__ == "__main__":
    main()
