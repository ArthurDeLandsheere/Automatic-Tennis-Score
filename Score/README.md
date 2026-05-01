J'ai demandé à un LLM de faire tout le "détail" du genre lire les json et les fusionner pour que je doive faire que la logique sans me faire chier pour les trucs nuls.


# Score

This folder is the third and final stage of the automatic tennis scoring pipeline.

It consumes:
- **Tracking JSON** — produced by `Tracking/scripts/track.py`
- **Action-spotting predictions JSON** — produced by `Action-Spotting` inference

and outputs a per-video **score JSON** with the computed game state (points, games, sets, final score).

## 1. Setup

No new environment needed — reuse the `tennis` conda env from the other stages.

```bash
conda activate tennis
```

No extra dependencies are required for the scaffolding. If your score logic needs additional packages (numpy, pandas, …) add them to `requirements.txt` and run:
```bash
pip install -r requirements.txt
```

## 2. Structure

```
Score
├── data
│   └── tennis
│       ├── tracks          # symlink or copy of Tracking/outputs/tracks/
│       └── predictions     # symlink or copy of Action-Spotting predictions
├── jobs
│   └── score.sbatch
├── outputs
│   ├── scores              # one <name>_score.json per video
│   └── logs
├── scripts
│   └── score.py            # CLI entry point (batch + single-video)
├── src
│   ├── io_utils.py         # loading, merging, saving — already done
│   └── score.py            # ScoreComputer — YOUR code goes here
├── requirements.txt
└── README.md
```

## 3. What is already done

### `src/io_utils.py`
- `load_tracking(path)` — loads a tracking JSON, logs basic metadata
- `load_predictions(path)` — loads action-spotting predictions, indexes by `frame_idx`
- `merge_frames(tracking, predictions)` — joins the two on `frame_idx`, adds an `events` list (classes above 0.5 threshold)
- `save_score_output(result, path)` — writes the result dict to JSON

### `scripts/score.py`
Full CLI with two modes:

**Single video:**
```bash
python -m scripts.score \
    --tracking  outputs/tracks/match1.json \
    --preds     ../Action-Spotting/checkpoints/tennis_rny008gsm_gru_rgb/match1_preds.json \
    --output    outputs/scores/match1_score.json
```

**Batch (all videos):**
```bash
python -m scripts.score \
    --tracks-dir  outputs/tracks/ \
    --preds-dir   ../Action-Spotting/checkpoints/tennis_rny008gsm_gru_rgb/ \
    --output-dir  outputs/scores/
```

Or on the cluster:
```bash
sbatch jobs/score.sbatch
```

## 4. What you need to implement

Open `src/score.py`. It contains the `ScoreComputer` class with two stubbed methods:

### `_process_frame(self, frame)`
Called once per frame (in order). Each `frame` dict contains:
```python
{
  "frame_idx": int,
  "players": [
    {"id": int, "bbox": [x1,y1,x2,y2], "conf": float, "side": "top"|"bottom"|"unknown"},
    {"id": int, "bbox": [...],          "conf": float, "side": "top"|"bottom"|"unknown"},
  ],
  "ball":   {"x": float, "y": float, "conf": float, "interpolated": bool} | None,
  "scores": {"Serve": float, "Bounce": float, ...} | None,  # raw action-spotting scores
  "events": ["Serve", ...],   # classes above the 0.5 detection threshold
}
```

### `_build_result(self)`
Called once after all frames are processed. Must return a dict that will be saved to disk.

Suggested output schema:
```json
{
  "video": "match1.mp4",
  "points": [
    {
      "point_idx":   0,
      "start_frame": 42,
      "end_frame":   130,
      "winner_id":   1,
      "winner_side": "bottom"
    }
  ],
  "games": [{"game_idx": 0, "score_top": 4, "score_bottom": 2}],
  "sets":  [{"set_idx":  0, "score_top": 6, "score_bottom": 3}],
  "final_score": "6-3"
}
```

## 5. Useful context

- `self.fps` — frames per second (use to convert frame offsets to seconds)
- `self.main_player_ids` — `[id_top, id_bottom]` persistent player IDs
- `self.court_polygon` — court boundary as `[[x,y], ...]`; use `cv2.pointPolygonTest` or the `point_in_court` helper in `Tracking/src/court.py` to check if the ball is inside
- The `side` field on each player is per-frame and flips at change-of-ends
- `ball` is `None` when undetected; `interpolated: true` means linear interp across a short gap

### Action-spotting event classes
Check `Action-Spotting/data/tennis/class.txt` for the exact labels. Typical ones:
`Serve`, `Bounce`, `NetHit`, `EmptyAction`, etc.

### Detection threshold
The default merge threshold is `0.5` (set in `io_utils.merge_frames`). You can lower it for noisier events or raise it for precision. You can also bypass `frame["events"]` and work directly from `frame["scores"]` for more nuanced logic.
