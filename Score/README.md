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

## 2. What is already done

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

## 3. What you need to implement

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

## 4. Useful context

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


## 5. Pipeline (`pipeline.py`)

The file `pipeline.py` implements an end-to-end process on a single video. It does the following:
1. Exracts `jpg` frames from the raw `mp4` video.
2. Runs action spotting on the extracted frames.
3. Runs tracking on a resized version of the video to match the input size of the tracking components.
4. Computes the score based on the outputs of action spotting and tracking by using the `score.py` file.
5. Annotates the video with tracking and action-spotting outputs.

**Important things**:
- The input video must be `1920x1080` to match the input of action spotting. It will be automatically resized to `1280x720` for the tracking part.
- Running the full `pipeline.py` file will produce five outputs in `outputs/predictions/<video_name>`: 3 for the action spotting, 1 for the tracking, and 1 for the final score.
- The input video must be in `mp4` format and placed in `Score/data/tennis/videos`.

To run the file, you simply have to specify the name of the video, and the action spotting model to use. **Run the file from the `Score` folder**. For example:
```bash
python pipeline.py \
      --video_name test_clip.mp4 \
      --as_model_name tennis_rny008gsm_gru_rgb
```

Here is what the `Score` folder looks like on my machine after I ran `pipeline.py` on `test_clip.mp4`:
```bash
Score
├── data
│   └── tennis
│       ├── frames
│       │   └── test_clip
|       |       ├── 000000.jpg
|       |       ├── 000001.jpg
|       |       └── ...
│       └── videos
│           └── test_clip.mp4
├── jobs
│   ├── pipeline.sbatch
│   └── score.sbatch
├── outputs
│   ├── logs
│   │   └── pipeline.log
│   └── predictions
│       └── test_clip
│           ├── action_spotting.json
│           ├── action_spotting.recall.json.gz
│           ├── action_spotting.score.json.gz
│           ├── score.json
│           └── tracking.json
├── scripts
|   └── score.py
├── io_utils.py
├── pipeline.py
├── score.py
└── README.md
```

Here are the flags available in the `pipeline.py` script:
* `--video_name`
  Name of the input video file to process (e.g., `match.mp4`). Required.

* `--as_model_name`
  Selects which pre-trained Action Spotting model to use. Required.
  Allowed values:

  * `tennis_rny002gsm_gru_rgb`
  * `tennis_rny008gsm_gru_rgb`

* `--force-score`
  Forces recomputation of the score, even if a `score.json` file already exists.

* `--force-tracking`
  Forces rerunning the tracking step, even if a `tracking.json` file already exists.

* `--force-action-spotting`
  Forces rerunning the action spotting step, even if an `action_spotting.json` file already exists.

* `--no-players`
  Skips player tracking during the tracking step.

* `--no-ball`
  Skips ball tracking during the tracking step.

* `--no-court`
  Skips court detection during the tracking step.

* `--initial-sets`
  Sets the starting set score in `FAR:NEAR` format (e.g., `1:0`).

* `--initial-games`
  Sets the starting game score in `FAR:NEAR` format (e.g., `3:2`).

* `--initial-points`
  Sets the starting point score in `FAR:NEAR` format (e.g., `2:1`).

* `--no-viz`
  Slip the visualization step.
* `--force-viz`
  Force the visualization step even if outputs already exist.

---

# Known Problems

When serving, it can be inside the court, but if a Let occured (the ball touching the net), it doesn't count. As it is really hard to track, we will just acknowledge the problem.

---
`[Originally made by Claude and then modified by the authors.]` 