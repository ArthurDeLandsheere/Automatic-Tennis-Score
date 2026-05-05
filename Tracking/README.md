J'ai demandé à un LLM de scinder le notebook en fichiers et de me générer ce readme après pour avoir la même structure que l'action-spotting.





# Tracking

This folder contains the **tracking component** of our automatic tennis scoring pipeline:

1. **Player detection & tracking** — YOLOv8m (pretrained on COCO) + ByteTrack
2. **Court-line detection** — TennisCourtDetector ([yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)): DL model with basic CV postprocessing steps.
3. **Ball tracking** — TrackNet V1 ([yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)) wrapped in a batched sliding-window inference loop (fixes the OOM in the reference repo)

The output is a single JSON per video, designed to be consumed by the score-prediction stage alongside the action-spotting predictions.

## 1. Setup

### 1.1 Environment

You can **reuse the `tennis` conda env from `Action-Spotting`** — just install the extra deps below into it. If you haven't created it yet:

```bash
cd Tracking
conda create -n tennis python=3.11
conda activate tennis
```

Install the requirements:
```bash
pip install -r requirements.txt
```

Then install the right version of torch (same line as action-spotting):
```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

And finally:
```bash
conda install ffmpeg
```

> **NB on `lap` vs `lapx`**: `requirements.txt` pins `lapx`, not `lap`. The legacy `lap` wheel pins numpy to a version that breaks scipy and crashes ByteTrack on its first call. If you accidentally installed `lap` already, run `pip uninstall -y lap` then restart your shell before continuing.

### 1.2 Videos

Drop your match videos into `data/tennis/videos/`. They should ideally be **1280×720** — TrackNet was trained on 720p and accuracy degrades silently on lower resolutions. To resize:
```bash
ffmpeg -i in.mp4 -vf scale=1280:720 out.mp4
```


### 1.3 Checkpoints

Three model checkpoints are needed:

**YOLOv8m**: auto-downloaded by `ultralytics` on first use. To pre-cache it:
```bash
mkdir -p checkpoints
cd checkpoints && wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt && cd ..
```

**TrackNet weights**: download `model_best.pt` from the Google Drive link in the [yastrebksv/TrackNet README](https://github.com/yastrebksv/TrackNet#pretrained-model) and place it at `checkpoints/tracknet/model_best.pt`. Using `gdown` (file ID is in the repo's README):
```bash
mkdir -p checkpoints/tracknet
gdown <FILE_ID> -O checkpoints/tracknet/model_best.pt
```

**TennisCourtDetecor weights**: download `model_tennis_court_det.pt` at https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view and place it at `checkpoints/court/model_tennis_court_det.pt`.

After this, `checkpoints/` should look like:
```
checkpoints
├── court
│   └── model_tennis_court_det.pt
├── tracknet
│   └── model_best.pt
└── yolov8m.pt
```

### 1.4 Structure

After setup, the `Tracking` repo should look like this:
```
Tracking
├── checkpoints                 
├── data
│   └── tennis
│       ├── videos              
│       │   └── ...
│       └── ground_truth
│           └── ...
├── jobs                        
│   ├── track.sbatch
│   ├── visualize.sbatch
│   └── evaluate.sbatch
├── outputs
│   ├── tracks                  
│   ├── videos                
│   └── logs                
├── scripts            
│   ├── track.py
│   ├── visualize.py
│   └── evaluate.py
├── src            
│   ├── ball.py                 # TrackNet inference + smoothing
│   ├── court.py                # homography-based + simple court detectors
│   ├── io_utils.py             # video metadata + JSON I/O
│   ├── metrics.py              # ε-threshold ball metric
│   ├── players.py              # YOLO + ByteTrack + main-player selection
│   ├── tracknet_model.py       # vendored BallTrackerNet architecture
│   └── viz.py                  # drawing + annotated video writer
├── .gitignore
├── README.md
└── requirements.txt
```

## 2. Run

### 2.1 Track

To run tracking on every mp4 in `data/tennis/videos/`:
```bash
sbatch jobs/track.sbatch
```

This produces one JSON per video at `outputs/tracks/<name>.json`. Each JSON contains, frame by frame: the 2 main player bboxes (with persistent IDs and per-frame top/bottom side labels), the ball position (raw + smoothed, with an `interpolated` flag), and the court polygon.

The job skips videos whose JSON already exists, so it's safe to re-run after adding new videos.

To run on a single video manually:
```bash
python -m scripts.track \
    --video data/tennis/videos/match1.mp4 \
    --output outputs/tracks/match1.json
```

### 2.2 Visualise

Once `outputs/tracks/` has JSONs, render annotated videos + sample frames + trajectory plots:
```bash
sbatch jobs/visualize.sbatch
```

For each tracks JSON, this produces:
- `outputs/videos/<name>/annotated.mp4` — full clip with boxes, ball trail, court overlay, HUD
- `outputs/videos/<name>/sample_frames.png` — 4 evenly-spaced frames
- `outputs/videos/<name>/trajectory.png` — ball path heatmap over a mid-clip frame

The annotated videos use `mp4v` codec; for inline playback in Colab/browsers re-encode to H.264 with the ffmpeg command printed at the end of each run.

### 2.3 Evaluate (optional)

If you have TrackNet-format ground-truth CSVs (columns: `file_name, visibility, x-coordinate, y-coordinate`), drop them in `data/tennis/ground_truth/` matching by stem (e.g. `match1.csv` ↔ `match1.json`) and run:
```bash
sbatch jobs/evaluate.sbatch
```

This computes precision / recall / accuracy / F1 at ε ∈ {4, 7, 10} pixels and writes a `_metrics.json` next to each tracks file. ε=4 is the threshold used in the TrackNet papers.

## 3. JSON output schema

The tracking JSON produced by `scripts/track.py` is the contract between this stage and the score-prediction stage:

```json
{
  "video": "match1.mp4",
  "video_path": "data/tennis/videos/match1.mp4",
  "fps": 30.0,
  "width": 1280,
  "height": 720,
  "n_frames": 900,
  "main_player_ids": [1, 4],
  "court_polygon": [[x1, y1], [x2, y2], ...],
  "frames": [
    {
      "frame_idx": 0,
      "players": [
        {"id": 1, "bbox": [x1, y1, x2, y2], "conf": 0.93, "side": "top"},
        {"id": 4, "bbox": [...],            "conf": 0.87, "side": "bottom"}
      ],
      "ball": {"x": 640.5, "y": 360.2, "conf": 0.72, "interpolated": false}
    },
    ...
  ]
}
```

Notes for the score-logic consumer:
- `main_player_ids` are persistent across the clip; `side` (`"top"` / `"bottom"` / `"unknown"`) is per-frame and flips at change-of-ends.
- `ball` is `null` when the ball is undetected and not recoverable by interpolation. `interpolated: true` means the position came from linear interp across a short gap, not from TrackNet directly.
- The first 2 frames of every video have `ball: null` regardless — TrackNet needs a 3-frame window.
- `court_polygon` is the projection of the reference court silhouette through the detected homography. Use it as a geometric filter (`cv2.pointPolygonTest`) when deciding whether ball/player positions are physically plausible — `src/court.py:point_in_court` is a one-liner helper.

## 4. Merging with action-spotting (next step)

Both pipelines now produce per-frame JSON outputs aligned by frame index:
- This folder → `outputs/tracks/<name>.json` with `frames[i].players` and `frames[i].ball`
- `Action-Spotting` → its own predictions JSON with per-frame event class scores

The score-prediction stage reads both, joins on `frame_idx`, and applies the rules of tennis. The natural place for that code is a new sibling folder, e.g. `Score/`, that imports from neither but consumes both JSON formats.

## Améliorations qu'on peut encore faire

Honest list of what's not finished:

1. **Court detection est maintenant faite sur chaque frame**: on peut ajouter un peu de smoothing et/ou d'interpolation si nécessaire mais c'est déjà pas mal. 
2. **Peut-être essayer TrackNet V3.**
3. **Peut-être Kalman filter pour le smoothing de la balle**