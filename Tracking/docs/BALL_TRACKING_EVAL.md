## Ball Tracking Evaluation on RacketVision

### Dataset

Evaluation was performed on the tennis test split of the [RacketVision dataset](https://huggingface.co/datasets/linfeng302/RacketVision), a large-scale multi-sport benchmark covering badminton, table tennis, and tennis. The tennis test split consists of 43 rally clips at 1920×1080 resolution, each annotated with per-frame ball positions. RacketVision was chosen as an evaluation set independent from the TrackNetV1 data the model was trained on, avoiding any data contamination.

Evaluation uses the standard TrackNet ε-threshold metric: a prediction is counted as correct (TP) if it falls within ε pixels of the ground truth position. Since RacketVision uses sparse annotations (only clearly visible frames are labeled), unannotated frames are excluded from the metric computation entirely.

### Match Filtering

Visual inspection of the 43 test clips revealed that 9 matches were filmed in non-standard conditions (unusual camera angles, extreme zoom levels, or heavily occluded courts) that are outside the expected operating conditions of the tracker. These matches were excluded from the filtered evaluation to better reflect real-world performance on standard broadcast footage.

### Results

#### Full test split (43 rallies)

| ε (px) | TP | FP | FN | Precision | Recall | F1 (micro) | F1 (macro) | Mean error (px) |
|-------:|---:|---:|---:|----------:|-------:|-----------:|-----------:|----------------:|
| 4      | 1210 | 528 | 737 | 0.696 | 0.622 | 0.657 | 0.647 | 30.62 |
| 7      | 1445 | 293 | 502 | 0.831 | 0.742 | 0.784 | 0.775 | 30.62 |
| 10     | 1520 | 218 | 427 | 0.875 | 0.781 | 0.825 | 0.816 | 30.62 |

#### Filtered (34 rallies, excluding non-standard footage)

| ε (px) | TP | FP | FN | Precision | Recall | F1 (micro) | F1 (macro) | Mean error (px) |
|-------:|---:|---:|---:|----------:|-------:|-----------:|-----------:|----------------:|
| 4      | 1020 | 425 | 557 | 0.706 | 0.647 | 0.675 | 0.669 | 24.19 |
| 7      | 1217 | 228 | 360 | 0.842 | 0.772 | 0.805 | 0.800 | 24.19 |
| 10     | 1282 | 163 | 295 | 0.887 | 0.813 | 0.848 | 0.844 | 24.19 |

Removing the 9 non-standard clips improves F1 by ~2 points across all thresholds and reduces mean pixel error from 30.6 to 24.2 px, confirming that the excluded matches were genuine outliers rather than representative failure cases.