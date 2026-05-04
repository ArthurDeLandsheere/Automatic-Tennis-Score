# Automatic-Tennis-Score

## 1. [Detection and Tracking](/Tracking/)

This folder contains the **tracking component** of our automatic tennis scoring pipeline:

1. **Player detection & tracking** — YOLOv8m (pretrained on COCO) + ByteTrack
2. **Court-line detection** — TennisCourtDetector [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector): DL model with basic CV postprocessing steps.
3. **Ball tracking** — TrackNet V1 ([yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)) wrapped in a batched sliding-window inference loop (fixes the OOM in the reference repo)

Look at the [README](/Tracking/README.md) file in the Tracking folder for details about the setup.


## 2. [Action Spotting](/Action-Spotting/)

For the Action Spotting task, I currently make use of the [Spotting Temporally Precise, Fine-Grained Events in Video](https://github.com/jhong93/spot) Github repository. Look at the [README](/Action-Spotting/README.md) file in the Action Spotting folder for details about the setup.