#!/usr/bin/env bash
#
# End-to-end: pipeline → annotated video.
# Submit from Score/:  sbatch jobs/pipeline_and_viz.sbatch
#
#SBATCH --job-name=tennis-pipeline-viz
#SBATCH --output=outputs/logs/pipeline_and_viz_%j.log
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=quadro

set -euo pipefail
cd $SLURM_SUBMIT_DIR
mkdir -p outputs/logs outputs/predictions

# --- Activate env --------------------------------------------------------
source /home/edlbb/anaconda3/etc/profile.d/conda.sh
conda activate tennis

nvidia-smi
which python

# --- Config --------------------------------------------------------------
VIDEO=test_clip4.mp4
MODEL=tennis_rny008gsm_gru_rgb
VIDEO_STEM="${VIDEO%.*}"

VIDEO_ABS="$(realpath data/tennis/videos/$VIDEO)"
PRED_DIR="$(realpath outputs/predictions/$VIDEO_STEM)"
TRACKING_JSON="$PRED_DIR/tracking.json"

# --- Step 1: pipeline ----------------------------------------------------
python -u pipeline.py \
    --video_name "$VIDEO" \
    --as_model_name "$MODEL"

# --- Step 2: render annotated video --------------------------------------
# The pipeline scales tracking coords back to 1080p but doesn't update
# `video_path` inside tracking.json, so we pass the original video
# explicitly here. (See note below.)
echo "[Viz] Rendering annotated video..."
python -u -m scripts.visualize_full \
    --video       "$VIDEO_ABS" \
    --tracking    "$TRACKING_JSON" \
    --predictions "$PRED_DIR/action_spotting.json" \
    --score       "$PRED_DIR/score.json" \
    --output      "$PRED_DIR/annotated.mp4"

echo "Done. Annotated outputs in: $VIZ_OUT"
