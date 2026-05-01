#!/bin/bash
#SBATCH --job-name=tennis-score
#SBATCH --output=../outputs/score.log
#SBATCH --error=../outputs/score.log
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# No GPU needed — pure Python logic

conda activate tennis

# ---- batch mode: process all videos ----
python -m scripts.score \
    --tracks-dir  outputs/tracks/ \
    --preds-dir   ../Action-Spotting/checkpoints/tennis_rny008gsm_gru_rgb/ \
    --output-dir  outputs/scores/

# ---- single-video mode (uncomment + adapt for quick tests) ----
# python -m scripts.score \
#     --tracking  outputs/tracks/match1.json \
#     --preds     ../Action-Spotting/checkpoints/tennis_rny008gsm_gru_rgb/match1_preds.json \
#     --output    outputs/scores/match1_score.json
