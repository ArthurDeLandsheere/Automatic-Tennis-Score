#!/bin/bash
#SBATCH --job-name=track
#SBATCH --output=outputs/logs/track_%j.log
#SBATCH --error=outputs/logs/track_%j.log
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ─── Adjust these for your cluster ────────────────────────────────────────
# (the partition/account flags above are intentionally omitted — copy whatever
# your action-spotting jobs use)

set -euo pipefail

# Activate the same conda env you use for action-spotting
source ~/miniconda3/etc/profile.d/conda.sh   # or /opt/conda/...
conda activate tennis

cd "$(dirname "$0")/.."   # → Tracking/

mkdir -p outputs/tracks outputs/logs

# Loop over every mp4 in data/tennis/videos/
shopt -s nullglob
videos=(data/tennis/videos/*.mp4)
if [ ${#videos[@]} -eq 0 ]; then
    echo "No videos found in data/tennis/videos/"
    exit 1
fi

echo "Processing ${#videos[@]} video(s)..."
for video in "${videos[@]}"; do
    name=$(basename "${video%.*}")
    out="outputs/tracks/${name}.json"
    if [ -f "$out" ]; then
        echo "  [skip] $name (output exists)"
        continue
    fi
    echo "  [run]  $name"
    python -m scripts.track \
        --video "$video" \
        --output "$out" \
        --yolo-weights checkpoints/yolov8m.pt \
        --tracknet-weights checkpoints/tracknet/model_best.pt
done

echo "All done."
