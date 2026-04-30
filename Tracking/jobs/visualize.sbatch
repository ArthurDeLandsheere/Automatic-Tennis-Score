#!/bin/bash
#SBATCH --job-name=track-viz
#SBATCH --output=outputs/logs/visualize_%j.log
#SBATCH --error=outputs/logs/visualize_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# (No GPU needed for visualisation)

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tennis

cd "$(dirname "$0")/.."

shopt -s nullglob
tracks=(outputs/tracks/*.json)
if [ ${#tracks[@]} -eq 0 ]; then
    echo "No tracks found in outputs/tracks/. Run track.sbatch first."
    exit 1
fi

for t in "${tracks[@]}"; do
    name=$(basename "${t%.*}")
    out_dir="outputs/videos/${name}"
    echo "[viz] $name → $out_dir"
    python -m scripts.visualize \
        --tracks "$t" \
        --out-dir "$out_dir"
done

echo "All done."
