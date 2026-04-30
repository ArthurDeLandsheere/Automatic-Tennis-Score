#!/bin/bash
#SBATCH --job-name=track-eval
#SBATCH --output=outputs/logs/evaluate_%j.log
#SBATCH --error=outputs/logs/evaluate_%j.log
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# (No GPU needed for metric computation)

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tennis

cd "$(dirname "$0")/.."

# Match each tracks JSON with a GT CSV by stem.
# data/tennis/ground_truth/match1.csv ↔ outputs/tracks/match1.json
shopt -s nullglob
tracks=(outputs/tracks/*.json)
if [ ${#tracks[@]} -eq 0 ]; then
    echo "No tracks found in outputs/tracks/."
    exit 1
fi

for t in "${tracks[@]}"; do
    name=$(basename "${t%.*}")
    gt="data/tennis/ground_truth/${name}.csv"
    if [ ! -f "$gt" ]; then
        echo "  [skip] $name — no GT at $gt"
        continue
    fi
    out="outputs/tracks/${name}_metrics.json"
    echo "  [eval] $name"
    python -m scripts.evaluate \
        --tracks "$t" \
        --gt "$gt" \
        --epsilon 4 7 10 \
        --save-json "$out"
done

echo "All done."
