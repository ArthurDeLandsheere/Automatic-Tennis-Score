#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=tennis-dl-videos        # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=outputs/dl-videos.log      # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0               # Number of GPU's
#SBATCH --time=04:00:00              # Max execution time
#

cd $SLURM_SUBMIT_DIR
# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/tennis

# Run your Python script
python download_videos.py