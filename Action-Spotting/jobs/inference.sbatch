#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=tennis-inference       # Name of the job 
#SBATCH --export=ALL                # Export all environment variables
#SBATCH --output=outputs/inference.log      # Log-file (important!)
#SBATCH --cpus-per-task=2           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1               # Number of GPU's
#SBATCH --time=04:00:00              # Max execution time
#

cd $SLURM_SUBMIT_DIR
# Activate your Anaconda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/adelandsheere/anaconda3/envs/tennis

# Run your Python script
python test_e2e.py checkpoints/tennis_rny008gsm_gru_rgb data/tennis/frames -s single --save