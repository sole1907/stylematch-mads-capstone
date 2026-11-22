#!/bin/bash
# Hyperparameter iteration #2: narrow num_steps follow-up (14 new
# configs, num_steps=[1000, 2000] x margin=[0.02, 0.05, 0.1, 0.15, 0.2,
# 0.25, 0.3], the same margin grid as the original sweep's stage 2 --
# swept again here, not fixed at 0.15, since that margin winner was
# found with num_steps fixed at only 30, so a longer training run
# could plausibly favor a different margin; num_steps=500 at
# margin=0.15 already exists in compatibility_staged_search_stage_3.csv,
# not re-run), see scripts/run_compatibility_num_steps_iteration2.py
# for why this exists (the original sweep's num_steps winner, 500, sat
# at the edge of the tested range).
#
# Same one-time setup as run_compatibility_sweep.slurm (repo already
# cloned, venv already built, real data already on this machine) --
# nothing new needed if that's already been done once.
#
# Submit:  sbatch scripts/run_compatibility_num_steps_iteration2.slurm
# Monitor: squeue -u $USER
# Output:  reports/compatibility_num_steps_iteration2_<jobid>.out/.err
#          reports/compatibility_num_steps_iteration2.csv (never
#            overwrites the original sweep's
#            reports/compatibility_staged_search_stage_{2,3}.csv)

#SBATCH --job-name=stylematch-compat-num-steps-iter2
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=reports/compatibility_num_steps_iteration2_%j.out
#SBATCH --error=reports/compatibility_num_steps_iteration2_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "num_steps iteration 2 starting: $(date)"
time python3 scripts/run_compatibility_num_steps_iteration2.py
echo "num_steps iteration 2 finished: $(date)"
