#!/bin/bash
# Hyperparameter iteration #2: narrow margin follow-up (10 new configs,
# margin=[0.7, 1.0] x num_steps=[30,60,100,200,500], the same num_steps
# grid as the original sweep's stage 3 — swept again here, not fixed at
# 60, since a bigger margin could plausibly peak at a different
# num_steps than margin=0.5's peak of 60; margin=0.5 across all 5
# num_steps values already exists in staged_search_stage_3.csv, not
# re-run), see scripts/run_similarity_margin_iteration2.py for why this
# exists (the original sweep's margin winner, 0.5, sat at the edge of
# the tested range).
#
# Same one-time setup as run_similarity_sweep.slurm (repo already
# cloned, venv already built, real data already on this machine) —
# nothing new needed if that's already been done once.
#
# Submit:  sbatch scripts/run_similarity_margin_iteration2.slurm
# Monitor: squeue -u $USER
# Output:  reports/margin_iteration2_<jobid>.out/.err
#          reports/margin_iteration2.csv (never overwrites the
#            original sweep's reports/staged_search_stage_{2,3}.csv)

#SBATCH --job-name=stylematch-similarity-margin-iter2
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=reports/margin_iteration2_%j.out
#SBATCH --error=reports/margin_iteration2_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Margin iteration 2 starting: $(date)"
time python3 scripts/run_similarity_margin_iteration2.py
echo "Margin iteration 2 finished: $(date)"
