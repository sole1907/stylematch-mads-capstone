#!/bin/bash
# One-off verification job, NOT the real sweep: confirms the whole
# pipeline actually runs on a real GPU node (everything so far has only
# been verified on the login node, which has no GPU at all), and gets
# one real per-step timing number to size run_similarity_sweep.slurm's
# --time before committing to the full 8-hour, 35-config run.
#
# Runs scripts/run_similarity_sweep_lite.py: a small 6-config search
# (2 learning rates x 2 freeze depths, then 2 margins), 5 training
# steps and 10 categories each, see sweep.py's run_staged_search_lite.
# Not meant to produce trustworthy hyperparameter conclusions, purely a
# cheap smoke test before the real thing.
#
# Submit:  sbatch scripts/run_similarity_sweep_lite_gpu_check.slurm
# Monitor: squeue -u $USER
# Output:  reports/similarity_sweep_lite_check_<jobid>.out/.err

#SBATCH --job-name=stylematch-similarity-gpu-check
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=reports/similarity_sweep_lite_check_%j.out
#SBATCH --error=reports/similarity_sweep_lite_check_%j.err

set -euo pipefail

source venv/bin/activate

# The actual thing we're here to confirm: a GPU is really visible to
# the job (not just requested), and which one.
python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

# Wall-clock timing around the whole run, printed explicitly so it's
# easy to find in the output without digging through every step's log
# line, used to size the real sweep's --time afterward.
echo "Lite sweep starting: $(date)"
time python3 scripts/run_similarity_sweep_lite.py
echo "Lite sweep finished: $(date)"
