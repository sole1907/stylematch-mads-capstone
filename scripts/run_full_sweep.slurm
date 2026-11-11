#!/bin/bash
# Submits sweep.py's full-scale staged hyperparameter search as a Great
# Lakes batch job (learning_rate x freeze_until_block jointly, then
# margin, then num_steps — see sweep.py's __main__ and
# dev-notes/architecture-decision.md for why staged this way).
#
# One-time setup BEFORE first use (venv/dependencies are OS-specific,
# a macOS-built venv will NOT run on Great Lakes' Linux nodes):
#   1. Get the repo onto Great Lakes: git clone <repo-url>, or git pull
#      if already cloned there.
#   2. module load python3.11-anaconda/2024.02   # or whatever `module
#      avail python` currently shows, module names change over time
#   3. python3 -m venv venv && source venv/bin/activate
#   4. pip install -r requirements.txt
#   5. Make sure data/raw/deepfashion/ actually exists on Great Lakes
#      storage (not just locally) — this script assumes it's already
#      there, it does not transfer it for you.
#
# Submit:  sbatch scripts/run_full_sweep.slurm
# Monitor: squeue -u $USER
# Output:  reports/sweep_full_<jobid>.out (this job's stdout/stderr)
#          reports/staged_search_stage_*.csv, staged_search_summary.csv
#            (sweep.py's own results, resumable if the job times out)

#SBATCH --job-name=stylematch-sweep
#SBATCH --account=YOUR_ACCOUNT_NAME        # see Open OnDemand's Account dropdown for the real value
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00                    # placeholder: time ONE config first (see note below), then size this for real
#SBATCH --output=reports/sweep_full_%j.out
#SBATCH --error=reports/sweep_full_%j.err

set -euo pipefail

module load python3.11-anaconda/2024.02
source venv/bin/activate

# Confirms a GPU is actually visible before committing hours to the
# real sweep — cheap sanity check, not optional.
python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

python3 -m stylematch.similarity.sweep
