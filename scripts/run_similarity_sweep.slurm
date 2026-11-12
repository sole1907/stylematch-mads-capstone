#!/bin/bash
# Submits sweep.py's full-scale staged hyperparameter search as a Great
# Lakes batch job (learning_rate x freeze_until_block jointly, then
# margin, then num_steps — see sweep.py's __main__ and
# dev-notes/architecture-decision.md for why staged this way).
#
# One-time setup BEFORE first use (venv/dependencies are OS-specific):
#   1. Get the repo onto Great Lakes: git clone <repo-url>, or git pull
#      if already cloned there.
#   2. module load python3.11-anaconda/2024.02   # or whatever `module
#      avail python` currently shows, module names change over time
#   3. python3 -m venv venv && source venv/bin/activate
#   4. pip install -r requirements.txt
#   5. Make sure data/raw/deepfashion/ actually exists on Great Lakes storage.
#
# Submit:  sbatch scripts/run_similarity_sweep.slurm
# Monitor: squeue -u $USER
# Output:  reports/similarity_sweep_full_<jobid>.out (this job's stdout/stderr)
#          reports/staged_search_stage_*.csv, staged_search_summary.csv
#            (sweep.py's own results, resumable if the job times out)

#SBATCH --job-name=stylematch-similarity-sweep
#SBATCH --account=YOUR_ACCOUNT_NAME
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/similarity_sweep_full_%j.out
#SBATCH --error=reports/similarity_sweep_full_%j.err

set -euo pipefail

module load python3.11-anaconda/2024.02
source venv/bin/activate

# Confirms a GPU is actually visible before committing hours to the
# real sweep — cheap sanity check, not optional.
python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

python3 -m stylematch.similarity.sweep
