#!/bin/bash
# Final similarity engine check: categories_per_batch, never tested
# before. See run_similarity_categories_per_batch_check.py for the grid.
#
# Submit:  sbatch scripts/run_similarity_categories_per_batch_check.slurm
# Monitor: squeue -u $USER
# If it times out: resubmit the same command, already-completed
#          configs (including reused ones) are skipped.
# Output:  reports/similarity_categories_per_batch_check_<jobid>.out/.err
#          reports/similarity_full_coverage_check.csv (shared, reused)

#SBATCH --job-name=stylematch-similarity-categories-per-batch
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/similarity_categories_per_batch_check_%j.out
#SBATCH --error=reports/similarity_categories_per_batch_check_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Categories-per-batch check starting: $(date)"
time python3 scripts/run_similarity_categories_per_batch_check.py
echo "Categories-per-batch check finished: $(date)"
