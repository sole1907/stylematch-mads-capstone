#!/bin/bash
# Checks whether category-size-weighted batch sampling actually helps,
# and whether the original sweep's num_steps=100 winner still holds
# under it -- see scripts/run_similarity_sampling_check.py for the
# full grid (18 configs: weight_by_category_size x num_steps x
# images_per_category) and why this exists.
#
# --time=08:00:00 is the account's hard per-job cap, requested as a
# safety buffer, not an estimate -- 18 configs on real GPU hardware
# should finish well under that; Slurm releases the allocation as
# soon as the job actually completes.
#
# Same one-time setup as the other similarity SLURM scripts (repo
# already cloned, venv already built, real DeepFashion data already on
# this machine) -- nothing new needed if that's already been done once.
#
# Submit:  sbatch scripts/run_similarity_sampling_check.slurm
# Monitor: squeue -u $USER
# If it times out: just run the submit command again -- same script,
#          same results file, already-completed configs are skipped.
# Output:  reports/similarity_sampling_check_<jobid>.out/.err
#          reports/similarity_sampling_check.csv (its own file, never
#            overwrites the original sweep's or margin_iteration2's
#            results)

#SBATCH --job-name=stylematch-similarity-sampling-check
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/similarity_sampling_check_%j.out
#SBATCH --error=reports/similarity_sampling_check_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Similarity sampling check starting: $(date)"
time python3 scripts/run_similarity_sampling_check.py
echo "Similarity sampling check finished: $(date)"
