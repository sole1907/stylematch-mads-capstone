#!/bin/bash
# Checks whether dropping thin categories (<500 images) entirely
# improves retrieval accuracy, and whether category-size weighting
# (which was tried on its own and decisively underperformed, see
# run_similarity_sampling_check.py) behaves differently once combined
# with dropping those thin categories. See
# run_similarity_category_threshold_check.py for the full grid (36
# configs: category_inclusion_threshold x weight_by_category_size x
# num_steps x images_per_category) and why this exists.
#
# --time=08:00:00 is the account's hard per-job cap, requested as a
# safety buffer, not an estimate -- the equivalent 18-config sampling
# check finished in ~30 minutes once the DeepFashion data gap was
# fixed; this grid is double that (36 configs), so expect roughly an
# hour, still comfortably within the cap.
#
# Same one-time setup as the other similarity SLURM scripts (repo
# already cloned, venv already built, real DeepFashion data already on
# this machine, confirmed complete -- see the missing-file fix for
# Striped_A-line_Dress) -- nothing new needed if that's already been
# done once.
#
# Submit:  sbatch scripts/run_similarity_category_threshold_check.slurm
# Monitor: squeue -u $USER
# If it times out: just run the submit command again -- same script,
#          same results file, already-completed configs are skipped.
# Output:  reports/similarity_category_threshold_check_<jobid>.out/.err
#          reports/similarity_category_threshold_check.csv (its own
#            file, never overwrites any earlier check's results)

#SBATCH --job-name=stylematch-similarity-category-threshold
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/similarity_category_threshold_check_%j.out
#SBATCH --error=reports/similarity_category_threshold_check_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Similarity category threshold check starting: $(date)"
time python3 scripts/run_similarity_category_threshold_check.py
echo "Similarity category threshold check finished: $(date)"
