#!/bin/bash
# Re-tests category_inclusion_threshold x images_per_category x
# num_steps under real full-coverage epochs + shared train/val
# category set. See run_similarity_full_coverage_check.py for the
# grid (45 configs) and why this exists.
#
# --time=08:00:00 is the account's hard per-job cap -- expect this to
# need multiple resubmissions given 45 configs up to num_steps=2000.
#
# Submit:  sbatch scripts/run_similarity_full_coverage_check.slurm
# Monitor: squeue -u $USER
# If it times out: just resubmit the same command -- already-completed
#          configs are skipped via the existing resume-from-CSV logic.
# Output:  reports/similarity_full_coverage_check_<jobid>.out/.err
#          reports/similarity_full_coverage_check.csv

#SBATCH --job-name=stylematch-similarity-full-coverage
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/similarity_full_coverage_check_%j.out
#SBATCH --error=reports/similarity_full_coverage_check_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Full coverage check starting: $(date)"
time python3 scripts/run_similarity_full_coverage_check.py
echo "Full coverage check finished: $(date)"
