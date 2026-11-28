#!/bin/bash
# Precomputes the Polyvore-via-SimilarityModel catalog. See
# scripts/build_polyvore_similarity_catalog.py for what it does.
# Requires models/similarity_final.pt to already exist.
#
# Submit:  sbatch scripts/build_polyvore_similarity_catalog.slurm
# Monitor: squeue -u $USER
# Output:  reports/polyvore_similarity_catalog_build_<jobid>.out/.err
#          models/polyvore_similarity_catalog.pt

#SBATCH --job-name=stylematch-polyvore-similarity-catalog
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/polyvore_similarity_catalog_build_%j.out
#SBATCH --error=reports/polyvore_similarity_catalog_build_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Polyvore similarity catalog build starting: $(date)"
time python3 scripts/build_polyvore_similarity_catalog.py
echo "Polyvore similarity catalog build finished: $(date)"
