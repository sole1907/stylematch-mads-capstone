#!/bin/bash
# Precomputes the AFRIFASHION1600 catalog via the similarity engine.
# See scripts/build_afrifashion_similarity_catalog.py for what it does.
# Requires models/similarity_final.pt and data/raw/afrifashion1600 to
# already exist.
#
# Submit:  sbatch scripts/build_afrifashion_similarity_catalog.slurm
# Monitor: squeue -u $USER
# Output:  reports/afrifashion_similarity_catalog_build_<jobid>.out/.err
#          models/afrifashion_similarity_catalog.pt

#SBATCH --job-name=stylematch-afrifashion-similarity-catalog
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/afrifashion_similarity_catalog_build_%j.out
#SBATCH --error=reports/afrifashion_similarity_catalog_build_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "AFRIFASHION1600 similarity catalog build starting: $(date)"
time python3 scripts/build_afrifashion_similarity_catalog.py
echo "AFRIFASHION1600 similarity catalog build finished: $(date)"
