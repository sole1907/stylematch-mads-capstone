#!/bin/bash
# Official, one-time test-set evaluation for the trained similarity
# engine. See scripts/evaluate_similarity_final.py for what it does.
# Requires models/similarity_final.pt to already exist.
#
# Submit:  sbatch scripts/evaluate_similarity_final.slurm
# Monitor: squeue -u $USER
# Output:  reports/similarity_final_evaluation_<jobid>.out/.err

#SBATCH --job-name=stylematch-similarity-final-eval
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=reports/similarity_final_evaluation_%j.out
#SBATCH --error=reports/similarity_final_evaluation_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Official test-set evaluation starting: $(date)"
time python3 scripts/evaluate_similarity_final.py
echo "Official test-set evaluation finished: $(date)"
