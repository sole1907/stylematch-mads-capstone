#!/bin/bash
# Official, one-time test-set evaluation for the trained compatibility
# engine. See scripts/evaluate_compatibility_final.py for what it does.
# Requires models/compatibility_final.pt to already exist.
#
# Submit:  sbatch scripts/evaluate_compatibility_final.slurm
# Monitor: squeue -u $USER
# Output:  reports/compatibility_final_evaluation_<jobid>.out/.err

#SBATCH --job-name=stylematch-compatibility-final-eval
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=reports/compatibility_final_evaluation_%j.out
#SBATCH --error=reports/compatibility_final_evaluation_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Official test-set evaluation starting: $(date)"
time python3 scripts/evaluate_compatibility_final.py
echo "Official test-set evaluation finished: $(date)"
