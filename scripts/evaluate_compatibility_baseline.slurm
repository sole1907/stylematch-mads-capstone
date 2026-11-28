#!/bin/bash
# Zero-shot baseline evaluation for the compatibility engine (raw
# backbone, no fine-tuning). See scripts/evaluate_compatibility_baseline.py
# for what it does and why.
#
# Submit:  sbatch scripts/evaluate_compatibility_baseline.slurm
# Monitor: squeue -u $USER
# Output:  reports/compatibility_baseline_evaluation_<jobid>.out/.err

#SBATCH --job-name=stylematch-compatibility-baseline-eval
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/compatibility_baseline_evaluation_%j.out
#SBATCH --error=reports/compatibility_baseline_evaluation_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Compatibility zero-shot baseline evaluation starting: $(date)"
time python3 scripts/evaluate_compatibility_baseline.py
echo "Compatibility zero-shot baseline evaluation finished: $(date)"
