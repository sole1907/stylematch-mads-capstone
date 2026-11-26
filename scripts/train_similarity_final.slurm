#!/bin/bash
# Final production training for the similarity engine. See
# scripts/train_similarity_final.py for the config and num_steps.
#
# Submit:  sbatch scripts/train_similarity_final.slurm
# Monitor: squeue -u $USER
# If it times out: resubmit the same command, it resumes from the
#          last saved checkpoint rather than starting over.
# Output:  reports/similarity_final_training_<jobid>.out/.err
#          models/similarity_final.pt (final checkpoint, written once
#              training actually completes)

#SBATCH --job-name=stylematch-similarity-final-training
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/similarity_final_training_%j.out
#SBATCH --error=reports/similarity_final_training_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Final similarity training starting: $(date)"
time python3 scripts/train_similarity_final.py
echo "Final similarity training finished: $(date)"
