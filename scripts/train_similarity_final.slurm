#!/bin/bash
# Final production training for the similarity engine, using the real
# winning config from the full-scale hyperparameter search. See
# scripts/train_similarity_final.py for the config and why it's final.
#
# Single run, num_steps=100 -- much smaller than the margin/num_steps
# follow-up sweep (10 configs, up to 500 steps each) that ran in ~3
# hours, so this should finish well within the hour requested.
#
# Same one-time setup as the other similarity SLURM scripts (repo
# already cloned, venv already built, real DeepFashion data already on
# this machine) -- nothing new needed if that's already been done once.
#
# Submit:  sbatch scripts/train_similarity_final.slurm
# Monitor: squeue -u $USER
# Output:  reports/similarity_final_training_<jobid>.out/.err
#          models/similarity_final.pt (the saved checkpoint)

#SBATCH --job-name=stylematch-similarity-final-training
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=reports/similarity_final_training_%j.out
#SBATCH --error=reports/similarity_final_training_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "Final similarity training starting: $(date)"
time python3 scripts/train_similarity_final.py
echo "Final similarity training finished: $(date)"
