#!/bin/bash
# Hyperparameter iteration #3: further follow-up (27 configs,
# num_steps=[2000, 3000, 4000] x margin=[0.005, 0.01, 0.02, 0.05, 0.1,
# 0.15, 0.2, 0.25, 0.3]), see
# scripts/run_compatibility_num_steps_iteration3.py for why this
# exists (iteration #2's winner, margin=0.02/num_steps=2000, had both
# values sitting at the edge of what was tested there).
#
# --time=08:00:00 is the account's hard per-job cap, not an estimate
# this fits in: at ~31-46 min/config across 27 configs (training time
# scales with num_steps, confirmed via the real per-step log from
# iteration #2's num_steps=2000 run), total is roughly 17 hours.
# Expect 2-3 resubmissions; already-completed configs are skipped via
# run_sweep's resume-from-CSV, reading back
# reports/compatibility_num_steps_iteration3.csv.
#
# Same one-time setup as run_compatibility_sweep.slurm (repo already
# cloned, venv already built, real data already on this machine) --
# nothing new needed if that's already been done once.
#
# Submit:  sbatch scripts/run_compatibility_num_steps_iteration3.slurm
# Monitor: squeue -u $USER
# If it times out: just run the submit command again -- same script,
#          same results file, already-completed configs are skipped.
# Output:  reports/compatibility_num_steps_iteration3_<jobid>.out/.err
#          reports/compatibility_num_steps_iteration3.csv (its own
#            file, never overwrites iteration #2's results)

#SBATCH --job-name=stylematch-compat-num-steps-iter3
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/compatibility_num_steps_iteration3_%j.out
#SBATCH --error=reports/compatibility_num_steps_iteration3_%j.err

set -euo pipefail

source venv/bin/activate

python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

echo "num_steps iteration 3 starting: $(date)"
time python3 scripts/run_compatibility_num_steps_iteration3.py
echo "num_steps iteration 3 finished: $(date)"
