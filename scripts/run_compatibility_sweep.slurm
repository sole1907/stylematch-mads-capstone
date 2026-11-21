#!/bin/bash
# Submits sweep.py's full-scale staged hyperparameter search as a Great
# Lakes batch job (learning_rate x freeze_until_block jointly, then
# margin, then num_steps -- see sweep.py's __main__ for why staged this
# way). No eval cap: every config evaluates the full 5,000 FITB
# questions / 10,000 compatibility outfits.
#
# --time is capped at 08:00:00 because that's siads699f26_class's real
# per-job wall-time ceiling (confirmed via `sacctmgr show assoc`, not
# assumed) -- requesting more gets the job rejected outright with
# AssocMaxWallDurationPerJobLimit before it ever queues. A real timed
# check showed the full sweep won't finish in one 8-hour job either, so
# this relies on sweep.py's resume-from-CSV behavior across however
# many resubmissions it takes. If a job times out, just resubmit the
# same script again; already-completed configs are reused from
# reports/compatibility_staged_search_stage_*.csv, not redone.
#
# One-time setup BEFORE first use (venv/dependencies are OS-specific,
# the repo lives in /home so results persist reliably):
#   1. Get the repo onto Great Lakes: git clone <repo-url>, or git pull
#      if already cloned there. Clone it into /home (e.g.
#      /home/$USER/stylematch-mads-capstone), not /scratch.
#   2. No module load needed: the login/compute shells already
#      auto-activate a system conda "base" with a working python3
#      (confirmed: /sw/pkgs/arc/mamba/py3.12, Python 3.12.0).
#   3. python3 -m venv venv && source venv/bin/activate
#   4. pip install -r requirements.txt
#   5. Make sure the real Polyvore Outfits data is downloaded (see
#      scripts/download_polyvore.py) -- POLYVORE_DIR defaults to
#      data/raw/polyvore_outfits under the repo root with no .env
#      override, which is where this runs from by default.
#
# Submit:   sbatch scripts/run_compatibility_sweep.slurm
# Monitor:  squeue -u $USER
# If it times out: just run the submit command again -- same script,
#           same results files, already-completed configs are skipped.
# Output:  reports/compatibility_sweep_full_<jobid>.out (this job's stdout/stderr)
#          reports/compatibility_staged_search_stage_*.csv,
#            compatibility_staged_search_summary.csv (sweep.py's own
#            results, resumable if the job times out)

#SBATCH --job-name=stylematch-compatibility-sweep
#SBATCH --account=siads699f26_class
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=reports/compatibility_sweep_full_%j.out
#SBATCH --error=reports/compatibility_sweep_full_%j.err

set -euo pipefail

# No module load needed: Great Lakes' login/compute shells already
# auto-activate a system conda "base" env with a working python3
# (confirmed: /sw/pkgs/arc/mamba/py3.12, Python 3.12.0), used directly
# below to build this project's own venv on top of it.
source venv/bin/activate

# Confirms a GPU is actually visible before committing hours to the
# real sweep -- cheap sanity check, not optional.
python3 -c "import torch; assert torch.cuda.is_available(), 'no GPU visible to this job'; print('GPU OK:', torch.cuda.get_device_name(0))"

python3 -m stylematch.compatibility.sweep
