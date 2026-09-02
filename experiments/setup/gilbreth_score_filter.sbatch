#!/bin/bash
#SBATCH --account=yunglu
#SBATCH --qos=standby
#SBATCH --partition=a100-80gb,a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=mer-scorefilter
#SBATCH --output=/scratch/gilbreth/dcharapa/mer/run/logs/slurm-%x-%j.out

# gilbreth_score_filter.sbatch -- one scorer run with the score-consistency
# filter (decoupled_scorer.py --score_filter 50): the same strict eps=0.05
# sweep, shipped-mode per-track F1 and 10^3 per-piece bootstrap as the main
# strict_eps05/shipped artifacts, on predictions with every deletion claim
# naming no score note within 50 ms removed. Standby QOS (no lab GPU quota);
# the GPU is unused, the partition is the one the standby template allows.
# Usage: sbatch --export=ALL,NAME=A_polytune_maestro,GT=gt_meta_maestro.json \
#               gilbreth_score_filter.sbatch
set -euo pipefail
MER=/scratch/gilbreth/dcharapa/mer
RUN=$MER/run
PY=$MER/envs/polytune/bin/python
NAME=${NAME:?set NAME via --export}
GT=${GT:?set GT via --export}
OUT=$RUN/results/${NAME}_scorefilter50.json

module load conda/2025.09
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== $NAME on $(hostname) $(date) ==="
"$PY" "$MER/repo/experiments/decoupled_scorer.py" \
  --pred_dir "$RUN/preds/$NAME" --gt_meta "$RUN/$GT" \
  --tau 50 75 100 150 200 500 --eps 0.05 --collapse strict \
  --shipped --bootstrap 1000 --seed 20260718 --score_filter 50 \
  --out "$OUT"
echo "=== done $(date) ==="
