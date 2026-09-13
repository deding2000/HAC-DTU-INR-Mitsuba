#!/bin/bash
### INR with convex prior, Mitsuba-rendered in both stages, real
### intensity data, held-out asteroids 4-10 (no ground truth).
### Config: STAGE1_STEPS=80, STAGE2_STEPS=150.
### See inr_convex_prior_with_mitsuba.py.
### Run `uv sync` once from the repo root first.
### Submit from this directory: cd submission && bsub < submit_inr_convex_prior_mitsuba_stage2_v2_asteroids4to10.sh

#BSUB -q c47511
#BSUB -J inr_stage2_v2_ast4to10
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/inr_stage2_v2_ast4to10_%J.out
#BSUB -e logs/inr_stage2_v2_ast4to10_%J.err

set -euo pipefail
cd "${LS_SUBCWD:-$(dirname "$0")}"

module load cuda/13.3.0

export OPTIX_CACHE_MAXSIZE=0
export STAGE1_STEPS=80
export STAGE2_STEPS=150
export EVAL_EVERY=10

for ASTEROID_ID in 4 5 6 7 8 9 10; do
    echo "=================== ASTEROID_ID=${ASTEROID_ID} ==================="
    export ASTEROID_ID
    uv run python -u inr_convex_prior_with_mitsuba.py
done
