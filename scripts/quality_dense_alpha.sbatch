#!/bin/bash
#SBATCH --job-name=quality_dense
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/slurm-%A_%a.out
#SBATCH --array=0-5
#SBATCH --gpus-per-node=1
#SBATCH --comment="Quality analysis for dense alpha 5-10 sweep"

source /mnt/polished-lake/home/oli/spd/.venv/bin/activate
cd /mnt/polished-lake/home/oli/spd

RUN_IDS=(c-3fb03112 c-b5c3e9a6 c-50f78277 c-54c5c625 c-747f15e8 c-c3b03afb)
ITERS=(6351 5812 5371 4992 4685 4423)
ALPHAS=(5.0 6.0 7.0 8.0 9.0 10.0)

IDX=$SLURM_ARRAY_TASK_ID
python scripts/analyze_cluster_quality.py \
    --run_id "${RUN_IDS[$IDX]}" --iter "${ITERS[$IDX]}" \
    --alpha "${ALPHAS[$IDX]}" --decay 0.8 \
    --snapshot_path /mnt/polished-lake/artifacts/mechanisms/spd/clustering/harvests/ch-2b9414a4 \
    --output_dir /mnt/polished-lake/artifacts/mechanisms/spd/clustering/quality_analysis
