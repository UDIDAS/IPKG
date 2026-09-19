#!/bin/bash
#SBATCH --job-name=vkg-remaining
#SBATCH --account=<allocation>
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --gpus-per-node=4 --cpus-per-task=64 --mem=200g
#SBATCH --time=24:00:00
#SBATCH --output=/path/to/VKG_data/logs/pipeline_slurm_%j.log
# Fallback if the OnDemand allocation is lost: same idempotent driver on a fresh node.
bash "$(dirname "$0")/run_remaining_pipeline.sh"
