#!/bin/bash
# Launch autonomous_infer_flare.py across all visible GPUs, WORKERS_PER_GPU processes each.
# Usage: bash run_autonomous_shards.sh [workers_per_gpu=2]   (logs -> $VKG_DATA/logs/)
set -e
WPG=${1:-2}
PY=${PY:-${CONDA_ENV:-$HOME/.conda/envs/medjepa}/bin/python}
export LD_LIBRARY_PATH=${CONDA_ENV:-$HOME/.conda/envs/medjepa}/lib:$LD_LIBRARY_PATH   # conda libstdc++ before the CUDA-SDK one (scipy ABI)
VKG_DATA=${VKG_DATA:-/path/to/VKG_data}
NGPU=$(nvidia-smi -L | wc -l); N=$((NGPU*WPG)); mkdir -p $VKG_DATA/logs
CPUS=$(( $(nproc) / N ))
cd "$(dirname "$0")"
for k in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$((k % NGPU)) OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS \
    nohup $PY autonomous_infer_flare.py --shard $k/$N > $VKG_DATA/logs/autonomous_shard$k.log 2>&1 &
  echo "shard $k/$N -> GPU $((k % NGPU)), $CPUS threads, pid $!"
done
