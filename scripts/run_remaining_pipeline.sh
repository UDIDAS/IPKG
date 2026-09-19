#!/bin/bash
# Unattended driver for the remaining TBD experiments. Idempotent: every stage checks for its own output
# and is skipped when present, so the script can be re-launched after any interruption (session drop,
# node loss) and only does what is missing.  Run DETACHED from the interactive session:
#     setsid nohup bash scripts/run_remaining_pipeline.sh > $VKG_DATA/logs/pipeline.log 2>&1 < /dev/null &
# or as a Slurm job: sbatch scripts/run_remaining_pipeline.sbatch
# Stages: [1] FLARE23 semi-oracle predicted masks (GPU, all visible GPUs, 2 workers/GPU)
#         [2] assemble corpora/corpus_predicted_flare23.json   [3] Tab. 6 FLARE23 rows
#         [4] Tab. 7/8 baselines incl. the FLARE23 sub-pool (GPU 0 for the ResNet50 embedding)
#         [5] LLM-judge study with Qwen3-32B (GPUs 0-1) if not already done
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export VKG_DATA=${VKG_DATA:-/path/to/VKG_data}
export HF_HOME=${HF_HOME:-/path/to/hf_cache}
export LD_LIBRARY_PATH=${CONDA_ENV:-$HOME/.conda/envs/medjepa}/lib:${LD_LIBRARY_PATH:-}
PY=${CONDA_ENV:-$HOME/.conda/envs/medjepa}/bin/python
LOGS=$VKG_DATA/logs; mkdir -p $LOGS $VKG_DATA/flare23_predicted
log(){ echo "[$(date '+%F %T')] $*"; }
# single instance: a second launch (e.g. after a reconnect) exits instead of duplicating running stages
exec 9>$LOGS/pipeline.lock; flock -n 9 || { log "another driver instance holds $LOGS/pipeline.lock — exiting"; exit 0; }
# never start on a partial case list: wait for the FLARE23 pool fetch to finish if it is still running
while pgrep -f "python[^ ]* [^ ]*fetch_zip_members.py" >/dev/null; do log "waiting for the FLARE23 pool fetch to finish"; sleep 60; done
NGPU=$(nvidia-smi -L | wc -l); WPG=${WPG:-2}; N=$((NGPU*WPG)); CPUS=$(( $(nproc) / N ))

# ---------- [1] FLARE23 semi-oracle inference ----------
want=$(ls $VKG_DATA/flare23_pool/labels/*.nii.gz 2>/dev/null | wc -l)
have=$(ls $VKG_DATA/flare23_predicted/*.json 2>/dev/null | wc -l)
if pgrep -f "python[^ ]* [^ ]*build_predicted_corpus_flare23.py infer" >/dev/null; then
  log "stage 1: shards already running elsewhere — waiting for them instead of relaunching"
  while pgrep -f "python[^ ]* [^ ]*build_predicted_corpus_flare23.py infer" >/dev/null; do sleep 60; done
  have=$(ls $VKG_DATA/flare23_predicted/*.json 2>/dev/null | wc -l)
fi
if [ "$have" -lt "$want" ]; then
  log "stage 1: FLARE23 inference — $have/$want done; launching $N shards on $NGPU GPUs"
  cd $REPO/scripts/kg
  for k in $(seq 0 $((N-1))); do
    CUDA_VISIBLE_DEVICES=$((k % NGPU)) OMP_NUM_THREADS=$CPUS MKL_NUM_THREADS=$CPUS \
      $PY build_predicted_corpus_flare23.py infer --shard $k/$N > $LOGS/flare23_pred_shard$k.log 2>&1 &
  done
  wait
  have=$(ls $VKG_DATA/flare23_predicted/*.json 2>/dev/null | wc -l)
  log "stage 1 finished: $have/$want cases"
  grep -l Traceback $LOGS/flare23_pred_shard*.log 2>/dev/null && log "WARNING: tracebacks in shard logs (re-run the driver to retry missing cases)"
else
  log "stage 1: already complete ($have/$want)"
fi

# ---------- [2] assemble corpus ----------
if [ ! -f $REPO/corpora/corpus_predicted_flare23.json ] || [ "$(ls $VKG_DATA/flare23_predicted/*.json | wc -l)" -gt "$($PY -c "import json;print(json.load(open('$REPO/corpora/corpus_predicted_flare23.json'))['n'])" 2>/dev/null || echo 0)" ]; then
  log "stage 2: assembling corpus_predicted_flare23.json"
  (cd $REPO/scripts/kg && $PY build_predicted_corpus_flare23.py assemble) 2>&1 | grep -vE "Warning|warn"
else log "stage 2: corpus present"; fi

# ---------- [3] Tab. 6 FLARE23 rows ----------
log "stage 3: query suite on FLARE23"
(cd $REPO/queries && $PY query_suite_flare23.py) 2>&1 | grep -vE "Warning|warn"

# ---------- [4] Tab. 7/8 baselines with the FLARE23 sub-pool ----------
if $PY -c "import json,sys; d=json.load(open('$REPO/results/retrieval/baselines_tab7_tab8.json')); sys.exit(0 if 'tab8_subpool' in d else 1)" 2>/dev/null; then
  log "stage 4: baselines (tab8 sub-pool) present"
else
  log "stage 4: baselines Tab. 7 + Tab. 8 sub-pool"
  (cd $REPO && CUDA_VISIBLE_DEVICES=0 $PY scripts/retrieval/baselines_tab7_tab8.py --workers 24) 2>&1 | grep -vE "Warning|warn"
fi

# ---------- [5] LLM judge ----------
while pgrep -f "python[^ ]* [^ ]*llm_expert_local.py" >/dev/null; do log "stage 5: a judge run is already in progress — waiting"; sleep 60; done
if [ -f $REPO/results/llm_expert/llm_expert_study_qwen3_32b.json ]; then log "stage 5: judge study present"
else
  log "stage 5: LLM judge (Qwen3-32B)"
  (cd $REPO && CUDA_VISIBLE_DEVICES=0,1 $PY scripts/llm_expert/llm_expert_local.py Qwen/Qwen3-32B qwen3_32b) 2>&1 | grep -vE "Warning|warn|it/s\]"
fi
log "ALL STAGES DONE"; touch $LOGS/pipeline.DONE
