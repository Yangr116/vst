#!/bin/bash
set -x

export SEED_MODELS_LOGGING_LEVEL=WARN
export OMNISTORE_LOGGING_LEVEL=ERROR
export BPEX_NO_WARN_ON_UNTUNED_CASE=1
export TOKENIZERS_PARALLELISM=false
export VESCALE_SINGLE_DEVICE_RAND=0
export TF_CPP_MIN_LOG_LEVEL=2
export CUDA_LAUNCH_BLOCKING=1

NNODES=${NNODES:=$ARNOLD_WORKER_NUM}
NPROC_PER_NODE=${NPROC_PER_NODE:=$ARNOLD_WORKER_GPU}
NPROC_PER_NODE=${NPROC_PER_NODE:=$ARNOLD_WORKER_GPU_PER_NODE}
NODE_RANK=${NODE_RANK:=$ARNOLD_ID}
MASTER_ADDR=${MASTER_ADDR:=$ARNOLD_WORKER_0_HOST}
MASTER_ADDR=${MASTER_ADDR:=$ARNOLD_EXECUTOR_0_HOST}
MASTER_PORT=${MASTER_PORT:=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)}
MASTER_PORT=${MASTER_PORT:=$(echo "$ARNOLD_EXECUTOR_0_PORT" | cut -d "," -f 1)}

model_path=$1
datetime=$(date +%Y-%m-%d_%H-%M-%S)

longest=""
IFS='/' read -ra arr <<< "$model_path"
for part in "${arr[@]}"; do
    if [[ ${#part} -gt ${#longest} ]]; then
        longest="$part"
    fi
done
basename=$longest
modified_string=$longest
work_dir="evaluation-cot-$basename-$datetime"
echo "$work_dir"

# vlmevalkit
work_dir="work_dirs/$work_dir"
mkdir -p $work_dir

torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT run.py --config config_vlm3dreasoner.json --model_path "$model_path" --work-dir "$work_dir" --use-vllm 2>&1 | tee "work_dirs/$modified_string-log.txt"
