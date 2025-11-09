#!/bin/bash


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


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(($NNODES * $NPROC_PER_NODE))

model_path=$1
fps=${3:-4}
output_dir="$2/vsibench_fps$fps"


PIDS=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "$IDX"
    CUDA_VISIBLE_DEVICES=$IDX python eval_vsibench.py \
        --model_name_or_path $model_path \
        --total-processor $CHUNKS \
        --processor-id $IDX \
        --batch-size 4 \
        --fps $fps \
        --output-dir $output_dir &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
    wait $pid
done

python utils/merge_vsibench.py --result-dir $output_dir
