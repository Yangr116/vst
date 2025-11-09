#!/bin/bash

set -x

# setup env
dir_path=$(pwd)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd $dir_path
pip install -r benchmark/libero/libero_requirements.txt
echo N | python -c "from libero.libero import benchmark"

sudo apt-get install libegl-dev -y

unset http_proxy
unset https_proxy


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

# model path
model_path=$1

task_suite_name=${2:-"libero_spatial"}
work_dir="$3/$task_suite_name"

PIDS=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "$IDX $task_suite_name $work_dir $CHUNKS"
    python vlm3d/vla/libero/run_libero_eval_mp.py \
        --model_family 'vst' \
        --pretrained_checkpoint $model_path \
        --task_suite_name $task_suite_name \
        --local_log_dir $work_dir \
        --total_processor $CHUNKS \
        --processor_id $IDX &
    PIDS+=($!)
done


for pid in "${PIDS[@]}"; do
    wait $pid
done


python benchmark/utils/merge_libero.py $work_dir > $work_dir/output_file.txt

cat $work_dir/output_file.txt
