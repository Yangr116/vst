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


NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_PORT=8888

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=$(($NNODES * $NPROC_PER_NODE))

# model path
model_path=$1

# model_path=/mnt/hdfs/rayyang/work_dirs/spatial/20250824_vla_qwen2_5vl_3b_spatial_sft_libero_packing4096_3/checkpoints/global_step_2700/hf_ckpt
task_suite_name=${2:-"libero_spatial"}
work_dir="$3/$task_suite_name"

PIDS=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    echo "$IDX $task_suite_name $work_dir $CHUNKS"
    python vlm3d/vla/libero/run_libero_eval_mp.py \
        --model_family 'vst' \
        --pretrained_checkpoint $model_path \
        --task_suite_name $task_suite_name \
        --norm_states prepare_data/vla/libero/norm_stats.json \
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
