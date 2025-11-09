#!/bin/bash
set -x

model_path=$1
modified_string="${model_path//[:\/]/-}"
datetime=$(date +%Y-%m-%d_%H-%M-%S)
basename=$(basename "$model_path")
work_dir="evaluation-$basename-$datetime"
echo "$work_dir"


# vlmevalkit
work_dir="work_dirs/$work_dir"
mkdir -p $work_dir

python run.py --config config_qwen2_5.json --model_path "$model_path" --work-dir "$work_dir" --use-vllm 2>&1 | tee "work_dirs/$modified_string-log.txt"

python run.py --config config_mmsi.json --model_path "$model_path" --work-dir "$work_dir" --use-vllm 2>&1 | tee "work_dirs/$modified_string-log.txt"

# vsibench
fps=${2:-4}
output_dir="$work_dir/vsibench_fps$fps"
mkdir -p $output_dir
python eval_vsibench.py \
    --model_name_or_path $model_path \
    --batch-size 4 \
    --fps $fps \
    --output-dir $output_dir \
    --tp-size 8

python utils/merge_vsibench.py --result-dir $output_dir
