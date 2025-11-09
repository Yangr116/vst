#!/bin/bash

source setup.sh

model_path=$1
model_path=$1
work_dir=${2:-results_libero}

mkdir -p $work_dir

longest=""
IFS='/' read -ra arr <<< "$model_path"
for part in "${arr[@]}"; do
    if [[ ${#part} -gt ${#longest} ]]; then
        longest="$part"
    fi
done
basename=$longest
modified_string=$longest
echo "$modified_string"
if [[ "$model_path" == *"hdfs"* && "$model_path" != /* ]]; then
    hdfs dfs get $model_path "$modified_string"
    model_path=$modified_string
fi


bash benchmark/libero/auto_run_vst_vla_libero.sh \
    $model_path \
    libero_spatial \
    $work_dir

bash benchmark/libero/auto_run_vst_vla_libero.sh \
    $model_path \
    libero_object \
    $work_dir

bash benchmark/libero/auto_run_vst_vla_libero.sh \
    $model_path \
    libero_goal \
    $work_dir

bash benchmark/libero/auto_run_vst_vla_libero.sh \
    $model_path \
    libero_10 \
    $work_dir


# copy the results into hdfs path
if [[ "$1" == *"hdfs"* ]]; then
    hdfs dfs put $work_dir $1
fi
