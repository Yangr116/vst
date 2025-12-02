#!/bin/bash

# export CUDA_LAUNCH_BLOCKING=1

set -x

export DEBUG='1'
torchrun --nnodes 1 --nproc-per-node 1 --node-rank 0 --master-port=8888 $@
