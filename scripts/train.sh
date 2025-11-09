#!/bin/bash

set -x

NNODES=1
NPROC_PER_NODE=8
NODE_RANK=0
MASTER_PORT=8888


MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt
