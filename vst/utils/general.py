# Copyright 2025 [Visual Spatial Tuning] Authors

import os
import json
from loguru import logger


def save_json(data, save_path):
    logger.info(f"Saving {len(data)} samples to {save_path}")
    with open(save_path, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)


def get_local_rank():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    return local_rank


def get_global_rank():
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    return global_rank


def load_json(path):
    
    with open(path, 'r') as fp:
        data = json.load(fp)

    return data
