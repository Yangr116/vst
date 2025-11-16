# Copyright 2025 [Visual Spatial Tuning] Authors

import os
import yaml
import glob
import random
from collections import defaultdict
from loguru import logger


def sample_data(file):
    data_paths = yaml.safe_load(open(file, 'r'))
    results = []
    for sample in data_paths:
        parquet_files = glob.glob(os.path.join(sample['data_dir'], sample['ann_path'], "*.parquet"))
        assert len(parquet_files) != 0, f"there isn't parquet files in {os.path.join(sample['data_dir'], sample['ann_path'])}"
        sampling_ratio = sample.get('sampling_ratio', '100%')
        sampling_num = max(int(float(sampling_ratio.strip('%')) / 100 * len(parquet_files)), 1)
        sampled_files = random.sample(parquet_files, k=sampling_num)
        if not isinstance(sampled_files, list):
            sampled_files = [sampled_files]
        results.extend(sampled_files)
        logger.info(f"sampling {sampling_num} {sampling_ratio} files from {os.path.join(sample['data_dir'], sample['ann_path'])}")
    return results
