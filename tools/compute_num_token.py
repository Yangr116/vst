from datasets import load_dataset
from tqdm import tqdm
import os
from glob import glob
import ujson as json
import multiprocessing
from PIL import Image
from io import BytesIO
import numpy as np
from functools import partial
import yaml
import copy
from typing import Dict, Any, List
import torch
from transformers import Qwen2VLProcessor
from qwen_vl_utils import smart_resize, process_vision_info
import concurrent.futures
import pyarrow.parquet as pq
import pyarrow as pa
from collections import defaultdict
from vst.preprocess import SampleTransform
from vst.chat_template import Qwen2_5VLChatTemplate
import gc 


MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 4096 * 28 * 28
SIZE_FACTOR = 28
IMAGE_TOKEN_ID=151655

SYS_PROMPT = """You are a helpful assistant."""

# Cache to store image token counts by dimensions
image_token_cache = {}


def get_token_num(sample: Dict[str, Any], processor: SampleTransform):
    tokenized_example = processor(sample)[0]
    length = len(tokenized_example['input_ids'])
    return length

def process_sample(sample: Dict[str, Any], processor: SampleTransform) -> int:
    """Calculate the total number of tokens in a sample, including image tokens."""
    try:
        return get_token_num(sample, processor)
    except Exception as e:
        print(f"Error: {e}")
        return 0

def position_id_func(*args, **kwargs):
    return {"position_ids": None}


def process_parquet_file(parquet_file, processor_path, batch_size=32, num_threads=8):
    """
    Processes a single Parquet file in a memory-efficient way.
    Reads the file in batches, processes them, and releases memory.
    """
    # 1. 每个工作进程只加载一次模型和处理器
    qwen_processor = Qwen2VLProcessor.from_pretrained(processor_path)
    template = Qwen2_5VLChatTemplate(qwen_processor.tokenizer)
    processor = SampleTransform(
        qwen_processor,
        template,
        position_id_func,
        max_seq_len=16384,
        fps=2
    )
    total_tokens = 0
    try:
        # 2. 使用 PyArrow 的 ParquetFile 对象，为流式读取做准备
        parquet_reader = pq.ParquetFile(parquet_file)
        num_row_groups = parquet_reader.num_row_groups
        with tqdm(total=parquet_reader.metadata.num_rows, desc=f"Processing {os.path.basename(parquet_file)}", leave=False) as pbar:
            # 3. 使用 iter_batches() 进行流式读取，显著降低内存占用
            # 这里的 batch_size 是 PyArrow 读取的行数，可以和你逻辑上的 batch_size 保持一致
            for record_batch in parquet_reader.iter_batches(batch_size=batch_size):
                # 将 PyArrow RecordBatch 转换为 Python 字典列表
                # to_pylist() 是高效的转换方式
                samples = record_batch.to_pylist()
                batch_tokens = 0
                # 4. 仍然使用线程池来加速单个批次的处理
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    # map 方法比 submit+as_completed 更简洁，且能保持顺序
                    results = executor.map(process_sample, samples, [processor] * len(samples))
                    for tokens in results:
                        batch_tokens += tokens
                        pbar.update(1)
                total_tokens += batch_tokens
                # 5. 在每个批次处理后，显式删除中间变量，并提示GC
                del samples
                del record_batch
                gc.collect()
    except Exception as e:
        print(f"Failed to process file {parquet_file}: {e}")
        return 0 # 返回0表示这个文件处理失败
    finally:
        # 6. 确保所有大型对象都被清理
        del qwen_processor, template, processor
        if 'parquet_reader' in locals():
            del parquet_reader
        gc.collect()
    return total_tokens

def process_data(parquet_list, processor_path, max_workers=None, threads_per_worker=4, batch_size=32):
    # Use ProcessPoolExecutor for parallel processing across files
    total_tokens = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_parquet_file, parquet, processor_path, batch_size, threads_per_worker): parquet 
                  for parquet in parquet_list}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                          desc="Processing parquet files"):
            parquet = futures[future]
            tokens = future.result()
            total_tokens += tokens
    
    return total_tokens

def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str, help="yamlfile")
    parser.add_argument('-p', '--processor_path', type=str, default="Qwen2.5-VL-3B-Instruct")
    parser.add_argument('-w', '--workers', type=int, default=None, help="Number of worker processes")
    parser.add_argument('-t', '--threads', type=int, default=4, help="Number of threads per worker process")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size for processing")
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    
    processor_path = args.processor_path
    data_paths = yaml.safe_load(open(args.train_path, 'r'))
    new_data_info = []

    debug = args.debug
    
    # Set default number of workers if not specified
    num_workers = args.workers or min(os.cpu_count(), 16)
    
    for data_info in data_paths:
        data_dir = os.path.join(data_info['data_dir'], data_info['ann_path'])
        parquet_list = glob(data_dir + "/*.parquet")
        
        if debug:
            qwen_processor = Qwen2VLProcessor.from_pretrained(processor_path)
            template = Qwen2_5VLChatTemplate(qwen_processor.tokenizer)
            processor = SampleTransform(
                qwen_processor,
                template,
                position_id_func,
                max_seq_len=16384,
                fps=2
            )
            table = pq.read_table(parquet_list[0])
            df = table.to_pandas()
            total_samples = len(df)
            batch_size = 32
            batches = [df.iloc[i:i+batch_size].to_dict('records') for i in range(0, total_samples, batch_size)]
            for batch in batches:
                for x in batch:
                    try:
                        results = get_token_num(x, processor)
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()
            import pdb; pdb.set_trace()
            import sys
            sys.exit()

        print(f"Processing {len(parquet_list)} parquet files from {data_dir}")
        total_length = process_data(
            parquet_list, 
            processor_path, 
            max_workers=num_workers,
            threads_per_worker=args.threads,
            batch_size=args.batch_size
        )
        
        print(f"Token number of {data_dir}: {total_length}")
        _data_info = copy.deepcopy(data_info)
        _data_info['token_num'] = int(total_length)
        new_data_info.append(_data_info)
    
    with open(args.train_path, 'w') as file:
        yaml.dump(new_data_info, file, default_flow_style=False, allow_unicode=True)
