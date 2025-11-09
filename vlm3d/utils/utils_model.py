# Copyright 2025 [Visual Spatial Tuning] Authors
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from tabulate import tabulate
from veomni.utils import helper


logger = helper.create_logger(__name__)


options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel, FullyShardedDataParallel)


def print_model_params(model):
    """
    Print trainable parameters of a model with their shapes and memory usage.
    Works with regular models and distributed models like DDP, DataParallel, and FSDP.
    
    Args:
        model: PyTorch model
        
    Returns:
        str: Formatted table of trainable parameters
    """
    # Unwrap model from distributed wrappers
    while isinstance(model, options):
        model = model.module
    
    params_trainable = []
    total_params = 0
    trainable_params = 0
    
    # Analyze all parameters
    for n, p in model.named_parameters():
        num_params = p.ds_numel if hasattr(p, "ds_numel") else p.numel()
        total_params += num_params
        
        if p.requires_grad:
            params_trainable.append([n, tuple(p.shape), num_params])
            trainable_params += num_params
    
    # Sort by parameter count (largest first)
    params_trainable.sort(key=lambda x: x[2], reverse=True)
    
    # Format numbers for display
    for row in params_trainable:
        row[2] = f"{row[2]:,}"  # Format with commas
    
    # Create table
    params_trainable_table = tabulate(
        params_trainable, 
        ["Name", "Shape", "Parameters"], 
        tablefmt="pretty", 
        stralign="left"
    )
    
    # Calculate memory in MB and percentage of trainable params
    trainable_percent = (trainable_params / total_params * 100) if total_params > 0 else 0
    
    # Log summary and table
    logger.info_rank0(
        "\n"
        f"\033[93mTrainable parameters: (~{trainable_params/1e6:.2f}M) / (~{total_params/1e6:.2f}M)"
        f"[{trainable_percent:.2f}%]\n{params_trainable_table}\n\033[0m"
    )
    
    return params_trainable_table
