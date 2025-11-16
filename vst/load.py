# Copyright 2025 [Visual Spatial Tuning] Authors

import os
from typing import Dict, Union, Optional, Callable, Any, Literal, List

import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from transformers import AutoConfig, PreTrainedModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig, Qwen2VLVisionConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from transformers.modeling_utils import no_init_weights

from veomni.models.transformers.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from veomni.models.transformers.qwen2_5vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from veomni.models.module_utils import init_empty_weights, _load_state_dict, _dispatch_buffer, _init_parameter, _find_submodule
from veomni.utils.helper import empty_cache


def _dispatch_parameter(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    dtensor_factory: Optional[Callable[[torch.Tensor, Any, Any], torch.Tensor]] = None,
) -> bool:
    """
    Assigns parameter to an empty model.
    
    Args:
        module: The module to assign the parameter to
        name: Parameter name
        tensor: Parameter tensor
        dtensor_factory: Factory function for distributed tensors
        
    Returns:
        bool: True if parameter was successfully loaded, False otherwise
        
    NOTE: FSDP module must use in-place operators.
    """
    try:
        module, name = _find_submodule(module, name)
        orig_tensor = module._parameters[name].data
        tensor = tensor.to(orig_tensor)
        
        if orig_tensor.shape != tensor.shape:
            logger.warning(f"{name} size mismatch {orig_tensor.shape} vs {tensor.shape}")
            return False
            
        if hasattr(orig_tensor, "device_mesh"):  # dtensor
            if orig_tensor.device.type == "cpu":
                raise ValueError("Cannot load dtensor on CPU.")

            device_mesh = getattr(orig_tensor, "device_mesh")
            placements = getattr(orig_tensor, "placements")
            module._parameters[name].data.copy_(dtensor_factory(tensor, device_mesh, placements))
        else:  # not dtensor
            module._parameters[name].data.copy_(tensor)
        return True
    except Exception as e:
        logger.error(f"Error dispatching parameter {name}: {str(e)}")
        return False


@torch.no_grad()
def load_model_weights(
    model: Union[nn.Module, PreTrainedModel],
    weights_path: str,
    init_device: Literal["cpu", "cuda"] = "cuda",
    dtensor_factory: Optional[Callable[[torch.Tensor, Any, Any], torch.Tensor]] = None,
    remove_start: Optional[str] = None,
) -> List[str]:
    """
    Loads pre-trained model states in transformers' format.
    
    Args:
        model: The model to load weights into
        weights_path: Path to the weights file or directory
        init_device: Device to initialize empty weights on
        dtensor_factory: Factory function for distributed tensors
        
    Returns:
        List[str]: List of missing parameter names that couldn't be loaded
    """
    # Store buffers and parameter names
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names = set(name for name, _ in model.named_parameters())
    
    # Move model to empty device
    model.to_empty(device=init_device)
    
    # Load state dict shards
    state_dict_iterators = _load_state_dict(weights_path)
    for state_dict_iterator in tqdm(
        state_dict_iterators, 
        desc="Loading checkpoint shards", 
        disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        for name, tensor in state_dict_iterator:
            if remove_start and remove_start in name:
                name = name[len(remove_start):]
            if name in buffer_dict:  # persistent buffers
                buffer_dict[name] = tensor.clone()
            elif name in parameter_names:
                is_loaded = _dispatch_parameter(model, name, tensor, dtensor_factory)
                if is_loaded:
                    parameter_names.remove(name)
            else:
                logger.warning(f"Unexpected key in state dict: {name}")

        # Cleanup to avoid memory leaks
        del state_dict_iterator
        empty_cache()

    # Load buffers
    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer)

    # Initialize missing parameters
    if parameter_names:
        logger.warning(f"Found {parameter_names} missing key(s) in state dict, initializing them.")
        for name in parameter_names:
            _init_parameter(model, name)
    
    return list(parameter_names)


def build_vst_custom_model(
    config_path: str,
    weights_path: Optional[str] = None,
    encoders: Dict[str, Any] = {},
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = "flash_attention_2",
    init_device: Literal["cpu", "cuda"] = "cuda",
):
    """
    Build a VLM3D model with configurations and weights.
    
    Args:
        config_path: Path to the LM config
        weights_path: Path to the model weights
        encoders: Configuration for vision encoders
        torch_dtype: Data type for model parameters
        attn_implementation: Attention implementation method
        init_device: Device to initialize model on
        
    Returns:
        Qwen2VLForConditionalGeneration or Qwen2_5_VLForConditionalGeneration: The constructed model
    """
    encoders = encoders.get("image", {})
    
    # Load language model config
    lm_config = AutoConfig.from_pretrained(config_path)
    vlm3d_config = lm_config.to_dict()
    
    # Remove unnecessary config keys
    for key in ["model_type", "architectures", "_name_or_path"]:
        vlm3d_config.pop(key, None)

    # Configure vision encoder
    is_qwen2_5 = False
    if encoders and "model_path" in encoders:
        vision_config_path = encoders['model_path']
        if '2_5' in vision_config_path.lower():
            is_qwen2_5 = True
            vision_config = Qwen2_5_VLVisionConfig.from_pretrained(vision_config_path)
            vision_config.out_hidden_size = vlm3d_config.get("hidden_size")
        else:
            vision_config = Qwen2VLVisionConfig.from_pretrained(vision_config_path)
            vision_config.hidden_size = vlm3d_config.get("hidden_size")
        vision_config.initializer_range = vlm3d_config.get("initializer_range", 0.02)        
        vision_config_dict = vision_config.to_dict()
        for key in ["model_type", "_name_or_path"]:
            vision_config_dict.pop(key, None)
        vlm3d_config["vision_config"] = vision_config_dict
    
    # Configure RoPE scaling based on model size
    head_dim = vlm3d_config['hidden_size'] / vlm3d_config['num_attention_heads']
    if head_dim == 128:  # 2B, 7B, 72B
        vlm3d_config['rope_scaling'] = {"type": "mrope", "mrope_section": [16, 24, 24]}
    elif head_dim == 64:  # 0.5B
        vlm3d_config['rope_scaling'] = {"type": "mrope", "mrope_section": [8, 12, 12]}
    else:
        logger.warning(f"Unknown head dimension: {head_dim}, no ROPE scaling applied")

    # Create config and set vision token IDs
    if is_qwen2_5:
        vlm3d_config = Qwen2_5_VLConfig().from_dict(vlm3d_config)
        model_cls = Qwen2_5_VLForConditionalGeneration
    else:
        vlm3d_config = Qwen2VLConfig().from_dict(vlm3d_config)
        model_cls = Qwen2VLForConditionalGeneration
    
    token_ids = {
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "vision_token_id": 151654,
        "image_token_id": 151655,
        "video_token_id": 151656,
    }
    for key, value in token_ids.items():
        setattr(vlm3d_config, key, value)
    
    vlm3d_config.torch_dtype = torch_dtype
    # Create model with empty weights
    with init_empty_weights(), no_init_weights():
        vlm3d_model = model_cls._from_config(
            config=vlm3d_config,
            attn_implementation=attn_implementation,
            torch_dtype=getattr(torch, torch_dtype),)
        vlm3d_model.to(dtype=getattr(torch, torch_dtype))
    
    if vlm3d_model.image_token_id != vlm3d_model.config.image_token_id:
        logger.warning(f"overwrite the model.config.image_token_id to {vlm3d_model.image_token_id}")
    if vlm3d_model.video_token_id != vlm3d_model.config.video_token_id:
        logger.warning(f"overwrite the model.config.video_token_id to {vlm3d_model.video_token_id}")
    
    # Load model weights
    if weights_path is not None:
        logger.info(f"Loading language model weights from {weights_path}...")
        load_model_weights(vlm3d_model, weights_path, init_device)
    
    # Load encoder weights if provided
    if encoders and "model_path" in encoders:
        logger.info(f"Loading vision encoder weights from {encoders['model_path']}...")
        load_model_weights(vlm3d_model.visual, encoders["model_path"], init_device, remove_start='visual.')
    
    # Tie embeddings if needed
    if getattr(vlm3d_model.config, "tie_word_embeddings", True):
        logger.info("Tying word embeddings")
        input_embeddings = vlm3d_model.get_input_embeddings()
        output_embeddings = vlm3d_model.get_output_embeddings()
        output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]

    if os.getenv('DEBUG') == '1':
        # check weigtgs
        # check_model_weights(vlm3d_model, weights_path=weights_path)
        import pdb; pdb.set_trace()
    return vlm3d_model


@torch.no_grad()
def check_model_weights(
    model: Union[nn.Module, PreTrainedModel],
    weights_path: str,
    remove_start: Optional[str] = None,
) -> List[str]:
    """
    Loads pre-trained model states in transformers' format.
    
    Args:
        model: The model to load weights into
        weights_path: Path to the weights file or directory
        init_device: Device to initialize empty weights on
        dtensor_factory: Factory function for distributed tensors
        
    Returns:
        List[str]: List of missing parameter names that couldn't be loaded
    """
    # Store buffers and parameter names
    buffer_dict = {name: buffer.clone() for name, buffer in model.named_buffers()}
    parameter_names = set(name for name, _ in model.named_parameters())
    
    # Load state dict shards
    state_dict_iterators = _load_state_dict(weights_path)
    for state_dict_iterator in tqdm(
        state_dict_iterators, 
        desc="Loading checkpoint shards", 
        disable=int(os.getenv("LOCAL_RANK", "-1")) > 0
    ):
        for name, tensor in state_dict_iterator:
            if remove_start and remove_start in name:
                name = name[len(remove_start):]
            if name in buffer_dict:  # persistent buffers
                buffer_dict[name] = tensor.clone()
            elif name in parameter_names:
                # is_loaded = _dispatch_parameter(model, name, tensor)
                module, _name = _find_submodule(model, name)
                orig_tensor = module._parameters[_name].data
                if torch.equal(orig_tensor.cpu(), tensor.cpu()):
                    parameter_names.remove(name)
                    continue
                else:
                    logger.debug(f"{module} is not loaded.")
            else:
                logger.warning(f"Unexpected key in state dict: {name}")

        # Cleanup to avoid memory leaks
        del state_dict_iterator
        empty_cache()

    # Load buffers
    for name, buffer in buffer_dict.items():
        _dispatch_buffer(model, name, buffer)

    # Initialize missing parameters
    if parameter_names:
        logger.warning(f"Found {parameter_names} missing key(s) in state dict, initializing them.")
        for name in parameter_names:
            _init_parameter(model, name)
    
    return list(parameter_names)
