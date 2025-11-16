# Copyright 2025 [Visual Spatial Tuning] Authors
import base64
import math
from copy import deepcopy
from io import BytesIO
from typing import Union

from PIL import Image

from qwen_vl_utils import smart_resize
from vst.prompt import SYS_PROMPT
from vst.dataset_iterative import build_mapping_dataset
from vst.prerpocess_box3d import (
    convert_prefix_prompt_to_fov,
    quat_prompt,
    uvd_prompt,
    xyz_prompt
)


def encode_image_to_base64(image: Union[Image.Image, str], resized_height=None, resized_width=None):
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy()

    if resized_height is not None and resized_width is not None:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=28,
        )
        img = img.resize((resized_width, resized_height))

    output_buffer = BytesIO()
    img.save(output_buffer, format="PNG")
    byte_data = output_buffer.getvalue()

    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return base64_str


def convert_qa2vllm(sample):
    if isinstance(sample["image"], list):
        base64_image_list = [encode_image_to_base64(image, resized_height=sample["image_resized_height"], resized_width=sample["image_resized_width"]) for image in sample["image"]]
    else:
        base64_image_list = [encode_image_to_base64(sample["image"], resized_height=sample["image_resized_height"], resized_width=sample["image_resized_width"])]
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYS_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}} for base64_image in base64_image_list]
                + [{
                    "type": "text",
                    "text": sample["question"],
                }],
        }
    ]


class VllmValPreprocess():
    def __init__(self, data_args):
        self.enable_fov = data_args.enable_fov
        self.no_resize = data_args.no_resize
        self.base_width = data_args.base_width
        self.base_fov = 69.16
        self.base_focal = self.base_width / (2 * math.tan(math.radians(self.base_fov) / 2))
        self.enable_uvd = data_args.enable_uvd
        self.enable_quat = data_args.enable_quat
    
    def __call__(self, sample):
        item = deepcopy(sample)
        item['image_resized_height'] = None
        item['image_resized_width'] = None
        if self.enable_fov:
            new_prompt, new_size, camera_params = convert_prefix_prompt_to_fov(
                item['question'], 
                new_focal_length=self.base_focal, 
                no_resize=self.no_resize)
            if new_size is not None:
                item['question_original'] = deepcopy(item['question'])
                item['question'] = new_prompt
                item['image_resized_height'] = new_size[1]
                item['image_resized_width'] = new_size[0]
            else:
                item['image_resized_height'] = None
                item['image_resized_width'] = None
        if self.enable_uvd:
            if "question_original" not in item:
                item['question_original'] = deepcopy(item['question']) 
            item['question'] = item['question'].replace(xyz_prompt, uvd_prompt)
        if self.enable_quat:
            if "question_original" not in item:
                item['question_original'] = deepcopy(item['question']) 
            item['question'] = item['question'].replace(xyz_prompt, quat_prompt)
        messages = convert_qa2vllm(item)
        return (messages, item)
