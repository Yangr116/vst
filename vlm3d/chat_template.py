# Copyright 2025 [Visual Spatial Tuning] Authors
# This file may have been modified by [Visual Spatial Tuning] Authors
# source file: https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/data/multimodal/multimodal_chat_template.py


from abc import ABC, abstractmethod
import torch
from typing import Sequence, Dict, List
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizer
from veomni.data.multimodal.multimodal_chat_template import MultimodalChatTemplate
from veomni.data.constants import IGNORE_INDEX, TYPE2INDEX


class Qwen2_5VLTemplate(MultimodalChatTemplate):
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs) -> None:
        super().__init__(tokenizer)
        self.image_pad = "<|image_pad|>"
        self.video_pad = "<|video_pad|>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_pad)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_pad)
        self.image_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.eos = self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
        logger.warning("Qwen2_5VLTemplate will not truncate sequence when longer than [max_seq_lens].")

    def image_pattern(self, token_num):
        if self.template_type == "conversation":
            return "<|vision_start|>" + self.image_pad * token_num + "<|vision_end|>"
        else:
            raise ValueError(f"Unknown template type: {self.template_type}")

    def video_pattern(self, token_num):
        if self.template_type == "conversation":
            return "<|vision_start|>" + self.video_pad * token_num + "<|vision_end|>"
        else:
            raise ValueError(f"Unknown template type: {self.template_type}")


    @abstractmethod
    def encode_messages(self, messages: Sequence[Dict[str, str]]) -> Dict[str, List[int]]:
        pass


class Qwen2_5VLChatTemplate(Qwen2_5VLTemplate):
    template_type = "conversation"
    system_prompt = "You are a helpful assistant."

    def _set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def _get_system_mesage(self):
        system_message = {
            "role": "system",
            "content": self.system_prompt,
            "loss_mask": 0,
        }
        return system_message

    def encode_messages(
        self,
        conversations: Sequence[Dict[str, str]],
        mm_num_tokens: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        image_index = 0
        video_index = 0
        messages = []
        image_token_num_list = mm_num_tokens.pop("image", [])
        video_token_num_list = mm_num_tokens.pop("video", [])
        assert len(mm_num_tokens) == 0
        # messages.append(self._get_system_mesage())
        for message in conversations:
            role = message[0]
            content = ""
            for value in message[1:]:
                if value[0] == "text":
                    content += value[1]
                elif value[0] == "video":
                    content += self.video_pattern(video_token_num_list[video_index])
                    video_index += 1
                else:
                    assert value[0] == "image"
                    content += self.image_pattern(image_token_num_list[image_index])
                    image_index += 1

            messages.append(
                {
                    "role": role,
                    "content": content,
                    "loss_mask": 1 if role == "assistant" else 0,
                }
            )

        input_ids, attention_mask, labels = [], [], []
        for message in messages:
            if message["content"] == "":  # eval
                content_str = "<|im_start|>" + message["role"] + "\n"
            else:
                content_str = "<|im_start|>" + message["role"] + "\n" + message["content"].strip() + "<|im_end|>\n"
            content_ids = self.tokenizer.encode(content_str, add_special_tokens=False)
            input_ids += content_ids
            attention_mask += [1] * len(content_ids)
            if message["loss_mask"] == 1:
                labels += content_ids
            else:
                labels += [IGNORE_INDEX] * len(content_ids)

        tokenized_example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}

        # change to veomni image token id
        image_mask = tokenized_example["input_ids"] == self.image_token_id
        input_mask = tokenized_example["labels"] == IGNORE_INDEX
        input_image_mask = image_mask & input_mask
        # output_image_mask = image_mask & ~input_mask
        tokenized_example["input_ids"][input_image_mask] = TYPE2INDEX["input"]["image"]
        # tokenized_example["input_ids"][output_image_mask] = TYPE2INDEX["output"]["image"]

        video_mask = tokenized_example["input_ids"] == self.video_token_id
        tokenized_example["input_ids"][video_mask] = TYPE2INDEX["input"]["video"]

        return tokenized_example
