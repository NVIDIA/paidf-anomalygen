# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import typing
from datetime import datetime
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor


class Captioner:
    """A class to generate captions for images using Cosmos Reason models."""

    def __init__(
        self,
        prompt_data: typing.Dict[str, str],
        model_name="nvidia/Cosmos-Reason1-7B",
        temperature=0.01,
        max_tokens=4096,
        seed=42,
        num_gpus=1,
    ):
        """Initialize the Captioner with model and prompt configuration.

        Args:
            prompt_data: A dictionary containing 'system_prompt' and
                'user_prompt'.
            model_name: The name of the pre-trained model to use. Defaults to
                `"nvidia/Cosmos-Reason1-7B"`.
            temperature: Sampling temperature. Defaults to `0.01`.
            max_tokens: Maximum number of tokens to generate. Defaults to
                `4096`.
            seed: Random seed for reproducibility. Defaults to `42`.
            num_gpus: Number of GPUs to shard the model across. Values `> 1`
                use accelerate's `device_map="auto"`. Defaults to `1`.
        """
        self.system_prompt = prompt_data["system_prompt"]
        self.user_prompt = prompt_data["user_prompt"]
        self.processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(model_name)
        # Batched generation on a decoder-only model requires left padding;
        # with the default right padding the pad tokens land between the prompt
        # and the continuation and the shorter samples decode to garbage.
        self.processor.tokenizer.padding_side = "left"
        self.model = self._load_model(model_name, num_gpus)
        self.model.eval()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    @staticmethod
    def _load_model(model_name: str, num_gpus: int):
        """Load the VLM, preferring flash-attn but tolerating its absence."""
        kwargs = {
            "dtype": torch.bfloat16,
            "device_map": "auto" if num_gpus > 1 else "cuda",
        }
        try:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, attn_implementation="flash_attention_2", **kwargs
            )
        except (ImportError, ValueError):
            # flash-attn is an optional CUDA extension. Fall back to the PyTorch
            # SDPA kernels, which are always available, rather than failing.
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, attn_implementation="sdpa", **kwargs
            )

    def _generation_kwargs(self) -> dict:
        """Build `generate()` kwargs, mapping temperature onto the sampling mode."""
        kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = self.temperature
        return kwargs

    def replace_placeholders(self, prompt: str, meta: dict) -> str:
        """Replace placeholders in the prompt with actual values from meta.

        Predefined placeholders are:
        - {image_type}: The type of the image.
        - {anomaly_type}: The type of anomaly present in the image.
        - {bboxes}: The list of bounding boxes in a string.
        """
        for key, value in meta.items():
            placeholder = "{" + key + "}"
            if placeholder in prompt and value is not None:
                prompt = prompt.replace(placeholder, str(value))
        return prompt

    def postprocess_response(self, response: str, num_bboxes: int) -> str:
        """Post-process the response to ensure it matches the number of bboxes."""
        start_tag = "<answer>"
        end_tag = "</answer>"
        # str.find / rfind return -1 when the tag is absent (they never raise),
        # so guard explicitly instead of relying on a dead try/except. Without
        # this, a response lacking the tags produced a corrupted slice
        # (response[-1 + len(start_tag) : -1]) rather than falling back to raw.
        start_pos = response.find(start_tag)
        end_pos = response.rfind(end_tag)
        if start_pos == -1 or end_pos == -1 or end_pos < start_pos + len(start_tag):
            return response
        content = response[start_pos + len(start_tag):end_pos].strip()
        # The delimiter is a blank line followed by the bolded "Anomaly " text
        delimiter = "\n\n**Anomaly "
        parts = content.split(delimiter)
        # The first part is always the caption
        caption = parts[0]
        # The rest of the parts are the anomalies
        anomalies = parts[1:]
        if not anomalies:
            return response
        # Keep only the desired number of anomalies from the list
        kept_anomalies = anomalies[:num_bboxes]
        rebuilt_parts = [caption]
        for anomaly in kept_anomalies:
            rebuilt_parts.append("**Anomaly " + anomaly)
        final_content = "\n\n".join(rebuilt_parts)
        return f"{start_tag}\n{final_content}\n{end_tag}"

    def batch_generate_caption(
        self,
        ori_image_paths: typing.List[Path],
        ori_mask_paths: typing.List[Path],
        gen_image_paths: typing.List[Path],
        metas: typing.List[dict],
    ) -> typing.List[str]:
        """Generate a caption for the given image and mask."""
        texts = []
        batch_images = []
        for ori_image_path, ori_mask_path, gen_image_path, meta in zip(
            ori_image_paths, ori_mask_paths, gen_image_paths, metas
        ):
            messages = [
                {
                    "role": "system",
                    "content": self.replace_placeholders(self.system_prompt, meta),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(ori_mask_path)},
                        {"type": "image", "image": str(ori_image_path)},
                        {"type": "image", "image": str(gen_image_path)},
                        {
                            "type": "text",
                            "text": self.replace_placeholders(self.user_prompt, meta),
                        },
                    ],
                },
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
            image_inputs, _ = process_vision_info(
                messages, return_video_kwargs=False
            )
            # The processor matches one flat image list against the image
            # placeholders across the whole batch, so extend rather than nest.
            if image_inputs:
                batch_images.extend(image_inputs)
        inputs = self.processor(
            text=texts,
            images=batch_images or None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        if self.seed is not None:
            torch.manual_seed(self.seed)
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **self._generation_kwargs())
        # Left padding makes the prompt width uniform, so one offset trims all.
        prompt_len = inputs.input_ids.shape[1]
        responses = self.processor.batch_decode(
            generated_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # Post-process the response to ensure it matches the number of bounding boxes.
        return [
            self.postprocess_response(response, int(meta["num_bboxes"]))
            for response, meta in zip(responses, metas)
        ]


def format_response(response: str, meta: dict) -> str:
    """Format the response by ensuring proper spacing around tags."""
    response = response.replace("<answer>", "", 1)
    response = response.replace("</answer>", "", 1)
    response = response.strip()

    # Add extra info to the response.
    response_with_extra_info = (
        f"Created: {datetime.now().isoformat()}\n"
        f"Meta:\n{json.dumps(meta, indent=4)}\n\n"
        f"{response}"
    )
    return response, response_with_extra_info
