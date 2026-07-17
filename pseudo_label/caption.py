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

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from vllm import LLM, RequestOutput, SamplingParams


class Captioner:
    """A class to generate captions for images using Cosmos Reason models."""

    def __init__(
        self,
        prompt_data: typing.Dict[str, str],
        model_name="nvidia/Cosmos-Reason1-7B",
        limit_mm_per_prompt={"image": 3, "video": 0},
        temperature=0.01,
        n=1,
        max_tokens=4096,
        seed=42,
        tensor_parallel_size=1,
    ):
        """Initialize the Captioner with model and prompt configuration.

        Args:
            prompt_data: A dictionary containing 'system_prompt' and
                'user_prompt'.
            model_name: The name of the pre-trained model to use. Defaults to
                `"nvidia/Cosmos-Reason1-7B"`.
            limit_mm_per_prompt: Limits for multi-modal inputs. Defaults to
                `{"image": 3, "video": 0}`.
            temperature: Sampling temperature. Defaults to `0.01`.
            n: Number of responses to generate. Defaults to `1`.
            max_tokens: Maximum number of tokens to generate. Defaults to
                `4096`.
            seed: Random seed for reproducibility. Defaults to `42`.
        """
        self.system_prompt = prompt_data["system_prompt"]
        self.user_prompt = prompt_data["user_prompt"]
        self.processor: Qwen2_5_VLProcessor = AutoProcessor.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            limit_mm_per_prompt=limit_mm_per_prompt,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, n=n, max_tokens=max_tokens, seed=seed
        )

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
        llm_inputs = []
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
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(
                messages, return_video_kwargs=False
            )
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            llm_inputs.append({"prompt": prompt, "multi_modal_data": mm_data})
        outputs: typing.List[RequestOutput] = self.llm.generate(
            llm_inputs, sampling_params=self.sampling_params, use_tqdm=False
        )
        responses = [output.outputs[0].text for output in outputs]
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
