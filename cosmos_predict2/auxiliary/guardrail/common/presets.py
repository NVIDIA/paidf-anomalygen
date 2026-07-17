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

import numpy as np
from PIL import Image

from cosmos_predict2.auxiliary.guardrail.common.core import GuardrailRunner
from imaginaire.utils import log

# NOTE: the concrete safety models (Blocklist, LlamaGuard3, VideoContentSafetyFilter,
# RetinaFaceFilter) are imported lazily inside the create_* functions below.
# Each pulls heavy, type-specific dependencies (e.g. nltk/better_profanity for the
# text blocklist, retinaface for face blur), so importing this module — and using
# one guardrail type — must not require the dependencies of the others. In
# particular, the image guardrail must remain importable without the text/face
# guardrail dependencies installed.


def create_text_guardrail_runner(checkpoint_dir: str, offload_model_to_cpu: bool) -> GuardrailRunner:
    """Create the text guardrail runner."""
    from cosmos_predict2.auxiliary.guardrail.blocklist.blocklist import Blocklist
    from cosmos_predict2.auxiliary.guardrail.llamaGuard3.llamaGuard3 import LlamaGuard3

    return GuardrailRunner(
        safety_models=[
            Blocklist(checkpoint_dir=checkpoint_dir),
            LlamaGuard3(checkpoint_dir=checkpoint_dir, offload_model_to_cpu=offload_model_to_cpu),
        ]
    )


def create_video_guardrail_runner(checkpoint_dir: str, offload_model_to_cpu: bool) -> GuardrailRunner:
    """Create the video guardrail runner."""
    from cosmos_predict2.auxiliary.guardrail.face_blur_filter.face_blur_filter import RetinaFaceFilter
    from cosmos_predict2.auxiliary.guardrail.video_content_safety_filter.video_content_safety_filter import (
        VideoContentSafetyFilter,
    )

    return GuardrailRunner(
        safety_models=[
            VideoContentSafetyFilter(checkpoint_dir=checkpoint_dir, offload_model_to_cpu=offload_model_to_cpu)
        ],
        postprocessors=[RetinaFaceFilter(checkpoint_dir=checkpoint_dir, offload_model_to_cpu=offload_model_to_cpu)],
    )


def create_image_guardrail_runner(checkpoint_dir: str, offload_model_to_cpu: bool) -> GuardrailRunner:
    """Create the image guardrail runner.

    Reuses the SigLIP-based content safety classifier (the same model used by
    the video guardrail) to score a single generated image. Unlike the video
    runner, no face-blur postprocessor is attached: generated anomaly images
    are checked, not rewritten — so this path does not require the face-blur
    (retinaface) or text-blocklist (nltk) dependencies.
    """
    from cosmos_predict2.auxiliary.guardrail.video_content_safety_filter.video_content_safety_filter import (
        VideoContentSafetyFilter,
    )

    return GuardrailRunner(
        safety_models=[
            VideoContentSafetyFilter(checkpoint_dir=checkpoint_dir, offload_model_to_cpu=offload_model_to_cpu)
        ],
        generic_safe_msg="Image is safe",
    )


def run_text_guardrail(prompt: str, guardrail_runner: GuardrailRunner) -> bool:
    """Run the text guardrail on the prompt, checking for content safety.

    Args:
        prompt: The text prompt.
        guardrail_runner: The text guardrail runner.

    Returns:
        bool: Whether the prompt is safe.
    """
    is_safe, message = guardrail_runner.run_safety_check(prompt)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
    return is_safe


def run_video_guardrail(frames: np.ndarray, guardrail_runner: GuardrailRunner) -> np.ndarray | None:
    """Run the video guardrail on the frames, checking for content safety and applying face blur.

    Args:
        frames: The frames of the generated video.
        guardrail_runner: The video guardrail runner.

    Returns:
        The processed frames if safe, otherwise None.
    """
    is_safe, message = guardrail_runner.run_safety_check(frames)
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
        return None

    frames = guardrail_runner.postprocess(frames)
    return frames


def run_image_guardrail(image: "Image.Image | np.ndarray", guardrail_runner: GuardrailRunner) -> bool:
    """Run the image guardrail on a single generated image, checking content safety.

    Args:
        image: The generated image, either a PIL image or an HxWxC uint8 array.
        guardrail_runner: The image guardrail runner.

    Returns:
        bool: Whether the image is safe.
    """
    if isinstance(image, Image.Image):
        frame = np.asarray(image.convert("RGB"))
    else:
        frame = np.asarray(image)
    # The SigLIP content-safety filter consumes an iterable of frames; a single
    # generated image is checked as a one-frame sequence.
    is_safe, message = guardrail_runner.run_safety_check([frame])
    if not is_safe:
        log.critical(f"GUARDRAIL BLOCKED: {message}")
    return is_safe
