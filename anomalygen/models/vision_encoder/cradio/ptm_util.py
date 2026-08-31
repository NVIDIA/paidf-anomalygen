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

import os
from typing import Any, Mapping, Union

import torch

try:
    # torch 2.6.0.
    from torch.serialization import FILE_LIKE
except ImportError:
    # torch 2.8.0.
    from torch.serialization import FileLike as FILE_LIKE
from safetensors.torch import load_file


def load_pretrained_weights(
    path_or_checkpoint: Union[FILE_LIKE, Mapping[str, Any]],
    map_location="cpu",
    weights_only=True,
    ptm_adapter=None,
    parser=None,
    **kwargs,
):
    """Load the pretrained weights.

    Args:
        path_or_checkpoint (str or dict): Path to the pretrained weights file or a checkpoint containing the
            weights.
        map_location (str): A function, `torch.device`, string or a dict specifying how to remap storage locations.
            Default: `"cpu"`.
        weights_only (bool): Indicates whether unpickler should be restricted to loading only tensors, primitive
            types, dictionaries and any types added via `torch.serialization.add_safe_globals`. Default: `True`.
            Defaults to restricted loading because a checkpoint is data, not code: unpickling one without
            this lets a crafted `.pt` run arbitrary code at load time. Pass `False` only for a checkpoint
            whose origin you trust and that genuinely needs a type outside the allowlist — prefer extending
            the allowlist with `torch.serialization.add_safe_globals` over disabling the check.
        ptm_adapter (StateDictAdapter): instance of StateDictAdapter to adapt the state dict of a TAO model.
        parser (function): function to parse the state dict for a custom/public model.
        kwargs: Additional arguments passed to the `torch.load` function.
    """
    # Get the checkpoint from the path.
    path = None
    if not isinstance(path_or_checkpoint, dict) and (
        isinstance(path_or_checkpoint, (str, os.PathLike)) or hasattr(path_or_checkpoint, "read")
    ):
        path = path_or_checkpoint
        # Support safetensors files.
        if isinstance(path, (str, os.PathLike)) and path.endswith(".safetensors"):
            checkpoint = load_file(path, device=map_location)
        else:
            checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only, **kwargs)
    else:
        checkpoint = path_or_checkpoint

    tao_model = checkpoint.get("tao_model", None)
    if tao_model is not None:  # for TAO models
        state_dict = ptm_adapter(tao_model, checkpoint["state_dict"])
    else:  # for public models
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        if parser is not None:
            state_dict = parser(state_dict)
    return state_dict
