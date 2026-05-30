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
"""Training-compatible CUDA graph utilities using TransformerEngine's make_graphed_callables."""
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
import torch
from transformer_engine.pytorch.graph import make_graphed_callables as _te_make_graphed_callables
from imaginaire.utils import log
__all__ = ["create_cuda_graph_training"]
_T = TypeVar("_T")
SingleOrTuple = Union[_T, Tuple[_T, ...]]
def _make_graphed_callables_training(
    modules: SingleOrTuple[Callable],
    sample_args: SingleOrTuple[Tuple[torch.Tensor, ...]],
    num_warmup_iters: int = 3,
    sample_kwargs: Optional[SingleOrTuple[Dict[str, Any]]] = None,
    pool: Optional[Tuple[int, ...]] = None,
) -> Union[Callable, Tuple[Callable, ...]]:
    """
    Make CUDA-graphed versions of modules for training (forward + backward).
    Delegates to TransformerEngine's ``make_graphed_callables`` which captures
    both forward and backward graphs. Temporarily sets modules to training mode
    during capture so TE captures backward graphs, even if the modules are
    frozen in eval mode (e.g. DiT blocks in anomaly-gen training).
    FP8 is explicitly disabled since DiT blocks use bf16/fp32.
    """
    # Canonicalize to tuple for uniform handling
    just_one = not isinstance(modules, tuple)
    if just_one:
        modules = (modules,)
    # TE checks c.training for all callables; we need True for backward capture.
    # Save original state and temporarily switch to train mode.
    original_training = [m.training for m in modules]
    for m in modules:
        m.train()
    try:
        result = _te_make_graphed_callables(
            modules if not just_one else modules[0],
            sample_args,
            num_warmup_iters=num_warmup_iters,
            allow_unused_input=True,
            sample_kwargs=sample_kwargs,
            fp8_enabled=False,
            pool=pool,
        )
    finally:
        for m, was_training in zip(modules, original_training):
            m.train(was_training)
    return result
def create_cuda_graph_training(
    cuda_graphs_storage: dict,
    blocks: torch.nn.ModuleList,
    x: torch.Tensor,
    affline_emb_B_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    rope_emb_L_1_1_D: torch.Tensor,
    adaln_lora_B_3D: torch.Tensor,
    extra_per_block_pos_emb: torch.Tensor,
) -> str:
    real_args = [arg for arg in [x, affline_emb_B_D, crossattn_emb] if arg is not None]
    real_kwargs = {
        k: v
        for k, v in {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_3D,
            "extra_per_block_pos_emb": extra_per_block_pos_emb,
        }.items()
        if v is not None
    }
    shapes_key = "_".join(
        [
            str(shape_component)
            for shape in [x.shape for x in real_args + list(real_kwargs.values())]
            for shape_component in shape
        ]
    )
    if shapes_key not in cuda_graphs_storage:
        callables = []
        sample_args = []
        sample_kwargs = []
        for block in blocks:
            callables.append(block)
            args = []
            kwargs = {}
            for arg in real_args:
                if arg.dtype == torch.int64:
                    dummy_arg = torch.randint(arg.min(), arg.max() + 1, arg.shape).type_as(arg)
                else:
                    dummy_arg = torch.randn(arg.shape).type_as(arg)
                    dummy_arg.requires_grad_(True)
                args.append(dummy_arg)
            for name, kwarg in real_kwargs.items():
                if kwarg.dtype == torch.int64:
                    dummy_kwarg = torch.randint(kwarg.min(), kwarg.max() + 1, kwarg.shape).type_as(kwarg)
                else:
                    dummy_kwarg = torch.randn(kwarg.shape).type_as(kwarg)
                    dummy_kwarg.requires_grad_(True)
                kwargs[name] = dummy_kwarg
            sample_args.append(args)
            sample_kwargs.append(kwargs)
        log.critical(f"Creating training graph for shape {shapes_key}")
        cuda_graphs_storage[shapes_key] = _make_graphed_callables_training(
            tuple(callables),
            tuple(sample_args),
            sample_kwargs=tuple(sample_kwargs),
            num_warmup_iters=11,
        )
        log.critical(f"Created training graph for shape {shapes_key}")
    return shapes_key
