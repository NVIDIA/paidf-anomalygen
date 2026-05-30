# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from imaginaire.utils import log


DEFAULT_PROCESS_GROUP_TIMEOUT_SEC = 1800.0
DEFAULT_FINALIZE_COLLECTIVE_BACKEND = "gloo"


@dataclass(frozen=True)
class InferenceRuntimeContext:
    rank: int
    world_size: int
    local_rank: int

    @property
    def is_multi_gpu(self) -> bool:
        return self.world_size > 1


@dataclass(frozen=True)
class SampleOutputPlan:
    global_order: int
    anomaly_offset: int
    num_outputs: int


def get_runtime_context() -> InferenceRuntimeContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return InferenceRuntimeContext(rank=rank, world_size=world_size, local_rank=local_rank)


def configure_local_cuda_device(runtime: InferenceRuntimeContext) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(runtime.local_rank)


def _get_backend_name() -> str:
    backend = dist.get_backend()
    if isinstance(backend, str):
        return backend.lower()
    return str(backend).split(".")[-1].lower()


def _get_finalize_collective_backend() -> str:
    backend = os.getenv(
        "ANOMALY_GEN_FINALIZE_BACKEND",
        DEFAULT_FINALIZE_COLLECTIVE_BACKEND,
    ).strip().lower()
    if backend not in {"gloo", "nccl"}:
        raise ValueError(
            "ANOMALY_GEN_FINALIZE_BACKEND must be one of {'gloo', 'nccl'}, "
            f"got {backend!r}"
        )
    if backend == "nccl" and not torch.cuda.is_available():
        log.warning("ANOMALY_GEN_FINALIZE_BACKEND=nccl requested without CUDA; falling back to gloo.")
        return "gloo"
    return backend


def initialize_distributed_collectives(
    runtime: InferenceRuntimeContext,
    timeout_sec: float = DEFAULT_PROCESS_GROUP_TIMEOUT_SEC,
) -> bool:
    if not runtime.is_multi_gpu:
        return False
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available for multi-GPU inference.")
    if dist.is_initialized():
        if dist.get_world_size() != runtime.world_size:
            raise RuntimeError(
                "torch.distributed was initialized with an unexpected world size: "
                f"{dist.get_world_size()} != {runtime.world_size}"
            )
        return False

    backend = _get_finalize_collective_backend()
    timeout_env = os.getenv("TORCH_DISTRIBUTED_TIMEOUT_SEC") or os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC")
    if timeout_env is not None:
        timeout_sec = float(timeout_env)
    init_kwargs = dict(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=timeout_sec),
    )
    if backend == "nccl":
        init_kwargs["device_id"] = torch.device("cuda", runtime.local_rank)
    dist.init_process_group(**init_kwargs)
    return True


def destroy_distributed_collectives(initialized_by_helper: bool) -> None:
    if initialized_by_helper and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank_work_items(total_items: int, rank: int, world_size: int) -> list[int]:
    if total_items < 0:
        raise ValueError(f"total_items must be >= 0, got {total_items}")
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, total_items, world_size))


def build_sample_output_plans(input_data: list[dict]) -> dict[int, SampleOutputPlan]:
    anomaly_offsets = defaultdict(int)
    plans = {}

    for global_order, sample in enumerate(input_data):
        sample_index = int(sample["index"])
        anomaly_type = sample["anomaly_type"]
        num_outputs = int(sample.get("num_generated_images", 1))

        plans[sample_index] = SampleOutputPlan(
            global_order=global_order,
            anomaly_offset=anomaly_offsets[anomaly_type],
            num_outputs=num_outputs,
        )
        anomaly_offsets[anomaly_type] += num_outputs

    return plans


def _to_json_safe(value):
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _require_collectives(world_size: int) -> bool:
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}")
    if world_size == 1:
        return False
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before using multi-GPU inference collectives."
        )
    if dist.get_world_size() != world_size:
        raise RuntimeError(
            "torch.distributed world size does not match the runtime context: "
            f"{dist.get_world_size()} != {world_size}"
        )
    return True


def _barrier() -> None:
    backend = _get_backend_name()
    if backend == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    elif backend == "gloo" and hasattr(dist, "monitored_barrier"):
        try:
            dist.monitored_barrier(wait_all_ranks=True)
        except TypeError:
            dist.monitored_barrier()
    else:
        dist.barrier()


def _merge_gathered_rank_rows(gathered_rows: list[list[dict]]) -> list[list]:
    merged_rows = []
    for rank_rows in gathered_rows:
        merged_rows.extend(rank_rows)
    merged_rows.sort(key=lambda item: tuple(item["sort_key"]))
    return [row["row"] for row in merged_rows]


def _sort_gathered_rank_timings(gathered_timings: list[dict]) -> list[dict]:
    merged_timings = [_to_json_safe(timing) for timing in gathered_timings]
    merged_timings.sort(key=lambda item: int(item.get("rank", 0)))
    return merged_timings


def wait_for_all_rank_rows(
    world_size: int,
) -> None:
    if _require_collectives(world_size):
        _barrier()


def wait_for_all_rank_timings(
    world_size: int,
) -> None:
    if _require_collectives(world_size):
        _barrier()


def merge_rank_rows(rows: list[dict], world_size: int) -> list[list]:
    payload = [_to_json_safe(row) for row in rows]
    if not _require_collectives(world_size):
        return _merge_gathered_rank_rows([payload])

    gathered_rows: list[list[dict]] = [[] for _ in range(world_size)]
    dist.all_gather_object(gathered_rows, payload)
    return _merge_gathered_rank_rows(gathered_rows)


def merge_rank_timings(timing: dict, world_size: int) -> list[dict]:
    payload = _to_json_safe(timing)
    if not _require_collectives(world_size):
        return _sort_gathered_rank_timings([payload])

    gathered_timings: list[dict] = [{} for _ in range(world_size)]
    dist.all_gather_object(gathered_timings, payload)
    return _sort_gathered_rank_timings(gathered_timings)


def aggregate_rank_timings(rank_timings: list[dict]) -> dict:
    if not rank_timings:
        raise ValueError("rank_timings must not be empty")

    active_rank_timings = [
        timing
        for timing in rank_timings
        if int(timing.get("assigned_samples", 0)) > 0 or int(timing.get("generated_images", 0)) > 0
    ]
    if not active_rank_timings:
        active_rank_timings = rank_timings

    def max_seconds(key: str, timings: list[dict]) -> float:
        return max(float(timing.get(key, 0.0)) for timing in timings)

    generated_images_total = sum(int(timing.get("generated_images", 0)) for timing in rank_timings)
    generation_seconds = max_seconds("generation_seconds", active_rank_timings)

    summary = {
        "aggregation_method": "single_rank" if len(rank_timings) == 1 else "max_rank_wall_time",
        "world_size": max(int(timing.get("world_size", 1)) for timing in rank_timings),
        "ranks_with_work": sum(int(timing.get("assigned_samples", 0)) > 0 for timing in rank_timings),
        "assigned_samples_total": sum(int(timing.get("assigned_samples", 0)) for timing in rank_timings),
        "generated_images_total": generated_images_total,
        "setup_seconds": max_seconds("setup_seconds", active_rank_timings),
        "model_init_seconds": max_seconds("model_init_seconds", active_rank_timings),
        "generation_seconds": generation_seconds,
        "finalize_seconds": max_seconds("finalize_seconds", rank_timings),
        "measured_total_seconds": max_seconds("measured_total_seconds", rank_timings),
        "rank_timings": [_to_json_safe(timing) for timing in rank_timings],
    }
    summary["generation_seconds_per_image"] = (
        generation_seconds / generated_images_total if generated_images_total > 0 else None
    )
    return summary
