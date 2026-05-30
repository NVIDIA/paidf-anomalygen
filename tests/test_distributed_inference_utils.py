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

from cosmos_predict2.inference.anomaly_gen import distributed_inference_utils as utils
from cosmos_predict2.inference.anomaly_gen.distributed_inference_utils import (
    aggregate_rank_timings,
    destroy_distributed_collectives,
    InferenceRuntimeContext,
    SampleOutputPlan,
    build_sample_output_plans,
    get_rank_work_items,
    initialize_distributed_collectives,
    merge_rank_timings,
    merge_rank_rows,
    wait_for_all_rank_timings,
    wait_for_all_rank_rows,
)


def test_get_rank_work_items_uses_round_robin_without_padding():
    assert get_rank_work_items(total_items=10, rank=0, world_size=3) == [0, 3, 6, 9]
    assert get_rank_work_items(total_items=10, rank=1, world_size=3) == [1, 4, 7]
    assert get_rank_work_items(total_items=10, rank=2, world_size=3) == [2, 5, 8]


def test_build_sample_output_plans_tracks_global_order_and_offsets():
    input_data = [
        {"index": 10, "anomaly_type": "metal_surface+MT_Blowhole", "num_generated_images": 1},
        {"index": 11, "anomaly_type": "metal_surface+MT_Blowhole", "num_generated_images": 2},
        {"index": 12, "anomaly_type": "metal_surface+MT_Crack", "num_generated_images": 1},
        {"index": 13, "anomaly_type": "metal_surface+MT_Blowhole", "num_generated_images": 1},
    ]

    plans = build_sample_output_plans(input_data)

    assert plans[10] == SampleOutputPlan(global_order=0, anomaly_offset=0, num_outputs=1)
    assert plans[11] == SampleOutputPlan(global_order=1, anomaly_offset=1, num_outputs=2)
    assert plans[12] == SampleOutputPlan(global_order=2, anomaly_offset=0, num_outputs=1)
    assert plans[13] == SampleOutputPlan(global_order=3, anomaly_offset=3, num_outputs=1)


def test_initialize_distributed_collectives_defaults_to_gloo_for_finalize(monkeypatch):
    runtime = InferenceRuntimeContext(rank=3, world_size=8, local_rank=3)
    init_kwargs = {}

    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)

    def fake_init_process_group(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(utils.dist, "init_process_group", fake_init_process_group)

    initialized = initialize_distributed_collectives(runtime)

    assert initialized is True
    assert init_kwargs["backend"] == "gloo"
    assert init_kwargs["init_method"] == "env://"
    assert "device_id" not in init_kwargs


def test_initialize_distributed_collectives_uses_local_cuda_device_for_nccl_override(monkeypatch):
    runtime = InferenceRuntimeContext(rank=3, world_size=8, local_rank=3)
    init_kwargs = {}

    monkeypatch.setenv("ANOMALY_GEN_FINALIZE_BACKEND", "nccl")
    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)

    def fake_init_process_group(**kwargs):
        init_kwargs.update(kwargs)

    monkeypatch.setattr(utils.dist, "init_process_group", fake_init_process_group)

    initialized = initialize_distributed_collectives(runtime)

    assert initialized is True
    assert init_kwargs["backend"] == "nccl"
    assert init_kwargs["init_method"] == "env://"
    assert init_kwargs["device_id"] == utils.torch.device("cuda", runtime.local_rank)


def test_destroy_distributed_collectives_only_destroys_helper_initialized_group(monkeypatch):
    destroy_calls = []

    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "destroy_process_group", lambda: destroy_calls.append("destroy"))

    destroy_distributed_collectives(initialized_by_helper=False)
    destroy_distributed_collectives(initialized_by_helper=True)

    assert destroy_calls == ["destroy"]


def test_wait_for_all_rank_rows_is_noop_for_single_rank():
    wait_for_all_rank_rows(world_size=1)


def test_wait_for_all_rank_rows_uses_monitored_barrier_for_gloo(monkeypatch):
    monitored_calls = []

    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(utils.dist, "get_backend", lambda: "gloo")

    def fake_monitored_barrier(**kwargs):
        monitored_calls.append(kwargs)

    monkeypatch.setattr(utils.dist, "monitored_barrier", fake_monitored_barrier)

    wait_for_all_rank_rows(world_size=2)

    assert monitored_calls == [{"wait_all_ranks": True}]


def test_merge_rank_rows_restores_global_order_with_all_gather(monkeypatch):
    local_rows = [
        {"sort_key": [0, 0], "row": ["rank0-row0"]},
        {"sort_key": [2, 1], "row": ["rank0-row1"]},
    ]
    gathered_rows = [
        local_rows,
        [
            {"sort_key": [1, 0], "row": ["rank1-row0"]},
            {"sort_key": [3, 0], "row": ["rank1-row1"]},
        ],
    ]

    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda: 2)

    def fake_all_gather_object(output, input_obj):
        assert input_obj == local_rows
        output[:] = gathered_rows

    monkeypatch.setattr(utils.dist, "all_gather_object", fake_all_gather_object)

    merged_rows = merge_rank_rows(local_rows, world_size=2)

    assert merged_rows == [
        ["rank0-row0"],
        ["rank1-row0"],
        ["rank0-row1"],
        ["rank1-row1"],
    ]


def test_merge_rank_rows_single_rank_sorts_without_distributed():
    merged_rows = merge_rank_rows(
        [
            {"sort_key": [1, 0], "row": ["rank0-row1"]},
            {"sort_key": [0, 0], "row": ["rank0-row0"]},
        ],
        world_size=1,
    )

    assert merged_rows == [["rank0-row0"], ["rank0-row1"]]


def test_merge_and_aggregate_rank_timings_uses_all_gather(monkeypatch):
    local_timing = {
        "rank": 0,
        "world_size": 2,
        "assigned_samples": 3,
        "generated_images": 3,
        "setup_seconds": 0.5,
        "model_init_seconds": 5.0,
        "generation_seconds": 18.0,
        "finalize_seconds": 2.0,
        "measured_total_seconds": 26.0,
    }
    rank1_timing = {
        "rank": 1,
        "world_size": 2,
        "assigned_samples": 2,
        "generated_images": 2,
        "setup_seconds": 0.3,
        "model_init_seconds": 4.0,
        "generation_seconds": 20.0,
        "finalize_seconds": 1.5,
        "measured_total_seconds": 25.8,
    }

    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda: 2)

    def fake_all_gather_object(output, input_obj):
        assert input_obj == local_timing
        output[:] = [local_timing, rank1_timing]

    monkeypatch.setattr(utils.dist, "all_gather_object", fake_all_gather_object)

    merged_timings = merge_rank_timings(local_timing, world_size=2)
    summary = aggregate_rank_timings(merged_timings)

    assert [timing["rank"] for timing in merged_timings] == [0, 1]
    assert summary["aggregation_method"] == "max_rank_wall_time"
    assert summary["world_size"] == 2
    assert summary["ranks_with_work"] == 2
    assert summary["assigned_samples_total"] == 5
    assert summary["generated_images_total"] == 5
    assert summary["setup_seconds"] == 0.5
    assert summary["model_init_seconds"] == 5.0
    assert summary["generation_seconds"] == 20.0
    assert summary["finalize_seconds"] == 2.0
    assert summary["measured_total_seconds"] == 26.0
    assert summary["generation_seconds_per_image"] == 4.0
    assert len(summary["rank_timings"]) == 2


def test_wait_for_all_rank_timings_uses_nccl_barrier(monkeypatch):
    barrier_calls = []

    monkeypatch.setattr(utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(utils.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(utils.torch.cuda, "current_device", lambda: 5)

    def fake_barrier(**kwargs):
        barrier_calls.append(kwargs)

    monkeypatch.setattr(utils.dist, "barrier", fake_barrier)

    wait_for_all_rank_timings(world_size=2)

    assert barrier_calls == [{"device_ids": [5]}]
