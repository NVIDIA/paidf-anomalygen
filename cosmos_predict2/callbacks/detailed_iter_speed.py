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

import time

import torch
from torch import Tensor

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed, log


class DetailedIterSpeed(EveryN):
    """A more detailed version of the IterSpeed callback that logs per sample time.

    Note that `DetailedIterSpeed` reports the time by each training step
    when the iteration is smaller than `hit_thres`. After that, it reports by
    each `every_n` iterations.

    In multi-gpu scenario, the per sample time is calculated as:

    `elapsed_time / batch_size / (world_size / context_parallel_size)`

    - `world_size` means how many GPUs are used in total.
    - `context_parallel_size` means how many replications are used in the data parallelism.

    For examples:

    1. If `batch_size` is 2, `world_size` is 8 and
    `context_parallel_size` is 4, the global batch size is 2 * 8 / 4 = 4.
    2. If `batch_size` is 1, `world_size` is 8 and `context_parallel_size` is 1
    (i.e., no CP), the global batch size is 1 * 8 / 1 = 8.

    Args:
        batch_size (int): Batch size.
        context_parallel_size (int): Context parallel size.
        hit_thres (int): Number of iterations to wait before logging.
    """

    def __init__(self, *args, hit_thres: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.hit_thres = hit_thres

        self.time = None
        self.hit_counter = 0
        self.name = self.__class__.__name__
        self.last_hit_time = time.time()

    def on_train_start(self, model, iteration=0):
        self.world_size = max(distributed.get_world_size(), 1)
        self.rank = distributed.get_rank()

    def _gather_data_list(self, data):
        data_list = [data] * self.world_size
        if self.world_size > 1:
            torch.distributed.all_gather_object(data_list, data)
            torch.distributed.barrier()
        return data_list

    def _aggregate_data_list(self, data_list):
        avg_loss = []
        batch_size = []
        for data in data_list:
            avg_loss.append(data["loss"])
            batch_size.append(data["batch_size"])
        total_batch_size = sum(batch_size)
        avg_loss = sum(avg_loss) / len(avg_loss)
        return total_batch_size, avg_loss

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.hit_counter < self.hit_thres:
            # Support both video and image batches
            if "video" in data_batch:
                batch_size = int(data_batch["video"].shape[0])
            elif "images" in data_batch:
                batch_size = int(data_batch["images"].shape[0])
            else:
                raise ValueError(f"Invalid data batch: {data_batch}")
            data_list = self._gather_data_list(
                {"batch_size": batch_size, "loss": loss.item()}
            )
            if self.rank == 0:
                total_batch_size, avg_loss = self._aggregate_data_list(data_list)
                elapsed_time = time.time() - self.last_hit_time
                per_sample_time = elapsed_time / total_batch_size
                log.info(
                    f"{self.name}: "
                    f"Iteration: {iteration} ({self.hit_counter + 1}/{self.hit_thres}) | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Time: {elapsed_time:.2f}s | "
                    f"Per sample time: {per_sample_time:.2f}s (bs={total_batch_size})"
                )
            self.hit_counter += 1
            self.last_hit_time = time.time()
            #! useful for large scale training and avoid oom crash in the first two iterations!!!
            torch.cuda.synchronize()
            return
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, Tensor],
        output_batch: dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        if self.time is None:
            self.time = time.time()
            return
        cur_time = time.time()

        # Support both video and image batches
        if "video" in data_batch:
            batch_size = int(data_batch["video"].shape[0])
        elif "images" in data_batch:
            batch_size = int(data_batch["images"].shape[0])
        else:
            raise ValueError(f"Invalid data batch: {data_batch}")

        data_list = self._gather_data_list(
            {"batch_size": batch_size, "loss": loss.item()}
        )
        if self.rank == 0:
            total_batch_size, avg_loss = self._aggregate_data_list(data_list)
            elapsed_time = (cur_time - self.time) / self.every_n / self.step_size
            per_sample_time = elapsed_time / total_batch_size
            log.info(
                f"{self.name}: "
                f"Iteration: {iteration} | "
                f"Loss: {avg_loss:.4f} | "
                f"Time: {elapsed_time:.2f}s | "
                f"Per sample time: {per_sample_time:.2f}s (bs={total_batch_size})"
            )
        self.time = cur_time
