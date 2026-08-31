# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warm-up floor in EarlyStop (pure Python; no model/GPU/distributed).

Early stopping and ``best_checkpoint.txt`` watch the same unsettled metric, so a floor on only one
of them is defeatable: see :meth:`EarlyStop._below_warmup`.
"""

from types import SimpleNamespace

from anomalygen.callbacks.early_stop import EarlyStop
from anomalygen.configs.texture.constants import CKPT_WARMUP_ITER


def _callback(max_iter, patience=5, metric="nn"):
    cb = EarlyStop(enabled=True, metric=metric, patience=patience)
    cb.config = SimpleNamespace(job=SimpleNamespace(path_local="/tmp"), trainer=SimpleNamespace(max_iter=max_iter))
    cb.on_train_start(model=None)
    return cb


def _kpi(nn):
    return {"Average": {"nn_score": nn}}


def _feed(cb, points):
    """Drive ``_rank0_decision`` over ``[(iteration, nn), ...]``; return the stopping iteration."""
    for iteration, nn in points:
        if cb._rank0_decision(_kpi(nn), iteration):
            return iteration
    return None


def test_a_spike_below_the_warmup_cannot_end_a_long_run():
    """The chain the floor exists to break: nn peaks at 1000 and is never beaten, so on a run planned
    to 15000 patience would otherwise exhaust at 6000 — below the floor, leaving best_checkpoint with
    no eligible iteration and falling back to exactly that spike."""
    cb = _callback(max_iter=15000, patience=5)
    points = [(1000, 0.9)] + [(i, 0.1) for i in range(2000, 7001, 1000)]
    assert _feed(cb, points) is None, "early stop fired below the warm-up floor"


def test_patience_counts_from_the_floor():
    cb = _callback(max_iter=15000, patience=2)
    # Below the floor nothing accumulates; the first validation at/after it seeds `best`, and only
    # then do two non-improving validations trigger.
    assert _feed(cb, [(1000, 0.9), (2000, 0.1), (3000, 0.1)]) is None
    assert cb.wait == 0 and cb.best_iteration is None
    assert _feed(cb, [(CKPT_WARMUP_ITER, 0.5), (8500, 0.1), (9500, 0.1)]) == 9500


def test_the_floor_does_not_set_best_from_an_early_spike():
    """`best` must not be seeded below the floor, or the spike still sets an unbeatable bar."""
    cb = _callback(max_iter=15000, patience=5)
    _feed(cb, [(1000, 0.9)])
    assert cb.best_iteration is None
    _feed(cb, [(CKPT_WARMUP_ITER, 0.4)])
    assert cb.best_iteration == CKPT_WARMUP_ITER and cb.best == 0.4


def test_short_runs_early_stop_normally():
    """The floor keys off the *planned* max_iter, so a dry run or a short fine-tune is unaffected —
    the same carve-out TrainingReport makes."""
    cb = _callback(max_iter=3000, patience=2)
    assert _feed(cb, [(1000, 0.9), (2000, 0.1), (3000, 0.1)]) == 3000


def test_floor_is_off_when_max_iter_equals_it():
    """`>` not `>=`: a run planned to exactly the floor is a short run."""
    cb = _callback(max_iter=CKPT_WARMUP_ITER, patience=1)
    assert _feed(cb, [(1000, 0.9), (2000, 0.1)]) == 2000


def test_planned_max_iter_is_read_before_trigger_shrinks_it():
    """_trigger rewrites trainer.max_iter, so a value read later would disable the floor retroactively."""
    cb = _callback(max_iter=15000, patience=1)
    cb.config.trainer.max_iter = 4000  # as _trigger would leave it
    assert cb._below_warmup(2000) is True
