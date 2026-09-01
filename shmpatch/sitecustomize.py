# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External DataLoader /dev/shm hardening.

Put this directory FIRST on PYTHONPATH; Python imports `sitecustomize` at every
interpreter startup, so the patch reaches the main training process AND every
DataLoader worker (forked or spawned).

cosmos-framework used to ship its own `sitecustomize.py`, which this file shadowed
and then re-executed by path. It is now a pip package and ships no such file, so
that re-exec was dead code. Its two behaviours — the COSMOS_DL_FILE_SYSTEM_SHARING
gate and the LOAD_TRACE tracer — are inlined at the tail of this file instead.

Activation: this dir is only ever placed on PYTHONPATH by launchers that WANT the
patch, so importing this module IS the opt-in — the patch is ON by default and
needs NO env var. Set COSMOS_DL_NO_SHM=0 to explicitly DISABLE it (e.g. an A/B
control group that keeps shmpatch on PYTHONPATH but wants stock torch shm IPC).

Why: with num_workers>=1 there are TWO independent /dev/shm/torch_* creation
sites, and on this cluster systemd RemoveIPC=yes wipes a user's /dev/shm the
moment ANY of their jobs on that node ends -> a co-located run racing in either
site dies with "could not unlink the shared memory file /torch_*" (SIGABRT).
  (A) collate: torch.utils.data._utils.collate.collate_tensor_fn pre-allocates
      the whole batch in shared memory via elem._typed_storage()._new_shared(n)
      -> UntypedStorage._new_shared -> _new_using_fd_cpu, INSIDE the worker,
      before transport, whenever a worker is active.
  (B) transport: the ForkingPickler that ships each batch to the main process
      reduces CPU tensors via reduce_storage -> _share_fd_cpu_ (worker _feed
      thread).
A 3-round fault-injection (2026-07-11) proved that patching only (B) still
crashed 2/3 rounds at (A). This module neutralizes BOTH: (B) by handing CPU
tensors to the main process BY VALUE (torch.save bytes over the queue pipe), and
(A) by returning a plain heap storage from _new_shared on CPU (by-value
transport never needs the storage to actually be shared). Result: zero
/dev/shm/torch_* files from the dataloader -> RemoveIPC has nothing to race.
Costs one extra copy per batch (hidden by prefetch). CUDA tensors keep torch's
native reducer (dataloaders here never produce them).
"""

import atexit
import os
import sys

if os.environ.get("COSMOS_DL_NO_SHM", "1") != "0":  # ON by default; =0 disables
    try:
        import io
        import traceback
        from multiprocessing.reduction import ForkingPickler

        import torch
        import torch.multiprocessing.reductions as _tmr

        # ---- Site B: transport reducer (ship CPU tensors by value) ----
        def _rebuild_tensor_byvalue(data):
            import io as _io

            import torch as _torch

            return _torch.load(_io.BytesIO(data), weights_only=True)

        def _reduce_tensor_byvalue(t):
            if t.device.type != "cpu":
                return _tmr.reduce_tensor(t)
            x = t.detach()
            # torch.save serializes the WHOLE backing storage; clone views so we
            # never ship more bytes than the tensor itself.
            if not x.is_contiguous() or x.untyped_storage().nbytes() != x.numel() * x.element_size():
                x = x.clone()
            buf = io.BytesIO()
            torch.save(x, buf)
            return (_rebuild_tensor_byvalue, (buf.getvalue(),))

        class _GuardedReducers(dict):
            # ForkingPickler.register() is just `_extra_reducers[type] = reduce`,
            # so this dict is the single choke point for reducer (re)registration:
            # any attempt to re-point a tensor class back at torch's shm reducer is
            # REFUSED and logged with the caller's stack.
            _protected = ()

            def __setitem__(self, key, value):
                if key in type(self)._protected and value is not _reduce_tensor_byvalue:
                    sys.stderr.write(
                        f"[COSMOS_DL_NO_SHM] blocked tensor-reducer clobber for {key!r} by:\n"
                        + "".join(traceback.format_stack(limit=10))
                    )
                    sys.stderr.flush()
                    return
                super().__setitem__(key, value)

        # Cover every tensor class dispatch can see, not just torch.Tensor: a batch
        # containing a Tensor SUBCLASS would otherwise still hit torch's shm reducer
        # (ForkingPickler dispatch is exact-type).
        _tensor_classes = {torch.Tensor, torch.nn.Parameter, *torch._tensor_classes}
        _GuardedReducers._protected = tuple(_tensor_classes)
        _guarded = _GuardedReducers(ForkingPickler._extra_reducers)
        for _t in _tensor_classes:
            dict.__setitem__(_guarded, _t, _reduce_tensor_byvalue)
        ForkingPickler._extra_reducers = _guarded

        # ---- Site A: collate pre-allocation (return a NON-shared storage) ----
        # collate_tensor_fn does `elem._typed_storage()._new_shared(numel)` inside
        # the worker, which allocates the batch in /dev/shm. With by-value transport
        # that storage never needs to be shared, so on CPU return a plain heap
        # storage -> no /torch_* file is ever created. Non-CPU (cuda/hpu) delegates
        # to torch's original (dataloaders here don't produce those).
        _orig_new_shared = torch.UntypedStorage._new_shared

        def _new_shared_byvalue(cls, size, *, device="cpu"):
            if torch.device(device).type == "cpu":
                return cls(size)  # plain, non-shared -> no /dev/shm/torch_* file
            return _orig_new_shared(size, device=device)

        torch.UntypedStorage._new_shared = classmethod(_new_shared_byvalue)
    except Exception:
        pass

# --- inlined from cosmos-framework's former sitecustomize.py -------------------
# Opt-in (COSMOS_DL_FILE_SYSTEM_SHARING=1): switch torch's DataLoader IPC from the
# default 'file_descriptor' strategy (which stages worker tensors in /dev/shm) to
# 'file_system'. On shm-constrained containers, large video batches overflow the
# small /dev/shm tmpfs and a worker dies mid-transfer -> the main process then sees
# "unable to open shared memory object ... No such file or directory".
# Guarded so non-training processes never import torch.
if os.environ.get("COSMOS_DL_FILE_SYSTEM_SHARING") == "1":
    try:
        import torch.multiprocessing as _tmp

        _tmp.set_sharing_strategy("file_system")
    except Exception:
        pass

# LOAD_TRACE_DIR: at exit, dump every loaded module whose real path is under
# LOAD_TRACE_ROOT to {LOAD_TRACE_DIR}/{LOAD_TRACE_TAG}_pid{PID}.txt.
_DIR = os.environ.get("LOAD_TRACE_DIR", "")
if _DIR:
    _TAG = os.environ.get("LOAD_TRACE_TAG", "default")
    _ROOT = os.path.realpath(os.environ.get("LOAD_TRACE_ROOT", os.getcwd()))
    os.makedirs(_DIR, exist_ok=True)

    def _dump() -> None:
        seen = set()
        for mod in list(sys.modules.values()):
            f = getattr(mod, "__file__", None)
            if not f:
                continue
            try:
                rp = os.path.realpath(f)
            except OSError:
                continue
            if rp.startswith(_ROOT):
                seen.add(rp)
        path = os.path.join(_DIR, f"{_TAG}_pid{os.getpid()}.txt")
        try:
            with open(path, "w") as h:
                for p in sorted(seen):
                    h.write(p + "\n")
        except OSError:
            pass

    atexit.register(_dump)
