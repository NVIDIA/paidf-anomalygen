# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint integrity sidecars, shared by the training and inference load paths.

``weights_only=True`` stops a checkpoint executing code; it does not say the checkpoint is the one
this run wrote. The base weights have assets/checkpoint_manifest.sha256 — this is the equivalent
for what we produce, which is also the file most likely to travel between machines.

A sidecar is not a signature (whoever replaces the .pt can replace the .sha256), so the two cases
differ: a *mismatch* is fatal, since only truncation, a bad sync or a careless swap produces it;
an *absent* digest only warns, because that is every pre-existing checkpoint and every copy of a
single file, and blocking would break those while a real attacker just writes the sidecar too.
"""

from __future__ import annotations

import hashlib
import os
import re

from cosmos_framework.utils import log

_DIGEST_SUFFIX = ".sha256"


def _file_digest(path: str) -> str:
    """SHA-256 of a file, read in chunks — a checkpoint shard does not fit comfortably in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(manifest_path, root, *, only_under: str | None = None) -> list[tuple[str, str]]:
    """Check a ``sha256sum``-format manifest against ``root``; return one entry per failure.

    This is the base and backbone counterpart to :func:`verify_digest`: those weights are recorded by
    ``scripts/download_checkpoints.sh`` but were previously only checked at download time, while
    ``checkpoints/`` is a bind mount that anything on the host can rewrite afterwards. Returning the
    failures instead of raising lets the caller decide — preflight reports all of them, the load path
    refuses on the first.

    ``only_under`` limits the check to entries at or below that relative path, so a run can verify the
    model size it is about to load without hashing the other one. A malformed line or an entry
    escaping ``root`` raises: those are defects in the manifest, not in the weights.
    """
    root = os.path.realpath(os.fspath(root))
    with open(manifest_path) as handle:
        lines = handle.read().splitlines()

    failures: list[tuple[str, str]] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        # Exactly what coreutils writes: digest, one space, then ' ' (text) or '*' (binary), then the
        # path. Both are accepted by `sha256sum -c`, which download_checkpoints.sh uses on the same
        # files — a parser that took only one of them would disagree with the tool that records them.
        entry = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        if not entry:
            raise ValueError(f"{manifest_path}:{lineno} is not a sha256sum line: {line!r}")
        digest, rel = entry.group(1), entry.group(2).strip()
        if only_under and rel != only_under and not rel.startswith(only_under.rstrip("/") + "/"):
            continue
        target = os.path.realpath(os.path.join(root, rel))
        if target != root and not target.startswith(root + os.sep):
            raise ValueError(f"{manifest_path}:{lineno} names {rel!r}, outside the checkpoint root {root}")
        if not os.path.isfile(target):
            failures.append((rel, f"missing: {target}"))
        elif _file_digest(target) != digest.lower():
            failures.append((rel, f"does not match the recorded digest in {os.path.basename(str(manifest_path))}"))
    return failures


def write_digest(path: str) -> None:
    """Record ``<path>.sha256`` in ``sha256sum`` format, so it can be checked by hand."""
    sidecar = path + _DIGEST_SUFFIX
    tmp = sidecar + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(f"{_file_digest(path)}  {os.path.basename(path)}\n")
    # Rename rather than write in place: a redirect truncates first, so an interrupt would leave a
    # half-written digest that fails every later load. Matches download_checkpoints.sh.
    os.replace(tmp, sidecar)


def verify_digest(path: str) -> None:
    """Check ``path`` against its sidecar digest.

    Raises ValueError when the digest is present and disagrees, and propagates the OSError from a
    sidecar that cannot be opened. Warns and returns when it is absent, empty or non-hex — see the
    module docstring for why those cases are treated differently.
    """
    sidecar = path + _DIGEST_SUFFIX
    if not os.path.exists(sidecar):
        log.warning(
            f"Checkpoint integrity: no digest beside {path}, so its bytes are unverified. "
            "Checkpoints written before digest recording have none; ones written from now on do. "
            "To record it for this checkpoint after confirming it is the one you expect:\n"
            f"  ( cd {os.path.dirname(path)!r} && sha256sum {os.path.basename(path)!r} > "
            f"{os.path.basename(path) + _DIGEST_SUFFIX!r} )"
        )
        return
    with open(sidecar) as handle:
        fields = handle.read().split()
    # `sha256sum f > f.sha256` leaves the sidecar empty when sha256sum fails. Warn rather than
    # compare, so a broken sidecar is not reported as a modified checkpoint.
    if not fields or not re.fullmatch(r"[0-9a-f]{64}", fields[0].lower()):
        log.warning(
            f"Checkpoint integrity: {sidecar} holds no readable SHA-256, so the bytes of {path} "
            "are unverified. The checkpoint itself may still be fine."
        )
        return
    recorded = fields[0].lower()
    actual = _file_digest(path)
    if actual != recorded:
        raise ValueError(
            f"Checkpoint integrity: {path} does not match its recorded digest.\n"
            f"  recorded {recorded}\n  actual   {actual}\n"
            "The checkpoint's bytes changed after it was written. Treat this as an integrity "
            "problem — do not re-record the digest to make it load."
        )
