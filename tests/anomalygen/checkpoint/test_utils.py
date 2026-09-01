# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the checkpoint integrity sidecars shared by the training and inference load paths."""

import hashlib
import shutil
import subprocess

import pytest

import anomalygen.checkpoint.utils as ckpt_utils
from anomalygen.checkpoint.utils import verify_manifest


def _write_checkpoint(tmp_path, payload=b"weights"):
    path = tmp_path / "iter_000000100.pt"
    path.write_bytes(payload)
    return str(path)


def test_write_digest_records_a_checkable_sha256sum_line(tmp_path):
    path = _write_checkpoint(tmp_path)
    ckpt_utils.write_digest(path)

    sidecar = tmp_path / "iter_000000100.pt.sha256"
    recorded, name = sidecar.read_text().split()
    assert name == "iter_000000100.pt", "sidecar must name the file, so `sha256sum -c` works by hand"
    assert recorded == ckpt_utils._file_digest(path)


def test_verify_digest_accepts_an_unmodified_checkpoint(tmp_path):
    path = _write_checkpoint(tmp_path)
    ckpt_utils.write_digest(path)

    ckpt_utils.verify_digest(path)  # must not raise


def test_verify_digest_rejects_modified_bytes(tmp_path):
    """The substitution this control exists for: same path, different content."""
    path = _write_checkpoint(tmp_path)
    ckpt_utils.write_digest(path)
    with open(path, "wb") as handle:
        handle.write(b"substituted")

    with pytest.raises(ValueError, match="does not match its recorded digest"):
        ckpt_utils.verify_digest(path)


def test_verify_digest_warns_but_proceeds_when_the_sidecar_is_absent(tmp_path, loguru_lines):
    """Absent is the normal state for a checkpoint written before this change, and for a copy of
    one file. Refusing would break existing runs while buying nothing: an attacker who can replace
    the .pt can replace the .sha256 too."""
    path = _write_checkpoint(tmp_path)

    ckpt_utils.verify_digest(path)  # must not raise

    assert any("no digest beside" in line for line in loguru_lines), "an unverified load must say so"
    assert any("sha256sum" in line for line in loguru_lines), "the warning must name how to record it"


def test_digest_write_is_atomic_leaving_no_tmp_file(tmp_path):
    path = _write_checkpoint(tmp_path)
    ckpt_utils.write_digest(path)

    assert not list(tmp_path.glob("*.tmp")), "a partial write must not survive as a .tmp"


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("empty", ""),
        ("whitespace", "   \n"),
        ("not hex", "not-a-digest  iter_000000100.pt\n"),
        ("truncated", "0" * 32 + "  iter_000000100.pt\n"),
    ],
)
def test_verify_digest_warns_but_proceeds_when_the_sidecar_is_unreadable(tmp_path, loguru_lines, label, content):
    """`sha256sum file > file.sha256` leaves the sidecar empty when sha256sum fails, and the
    recipe this module prints is exactly that. Reading it must not raise, and must not report a
    mismatch — the checkpoint is not what went wrong."""
    path = _write_checkpoint(tmp_path)
    (tmp_path / "iter_000000100.pt.sha256").write_text(content)

    ckpt_utils.verify_digest(path)  # must not raise

    assert any("no readable SHA-256" in line for line in loguru_lines), f"{label} must say so"
    assert not any("does not match" in line for line in loguru_lines), "must not blame the checkpoint"


def test_verify_digest_accepts_an_uppercase_recorded_digest(tmp_path):
    """Hand-written sidecars turn up uppercase; the bytes are what matter, not the casing."""
    path = _write_checkpoint(tmp_path)
    digest = ckpt_utils._file_digest(path)
    (tmp_path / "iter_000000100.pt.sha256").write_text(f"{digest.upper()}  iter_000000100.pt\n")

    ckpt_utils.verify_digest(path)  # must not raise


# --- manifest verification (base + backbone weights) ---------------------------------------------


def _manifest_tree(tmp_path):
    """A checkpoint root with two weights and a sha256sum-format manifest covering both."""
    root = tmp_path / "ckpts"
    (root / "sub").mkdir(parents=True)
    a, b = root / "a.safetensors", root / "sub" / "b.distcp"
    a.write_bytes(b"weight-a")
    b.write_bytes(b"weight-b")
    manifest = tmp_path / "manifest.sha256"
    manifest.write_text(
        f"{hashlib.sha256(b'weight-a').hexdigest()}  a.safetensors\n"
        f"{hashlib.sha256(b'weight-b').hexdigest()}  sub/b.distcp\n"
    )
    return manifest, root, a


def test_verify_manifest_accepts_an_untouched_tree(tmp_path):
    manifest, root, _ = _manifest_tree(tmp_path)
    assert verify_manifest(manifest, root) == []


def test_verify_manifest_reports_a_modified_weight(tmp_path):
    manifest, root, a = _manifest_tree(tmp_path)
    a.write_bytes(b"tampered")
    failures = verify_manifest(manifest, root)
    assert [path for path, _ in failures] == ["a.safetensors"]
    assert "does not match" in failures[0][1]


def test_verify_manifest_reports_a_missing_weight(tmp_path):
    manifest, root, a = _manifest_tree(tmp_path)
    a.unlink()
    failures = verify_manifest(manifest, root)
    assert [path for path, _ in failures] == ["a.safetensors"]
    assert "missing" in failures[0][1]


def test_verify_manifest_reports_every_failure_not_just_the_first(tmp_path):
    manifest, root, a = _manifest_tree(tmp_path)
    a.write_bytes(b"tampered")
    (root / "sub" / "b.distcp").unlink()
    assert len(verify_manifest(manifest, root)) == 2


def test_verify_manifest_rejects_a_malformed_line(tmp_path):
    manifest, root, _ = _manifest_tree(tmp_path)
    manifest.write_text("not-a-digest  a.safetensors\n")
    with pytest.raises(ValueError, match="not a sha256sum"):
        verify_manifest(manifest, root)


def test_verify_manifest_refuses_a_path_escaping_the_root(tmp_path):
    """The manifest is a file like any other; a relative entry must stay under the checkpoint root."""
    manifest, root, _ = _manifest_tree(tmp_path)
    manifest.write_text(f"{hashlib.sha256(b'x').hexdigest()}  ../outside.bin\n")
    with pytest.raises(ValueError, match="outside the checkpoint root"):
        verify_manifest(manifest, root)


def test_verify_manifest_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_manifest(tmp_path / "nope.sha256", tmp_path)


def test_verify_manifest_only_under_limits_what_is_hashed(tmp_path):
    """A run loads one model size, so the load-time check should not hash the other size's shards."""
    manifest, root, a = _manifest_tree(tmp_path)
    a.write_bytes(b"tampered")  # outside the requested subtree
    assert verify_manifest(manifest, root, only_under="sub") == []
    assert [path for path, _ in verify_manifest(manifest, root, only_under="a.safetensors")] == ["a.safetensors"]


@pytest.mark.parametrize("mode", ["text", "binary"])
def test_verify_manifest_reads_what_coreutils_sha256sum_writes(tmp_path, mode):
    """download_checkpoints.sh records manifests with coreutils `sha256sum` and re-checks them with
    `sha256sum -c`. This parser has to accept exactly what that writes, or the two disagree about a
    file `sha256sum -c` would happily verify."""
    sha256sum = shutil.which("sha256sum")
    if sha256sum is None:
        pytest.skip("coreutils sha256sum not available")
    root = tmp_path / "ckpts"
    root.mkdir()
    (root / "w.safetensors").write_bytes(b"weight")
    flag = ["-b"] if mode == "binary" else []
    out = subprocess.run(
        [sha256sum, *flag, "w.safetensors"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    manifest = tmp_path / "m.sha256"
    manifest.write_text(out)

    assert verify_manifest(manifest, root) == []
