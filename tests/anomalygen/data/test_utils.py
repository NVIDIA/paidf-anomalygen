# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for anomalygen.data.utils (caption/tokenizer/path helpers and the source item)."""

import pytest
import torch

from anomalygen.data.utils import (
    CAPTION_TEMPLATE,
    build_caption,
    build_source_item,
    list_image_mask_pairs,
    pad_or_truncate,
    resolve_word_token_id,
    validate_anomaly_type,
)


@pytest.mark.parametrize(
    "anomaly_type",
    ["wood+scratch", "metal_surface+MT_Blowhole", "passive_component+excess_solder", "Phone+oil", "IC+bridge"],
)
def test_validate_anomaly_type_accepts_the_real_recipe_values(anomaly_type):
    assert validate_anomaly_type(anomaly_type) == anomaly_type


@pytest.mark.parametrize(
    "anomaly_type",
    [
        "../escape+x",
        "wood+../../escape",
        "a/b+c",
        "wood+scratch/../..",
        "..+..",
        "wood+scratch\x00",
    ],
)
def test_validate_anomaly_type_rejects_a_traversing_value(anomaly_type):
    with pytest.raises(ValueError, match="anomaly type"):
        validate_anomaly_type(anomaly_type)


def test_validate_anomaly_type_rejects_an_absolute_value():
    # pathlib.Path(base, seg) DISCARDS base when seg is absolute — the worst case.
    with pytest.raises(ValueError, match="anomaly type"):
        validate_anomaly_type("/tmp/escape+x")


@pytest.mark.parametrize("anomaly_type", ["", "scratch", "+scratch", "wood+", "wood+a+b"])
def test_validate_anomaly_type_rejects_a_malformed_key(anomaly_type):
    with pytest.raises(ValueError, match="anomaly type"):
        validate_anomaly_type(anomaly_type)


def test_validate_anomaly_type_names_the_field_it_rejected():
    with pytest.raises(ValueError, match="defect_type"):
        validate_anomaly_type("../escape+x", field="defect_type")


def test_build_caption_normalises_separators():
    caption = build_caption(defect="missing-solder", texture="pcb_board")
    assert "missing solder" in caption
    assert "pcb board" in caption
    assert caption == CAPTION_TEMPLATE.format(defect="missing solder", texture="pcb board")


class _FakeTokenizer:
    """Minimal tokenizer stub exercising both resolve_word_token_id branches."""

    def __init__(self, vocab, encode_map, unk_token_id=0):
        self._vocab = vocab
        self._encode_map = encode_map
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)

    def encode(self, word, add_special_tokens=False):
        return self._encode_map.get(word, [])


def test_resolve_word_token_id_direct_vocab_hit():
    tok = _FakeTokenizer(vocab={"anomaly": 42}, encode_map={}, unk_token_id=0)
    assert resolve_word_token_id(tok, "anomaly") == 42


def test_resolve_word_token_id_falls_back_to_last_piece():
    # Not a standalone vocab entry (maps to unk) -> use the last BPE piece.
    tok = _FakeTokenizer(vocab={}, encode_map={"anomaly": [11, 12, 13]}, unk_token_id=0)
    assert resolve_word_token_id(tok, "anomaly") == 13


def test_resolve_word_token_id_unresolvable_raises():
    tok = _FakeTokenizer(vocab={}, encode_map={"anomaly": []}, unk_token_id=0)
    with pytest.raises(ValueError):
        resolve_word_token_id(tok, "anomaly")


def test_pad_or_truncate_pads_and_truncates():
    assert pad_or_truncate([1, 2], length=4, pad_token_id=9) == [1, 2, 9, 9]
    assert pad_or_truncate([1, 2, 3, 4, 5], length=3, pad_token_id=9) == [1, 2, 3]
    assert pad_or_truncate([1, 2, 3], length=3, pad_token_id=9) == [1, 2, 3]


def _touch(path, data=b"x"):
    path.write_bytes(data)


def test_list_image_mask_pairs_missing_dir_returns_empty():
    assert list_image_mask_pairs("/definitely/not/a/dir", "/nope") == []


def test_list_image_mask_pairs_pairs_and_fallback(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Two images (sorted), plus a non-image that must be ignored.
    _touch(img_dir / "b.jpg")
    _touch(img_dir / "a.png")
    _touch(img_dir / "notes.txt")

    # a: same-extension mask exists; b: only the .png fallback exists.
    _touch(mask_dir / "a.png")
    _touch(mask_dir / "b.png")

    pairs = list_image_mask_pairs(str(img_dir), str(mask_dir))

    assert [p[0] for p in pairs] == [str(img_dir / "a.png"), str(img_dir / "b.jpg")]
    assert pairs[0][1] == str(mask_dir / "a.png")
    assert pairs[1][1] == str(mask_dir / "b.png")  # jpg image -> .png mask fallback


def test_list_image_mask_pairs_returns_missing_mask_path(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    _touch(img_dir / "x.png")  # no corresponding mask on disk

    pairs = list_image_mask_pairs(str(img_dir), str(mask_dir))
    assert len(pairs) == 1
    # Path is returned even though the mask does not exist (falls back to .png form).
    assert pairs[0][1] == str(mask_dir / "x.png")


def test_list_image_mask_pairs_honours_suffix(tmp_path):
    img_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    _touch(img_dir / "a.png")
    _touch(mask_dir / "a_mask.png")

    pairs = list_image_mask_pairs(str(img_dir), str(mask_dir), mask_suffix="_mask")
    assert pairs[0][1] == str(mask_dir / "a_mask.png")


# --- build_source_item --------------------------------------------------------------------------


def _base_and_mask():
    """A CPU ``([3,1,H,W] base, stride-0 expanded m3)`` pair shaped exactly as inpaint.py builds it."""
    base = torch.linspace(-1.0, 1.0, 3 * 1 * 4 * 4).reshape(3, 1, 4, 4)
    m = torch.zeros(4, 4)
    m[1:3, 1:3] = 1.0
    m3 = m[None, None].expand(base.shape)  # stride-0 view, not contiguous
    return base, m3


def test_build_source_item_keeps_background_and_noises_defect():
    base, m3 = _base_and_mask()
    clone = base.clone()

    out = build_source_item(base, m3, seed=7)

    inside = m3 > 0
    assert torch.equal(out[~inside], base[~inside])  # background is bit-exact clean pixels
    assert (out[inside] >= -1.0).all() and (out[inside] <= 1.0).all()
    assert not torch.any(out[inside] == -1.0)  # noise, not the dropout constant
    assert torch.equal(base, clone)  # base never mutated


def test_build_source_item_background_dropout_blacks_out_background():
    base, m3 = _base_and_mask()
    clone = base.clone()

    out = build_source_item(base, m3, True, seed=7)

    inside = m3 > 0
    assert torch.all(out[~inside] == -1.0)  # background dropped to black
    assert (out[inside] >= -1.0).all() and (out[inside] <= 1.0).all()
    assert not torch.any(out[inside] == -1.0)
    assert torch.equal(base, clone)


def test_build_source_item_same_seed_is_bit_identical():
    base, m3 = _base_and_mask()
    assert torch.equal(build_source_item(base, m3, seed=123), build_source_item(base, m3, seed=123))


def test_build_source_item_different_seed_gives_different_noise():
    base, m3 = _base_and_mask()
    a = build_source_item(base, m3, seed=123)
    b = build_source_item(base, m3, seed=124)
    assert not torch.equal(a, b)


def test_build_source_item_without_seed_stays_stochastic():
    """The training path passes no seed and must draw fresh noise on every call."""
    base, m3 = _base_and_mask()
    assert not torch.equal(build_source_item(base, m3), build_source_item(base, m3))


def test_build_source_item_seed_does_not_advance_global_rng():
    """A seeded draw uses a private generator, so it must not perturb the global RNG stream."""
    base, m3 = _base_and_mask()
    torch.manual_seed(0)
    expected = torch.rand(4)

    torch.manual_seed(0)
    build_source_item(base, m3, seed=99)
    assert torch.equal(torch.rand(4), expected)
