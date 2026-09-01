# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anomaly inpainting training dataset.

Each sample is a two-vision-item image-edit example ``images = [source, target]``, where the
source is the clean image with the defect region substituted by noise — its background dropped to
-1 (black) per-sample with probability ``background_dropout_prob`` — and the target is the clean
image the model learns to produce, plus per-sample ``edit_mask`` and ``anomaly_class_id`` keys.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.data.generator.sequence_packing.modalities import add_special_tokens
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import tokenize_caption
from cosmos_framework.utils import log
from cosmos_framework.utils.lazy_config import instantiate as lazy_instantiate
from PIL import Image

from anomalygen.data.augmentations import (
    RandomHorizontalFlip,
    RandomInstanceDrop,
    RandomOrthogonalRotation,
    RandomRatioCrop,
    RandomRingJitter,
    RandomRotation,
    RandomVerticalFlip,
)
from anomalygen.data.utils import (
    MASK_FG_THRESHOLD,
    build_caption,
    build_source_item,
    list_image_mask_pairs,
    pad_or_truncate,
    resolve_word_token_id,
)


def _to_chw(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [3,H,W] in [-1, 1] (VAE convention)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _to_edit_mask(mask: Image.Image) -> torch.Tensor:
    """PIL 'L' mask -> binary float tensor [1,H,W] (1 inside the defect region)."""
    return (torch.from_numpy(np.asarray(mask, dtype=np.float32)) >= MASK_FG_THRESHOLD).float().unsqueeze(0)


class InpaintingDataset(torch.utils.data.Dataset):
    """Anomaly inpainting dataset (noise-substituted source, optional background dropout).

    On-disk layout (one entry per (texture, defect) class):
        ``{dataset_dir}/{texture}/anomaly_image/{defect}/<name>.png``  +
        ``{dataset_dir}/{texture}/mask/{defect}/<name>_mask.png``
    Override ``_index`` to match a different directory structure.

    Each item: RandomRatioCrop the (image, mask) pair, resize to ``image_size``, build the source
    (defect region noise-substituted; background dropped to -1 (black) per ``background_dropout_prob``)
    and the clean target, and emit them as the two vision items plus the binary ``edit_mask`` and
    integer ``anomaly_class_id``.
    """

    def __init__(
        self,
        dataset_dir: str,
        anomaly_types: Sequence[Sequence[str]],
        image_size: Sequence[int] = (512, 512),
        ratio_range: Sequence[float] = (1.5, 8.0),
        background_dropout_prob: float = 0.5,
        inst_aug_prob: float = 0.5,
        ring_jitter_prob: float = 0.5,
        tokenizer_config=None,
        max_caption_tokens: int = 1024,
        text_prompt_pad_to: int | None = None,
        text_prompt_pad_word: str = "anomaly",
        dataset_name: str = "anomaly",
        seed: int = 1,
        repeat: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_dir = dataset_dir
        self.anomaly_types = [tuple(a) for a in anomaly_types]
        self.image_size = (int(image_size[0]), int(image_size[1]))
        # Per-sample probability for augmentations.RandomInstanceDrop.
        self.inst_aug_prob = float(inst_aug_prob)
        if not 0.0 <= self.inst_aug_prob <= 1.0:
            raise ValueError(f"inst_aug_prob must be in [0, 1], got {self.inst_aug_prob}")
        # Per-sample probability for augmentations.RandomRingJitter, plus its band width in ORIGINAL-image
        # pixels (still an env knob — it is swept far more often than it is configured). Both are
        # validated here so a bad value fails at dataset build, not inside a worker.
        self.ring_jitter_prob = float(ring_jitter_prob)
        if not 0.0 <= self.ring_jitter_prob <= 1.0:
            raise ValueError(f"ring_jitter_prob must be in [0, 1], got {self.ring_jitter_prob}")
        self.ring_jitter_px = int(os.environ.get("ANOMALYGEN_RING_JITTER_PX", "10"))
        if self.ring_jitter_px < 0:
            raise ValueError(f"ANOMALYGEN_RING_JITTER_PX must be >= 0, got {self.ring_jitter_px}")
        # Per-sample prob the source item's background is dropped to -1 (black; the defect region is noise
        # either way); a training-only augmentation, inference never drops the background.
        self.background_dropout_prob = float(background_dropout_prob)
        if not 0.0 <= self.background_dropout_prob <= 1.0:
            raise ValueError(f"background_dropout_prob must be in [0, 1], got {self.background_dropout_prob}")
        self.tokenizer_config = tokenizer_config
        self.max_caption_tokens = int(max_caption_tokens)
        # When learnable text-prompt tuning is enabled, every caption is padded/truncated to a
        # fixed length so each sample contributes the same number of text positions (the packed
        # text block is then a constant size and the per-defect-type learnable prompt scatters cleanly
        # into the first ``text_prompt_pad_to`` rows). None -> native caption length (feature off).
        self.text_prompt_pad_to = None if text_prompt_pad_to is None else int(text_prompt_pad_to)
        self.text_prompt_pad_word = str(text_prompt_pad_word)
        self._pad_token_id = None  # resolved lazily from the (per-worker) tokenizer
        self.dataset_name = dataset_name
        self.repeat = max(1, int(repeat))
        self.ratio_range = tuple(ratio_range)
        self.class_to_id = {f"{t}+{d}": i for i, (t, d) in enumerate(self.anomaly_types)}
        self._base_seed = int(seed)
        self._rng = None  # built lazily per worker (see the ``rng`` property)
        # built lazily per worker (binds the per-worker rng; see the ``augmentations`` property)
        self._augmentations = None
        self._tokenizer = None  # built lazily per worker (avoids pickling across workers)
        self.samples: List[Dict] = self._index()

        # Fail fast: an empty dataset would make the data loader hang with no batch emitted.
        if not self.samples:
            raise FileNotFoundError(
                f"InpaintingDataset found 0 samples under dataset_dir={self.dataset_dir!r} "
                f"(resolved from cwd={os.getcwd()!r}) for anomaly_types={self.anomaly_types}. "
                f"Expected {{dataset_dir}}/{{texture}}/anomaly_image/{{defect}}/*.png + "
                f"{{texture}}/mask/{{defect}}/*_mask.png. Use an ABSOLUTE DATASET_PATH."
            )

        # Group sample indices by class for class-balanced sampling in ``__getitem__``: draw a class
        # uniformly, then a sample within it, giving P(sample) = 1/(num_classes * class_count). This
        # matches a WeightedRandomSampler with weight 1/class_count, done here because the framework
        # builds the DataLoader and takes no external sampler. Fixes class imbalance and, by ignoring
        # the sequential index, the lack of shuffling.
        _pools: Dict[str, List[int]] = defaultdict(list)
        for i, rec in enumerate(self.samples):
            _pools[rec["class_name"]].append(i)
        self._class_pools: List[np.ndarray] = [np.asarray(v, dtype=np.int64) for v in _pools.values()]

        log.info(
            f"InpaintingDataset: {len(self.samples)} samples across {len(self._class_pools)} populated "
            f"classes (virtual length {len(self.samples) * self.repeat} with repeat={self.repeat}); "
            f"class-balanced sampling "
            f"(per-class counts: {{{', '.join(f'{c!r}: {len(p)}' for c, p in zip(_pools, self._class_pools))}}})."
        )

    @property
    def rng(self) -> np.random.Generator:
        """Per-worker NumPy generator, seeded from the experiment seed.

        Built lazily on first access so it is seeded *inside* each dataloader worker (after
        fork) rather than once in the parent. Mixing the base seed with the shard rank and
        worker id yields a stream that is reproducible from the experiment seed yet independent
        across ranks and workers. ``shard_rank`` is set by RankPartitionedDataLoader after
        construction; absent (e.g. num_workers=0, single process) it defaults to 0.
        """
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker_id = 0 if info is None else info.id
            shard_rank = getattr(self, "shard_rank", 0)
            self._rng = np.random.default_rng((self._base_seed, shard_rank, worker_id))
        return self._rng

    @property
    def augmentations(self):
        """The read-time augmentation pipeline, in application order.

        Built lazily so every stage binds THIS worker's ``rng`` (see ``rng``), and shares one
        generator so the draw sequence is deterministic per worker. Order matters: ring jitter cuts
        its bands against the full mask, so it must precede the instance drop; both run before any
        geometry so their pixels are flipped and cropped like ordinary content. The crop MUST stay
        last — ``__getitem__`` peels it off to re-draw it on an empty mask.

        Stages left at ``p=0`` cost nothing and draw no randomness (see ``augmentations._fires``), so
        the disabled rotation does not perturb any other stage's draws.
        """
        if self._augmentations is None:
            rng = self.rng
            self._augmentations = [
                RandomRingJitter(self.ring_jitter_prob, self.ring_jitter_px, rng=rng),
                RandomInstanceDrop(self.inst_aug_prob, self.ring_jitter_px, rng=rng),
                RandomVerticalFlip(0.5, rng=rng),
                RandomHorizontalFlip(0.5, rng=rng),
                RandomOrthogonalRotation(rng=rng),
                # Off (p=0) — listed so the stage is discoverable and enabling it is a one-word edit.
                # It must sit immediately before the crop, which trims the black corners it leaves.
                RandomRotation(max_angle=20, p=0.0, rng=rng),
                RandomRatioCrop(final_crop_size=self.image_size[0], ratio_range=self.ratio_range, rng=rng),
            ]
        return self._augmentations

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            proc = lazy_instantiate(self.tokenizer_config)
            tok = getattr(proc, "tokenizer", proc)
            tok, _ = add_special_tokens(tok)
            self._tokenizer = tok
        return self._tokenizer

    @property
    def pad_token_id(self) -> int:
        """Token id used to pad the learnable text-prompt region (the ``"anomaly"`` word)."""
        if self._pad_token_id is None:
            self._pad_token_id = resolve_word_token_id(self.tokenizer, self.text_prompt_pad_word)
        return self._pad_token_id

    def _tokenize(self, caption: str) -> torch.Tensor:
        ids = tokenize_caption(caption, self.tokenizer, is_video=False, use_system_prompt=False)
        ids = ids[: self.max_caption_tokens]
        if self.text_prompt_pad_to is not None:
            ids = pad_or_truncate(ids, self.text_prompt_pad_to, self.pad_token_id)
        return torch.tensor(ids, dtype=torch.long)

    def _index(self) -> List[Dict]:
        samples: List[Dict] = []
        n_missing_mask = 0
        for texture, defect in self.anomaly_types:
            img_dir = os.path.join(self.dataset_dir, texture, "anomaly_image", defect)
            mask_dir = os.path.join(self.dataset_dir, texture, "mask", defect)
            if not os.path.isdir(img_dir):
                log.warning(f"InpaintingDataset: missing image dir {img_dir}")
                continue

            for img_fn, mask_fn in list_image_mask_pairs(img_dir, mask_dir, mask_suffix="_mask"):
                if not os.path.exists(mask_fn):
                    log.warning(f"InpaintingDataset: no mask for {img_fn} (looked for {mask_fn}); skipping.")
                    n_missing_mask += 1
                    continue

                samples.append(
                    {
                        "image": img_fn,
                        "mask": mask_fn,
                        "texture": texture,
                        "defect": defect,
                        "class_name": f"{texture}+{defect}",
                    }
                )

        if n_missing_mask:
            log.warning(f"InpaintingDataset: skipped {n_missing_mask} image(s) with no matching mask on disk.")

        return samples

    def __len__(self) -> int:
        # Virtual length: real samples cycled ``repeat`` times, so the loader's iterator
        # does not exhaust mid-run. __getitem__ maps the index back into the real samples.
        return len(self.samples) * self.repeat

    def __getitem__(self, idx: int) -> Dict:
        # Class-balanced draw: uniform class, then uniform sample within it. ``idx`` (the framework's
        # sequential counter) is ignored so items are shuffled and every class is equally likely.
        pool = self._class_pools[int(self.rng.integers(len(self._class_pools)))]
        rec = self.samples[int(pool[int(self.rng.integers(len(pool)))])]
        image = Image.open(rec["image"]).convert("RGB")
        mask = Image.open(rec["mask"]).convert("L")

        # Read-time augmentation, in pipeline order (see the ``augmentations`` property): ring
        # jitter, instance drop, V/H flip, quarter turn, free rotation (off), zoom-crop. Photometric
        # stages run before any geometry so their pixels are flipped and cropped like ordinary content.
        # The crop is held back so the empty-mask retry below can re-draw it from the pre-crop pair.
        *pre_crop, crop = self.augmentations
        for aug in pre_crop:
            image, mask = aug(image, mask)
        aug_image, aug_mask = crop(image, mask)
        aug_image = aug_image.resize(self.image_size, Image.Resampling.BICUBIC)
        aug_mask = aug_mask.resize(self.image_size, Image.Resampling.NEAREST)
        edit = _to_edit_mask(aug_mask)  # [1,H,W]

        # A defect can end up absent from the crop — with the small-angle rotation disabled the
        # only way in is an all-zero mask ON DISK, which re-cropping cannot recover (the crop
        # early-returns on an empty mask), but the guard is kept for when rotation is enabled.
        if float(edit.sum()) == 0.0:
            aug_image, aug_mask = crop(image, mask)
            aug_image = aug_image.resize(self.image_size, Image.Resampling.BICUBIC)
            aug_mask = aug_mask.resize(self.image_size, Image.Resampling.NEAREST)
            edit = _to_edit_mask(aug_mask)

        image = aug_image
        target = _to_chw(image)  # [3,H,W] in [-1,1]  (photometric already applied at read-time)

        # Source item: defect region noise-substituted, background dropped to -1 (black) with prob
        # background_dropout_prob. Guard so prob=0 draws no rng, keeping the augmentation stream
        # identical to background-preserving-only.
        m3 = edit.expand_as(target)
        drop_background = self.background_dropout_prob > 0.0 and self.rng.random() < self.background_dropout_prob
        source = build_source_item(target, m3, drop_background)

        caption = build_caption(defect=rec["defect"], texture=rec["texture"])
        h, w = self.image_size

        return {
            "images": [source, target],  # list of two [3,H,W] tensors → two vision items
            "text_token_ids": self._tokenize(caption),
            "ai_caption": caption,
            "image_size": [
                torch.tensor([h, w, h, w], dtype=torch.float32),
                torch.tensor([h, w, h, w], dtype=torch.float32),
            ],
            "fps": 30.0,
            "num_frames": 2,
            "sequence_plan": SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=[]),
            "edit_mask": edit,  # [1,H,W]
            "anomaly_class_id": int(self.class_to_id[rec["class_name"]]),
            "dataset_name": self.dataset_name,
        }


def get_inpainting_dataset(**kwargs) -> InpaintingDataset:
    """LazyCall target used by the experiment config."""
    return InpaintingDataset(**kwargs)


class _EmptyValidationDataset(torch.utils.data.Dataset):
    """Zero-length dataset, so the validation loop is a no-op (KPI is done in a callback)."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx):  # pragma: no cover - never called for a 0-length dataset
        raise IndexError(idx)


def get_empty_val_dataloader() -> torch.utils.data.DataLoader:
    """LazyCall target for ``dataloader_val`` — an iterable that yields nothing."""
    return torch.utils.data.DataLoader(_EmptyValidationDataset(), batch_size=1, num_workers=0)
