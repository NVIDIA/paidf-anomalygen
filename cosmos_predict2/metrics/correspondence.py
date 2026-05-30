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

"""
Defect correspondence scoring using frozen DINOv2 ViT-L/14 patch features.

For each generated image, we extract patch-level features from only the masked
defect region, then measure how well those features match real defect patches of
the same anomaly type via nearest-neighbour cosine similarity.

Two complementary scores are produced:
  - nn_score  : mean best-match similarity for every generated patch   (higher = better)
  - mnn_score : same, but restricted to mutual nearest-neighbour pairs (stricter, higher = better)

These are added as extra rows in valid_kpi.csv alongside FID.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from transformers import AutoModel

from imaginaire.utils import log


# ── Backbone constants (DINOv2 ViT-L/14) ─────────────────────────────────────
BackboneSpec = {
    "facebook/dinov2-large": {
        "patch_size": 14,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "checkpoints/facebook/dinov2-large": {
        "patch_size": 14,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    }
}
DEFAULT_BACKBONE = "checkpoints/facebook/dinov2-large"

# Module-level cache: {model_id: model} — models live on GPU.
_model_cache: dict = {}


def _get_model(model_id: str = DEFAULT_BACKBONE) -> torch.nn.Module:
    """Load backbone once per model_id on GPU; raises RuntimeError if unavailable."""
    if model_id not in _model_cache:
        log.info(f"Loading {model_id} for correspondence scoring (auto-downloads if not cached)...")
        try:
            model = AutoModel.from_pretrained(model_id, device_map="cuda")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load correspondence backbone ({model_id}). "
                "Please follow the instruction (`scripts/download_checkpoints.py`) in the tutorial notebook to "
                "pre-download the checkpoints."
            ) from exc

        if "facebook/dinov2-large" in model_id:
            log.info(f"torch.compile: {model_id}")
            model = torch.compile(model, mode="default", fullgraph=False)

        model.eval()
        _model_cache[model_id] = model
        log.info(f"{model_id} loaded successfully.")

    return _model_cache[model_id]


def prefetch_model(model_id: str = DEFAULT_BACKBONE) -> None:
    """
    Eagerly load the backbone so any weight-unavailability error surfaces at
    startup rather than mid-training on the first validation call.
    """
    _get_model(model_id)


# ── Tensor preparation ────────────────────────────────────────────────────────


def _image_mask_to_tensor(
    image_arr: np.ndarray,
    mask_arr: np.ndarray,
    min_size: int = 224,
    max_size: int = 518 * 2,
    patch_size: int = 14,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """float32 numpy (H×W×3, 0-1) → normalised [3, H, W] tensor."""
    image = Image.fromarray((image_arr * 255).astype(np.uint8)).convert("RGB")
    mask = Image.fromarray((mask_arr * 255).astype(np.uint8)).convert("L")
    height, width = image.height, image.width
    if height < min_size or width < min_size:
        # Upscale so the short side reaches min_size while preserving aspect ratio.
        scale = min_size / min(height, width)
        height = int(round(height * scale))
        width = int(round(width * scale))
    if height > max_size or width > max_size:
        # Downscale so the long side fits in max_size while preserving aspect ratio.
        scale = max_size / max(height, width)
        height = int(round(height * scale))
        width = int(round(width * scale))
    # Snap to a multiple of the patch size so the strided ViT conv doesn't drop pixels.
    # Ref: https://github.com/facebookresearch/dinov2/issues/86
    new_height = height - height % patch_size
    new_width = width - width % patch_size
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
    return (
        TF.normalize((TF.to_tensor(image)), mean=mean, std=std),  # [3, H, W]
        (TF.to_tensor(mask)[0] > 0.5).float(),  # [H, W]
        new_height,
        new_width,
    )


# ── Patch-level operations ────────────────────────────────────────────────────

def _patch_grid_mask(
    mask_tensor: torch.Tensor,
    height_patch: int,
    width_patch: int,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Project a binary pixel mask [H, W] onto the ViT patch grid [H_p, W_p].
    Uses max-pooling: a patch is active if any pixel inside it is active.
    """
    # Crop to the exact pixel area the ViT conv covers (handles non-divisible sizes).
    patch_view = mask_tensor[
        : height_patch * patch_size, : width_patch * patch_size
    ].reshape(height_patch, patch_size, width_patch, patch_size)
    return patch_view.amax(dim=(1, 3))   # [H_p, W_p]


def _extract_defect_features(
    model:     torch.nn.Module,
    image_arr: np.ndarray,   # H×W×3, float32, 0-1
    mask_arr:  np.ndarray,   # H×W,   float32, 0-1
    patch_size: int = 14,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor | None:
    """
    Return patch features for pixels inside the defect mask, shape [N_defect, D].
    Returns None if the mask is empty after downsampling to patch resolution.
    """
    x, mask_tensor, new_height, new_width = _image_mask_to_tensor(
        image_arr, mask_arr, patch_size=patch_size, mean=mean, std=std
    )
    x = x.unsqueeze(0).to("cuda") # [1, 3, H, W]
    mask_tensor = mask_tensor.to("cuda") # [H, W]

    with torch.no_grad():
        out = model(pixel_values=x)
        patch_tokens = out.last_hidden_state[:, 1:, :].squeeze(0)   # [N, D]

    H_p = new_height // patch_size
    W_p = new_width // patch_size
    grid = patch_tokens.reshape(H_p, W_p, -1)                       # [H_p, W_p, D]

    mask_at_patch_res = _patch_grid_mask(mask_tensor, H_p, W_p, patch_size)
    defect_mask = mask_at_patch_res > 0.5                           # [H_p, W_p]
    if not defect_mask.any():
        return None

    # Move back to CPU to avoid massive memory usage for storing the features.
    return grid[defect_mask].to("cpu")                              # [N_defect, D]

# ── Correspondence scores ─────────────────────────────────────────────────────

def _nn_mnn_scores(f_g: torch.Tensor, f_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """NN and MNN cosine similarity scores between *L2-normalised* features.

    The caller is responsible for normalising the inputs (so per-gen and
    per-real normalisation happens once, not on every gen×real pair).

    Returns 0-D GPU tensors so the caller can batch device→host syncs:
      - NN  : mean best-match similarity, gen → real direction.
      - MNN : mean similarity of mutual nearest-neighbour pairs only;
              0.0 (as a tensor) when no mutual pair exists.
    """
    sim = f_g @ f_r.T                                # [N, M]

    # NN: best real match per gen patch (values + indices in one kernel).
    nn_values, nn_g2r = sim.max(dim=1)               # [N], [N]
    nn = nn_values.mean()

    # MNN: reuse sim; check mutual pairs without a host-side branch.
    nn_r2g = sim.argmax(dim=0)                       # [M]
    idx    = torch.arange(f_g.shape[0], device=sim.device)
    mutual = nn_r2g[nn_g2r] == idx                   # [N], bool
    sim_diag = sim[idx, nn_g2r]                      # [N]
    mut_f = mutual.float()
    # safe mean → 0 when there are no mutual pairs (sum is 0 too).
    mnn = (sim_diag * mut_f).sum() / mut_f.sum().clamp(min=1)
    return nn, mnn


# ── Public API ────────────────────────────────────────────────────────────────

def compute_correspondence_kpi(
    real_images_dict:      dict,
    generated_images_dict: dict,
    backbone:              str = DEFAULT_BACKBONE,
    top_k:                 int = 3,
) -> dict:
    """
    Compute nn_score and mnn_score for each anomaly type and their macro average.

    Scoring strategy (pairwise + top-K):
        - For every generated sample, compute (nn, mnn) against each real reference
          of the same anomaly type individually.
        - Sort those pair scores by nn descending; take the top-K pairs; mean their
          nn and mnn separately to get the generated sample's (nn_score, mnn_score).
        - Per anomaly type, mean across all generated samples.
        - "Average" is the macro-mean across anomaly types.

    Args:
        backbone: HuggingFace model ID for the feature extractor.
                  Defaults to 'facebook/dinov2-large' (ViT-L/14).
        top_k:    Number of best-matching real references to average over per
                  generated sample. -1 (or a value ≥ the number of available
                  references) means use all pairs.

    Inputs use the same dict structure as compute_kpi() in metrics/utils.py:
        real_images_dict[anomaly_name]["original_image"]  — list of numpy (H×W×3, 0-1)
        real_images_dict[anomaly_name]["original_mask"]   — list of numpy (H×W, 0-1)
        generated_images_dict[anomaly_name]["reconstructed_image"] — same format
        generated_images_dict[anomaly_name]["original_mask"]       — inpainting-input masks

    Returns a dict mirroring compute_kpi output:
        {anomaly_name: {"nn_score": float, "mnn_score": float, "per_sample": [...]},
         "Average": {"nn_score": float, "mnn_score": float}}

    `per_sample` is always populated for every anomaly type — one row per
    generated sample with `path`, `nn_score`, `mnn_score`. Samples that fail
    feature extraction are recorded with NaN scores.

    All anomaly types present in real_images_dict are guaranteed to appear in the
    result (using float("nan") for any type where scoring failed), so that the
    valid_kpi.csv writer can iterate safely.
    """
    if top_k == 0 or top_k < -1:
        raise ValueError(
            f"top_k must be -1 (use all) or a positive integer, got {top_k}."
        )

    model = _get_model(model_id=backbone)
    if backbone not in BackboneSpec:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Available: {list(BackboneSpec.keys())}"
        )
    patch_size = BackboneSpec[backbone]["patch_size"]
    mean = BackboneSpec[backbone]["mean"]
    std = BackboneSpec[backbone]["std"]

    result: dict = {}
    nn_vals, mnn_vals = [], []

    for anomaly_name in sorted(real_images_dict.keys()):
        real_images = real_images_dict[anomaly_name].get("original_image", [])
        real_masks  = real_images_dict[anomaly_name].get("original_mask",  [])
        gen_images  = generated_images_dict[anomaly_name].get("reconstructed_image", [])
        gen_masks   = generated_images_dict[anomaly_name].get("original_mask")
        if gen_masks is None:
            # Each gen image is paired with the input mask used for its inpainting;
            # falling back to real_masks would mis-pair gen features with unrelated masks.
            log.warning(
                f"[{anomaly_name}] generated_images_dict has no 'original_mask' key — "
                "skipping correspondence score."
            )
            result[anomaly_name] = {"nn_score": float("nan"), "mnn_score": float("nan"), "per_sample": []}
            continue

        # Extract per-reference real features once and pre-normalise; reused
        # across every generated sample of this anomaly type.
        real_feats_list = []
        for img, mask in zip(real_images, real_masks):
            feats = _extract_defect_features(
                model, img, mask, patch_size=patch_size, mean=mean, std=std
            )
            if feats is not None:
                real_feats_list.append(F.normalize(feats, dim=-1))

        if not real_feats_list:
            log.warning(f"[{anomaly_name}] No real defect patches — skipping correspondence score.")
            result[anomaly_name] = {"nn_score": float("nan"), "mnn_score": float("nan"), "per_sample": []}
            continue

        # Resolve top-K once per anomaly type (number of refs is fixed for this loop).
        n_refs = len(real_feats_list)
        effective_k = n_refs if top_k == -1 or top_k >= n_refs else top_k

        # Score each gen image against every real reference, then take top-K by NN.
        gen_paths = generated_images_dict[anomaly_name].get("img_path", [None] * len(gen_images))
        type_nn, type_mnn, per_sample = [], [], []
        for path, img, mask in zip(gen_paths, gen_images, gen_masks):
            feats_gen = _extract_defect_features(
                model, img, mask, patch_size=patch_size, mean=mean, std=std
            )
            if feats_gen is None:
                per_sample.append({"path": path, "nn_score": float("nan"), "mnn_score": float("nan")})
                continue
            f_g = F.normalize(feats_gen, dim=-1)   # normalise once per gen image

            # Collect per-pair scores as GPU tensors; defer the host sync until
            # after the top-K mean so we sync twice per gen image, not 2×n_refs.
            nn_per_ref, mnn_per_ref = [], []
            for f_r in real_feats_list:
                nn, mnn = _nn_mnn_scores(f_g, f_r)
                nn_per_ref.append(nn)
                mnn_per_ref.append(mnn)
            nn_t  = torch.stack(nn_per_ref)    # [n_refs]
            mnn_t = torch.stack(mnn_per_ref)   # [n_refs]

            # Sort by NN desc, average both metrics over the same top-K refs.
            topk_nn, topk_idx = nn_t.topk(effective_k)
            nn_s  = topk_nn.mean().item()
            mnn_s = mnn_t[topk_idx].mean().item()
            type_nn.append(nn_s)
            type_mnn.append(mnn_s)
            per_sample.append({"path": path, "nn_score": nn_s, "mnn_score": mnn_s})

        if not type_nn:
            log.warning(f"[{anomaly_name}] No generated defect patches — skipping correspondence score.")
            result[anomaly_name] = {"nn_score": float("nan"), "mnn_score": float("nan"), "per_sample": per_sample}
            continue

        result[anomaly_name] = {
            "nn_score":  float(np.mean(type_nn)),
            "mnn_score": float(np.mean(type_mnn)),
            "per_sample": per_sample,
        }
        nn_vals.append(result[anomaly_name]["nn_score"])
        mnn_vals.append(result[anomaly_name]["mnn_score"])
        log.info(
            f"[{anomaly_name}] nn_score={result[anomaly_name]['nn_score']:.4f}  "
            f"mnn_score={result[anomaly_name]['mnn_score']:.4f}  "
            f"(top_k={effective_k}/{n_refs})"
        )

    if nn_vals:
        result["Average"] = {
            "nn_score":  float(np.mean(nn_vals)),
            "mnn_score": float(np.mean(mnn_vals)),
        }

    return result
