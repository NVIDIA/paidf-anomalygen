# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Automatic-mask-placement CLIs, driving :mod:`anomalygen.auto_mask_placement`.

- ``place.py`` — single/batch AMP over a submask + ROI image(s).
- ``roi_allocate.py`` — preflight-validate the AMP input triple, then distribute
  ``num_sdg`` across defect types → allocation.json (uniform / proportional / explicit
  per-defect counts).
- ``roi_pair.py`` — build the (clean, submask) pairing JSON (+ n_seeds) from a
  dataset and an allocation, i.e. the ``--input_pair_path`` consumed by ``roi_place``.
- ``roi_place.py`` — unified ROI→AMP pipeline that auto-routes samples between
  cad2roi, text2roi, and whole-image ("free") ROI generation.

Full pipeline: ``roi_allocate`` → ``roi_pair`` → ``roi_place``.

Run by module path, e.g.::

    python -m anomalygen.scripts.auto_mask_placement.place -h
"""
