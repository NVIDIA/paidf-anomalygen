#!/usr/bin/env bash
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
# Sanity-check an SDG output directory against its input JSONL.
#
# Checks:
#   - SDG_result.csv exists and has the expected row count
#   - reconstructed_image/ count matches JSONL entry count
#
# Usage: scripts/verify_output.sh <input_jsonl> <output_dir>
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <input_jsonl> <output_dir>" >&2
    exit 2
fi

input_jsonl="$1"
output_dir="$2"

if [[ ! -f "$input_jsonl" ]]; then
    echo "error: input JSONL not found: $input_jsonl" >&2
    exit 1
fi
# `|| true`: grep -c exits 1 on zero matches, which would silently kill the
# script under `set -e` before any diagnostic prints.
expected=$(grep -c '[^[:space:]]' "$input_jsonl" || true)
if [[ -z "$expected" ]]; then
    # grep exit 2 (e.g. unreadable file) prints nothing to stdout.
    echo "error: cannot read $input_jsonl" >&2
    exit 1
fi
if (( expected == 0 )); then
    echo "error: $input_jsonl contains no non-empty lines — nothing to verify" >&2
    exit 1
fi
csv="$output_dir/SDG_result.csv"

if [[ ! -f "$csv" ]]; then
    echo "error: $csv not found — SDG did not finish?" >&2
    exit 1
fi
csv_rows=$(($(wc -l < "$csv") - 1))   # minus header

recon_dir="$output_dir/reconstructed_image"
if [[ ! -d "$recon_dir" ]]; then
    echo "error: $recon_dir not found" >&2
    exit 1
fi
image_count=$(find "$recon_dir" -type f \( -name '*.png' -o -name '*.jpg' \) | wc -l)

printf "expected: %d\ncsv rows: %d\nimages  : %d\n" \
    "$expected" "$csv_rows" "$image_count"

if (( csv_rows != expected )) || (( image_count != expected )); then
    echo "error: count mismatch. SDG may have been interrupted;" >&2
    echo "       nn_score eval on this output will be unreliable." >&2
    exit 1
fi

echo "OK: counts match"
