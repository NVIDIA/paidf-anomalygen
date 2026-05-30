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

import argparse
import csv

from cosmos_predict2.metrics.utils import compute_kpi, log_kpi_table, load_real_images, load_generated_images
from imaginaire.utils import log


def main():
    parser = argparse.ArgumentParser(description="Evaluate KPIs between real and generated anomaly images.")
    parser.add_argument('--real_path', type=str, required=True, help='Path to real images dataset')
    parser.add_argument('--generated_path', type=str, required=True, help='Path to generated images')
    parser.add_argument('--backbone', type=str, default="cradio_v3_base", help='Feature Backbone')
    parser.add_argument('--anomaly_types', type=str, nargs='+', required=True,
                        help='List of anomalies in the format TEXTURE+TYPE (e.g., SEM_IC+crack)')
    parser.add_argument('--per_sample_csv', type=str, default=None,
                        help='If set, also write per-sample nn_score / mnn_score to this CSV path.')
    args = parser.parse_args()

    log.init_loguru_file(f"{args.generated_path}/eval_stdout.log")
    anomaly_types =  [entry.split("+") for entry in args.anomaly_types]

    log.info("Loading real images...")
    real_dict = load_real_images(args.real_path, anomaly_types)

    log.info("Loading generated images...")
    gen_dict = load_generated_images(args.generated_path, anomaly_types)

    log.info("Computing KPIs...")
    kpis = compute_kpi(real_dict, gen_dict, backbone_name=args.backbone)
    log_kpi_table(kpis)

    if args.per_sample_csv:
        log.info(f"Writing per-sample correspondence scores → {args.per_sample_csv}")
        with open(args.per_sample_csv, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["anomaly_type", "path", "nn_score", "mnn_score"])
            for anomaly_name in sorted(k for k in kpis if k != "Average"):
                for row in kpis[anomaly_name].get("per_sample", []):
                    writer.writerow([anomaly_name, row["path"], row["nn_score"], row["mnn_score"]])

if __name__ == "__main__":
    main()
