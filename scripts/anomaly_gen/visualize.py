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

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_validation_kpi_curve(root, anomaly_types, output_dir="plots", validation_iter=5000, max_iter=100000):
    steps = list(range(0, max_iter + 1, validation_iter))
    kpis = ["cradio_v3_base_fid"]
    exps = [f"{step}" for step in steps]

    curve = {kpi: {anomaly_type: [] for anomaly_type in anomaly_types + ["mean"]} for kpi in kpis}

    for exp in exps:
        csv_file = os.path.join(root, exp, "valid_kpi.csv")
        if not os.path.exists(csv_file):
            print(f"Missing file: {csv_file}")
            for kpi in kpis:
                for anomaly_type in anomaly_types:
                    curve[kpi][anomaly_type].append(float("nan"))
                curve[kpi]["mean"].append(float("nan"))
            continue

        df = pd.read_csv(csv_file)

        for kpi in kpis:
            values = []
            for anomaly_type in anomaly_types:
                try:
                    value = float(df.loc[df["kpi"] == kpi, anomaly_type].values[0])
                except Exception:
                    value = float("nan")
                curve[kpi][anomaly_type].append(value)
                if pd.notna(value):
                    values.append(value)
            # Append mean (excluding NaN)
            mean_value = sum(values) / len(values) if values else float("nan")
            curve[kpi]["mean"].append(mean_value)

    os.makedirs(output_dir, exist_ok=True)
    for kpi in kpis:
        plt.figure(figsize=(10, 6))
        for anomaly_type in anomaly_types + ["mean"]:
            plt.plot(
                steps,
                curve[kpi][anomaly_type],
                label=anomaly_type,
                linewidth=3.0 if anomaly_type == "mean" else 1.2
            )
        plt.xlabel("Training Step")
        plt.ylabel(kpi)
        plt.title(f"{kpi} over Training Steps")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{kpi}_curve.png"))
        plt.close()

    print(f"Done. Plots saved in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot KPI curves over training steps.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing validation KPI CSVs.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save plots.")
    parser.add_argument("--validation_iter", type=int, default=5000, help="Step interval (in iterations) between validation checkpoints.")
    parser.add_argument("--max_iter", type=int, default=100000, help="Final training iteration to include when plotting.")
    parser.add_argument('--anomaly_types', type=str, nargs='+', required=True,
                        help='List of anomalies in the format TEXTURE+TYPE (e.g., SEM_IC+crack)')
    args = parser.parse_args()

    plot_validation_kpi_curve(
        anomaly_types=args.anomaly_types,
        root=args.root,
        output_dir=args.output_dir,
        validation_iter=args.validation_iter,
        max_iter=args.max_iter
    )
