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

"""Diagnostic plot for training validation metrics.

Plots FID, NN, MNN trajectories on a twin-y-axis figure (sky / green / red
palette shared with the UC1 report):

  * ``triggered=True`` — the early-stop mechanism fired. The criteria metric
    is drawn thicker, best-star + stop-X markers are added, a patience window
    is shaded, and when the skipped tail is ≥ 20% of ``max_iter_planned`` the
    x-axis is split with a broken-axis cue so the trained range keeps most of
    the horizontal real estate (the freed strip on the right hosts the
    original ``max_iter`` marker plus the legend).
  * ``triggered=False`` — training ran to completion or early stop was off.
    Only the three metric curves are drawn; no criteria highlight, no
    best / stop markers, no broken-axis — the caller does not need to supply
    ``criteria`` or any best / stop fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence


import matplotlib.pyplot as plt

from cosmos_predict2.utils.metric_specs import METRIC_SPECS


# Metric → direction of improvement. Derived from METRIC_SPECS so adding
# a new metric is a one-place change in metric_specs.py.
_MODE_BY_METRIC = {metric: mode for metric, (_, mode) in METRIC_SPECS.items()}

# Palette shared with generate_uc1_report (sky / green / red).
_COLORS = {
    "nn":  "#38bdf8",
    "mnn": "#34d399",
    "fid": "#f87171",
}


def _display_name(metric: str) -> str:
    """Pretty-print metric key: upper-case all three."""
    return metric.upper()


def plot_early_stop(
    iters: Sequence[int],
    metrics: Dict[str, Sequence[float]],  # keys in {"fid","mnn","nn"}; each len(iters)
    *,
    max_iter_planned: int,
    out_path: Path,
    triggered: bool = False,
    criteria: Optional[str] = None,
    patience: Optional[int] = None,
    best_iteration: Optional[int] = None,
    stop_iteration: Optional[int] = None,
    last_improved_iteration: Optional[int] = None,
    min_delta: Optional[float] = None,
    min_delta_mode: Optional[str] = None,
    cumulative_delta: Optional[bool] = None,
) -> Path:
    """Render metric trajectories, with early-stop decoration when triggered.

    The criteria curve is the single source of truth: best/stop markers,
    axhline, and the no-improve band are all positioned by looking up
    `metrics[criteria]` at `best_iteration` / `stop_iteration` — the caller
    does not pass a separate y-value, removing one chance for inconsistency.
    """
    iters = list(iters)
    metrics = {k: list(v) for k, v in metrics.items()}
    if not iters:
        raise ValueError("iters must contain at least one validation iteration")

    if triggered:
        required = {
            "criteria": criteria, "patience": patience,
            "best_iteration": best_iteration,
            "stop_iteration": stop_iteration,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            raise ValueError(
                f"triggered=True requires non-None: {', '.join(missing)}"
            )
        if criteria not in metrics:
            raise ValueError(
                f"criteria '{criteria}' not in metrics keys {list(metrics)}"
            )
        if best_iteration not in iters:
            raise ValueError(
                f"best_iteration={best_iteration} is not in iters; the plot "
                "looks up the criteria curve's y-value at that point and "
                "expects it to be one of the recorded validations."
            )
        if stop_iteration not in iters:
            raise ValueError(
                f"stop_iteration={stop_iteration} is not in iters; same "
                "contract as best_iteration above."
            )

    # Broken-axis mode only when early stop fired far from max_iter.
    skipped = max_iter_planned - stop_iteration if triggered else 0
    use_break = (
        triggered
        and stop_iteration > iters[0]
        and skipped / max_iter_planned >= 0.2
    )

    # --- figure + axes ---
    if use_break:
        fig = plt.figure(figsize=(14.0, 5.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 2.0], wspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1r = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax2 = ax1.twinx()
        ax2r = ax1r.twinx()
        ax1.spines["right"].set_visible(False)
        ax1r.spines["left"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2r.spines["left"].set_visible(False)
        ax1r.tick_params(axis="y", left=False, labelleft=False)
        ax2.tick_params(axis="y", right=False, labelright=False)
        # Break markers on both spines at each inner corner (matplotlib
        # broken-axis idiom: a short diagonal slash drawn via a path-shaped
        # marker at top and bottom of the two panels).
        d = 1.0  # vertical:horizontal ratio of the slash (1.0 ≈ 45°)
        slash_kw = dict(
            marker=[(-1, -d), (1, d)], markersize=10,
            linestyle="none", color="#64748b",
            mec="#64748b", mew=1.1, clip_on=False,
        )
        ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **slash_kw)
        ax1r.plot([0, 0], [0, 1], transform=ax1r.transAxes, **slash_kw)
    else:
        fig, ax1 = plt.subplots(figsize=(14.0, 5.8))
        ax1r = None
        ax2 = ax1.twinx()
        ax2r = None

    primary_axes = [ax1] + ([ax1r] if use_break else [])
    fid_axes = [ax2] + ([ax2r] if use_break else [])

    # --- draw metric curves ---
    # When triggered, the criteria curve is thicker/opaque and the others
    # are reference lines. When not triggered, all three are equal peers.
    for name, values in metrics.items():
        if name not in _COLORS:
            continue
        is_criteria = (triggered and name == criteria)
        target_axes = fid_axes if name == "fid" else primary_axes
        for i, ax in enumerate(target_axes):
            if triggered:
                linewidth = 2.8 if is_criteria else 1.4
                alpha = 1.0 if is_criteria else 0.45
                markersize = 6.5 if is_criteria else 3.5
                label_first = (
                    f"{_display_name(name)}  ★ criteria" if is_criteria
                    else _display_name(name)
                )
            else:
                linewidth = 1.8
                alpha = 0.95
                markersize = 5.0
                label_first = _display_name(name)
            ax.plot(
                iters, values,
                marker="o",
                linewidth=linewidth,
                alpha=alpha,
                markersize=markersize,
                color=_COLORS[name],
                linestyle="--" if name == "fid" else "-",
                label=label_first if i == 0 else "_nolegend_",
                zorder=3 if is_criteria else 2,
            )

    # --- early-stop decoration (triggered only) ---
    if triggered:
        criteria_ax = ax2 if criteria == "fid" else ax1
        criteria_values = metrics[criteria]
        best_idx = iters.index(best_iteration)
        # Pin both markers to the criteria curve's actual y-values — the
        # curve is the single source of truth, so ★ and X are guaranteed
        # to land on it.
        best_y = criteria_values[best_idx]
        stop_y = criteria_values[iters.index(stop_iteration)]

        def _fmt(name: str, val: float) -> str:
            # FID range is typically 20–80; 2 decimals is enough.
            # nn / mnn live in [0, 1]; keep 4 decimals.
            return f"{val:.2f}" if name == "fid" else f"{val:.4f}"

        best_summary = "   ".join(
            f"{_display_name(name)}={_fmt(name, metrics[name][best_idx])}"
            + (" ★" if name == criteria else "")
            for name in ("fid", "nn", "mnn") if name in metrics
        )
        best_label = f"best: iter={best_iteration}\n  {best_summary}"

        # When best and stop coincide (typical under cumulative_delta=False),
        # the X marker would otherwise hide the ★. Bump the ★ size and shrink
        # the X so the green star tips peek out around the red cross — both
        # remain identifiable while still pinned to the curve point.
        overlap = best_iteration == stop_iteration
        best_s = 520 if overlap else 280
        stop_s = 140 if overlap else 200
        criteria_ax.scatter(
            [best_iteration], [best_y],
            color="#16a34a", s=best_s, zorder=6, marker="*",
            edgecolors="white", linewidths=1.4,
            label=best_label,
        )
        criteria_ax.scatter(
            [stop_iteration], [stop_y],
            color="#ef4444", s=stop_s, zorder=7, marker="X",
            edgecolors="white", linewidths=1.0 if overlap else 1.3,
            label=f"early stop: iter={stop_iteration}",
        )
        criteria_ax.axhline(
            y=best_y, color="#16a34a", linestyle=":",
            linewidth=1.2, alpha=0.7,
        )
        # Patience window: shaded from the last improvement (the validation
        # that last reset ``wait`` to 0) through ``stop_iteration``.
        # This shows the full stagnation context: "improvement was here,
        # then patience consecutive no-improve validations led to stop."
        #
        #     epoch:  …   5    6   7   8   9
        #     val:    … impr  +1  +2 +3 +4   ← patience=4 → trigger at 9
        #                  ↑──── window ────↑
        #
        # ``last_improved_iteration`` differs from ``best_iteration`` when
        # ``cumulative_delta=False`` and strict-better updates creep
        # ``best`` forward without resetting the wait counter.
        #
        # Fallback for callers that don't supply the new field: use
        # ``iters[-patience]`` (the first bad validation) as left edge.
        if last_improved_iteration is not None and last_improved_iteration in iters:
            streak_start = last_improved_iteration
        else:
            streak_start = iters[-patience] if len(iters) >= patience else iters[0]
        if stop_iteration > streak_start:
            window_lines = [f"patience window  (patience={patience})"]
            if cumulative_delta is not None:
                window_lines.append(f"cumulative_delta={cumulative_delta}")
            if min_delta is not None:
                delta_str = f"min_delta={min_delta}"
                if min_delta_mode:
                    delta_str += f" ({min_delta_mode})"
                window_lines.append(delta_str)
            window_label = "\n".join(window_lines)
            ax1.axvspan(
                streak_start, stop_iteration,
                alpha=0.08, color="#ef4444",
                label=window_label,
            )

    # --- x-axis limits ---
    x_margin = (iters[1] - iters[0]) * 0.5 if len(iters) >= 2 else 0.5
    if use_break:
        ax1.set_xlim(iters[0] - x_margin, stop_iteration + x_margin)
        right_pad = (iters[1] - iters[0]) * 1.2 if len(iters) >= 2 else 1.0
        ax1r.set_xlim(max_iter_planned - right_pad, max_iter_planned + right_pad)
        ax1r.set_xticks([max_iter_planned])
        ax1r.tick_params(axis="x", labelsize=9)
        ax2r.set_ylim(ax2.get_ylim())
        ax1r.axvline(
            x=max_iter_planned,
            color="#64748b", linestyle="--", linewidth=1.2, alpha=0.55,
        )
        ax1r.annotate(
            f"original max_iter={max_iter_planned:,}\n(skipped {skipped:,} iters)",
            xy=(max_iter_planned, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, -6), textcoords="offset points",
            ha="center", va="top",
            fontsize=9, color="#475569",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e1", alpha=0.95),
            zorder=7,
        )
    else:
        ax1.set_xlim(iters[0] - x_margin, max_iter_planned + x_margin)
        if triggered and stop_iteration < max_iter_planned:
            ax1.axvline(
                x=max_iter_planned,
                color="#64748b", linestyle="--", linewidth=1.2, alpha=0.5,
            )
            ax1.annotate(
                f"original max_iter={max_iter_planned:,}\n(skipped {skipped:,} iters)",
                xy=(max_iter_planned, ax1.get_ylim()[1]),
                xytext=(-6, -8), textcoords="offset points",
                ha="right", va="top",
                fontsize=9, color="#475569",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e1", alpha=0.9),
            )

    # --- labels / title ---
    ax1.set_xlabel("Training iteration")
    ax1.set_ylabel("Correspondence score  (higher = better)")
    fid_label_ax = ax2r if use_break else ax2
    fid_label_ax.set_ylabel("FID  (lower = better)")

    if triggered:
        mode = _MODE_BY_METRIC[criteria]
        title = (
            f"Early Stopping diagnostic  —  "
            f"criteria: {_display_name(criteria)} "
            f"({'lower' if mode == 'min' else 'higher'}=better)"
        )
    else:
        title = "Training validation metrics"
    subtitle = None
    if len(iters) >= 2:
        spacing = iters[1] - iters[0]
        subtitle = f"each marker = 1 validation  ·  spacing = {spacing:,} iterations"

    if use_break:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.985)
        if subtitle:
            fig.text(
                0.5, 0.935, subtitle,
                fontsize=9, color="#64748b",
                ha="center", va="bottom", style="italic",
            )
    else:
        ax1.set_title(title, fontsize=13, fontweight="bold", pad=22)
        if subtitle:
            ax1.text(
                0.5, 1.01, subtitle,
                transform=ax1.transAxes,
                fontsize=9, color="#64748b",
                ha="center", va="bottom", style="italic",
            )

    ax1.grid(True, alpha=0.3)
    if use_break:
        ax1r.grid(True, alpha=0.3)

    # --- combined legend (dedupe across all axes) ---
    handles_all: List = []
    labels_all: List[str] = []
    legend_source_axes = [ax1, ax2] + ([ax1r, ax2r] if use_break else [])
    for ax in legend_source_axes:
        hh, ll = ax.get_legend_handles_labels()
        for h, l in zip(hh, ll):
            if l and not l.startswith("_") and l not in labels_all:
                handles_all.append(h)
                labels_all.append(l)

    # Group: (A) metric curves, (B) diagnostic info. No blank separator.
    metric_first_tokens = {"nn", "mnn", "fid"}
    metrics_block, info_block = [], []
    for h, l in zip(handles_all, labels_all):
        if l.split()[0].lower() in metric_first_tokens:
            metrics_block.append((h, l))
        else:
            info_block.append((h, l))
    ordered = metrics_block + info_block
    handles = [h for h, _ in ordered]
    labels = [l for _, l in ordered]

    if use_break:
        ax1r.legend(
            handles, labels,
            loc="lower right",
            bbox_to_anchor=(0.985, 0.02),
            fontsize=8.5,
            framealpha=0.96,
            borderpad=0.85,
            labelspacing=0.65,
        )
    elif triggered and stop_iteration < max_iter_planned:
        # Place legend in the empty strip between stop_iteration and
        # max_iter_planned — the area where no curve data exists.
        x_lo, x_hi = ax1.get_xlim()
        x_range = x_hi - x_lo
        legend_x = (stop_iteration - x_lo) / x_range  # left edge in axes fraction
        legend_x = min(legend_x + 0.02, 0.98)         # small pad from stop line
        ax1.legend(
            handles, labels,
            loc="upper left",
            bbox_to_anchor=(legend_x, 0.98),
            fontsize=8.5,
            framealpha=0.94,
            borderpad=0.9,
            labelspacing=0.65,
        )
    else:
        # Plain case (no trigger): curves cover the full x-range, so park
        # the legend outside the plot on the right.
        ax2.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(1.12, 0.5),
            fontsize=9.5,
            framealpha=0.94,
            borderpad=0.9,
            labelspacing=0.7,
        )

    fig.tight_layout(pad=1.8)
    if use_break:
        fig.subplots_adjust(top=0.90)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
