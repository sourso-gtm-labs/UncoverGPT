#!/usr/bin/env python3
"""Plot leave-one-run-out recurrence and held-out citation propensity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "result/entity_runlevel_official_gpt41mini_v1/results.json"
)
DEFAULT_OUTPUT = (
    ROOT / "figures/recurrence_citation.pdf"
)

LEVELS = (
    ("url", "URL", "#0072B2", "o", "-"),
    ("hostname", "Hostname", "#D55E00", "s", "--"),
    ("entity", "Entity", "#009E73", "^", "-."),
)
MODEL_LABELS = {
    "dataset-5.3-1600": "(a) GPT-5.3 Instant",
    "dataset-5.5-1600": "(b) GPT-5.5 Instant",
}
BIN_LABELS = ("0", "1-9", "10-29", "30-49", "50-69", "70-89", "90-100")


def load_results(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or len(value.get("models", [])) != 2:
        raise ValueError(f"Expected two-model experiment results: {path}")
    return value


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(results: dict[str, Any], output: Path, preview: Path | None) -> None:
    configure_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.40),
        sharex=True,
        sharey=True,
    )
    x_values = list(range(len(BIN_LABELS)))

    for axis, model in zip(axes, results["models"], strict=True):
        loo = model["leave_one_run_out"]
        auc_parts = []
        for key, label, color, marker, linestyle in LEVELS:
            bins = loo[key]["query_macro_bins"]
            rates = [100 * row["macro_rate"] for row in bins]
            lows = [100 * row["macro_ci_low"] for row in bins]
            highs = [100 * row["macro_ci_high"] for row in bins]
            auc = loo[key]["query_macro_auc"]["summary"]["mean"]
            auc_parts.append(f"{label} {auc:.3f}")

            axis.fill_between(
                x_values,
                lows,
                highs,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            axis.plot(
                x_values,
                rates,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4.6,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                label=label,
                zorder=3,
            )

        axis.set_title(MODEL_LABELS[model["dataset"]], pad=5, fontweight="bold")
        axis.text(
            0.02,
            0.96,
            "AUC: " + "  |  ".join(auc_parts),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.1,
            color="#303030",
        )
        axis.set_xticks(x_values, BIN_LABELS)
        axis.set_ylim(10, 82)
        axis.set_yticks(range(10, 81, 10))
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", length=2.8, width=0.7)

    figure.supylabel("Held-out citation rate (%)", x=0.005, fontsize=9)
    figure.supxlabel(
        "Prior occurrence in the other 19 runs (%)",
        y=0.015,
        fontsize=9,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.6,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.18,
        top=0.84,
        wspace=0.11,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot(load_results(args.input), args.output, args.preview)
    print(f"Wrote {args.output}")
    if args.preview is not None:
        print(f"Wrote {args.preview}")


if __name__ == "__main__":
    main()
