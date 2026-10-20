#!/usr/bin/env python3
"""Plot cross-run candidate recurrence and set stability."""

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
    ROOT / "figures/candidate_stability.pdf"
)

LEVELS = ("URL", "Hostname", "Entity")
MODELS = ("5.3", "5.5")
MODEL_DATA: dict[str, dict[str, dict[str, float]]] = {
    "5.3": {
        "URL": {
            "unique": 96.4,
            "one_off": 40.8,
            "stable": 9.4,
            "jaccard": 0.303,
        },
        "Hostname": {
            "unique": 52.3,
            "one_off": 31.3,
            "stable": 17.0,
            "jaccard": 0.408,
        },
        "Entity": {},
    },
    "5.5": {
        "URL": {
            "unique": 132.2,
            "one_off": 44.5,
            "stable": 7.7,
            "jaccard": 0.292,
        },
        "Hostname": {
            "unique": 64.3,
            "one_off": 35.2,
            "stable": 15.1,
            "jaccard": 0.402,
        },
        "Entity": {},
    },
}
DATASET_TO_MODEL = {
    "dataset-5.3-1600": "5.3",
    "dataset-5.5-1600": "5.5",
}

RECURRENCE_STYLES = (
    ("one_off", "One-off", "#D55E00"),
    ("intermittent", "Intermittent", "#B8B8B8"),
    ("stable", "Stable core", "#009E73"),
)
MODEL_STYLES = {
    "5.3": ("GPT-5.3 Instant", "#0072B2", "o", "-"),
    "5.5": ("GPT-5.5 Instant", "#CC79A7", "s", "--"),
}


def load_data(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    with path.open(encoding="utf-8") as stream:
        results: dict[str, Any] = json.load(stream)

    data = {
        model: {level: values.copy() for level, values in levels.items()}
        for model, levels in MODEL_DATA.items()
    }
    seen_models: set[str] = set()
    for result in results.get("models", []):
        model = DATASET_TO_MODEL.get(result.get("dataset"))
        if model is None:
            continue
        stability = result["entity_candidate_stability"]
        data[model]["Entity"] = {
            "unique": stability["mean_unique_per_query"],
            "one_off": 100 * stability["mean_one_run_share"],
            "stable": 100 * stability["mean_stable_core_share"],
            "jaccard": stability["mean_pairwise_jaccard"],
        }
        seen_models.add(model)

    if seen_models != set(MODELS):
        raise ValueError(f"Missing entity stability results for: {set(MODELS) - seen_models}")

    for model in MODELS:
        for level in LEVELS:
            values = data[model][level]
            values["intermittent"] = (
                100 - values["one_off"] - values["stable"]
            )
            if not 99.999 <= sum(
                values[key] for key in ("one_off", "intermittent", "stable")
            ) <= 100.001:
                raise ValueError(f"Invalid recurrence composition: {model} {level}")
    return data


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot(
    data: dict[str, dict[str, dict[str, float]]],
    output: Path,
    preview: Path | None,
) -> None:
    configure_style()
    figure, (composition_axis, jaccard_axis) = plt.subplots(
        1,
        2,
        figsize=(7.15, 3.45),
        gridspec_kw={"width_ratios": (1.28, 1.0)},
    )

    group_centers = [0.0, 1.2, 2.4]
    positions = [
        center + offset
        for center in group_centers
        for offset in (-0.22, 0.22)
    ]
    left = [0.0] * len(positions)
    flat_values = [
        data[model][level]
        for level in LEVELS
        for model in MODELS
    ]

    for key, label, color in RECURRENCE_STYLES:
        heights = [values[key] for values in flat_values]
        bars = composition_axis.bar(
            positions,
            heights,
            bottom=left,
            width=0.36,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            label=label,
        )
        if key != "intermittent":
            text_color = "white"
            for bar, height in zip(bars, heights, strict=True):
                composition_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{height:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=7.1,
                    fontweight="bold",
                )
        left = [
            start + height
            for start, height in zip(left, heights, strict=True)
        ]

    composition_axis.set_xticks(
        positions,
        [model for _ in LEVELS for model in MODELS],
    )
    for center, level in zip(group_centers, LEVELS, strict=True):
        composition_axis.text(
            center,
            -0.14,
            level,
            transform=composition_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )
    composition_axis.axvline(0.6, color="#D5D5D5", linewidth=0.65)
    composition_axis.axvline(1.8, color="#D5D5D5", linewidth=0.65)
    composition_axis.set_xlim(-0.52, 2.92)
    composition_axis.set_ylim(0, 100)
    composition_axis.set_yticks(range(0, 101, 20))
    composition_axis.set_ylabel("Share of distinct candidates (%)")
    composition_axis.set_title(
        "(a) Recurrence composition",
        loc="left",
        pad=7,
        fontweight="bold",
    )
    composition_axis.grid(
        axis="y",
        color="#E0E0E0",
        linewidth=0.55,
        alpha=0.8,
    )
    composition_axis.set_axisbelow(True)
    composition_axis.spines["top"].set_visible(False)
    composition_axis.spines["right"].set_visible(False)
    composition_axis.tick_params(axis="both", length=2.8, width=0.7)
    composition_handles, composition_labels = (
        composition_axis.get_legend_handles_labels()
    )

    x_values = list(range(len(LEVELS)))
    for model in MODELS:
        label, color, marker, linestyle = MODEL_STYLES[model]
        values = [data[model][level]["jaccard"] for level in LEVELS]
        jaccard_axis.plot(
            x_values,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.9,
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=label,
            zorder=3,
        )

    jaccard_axis.set_xticks(x_values, LEVELS)
    jaccard_axis.set_xlim(-0.25, 2.25)
    jaccard_axis.set_ylim(0.28, 0.43)
    jaccard_axis.set_yticks([0.30, 0.35, 0.40])
    jaccard_axis.set_ylabel("Pairwise Jaccard")
    jaccard_axis.set_title(
        "(b) Set stability by granularity",
        loc="left",
        pad=7,
        fontweight="bold",
    )
    jaccard_axis.grid(
        axis="y",
        color="#D9D9D9",
        linewidth=0.55,
        alpha=0.8,
    )
    jaccard_axis.set_axisbelow(True)
    jaccard_axis.spines["top"].set_visible(False)
    jaccard_axis.spines["right"].set_visible(False)
    jaccard_axis.tick_params(axis="both", length=2.8, width=0.7)
    model_handles, model_labels = jaccard_axis.get_legend_handles_labels()
    figure.legend(
        composition_handles + model_handles,
        composition_labels + model_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.25,
        borderaxespad=0,
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.23,
        top=0.78,
        wspace=0.25,
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
    plot(load_data(args.input), args.output, args.preview)
    print(f"Wrote {args.output}")
    if args.preview is not None:
        print(f"Wrote {args.preview}")


if __name__ == "__main__":
    main()
