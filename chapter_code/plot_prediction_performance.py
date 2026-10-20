#!/usr/bin/env python3
"""Plot citation-prediction performance from the final experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTION = ROOT / "result/prediction_experiment_1600run.json"
DEFAULT_RRF = ROOT / "result/prediction_overall_rrf_ablation_1600run.json"
DEFAULT_OUTPUT_DIR = ROOT / "figures"

DATASETS = (
    ("dataset-5.3-1600", "GPT-5.3 Instant"),
    ("dataset-5.5-1600", "GPT-5.5 Instant"),
)
GRANULARITIES = (
    ("url", "URL"),
    ("domain", "Hostname"),
    ("entity_brand", "Entity/brand"),
)
METHODS = (
    ("Full XGBoost", "XGBoost", "#0072B2", "o", "-"),
    ("Overall RRF", "Overall_RRF", "#D55E00", "s", "-"),
    ("G-A RRF", "G_A_RRF", "#009E73", "^", "--"),
    ("A-citation only", "A_citation_only", "#CC79A7", "D", "-."),
    ("Random", "Random", "#777777", "x", ":"),
)
TOP_K = (1, 3, 5, 10)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def index_models(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models = results.get("models", [])
    indexed = {model["dataset"]: model for model in models}
    expected = {dataset for dataset, _ in DATASETS}
    if set(indexed) != expected:
        raise ValueError(f"Unexpected datasets: {sorted(indexed)}")
    return indexed


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.8,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_metrics(
    prediction_models: dict[str, dict[str, Any]],
    rrf_models: dict[str, dict[str, Any]],
    dataset: str,
    granularity: str,
    method_key: str,
) -> dict[str, Any]:
    source = rrf_models if method_key == "Overall_RRF" else prediction_models
    return source[dataset]["granularities"][granularity]["methods"][method_key]


def plot_auc_ap(
    prediction_models: dict[str, dict[str, Any]],
    rrf_models: dict[str, dict[str, Any]],
    output: Path,
    preview: Path | None,
) -> None:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.15))
    groups = [
        (dataset, granularity)
        for dataset, _ in DATASETS
        for granularity, _ in GRANULARITIES
    ]
    x_values = np.arange(len(groups), dtype=float)
    offsets = np.linspace(-0.25, 0.25, len(METHODS))
    panels = (
        ("roc_auc", "(a) Query-macro ROC-AUC", (0.45, 0.84)),
        ("average_precision", "(b) Query-macro Average Precision", (0.05, 0.56)),
    )

    for axis, (metric, title, limits) in zip(axes, panels, strict=True):
        for offset, (label, key, color, marker, _) in zip(
            offsets, METHODS, strict=True
        ):
            summaries = [
                method_metrics(
                    prediction_models,
                    rrf_models,
                    dataset,
                    granularity,
                    key,
                )[metric]
                for dataset, granularity in groups
            ]
            means = np.asarray([row["mean"] for row in summaries])
            low = np.asarray([row["ci_low"] for row in summaries])
            high = np.asarray([row["ci_high"] for row in summaries])
            axis.errorbar(
                x_values + offset,
                means,
                yerr=np.vstack((means - low, high - means)),
                fmt=marker,
                color=color,
                markerfacecolor="white" if key not in {"XGBoost", "Overall_RRF"} else color,
                markeredgecolor=color,
                markersize=4.5,
                markeredgewidth=0.9,
                elinewidth=0.9,
                capsize=1.8,
                capthick=0.8,
                label=label,
                zorder=3,
            )

        axis.axvline(2.5, color="#A8A8A8", linewidth=0.7, linestyle="--")
        if metric == "roc_auc":
            axis.axhline(0.5, color="#B0B0B0", linewidth=0.7, linestyle=":")
        axis.set_title(title, pad=17, fontweight="bold")
        axis.set_xlim(-0.55, 5.55)
        axis.set_ylim(*limits)
        axis.set_xticks(
            x_values,
            [
                "URL",
                "Hostname",
                "Entity/\nbrand",
                "URL",
                "Hostname",
                "Entity/\nbrand",
            ],
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.85)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", length=2.7, width=0.7)
        axis.text(
            1.0,
            1.025,
            DATASETS[0][1],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#303030",
        )
        axis.text(
            4.0,
            1.025,
            DATASETS[1][1],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8.2,
            color="#303030",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.15,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.16,
        top=0.78,
        wspace=0.16,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def add_f1_contours(axis: plt.Axes, label_contours: bool) -> None:
    recall = np.linspace(0.001, 0.52, 700)
    for f1 in (0.1, 0.2, 0.3, 0.4):
        valid = recall > f1 / 2
        precision = np.full_like(recall, np.nan)
        precision[valid] = f1 * recall[valid] / (2 * recall[valid] - f1)
        visible = (precision > 0) & (precision <= 0.9)
        axis.plot(
            recall[visible],
            precision[visible],
            color="#D0D0D0",
            linewidth=0.55,
            linestyle="--",
            zorder=0,
        )
        if label_contours:
            label_x = 0.47
            label_y = f1 * label_x / (2 * label_x - f1)
            axis.text(
                label_x,
                label_y,
                f"F1={f1:.1f}",
                ha="right",
                va="bottom",
                fontsize=5.9,
                color="#9A9A9A",
            )


def plot_topk_pr(
    prediction_models: dict[str, dict[str, Any]],
    rrf_models: dict[str, dict[str, Any]],
    output: Path,
    preview: Path | None,
) -> None:
    configure_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.15, 4.75),
        sharex=True,
        sharey=True,
    )
    marker_sizes = (3.1, 4.0, 5.0, 6.1)

    for row_index, (dataset, model_label) in enumerate(DATASETS):
        for col_index, (granularity, granularity_label) in enumerate(
            GRANULARITIES
        ):
            axis = axes[row_index, col_index]
            add_f1_contours(axis, label_contours=col_index == 2)
            for label, key, color, marker, linestyle in METHODS:
                metrics = method_metrics(
                    prediction_models,
                    rrf_models,
                    dataset,
                    granularity,
                    key,
                )
                precision = [metrics[f"precision@{k}"]["mean"] for k in TOP_K]
                recall = [metrics[f"recall@{k}"]["mean"] for k in TOP_K]
                alpha = 0.72 if key == "Random" else 0.95
                axis.plot(
                    recall,
                    precision,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.25,
                    alpha=alpha,
                    zorder=2,
                )
                for x_value, y_value, size in zip(
                    recall, precision, marker_sizes, strict=True
                ):
                    axis.plot(
                        x_value,
                        y_value,
                        marker=marker,
                        color=color,
                        markerfacecolor=(
                            "white"
                            if key not in {"XGBoost", "Overall_RRF"}
                            else color
                        ),
                        markeredgecolor=color,
                        markeredgewidth=0.75,
                        markersize=size,
                        alpha=alpha,
                        zorder=3,
                    )

            if row_index == 0:
                axis.set_title(granularity_label, pad=5, fontweight="bold")
            if col_index == 0:
                axis.set_ylabel(f"{model_label}\nPrecision", labelpad=5)
            if row_index == 1:
                axis.set_xlabel("Recall")

            axis.set_xlim(0, 0.50)
            axis.set_ylim(0, 0.90)
            axis.set_xticks(np.arange(0, 0.51, 0.1))
            axis.set_yticks(np.arange(0, 0.91, 0.2))
            axis.grid(color="#E1E1E1", linewidth=0.45, alpha=0.8)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(axis="both", length=2.5, width=0.65)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=(
                "white" if key not in {"XGBoost", "Overall_RRF"} else color
            ),
            markeredgecolor=color,
            linewidth=1.25,
            markersize=4.5,
            label=label,
        )
        for label, key, color, marker, linestyle in METHODS
    ]
    figure.legend(
        handles=method_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        handletextpad=0.4,
        columnspacing=1.15,
    )
    figure.text(
        0.995,
        0.01,
        "Marker size: k = 1, 3, 5, 10",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.995,
        bottom=0.105,
        top=0.90,
        wspace=0.13,
        hspace=0.18,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--rrf", type=Path, default=DEFAULT_RRF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_models = index_models(load_json(args.prediction))
    rrf_models = index_models(load_json(args.rrf))

    auc_ap_output = args.output_dir / "prediction_auc_ap.pdf"
    topk_output = args.output_dir / "prediction_topk_pr.pdf"
    auc_ap_preview = (
        args.preview_dir / "prediction_auc_ap.png"
        if args.preview_dir is not None
        else None
    )
    topk_preview = (
        args.preview_dir / "prediction_topk_pr.png"
        if args.preview_dir is not None
        else None
    )

    plot_auc_ap(
        prediction_models,
        rrf_models,
        auc_ap_output,
        auc_ap_preview,
    )
    plot_topk_pr(
        prediction_models,
        rrf_models,
        topk_output,
        topk_preview,
    )
    print(f"Wrote {auc_ap_output}")
    print(f"Wrote {topk_output}")


if __name__ == "__main__":
    main()
