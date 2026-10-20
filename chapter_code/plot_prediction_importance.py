#!/usr/bin/env python3
"""Plot normalized SHAP diagnostics for the citation-prediction models."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "result/prediction_experiment_1600run.json"
DEFAULT_OUTPUT = ROOT / "figures/prediction_shap_importance.pdf"

DATASETS = (
    ("dataset-5.3-1600", "5.3"),
    ("dataset-5.5-1600", "5.5"),
)
GRANULARITIES = (
    ("url", "URL"),
    ("domain", "Hostname"),
    ("entity_brand", "Entity/brand"),
)
FAMILIES = (
    ("Rank", "#0072B2"),
    ("Frequency", "#56B4E9"),
    ("Conversion", "#D55E00"),
    ("Support", "#009E73"),
    ("Snippet", "#E69F00"),
    ("Query controls", "#888888"),
)
FAMILY_COLORS = dict(FAMILIES)
FEATURES = (
    ("rrf_score", "G-A RRF", "Rank"),
    ("api_best_rank_log", "API best rank", "Rank"),
    ("api_appear_rate", "API appearance", "Frequency"),
    ("api_exposure_per_run", "API entity exposure", "Frequency"),
    (
        "api_citation_given_candidate",
        "API citation conversion",
        "Conversion",
    ),
    ("api_citation_rate", "API entity citation rate", "Conversion"),
    ("api_url_count_log", "API URL support (host)", "Support"),
    ("api_url_support_log", "API URL support (entity)", "Support"),
    ("source_consensus", "G-A entity consensus", "Support"),
    ("snippet_length_log", "Snippet length", "Snippet"),
    ("intent_factual_lookup", "Factual intent", "Query controls"),
    ("intent_info_synthesis", "Synthesis intent", "Query controls"),
    ("intent_recommendation", "Recommendation intent", "Query controls"),
    ("language_chinese", "Chinese query", "Query controls"),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.1,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.2,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def index_models(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed = {model["dataset"]: model for model in results["models"]}
    expected = {dataset for dataset, _ in DATASETS}
    if set(indexed) != expected:
        raise ValueError(f"Unexpected datasets: {sorted(indexed)}")
    return indexed


def normalized_shap(
    models: dict[str, dict[str, Any]],
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], dict[str, float]]]:
    columns: list[tuple[str, str]] = []
    values: dict[tuple[str, str], dict[str, float]] = {}
    for granularity, _ in GRANULARITIES:
        for dataset, _ in DATASETS:
            rows = models[dataset]["granularities"][granularity][
                "xgboost_diagnostics"
            ]["shap"]
            total = sum(float(row["mean_abs_shap"]) for row in rows)
            if total <= 0:
                raise ValueError(f"Non-positive SHAP total: {dataset}/{granularity}")
            column = (dataset, granularity)
            columns.append(column)
            values[column] = {
                row["feature"]: float(row["mean_abs_shap"]) / total
                for row in rows
            }
    return columns, values


def family_shares(
    columns: list[tuple[str, str]],
    values: dict[tuple[str, str], dict[str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    feature_family = {feature: family for feature, _, family in FEATURES}
    shares: dict[tuple[str, str], dict[str, float]] = {}
    for column in columns:
        grouped: dict[str, float] = defaultdict(float)
        for feature, value in values[column].items():
            grouped[feature_family[feature]] += value
        shares[column] = dict(grouped)
    return shares


def add_group_guides(axis: plt.Axes) -> None:
    for boundary in (1.5, 3.5):
        axis.axvline(
            boundary,
            color="#B7B7B7",
            linewidth=0.65,
            linestyle="--",
            zorder=0,
        )


def plot_importance(
    results: dict[str, Any],
    output: Path,
    preview: Path | None,
) -> None:
    configure_style()
    models = index_models(results)
    columns, values = normalized_shap(models)
    shares = family_shares(columns, values)

    figure = plt.figure(figsize=(7.15, 5.35))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.0, 2.65), hspace=0.46)
    family_axis = figure.add_subplot(grid[0])
    feature_axis = figure.add_subplot(grid[1])
    x_values = np.arange(len(columns), dtype=float)

    bottoms = np.zeros(len(columns), dtype=float)
    for family, color in FAMILIES:
        heights = np.asarray([shares[column].get(family, 0.0) for column in columns])
        family_axis.bar(
            x_values,
            heights * 100,
            bottom=bottoms * 100,
            width=0.68,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=family,
        )
        bottoms += heights

    family_axis.set_title(
        "(a) SHAP composition by signal family",
        loc="left",
        pad=4,
        fontweight="bold",
    )
    family_axis.set_ylabel("Share of mean |SHAP| (%)")
    family_axis.set_ylim(0, 100)
    family_axis.set_yticks((0, 25, 50, 75, 100))
    family_axis.set_xlim(-0.55, len(columns) - 0.45)
    short_granularity = {"url": "URL", "domain": "Host", "entity_brand": "Entity"}
    family_axis.set_xticks(
        x_values,
        [
            f"{model}\n{short_granularity[granularity]}"
            for granularity, _ in GRANULARITIES
            for _, model in DATASETS
        ],
    )
    family_axis.grid(axis="y", color="#E1E1E1", linewidth=0.5, alpha=0.8)
    family_axis.set_axisbelow(True)
    family_axis.spines["top"].set_visible(False)
    family_axis.spines["right"].set_visible(False)
    family_axis.tick_params(axis="both", length=2.5, width=0.65)
    add_group_guides(family_axis)

    feature_axis.set_title(
        "(b) Feature-level normalized mean |SHAP|",
        loc="left",
        pad=5,
        fontweight="bold",
    )
    row_count = len(FEATURES)
    feature_axis.set_xlim(-0.65, len(columns) - 0.45)
    feature_axis.set_ylim(row_count - 0.5, -0.5)
    feature_axis.set_xticks(
        x_values,
        [
            f"{model}\n{granularity}"
            for _, granularity in GRANULARITIES
            for _, model in DATASETS
        ],
    )
    feature_axis.xaxis.tick_top()
    feature_axis.tick_params(axis="x", pad=3)
    feature_axis.set_yticks(
        np.arange(row_count),
        [label for _, label, _ in FEATURES],
    )
    feature_axis.grid(
        axis="both",
        color="#E7E7E7",
        linewidth=0.45,
        alpha=0.9,
        zorder=0,
    )
    feature_axis.tick_params(axis="both", length=0)
    for spine in feature_axis.spines.values():
        spine.set_visible(False)
    add_group_guides(feature_axis)

    for row_index, (feature, _, family) in enumerate(FEATURES):
        feature_axis.scatter(
            [-0.50],
            [row_index],
            s=18,
            marker="s",
            color=FAMILY_COLORS[family],
            edgecolors="none",
            clip_on=False,
            zorder=3,
        )
        for column_index, column in enumerate(columns):
            if feature not in values[column]:
                continue
            share = values[column][feature]
            feature_axis.scatter(
                [column_index],
                [row_index],
                s=16 + 780 * share,
                color=FAMILY_COLORS[family],
                alpha=0.82,
                edgecolors="white",
                linewidths=0.55,
                zorder=3,
            )
            if share >= 0.15:
                feature_axis.text(
                    column_index,
                    row_index,
                    f"{share * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.1,
                    color="white",
                    fontweight="bold",
                    zorder=4,
                )

    for boundary in (1.5, 3.5, 5.5, 8.5, 9.5):
        feature_axis.axhline(boundary, color="#B8B8B8", linewidth=0.55)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=6.0,
            label=family,
        )
        for family, color in FAMILIES
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=6,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.05,
    )
    figure.text(
        0.99,
        0.012,
        r"Bubble labels show within-model SHAP share (%) when $\geq$15%; blank cells are not applicable.",
        ha="right",
        va="bottom",
        fontsize=6.7,
        color="#666666",
    )
    figure.subplots_adjust(
        left=0.255,
        right=0.995,
        bottom=0.055,
        top=0.91,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview, dpi=240, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_importance(
        load_json(args.results),
        args.output,
        args.preview,
    )


if __name__ == "__main__":
    main()
