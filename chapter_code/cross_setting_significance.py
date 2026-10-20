"""Test the matched GPT-5.3 versus GPT-5.5 effects highlighted in the paper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECURRENCE_PATH = (
    ROOT / "result/entity_runlevel_official_gpt41mini_v1/results.json"
)
PROXY_PATH = ROOT / "result/proxy_effectiveness_1600run.json"
OUTPUT_PATH = ROOT / "chapter_code/generated/cross_setting_significance.json"
N_PERMUTATIONS = 1_000_000
N_BOOTSTRAPS = 2_000


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def paired_test(differences: list[float], family: str, level: str) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(stable_seed(family, level, "sign_flip"))
    extreme = 0
    chunk_size = 10_000
    for _ in range(N_PERMUTATIONS // chunk_size):
        signs = rng.choice((-1.0, 1.0), size=(chunk_size, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))

    bootstrap_rng = np.random.default_rng(
        stable_seed(family, level, "paired_bootstrap")
    )
    indices = bootstrap_rng.integers(
        0, len(values), size=(N_BOOTSTRAPS, len(values))
    )
    bootstrap_means = np.sort(values[indices].mean(axis=1))

    return {
        "n_queries": int(len(values)),
        "mean_gpt_5_5_minus_gpt_5_3": float(values.mean()),
        "paired_bootstrap_95_ci": [
            float(bootstrap_means[int(0.025 * N_BOOTSTRAPS)]),
            float(bootstrap_means[int(0.975 * N_BOOTSTRAPS) - 1]),
        ],
        "raw_p_value": (extreme + 1) / (N_PERMUTATIONS + 1),
    }


def apply_holm(results: dict[str, dict[str, Any]]) -> None:
    ordered = sorted(results, key=lambda level: results[level]["raw_p_value"])
    running_max = 0.0
    family_size = len(ordered)
    for rank, level in enumerate(ordered):
        adjusted = (family_size - rank) * results[level]["raw_p_value"]
        running_max = max(running_max, adjusted)
        results[level]["holm_adjusted_p_value"] = min(1.0, running_max)


def recurrence_family() -> dict[str, dict[str, Any]]:
    data = json.loads(RECURRENCE_PATH.read_text())
    models = {model["dataset"]: model for model in data["models"]}
    earlier = models["dataset-5.3-1600"]
    later = models["dataset-5.5-1600"]
    results: dict[str, dict[str, Any]] = {}
    for level in ("url", "hostname", "entity"):
        left = earlier["leave_one_run_out"][level]["query_macro_auc"]["per_query"]
        right = later["leave_one_run_out"][level]["query_macro_auc"]["per_query"]
        shared = sorted(set(left) & set(right))
        differences = [right[query] - left[query] for query in shared]
        results[level] = paired_test(differences, "recurrence_auc", level)
    apply_holm(results)
    return results


def proxy_precision_family() -> dict[str, dict[str, Any]]:
    data = json.loads(PROXY_PATH.read_text())
    models = {model["dataset"]: model for model in data["models"]}
    earlier = {
        (row["subdataset"], row["qid"]): row
        for row in models["dataset-5.3-1600"]["per_question"]
    }
    later = {
        (row["subdataset"], row["qid"]): row
        for row in models["dataset-5.5-1600"]["per_question"]
    }
    shared = sorted(set(earlier) & set(later))
    level_names = {
        "url": "url",
        "hostname": "domain",
        "entity": "entity_brand",
    }
    results: dict[str, dict[str, Any]] = {}
    for output_level, stored_level in level_names.items():
        differences = []
        for query in shared:
            left = earlier[query]["granularities"][stored_level]["proxies"]
            right = later[query]["granularities"][stored_level]["proxies"]
            differences.append(
                right["A_citation"]["visibility"]["precision"]
                - left["A_citation"]["visibility"]["precision"]
            )
        results[output_level] = paired_test(
            differences, "api_citation_precision", output_level
        )
    apply_holm(results)
    return results


def main() -> None:
    output = {
        "protocol": {
            "unit": "query",
            "test": "two-sided paired sign-flip permutation test",
            "permutations": N_PERMUTATIONS,
            "confidence_interval": f"{N_BOOTSTRAPS} paired query bootstraps",
            "multiple_comparisons": "Holm correction within each metric family",
            "significance_threshold": 0.05,
        },
        "recurrence_auc": recurrence_family(),
        "api_citation_precision": proxy_precision_family(),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
