"""Evaluate Google and Responses API proxies against ChatGPT Web citations.

This is the unified experiment for paper sections 4.2 and 4.3.  For every
question it treats ChatGPT Web (W) citations as the target and evaluates three
feasible proxies plus one visibility-only union:

* G: Google SerpAPI organic result pool.
* A_candidate: all Responses API web-search sources (plus citation-only URLs).
* A_citation: URLs actually cited by the Responses API answer.
* G_union_A_candidate: the set union of G and A_candidate.

At URL, hostname, and entity/brand granularity the script reports:

* Visibility: precision, recall, and F1 over unique per-question item sets.
* Ranking: NDCG@1/@3/@5/@10 with W citation frequency as graded relevance.

All metrics are computed per question and then macro-averaged.  URL matching
uses the project's ``deep_normalize`` convention.  Entity/brand counts reuse
the cached v11-style spaCy + GPT-4.1-mini extraction results produced by each
dataset's ``v9_trend_analysis`` pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlparse, urlunparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("dataset-5.3-1600", "dataset-5.5-1600")
SUBDATASETS = (
    "v9_analysis",
    "v9_factual_lookup",
    "v9_info_synthesis",
    "v9_recommendation",
)
RANKED_PROXIES = ("G", "A_candidate", "A_citation")
UNION_PROXY = "G_union_A_candidate"
PROXIES = (*RANKED_PROXIES, UNION_PROXY)
GRANULARITIES = ("url", "domain", "entity_brand")
K_VALUES = (1, 3, 5, 10)
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "srsltid",
    "ref",
    "fbclid",
    "gclid",
    "source",
    "from",
}


def deep_normalize(url: str) -> str:
    """Apply the URL normalization used by the v9 comparison scripts."""
    try:
        parsed = urlparse(unquote(url or ""))
        scheme = parsed.scheme.lower() or "https"
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        host = re.sub(r":443$", "", host)
        host = re.sub(r":80$", "", host)
        path = parsed.path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        params = parse_qs(parsed.query, keep_blank_values=True)
        for key in TRACKING_PARAMS:
            params.pop(key, None)
        query = "&".join(f"{key}={values[0]}" for key, values in sorted(params.items()))
        return urlunparse((scheme, host, path, "", query, ""))
    except Exception:
        return url


def hostname(url: str) -> str:
    """Return a normalized hostname while retaining subdomains."""
    try:
        host = (urlparse(url).hostname or "").lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def dcg_at_k(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], 1))


def ndcg_at_k(relevance: Counter[str], ranking: list[str], k: int) -> float | None:
    ideal = dcg_at_k(sorted(relevance.values(), reverse=True), k)
    if ideal <= 0:
        return None
    observed = dcg_at_k([relevance.get(item, 0) for item in ranking], k)
    return observed / ideal


def visibility_metrics(target: set[str], proxy: set[str]) -> dict[str, float | int] | None:
    """Return binary set P/R/F1, or None when the target has no positives."""
    if not target:
        return None
    intersection = len(target & proxy)
    precision = intersection / len(proxy) if proxy else 0.0
    recall = intersection / len(target)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "target_size": len(target),
        "proxy_size": len(proxy),
        "intersection": intersection,
    }


def top_k_visibility_metrics(
    target: set[str], ranking: list[str], k: int
) -> dict[str, float] | None:
    """Return binary P/R/F1@k using a fixed-k precision denominator."""
    if not target:
        return None
    if k <= 0:
        raise ValueError("k must be positive")
    hits = len(target & set(ranking[:k]))
    precision = hits / k
    recall = hits / len(target)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "n": 0,
        }
    return {
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "std": statistics.pstdev(clean),
        "min": min(clean),
        "max": max(clean),
        "n": len(clean),
    }


def _api_root(dataset_root: Path) -> Path:
    for name in ("api_responses", "api_responses_parsed"):
        candidate = dataset_root / "v9_api_pool" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No parsed API response directory under {dataset_root}")


def _question_dirs(dataset_root: Path) -> list[tuple[str, str, Path]]:
    questions: list[tuple[str, str, Path]] = []
    for subdataset in SUBDATASETS:
        search_root = dataset_root / subdataset / "search_detail"
        if not search_root.exists():
            continue
        for question_dir in sorted(search_root.iterdir()):
            if question_dir.is_dir() and question_dir.name.startswith("q"):
                questions.append((subdataset, question_dir.name, question_dir))
    return questions


def _load_web_relevance(question_dir: Path) -> tuple[dict[str, Counter[str]], int]:
    url_relevance: Counter[str] = Counter()
    domain_relevance: Counter[str] = Counter()
    files = sorted(question_dir.glob("*_detail.json"))
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        cited_urls: set[str] = set()
        cited_domains: set[str] = set()
        for item in data.get("all_urls", []):
            if not item.get("cited"):
                continue
            normalized = deep_normalize(item.get("url") or "")
            domain = hostname(normalized)
            if normalized:
                cited_urls.add(normalized)
            if domain:
                cited_domains.add(domain)
        url_relevance.update(cited_urls)
        domain_relevance.update(cited_domains)
    return {"url": url_relevance, "domain": domain_relevance}, len(files)


def _mean_positions(position_lists: dict[str, list[int]]) -> dict[str, float]:
    return {
        item: statistics.fmean(positions)
        for item, positions in position_lists.items()
        if positions
    }


def _rank_by_frequency(
    primary: Counter[str],
    secondary: Counter[str] | None = None,
    positions: dict[str, float] | None = None,
) -> list[str]:
    secondary = secondary or Counter()
    positions = positions or {}
    return sorted(
        (item for item, count in primary.items() if count > 0),
        key=lambda item: (
            -primary[item],
            -secondary.get(item, 0),
            positions.get(item, math.inf),
            item,
        ),
    )


def _load_api_proxies(
    api_question_dir: Path,
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    candidate_url: Counter[str] = Counter()
    citation_url: Counter[str] = Counter()
    candidate_domain: Counter[str] = Counter()
    citation_domain: Counter[str] = Counter()
    url_positions: dict[str, list[int]] = defaultdict(list)
    domain_positions: dict[str, list[int]] = defaultdict(list)
    citation_only_instances = 0
    citation_only_urls: set[str] = set()
    files = sorted(api_question_dir.glob("run_*.json")) if api_question_dir.exists() else []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sources = data.get("sources") or []
        citations = data.get("citations") or []
        candidate_positions: dict[str, int] = {}

        for position, source in enumerate(sources, 1):
            raw = source.get("url") if isinstance(source, dict) else source
            normalized = deep_normalize(raw or "")
            if normalized and (normalized not in candidate_positions or position < candidate_positions[normalized]):
                candidate_positions[normalized] = position

        source_urls = set(candidate_positions)
        for citation_position, citation in enumerate(citations, 1):
            normalized = deep_normalize(citation.get("url") or "")
            if not normalized:
                continue
            if normalized not in source_urls:
                citation_only_instances += 1
                citation_only_urls.add(normalized)
            candidate_positions.setdefault(normalized, len(sources) + citation_position)

        candidate_domains_this_run: dict[str, int] = {}
        for normalized, position in candidate_positions.items():
            candidate_url[normalized] += 1
            url_positions[normalized].append(position)
            domain = hostname(normalized)
            if domain and (
                domain not in candidate_domains_this_run
                or position < candidate_domains_this_run[domain]
            ):
                candidate_domains_this_run[domain] = position
        for domain, position in candidate_domains_this_run.items():
            candidate_domain[domain] += 1
            domain_positions[domain].append(position)

        cited_urls_this_run = {
            deep_normalize(item.get("url") or "")
            for item in citations
            if deep_normalize(item.get("url") or "")
        }
        cited_domains_this_run = {hostname(url) for url in cited_urls_this_run if hostname(url)}
        citation_url.update(cited_urls_this_run)
        citation_domain.update(cited_domains_this_run)

    avg_url_position = _mean_positions(url_positions)
    avg_domain_position = _mean_positions(domain_positions)
    output: dict[str, dict[str, object]] = {}
    for granularity, candidates, citations, positions in (
        ("url", candidate_url, citation_url, avg_url_position),
        ("domain", candidate_domain, citation_domain, avg_domain_position),
    ):
        output[granularity] = {
            "candidate_counts": candidates,
            "citation_counts": citations,
            "candidate_set": {item for item, count in candidates.items() if count > 0},
            "citation_set": {item for item, count in citations.items() if count > 0},
            "candidate_ranking": _rank_by_frequency(candidates, positions=positions),
            "citation_ranking": _rank_by_frequency(citations, candidates, positions),
        }
    quality = {
        "n_runs": len(files),
        "citation_only_instances": citation_only_instances,
        "citation_only_unique_urls": len(citation_only_urls),
    }
    return output, quality


def _load_google_proxy(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    url_position: dict[str, int] = {}
    global_position = 0
    for page in data.get("pages") or []:
        for result in page.get("results") or []:
            global_position += 1
            normalized = deep_normalize(result.get("link") or "")
            if normalized:
                url_position.setdefault(normalized, global_position)
    if not url_position:
        for global_position, raw in enumerate(data.get("all_urls") or [], 1):
            normalized = deep_normalize(raw)
            if normalized:
                url_position.setdefault(normalized, global_position)

    domain_position: dict[str, int] = {}
    for url, position in url_position.items():
        domain = hostname(url)
        if domain and (domain not in domain_position or position < domain_position[domain]):
            domain_position[domain] = position

    return {
        "url": {
            "set": set(url_position),
            "ranking": [item for item, _ in sorted(url_position.items(), key=lambda row: (row[1], row[0]))],
        },
        "domain": {
            "set": set(domain_position),
            "ranking": [item for item, _ in sorted(domain_position.items(), key=lambda row: (row[1], row[0]))],
        },
    }


def _load_entity_brand_proxy(cache_path: Path) -> tuple[Counter[str], dict[str, dict[str, object]], str]:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing entity cache {cache_path}; run the dataset's c_entity.py first"
        )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    counts = payload["counts"]
    w_cite = Counter({item: count for item, count in counts["W_cite"].items() if count > 0})
    a_candidate = Counter({item: count for item, count in counts["A_exp"].items() if count > 0})
    a_citation = Counter({item: count for item, count in counts["A_cite"].items() if count > 0})
    g_exposure = Counter({item: count for item, count in counts["G_exp"].items() if count > 0})
    g_position = {item: int(position) for item, position in counts["G_pos"].items()}
    a_n_urls = Counter(counts.get("A_n") or {})
    g_n_urls = Counter(counts.get("G_n") or {})

    g_ranking = sorted(
        g_exposure,
        key=lambda item: (
            -g_exposure[item],
            -g_n_urls.get(item, 0),
            g_position.get(item, math.inf),
            item,
        ),
    )
    a_candidate_ranking = _rank_by_frequency(a_candidate, a_n_urls)
    a_citation_ranking = _rank_by_frequency(a_citation, a_candidate)
    proxies = _with_candidate_union({
        "G": {"set": set(g_exposure), "ranking": g_ranking},
        "A_candidate": {"set": set(a_candidate), "ranking": a_candidate_ranking},
        "A_citation": {"set": set(a_citation), "ranking": a_citation_ranking},
    })
    return w_cite, proxies, payload.get("method", "unknown")


def _with_candidate_union(
    proxies: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Add the unranked set union of Google and API candidates."""
    merged = dict(proxies)
    merged[UNION_PROXY] = {
        "set": set(proxies["G"]["set"]) | set(proxies["A_candidate"]["set"]),
        "ranking": None,
    }
    return merged


def _score_granularity(
    relevance: Counter[str],
    proxies: dict[str, dict[str, object]],
) -> dict[str, object]:
    target = {item for item, count in relevance.items() if count > 0}
    scored: dict[str, object] = {}
    for proxy_name in PROXIES:
        proxy_set = set(proxies[proxy_name]["set"])
        visibility = visibility_metrics(target, proxy_set)
        ranking_value = proxies[proxy_name].get("ranking")
        if ranking_value is None:
            ranking_metrics = {
                metric: None
                for k in K_VALUES
                for metric in (
                    f"precision@{k}",
                    f"recall@{k}",
                    f"f1@{k}",
                    f"ndcg@{k}",
                )
            }
        else:
            ranking = list(ranking_value)
            if len(ranking) != len(set(ranking)):
                raise ValueError(f"Duplicate items in {proxy_name} ranking")
            if set(ranking) != proxy_set:
                raise ValueError(f"{proxy_name} ranking and visibility set disagree")
            ranking_metrics = {}
            for k in K_VALUES:
                top_k = top_k_visibility_metrics(target, ranking, k)
                ranking_metrics.update(
                    {
                        f"precision@{k}": None if top_k is None else top_k["precision"],
                        f"recall@{k}": None if top_k is None else top_k["recall"],
                        f"f1@{k}": None if top_k is None else top_k["f1"],
                        f"ndcg@{k}": ndcg_at_k(relevance, ranking, k),
                    }
                )
        scored[proxy_name] = {
            "visibility": visibility,
            "ranking": ranking_metrics,
        }
    return {
        "n_target_items": len(target),
        "sum_target_relevance": sum(relevance.values()),
        "proxies": scored,
    }


def _macro_average(per_question: list[dict[str, object]]) -> dict[str, object]:
    macro: dict[str, object] = {}
    for granularity in GRANULARITIES:
        granularity_out: dict[str, object] = {}
        for proxy_name in PROXIES:
            rows = [
                question["granularities"][granularity]["proxies"][proxy_name]
                for question in per_question
            ]
            visibility_rows = [row["visibility"] for row in rows if row["visibility"] is not None]
            granularity_out[proxy_name] = {
                "visibility": {
                    metric: summarize(row[metric] for row in visibility_rows)
                    for metric in (
                        "precision",
                        "recall",
                        "f1",
                        "target_size",
                        "proxy_size",
                        "intersection",
                    )
                },
                "ranking": {
                    metric: summarize(row["ranking"][metric] for row in rows)
                    for k in K_VALUES
                    for metric in (
                        f"precision@{k}",
                        f"recall@{k}",
                        f"f1@{k}",
                        f"ndcg@{k}",
                    )
                },
            }
        macro[granularity] = granularity_out
    return macro


def evaluate_dataset(dataset_root: Path) -> dict[str, object]:
    api_root = _api_root(dataset_root)
    question_rows: list[dict[str, object]] = []
    total_w_runs = 0
    total_api_runs = 0
    citation_only_instances = 0
    citation_only_urls = 0
    entity_methods: set[str] = set()

    for subdataset, qid, question_dir in _question_dirs(dataset_root):
        web_relevance, n_w_runs = _load_web_relevance(question_dir)
        api, api_quality = _load_api_proxies(api_root / subdataset / qid)
        google = _load_google_proxy(
            dataset_root / "v9_google_pool" / "google_pools" / f"{subdataset}__{qid}.json"
        )
        cache_path = (
            dataset_root
            / "v9_trend_analysis"
            / "results"
            / "entity_llm_cache"
            / f"{subdataset}__{qid}.json"
        )
        entity_relevance, entity_proxies, entity_method = _load_entity_brand_proxy(cache_path)
        entity_methods.add(entity_method)

        granularities: dict[str, object] = {}
        for granularity in ("url", "domain"):
            if not set(api[granularity]["citation_set"]).issubset(
                api[granularity]["candidate_set"]
            ):
                raise ValueError(f"A_citation is not a subset of A_candidate for {subdataset}/{qid}")
            proxies = _with_candidate_union({
                "G": google[granularity],
                "A_candidate": {
                    "set": api[granularity]["candidate_set"],
                    "ranking": api[granularity]["candidate_ranking"],
                },
                "A_citation": {
                    "set": api[granularity]["citation_set"],
                    "ranking": api[granularity]["citation_ranking"],
                },
            })
            granularities[granularity] = _score_granularity(
                web_relevance[granularity], proxies
            )
        granularities["entity_brand"] = _score_granularity(
            entity_relevance, entity_proxies
        )

        total_w_runs += n_w_runs
        total_api_runs += int(api_quality["n_runs"])
        citation_only_instances += int(api_quality["citation_only_instances"])
        citation_only_urls += int(api_quality["citation_only_unique_urls"])
        question_rows.append(
            {
                "subdataset": subdataset,
                "qid": qid,
                "n_w_runs": n_w_runs,
                "n_api_runs": api_quality["n_runs"],
                "citation_only_api_instances": api_quality["citation_only_instances"],
                "granularities": granularities,
            }
        )

    if not question_rows:
        raise ValueError(f"No questions found in {dataset_root}")
    return {
        "dataset": dataset_root.name,
        "n_questions": len(question_rows),
        "n_w_runs": total_w_runs,
        "n_api_runs": total_api_runs,
        "citation_only_api_instances": citation_only_instances,
        "citation_only_api_unique_urls_sum": citation_only_urls,
        "entity_methods": sorted(entity_methods),
        "macro": _macro_average(question_rows),
        "per_question": question_rows,
    }


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


def _decimal(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Unified Proxy Effectiveness: Visibility and Ranking",
        "",
        "W citations are the target. G, A_candidate, A_citation, and the visibility-only "
        "G_union_A_candidate set are evaluated at URL, hostname, and entity/brand granularity. "
        "Every metric is computed per question and macro-averaged within each dataset.",
        "",
        "- **Visibility:** precision, recall, and F1 over unique item sets.",
        "- **Top-K visibility:** binary precision, recall, and F1 at each cutoff; precision uses "
        "the fixed K denominator.",
        "- **Ranking:** linear-gain NDCG with W citation frequency/cite-exposure as relevance.",
        "- **A_candidate ranking:** cross-run candidate appearance frequency at URL/domain "
        "level; entity/brand scores sum the exposure counts of containing URLs. Ties use native "
        "source position or entity URL support.",
        "- **A_citation ranking:** cross-run citation frequency at URL/domain level and summed "
        "cite-exposure at entity/brand level; ties use candidate exposure.",
        "- **G ranking:** native SERP order at URL/domain level; entity-result exposure with "
        "best SERP position as a tie-break at entity/brand level.",
        "- **G_union_A_candidate:** set union used only for visibility P/R/F1; NDCG is not "
        "reported because no rank-fusion rule is assumed.",
        "",
    ]
    for model in result["models"]:
        lines.extend(
            [
                f"## {model['dataset']}",
                "",
                f"Questions: {model['n_questions']}; W runs: {model['n_w_runs']}; "
                f"API runs: {model['n_api_runs']}; citation-only API instances retained: "
                f"{model['citation_only_api_instances']}.",
                "",
            ]
        )
        for granularity in GRANULARITIES:
            title = {
                "url": "URL",
                "domain": "Hostname",
                "entity_brand": "Entity / Brand",
            }[granularity]
            lines.extend(
                [
                    f"### {title}",
                    "",
                    "| Proxy | Precision | Recall | F1 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for proxy_name in PROXIES:
                row = model["macro"][granularity][proxy_name]
                visibility = row["visibility"]
                ranking = row["ranking"]
                lines.append(
                    f"| {proxy_name} "
                    f"| {_pct(visibility['precision']['mean'])} "
                    f"| {_pct(visibility['recall']['mean'])} "
                    f"| {_pct(visibility['f1']['mean'])} "
                    + " ".join(
                        f"| {_decimal(ranking[f'ndcg@{k}']['mean'])}"
                        for k in K_VALUES
                    )
                    + " |"
                )
            lines.extend(
                [
                    "",
                    "| Proxy | P/R/F1@1 | P/R/F1@3 | P/R/F1@5 | P/R/F1@10 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for proxy_name in PROXIES:
                ranking = model["macro"][granularity][proxy_name]["ranking"]
                lines.append(
                    f"| {proxy_name} "
                    + " ".join(
                        f"| {_pct(ranking[f'precision@{k}']['mean'])}"
                        f"/{_pct(ranking[f'recall@{k}']['mean'])}"
                        f"/{_pct(ranking[f'f1@{k}']['mean'])}"
                        for k in K_VALUES
                    )
                    + " |"
                )
            lines.extend(
                [
                    "",
                    "| Proxy | Mean W target size | Mean proxy size | Mean intersection |",
                    "|---|---:|---:|---:|",
                ]
            )
            for proxy_name in PROXIES:
                visibility = model["macro"][granularity][proxy_name]["visibility"]
                lines.append(
                    f"| {proxy_name} | {visibility['target_size']['mean']:.1f} "
                    f"| {visibility['proxy_size']['mean']:.1f} "
                    f"| {visibility['intersection']['mean']:.1f} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "P/R/F1 measures whether a proxy makes the same items visible at all; NDCG measures "
            "whether the proxy's induced order prioritizes items that W cites frequently. The two "
            "families answer different questions and should be reported together. P/R/F1@K adds "
            "the cutoff-aware visibility view used by the prediction experiment.",
            "",
            "Hostname P/R/F1 is computed symmetrically over unique normalized hostname sets on "
            "both the W target and each proxy.",
            "",
            "Entity/brand is the combined named-entity layer produced by the existing v11-style "
            "pipeline (English spaCy candidates plus GPT-4.1-mini filtering; Chinese direct "
            "GPT-4.1-mini extraction). It is not a separate manual brand-only annotation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset directories relative to the repository root.",
    )
    parser.add_argument(
        "--output-prefix",
        default="result/proxy_effectiveness_1600run",
        help="Output path without extension, relative to the repository root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = []
    for dataset in args.datasets:
        dataset_root = (ROOT / dataset).resolve()
        print(f"Evaluating {dataset_root.name}...")
        models.append(evaluate_dataset(dataset_root))
    result = {
        "method": {
            "target": "ChatGPT Web cited items",
            "proxies": list(PROXIES),
            "ranked_proxies": list(RANKED_PROXIES),
            "visibility_only_proxies": [UNION_PROXY],
            "granularities": list(GRANULARITIES),
            "aggregation": "per-question metric, then macro-average",
            "url_normalization": "deep_normalize",
            "domain_normalization": "normalized hostname with subdomains retained",
            "ndcg_gain": "linear W citation frequency/cite-exposure",
            "top_k_relevance": "binary membership in the W cited-item set",
            "precision_at_k_denominator": "fixed k",
            "entity_brand_relevance": "sum of W cited-URL exposure counts per extracted entity",
            "k_values": list(K_VALUES),
        },
        "models": models,
    }
    output_prefix = (ROOT / args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    markdown_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
