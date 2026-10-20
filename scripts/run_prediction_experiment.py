"""Predict ChatGPT Web citations from Google and Responses API evidence.

The experiment implements the Prediction design documented on the Geo Notion
page.  It is deliberately a conditional prediction task: candidates are drawn
from G union A_candidate, while binary labels come from ChatGPT Web citations.
Models are trained independently for each model version and granularity with
query-grouped five-fold out-of-fold evaluation.

No ChatGPT Web feature is exposed to the predictor.  W data is used only for
the binary label and candidate-pool coverage ceiling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from xgboost import XGBClassifier

try:
    from evaluate_proxy_effectiveness import deep_normalize, hostname
except ModuleNotFoundError:  # Imported as scripts.run_prediction_experiment.
    from scripts.evaluate_proxy_effectiveness import deep_normalize, hostname


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("dataset-5.3-1600", "dataset-5.5-1600")
SUBDATASETS = (
    "v9_analysis",
    "v9_factual_lookup",
    "v9_info_synthesis",
    "v9_recommendation",
)
GRANULARITIES = ("url", "domain", "entity_brand")
METHODS = ("XGBoost", "A_citation_only", "G_A_RRF", "Random")
N_FOLDS = 5
RRF_K0 = 60.0
RANDOM_REPEATS = 100
BOOTSTRAP_REPEATS = 2000
SEED = 20260713
TOP_K_VALUES = (1, 3, 5, 10)
CONTEXT_FEATURES = (
    "intent_factual_lookup",
    "intent_info_synthesis",
    "intent_recommendation",
    "language_chinese",
)
COMPACT_FEATURES = {
    "url": (
        "rrf_score",
        "api_best_rank_log",
        "api_appear_rate",
        "api_citation_given_candidate",
        "snippet_length_log",
        *CONTEXT_FEATURES,
    ),
    "domain": (
        "rrf_score",
        "api_best_rank_log",
        "api_appear_rate",
        "api_citation_given_candidate",
        "snippet_length_log",
        "api_url_count_log",
        *CONTEXT_FEATURES,
    ),
    "entity_brand": (
        "rrf_score",
        "api_exposure_per_run",
        "api_citation_rate",
        "source_consensus",
        "api_url_support_log",
        *CONTEXT_FEATURES,
    ),
}
MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "what",
    "which",
    "with",
}


@dataclass
class SourceObservation:
    """Aggregated non-W evidence for one normalized URL."""

    positions: list[float] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    snippets: list[str] = field(default_factory=list)
    recency_days: list[float] = field(default_factory=list)
    appear_runs: int = 0
    citation_runs: int = 0
    positions_by_run: dict[str, float] = field(default_factory=dict)
    citation_run_ids: set[str] = field(default_factory=set)


@dataclass
class CandidateRow:
    """One candidate x query example."""

    group: str
    item: str
    label: int
    relevance: float
    features: dict[str, float]
    a_citation_score: float
    rrf_score: float


@dataclass
class QuestionBundle:
    """All source-side evidence and W targets for one question."""

    dataset: str
    subdataset: str
    qid: str
    query: str
    language: str
    intent: str
    n_api_runs: int
    web_url_relevance: Counter[str]
    web_domain_relevance: Counter[str]
    google_urls: dict[str, SourceObservation]
    api_urls: dict[str, SourceObservation]
    entity_cache: dict[str, object]

    @property
    def group(self) -> str:
        return f"{self.subdataset}/{self.qid}"


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def reciprocal_rank(rank: float | None) -> float:
    return 0.0 if rank is None or not math.isfinite(rank) else 1.0 / rank


def rrf_score(google_rank: float | None, api_rank: float | None) -> float:
    score = 0.0
    if google_rank is not None and math.isfinite(google_rank):
        score += 1.0 / (RRF_K0 + google_rank)
    if api_rank is not None and math.isfinite(api_rank):
        score += 1.0 / (RRF_K0 + api_rank)
    return score


def _safe_mean(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else math.nan


def _safe_min(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return min(clean) if clean else math.nan


def _safe_max(values: Iterable[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return max(clean) if clean else math.nan


def _log_value(value: float | None) -> float:
    if value is None or not math.isfinite(value) or value < 0:
        return math.nan
    return math.log1p(value)


def _text_tokens(text: str) -> set[str]:
    lowered = text.lower()
    latin = {
        token
        for token in re.findall(r"[a-z0-9]+", lowered)
        if token not in STOPWORDS and len(token) > 1
    }
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    cjk = {
        run[index : index + 2]
        for run in cjk_runs
        for index in range(max(1, len(run) - 1))
        if len(run[index : index + 2]) == 2
    }
    return latin | cjk


def query_overlap(query: str, text: str) -> float:
    query_tokens = _text_tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _text_tokens(text)) / len(query_tokens)


def _parse_date_candidate(text: str, anchor: datetime) -> float | None:
    """Return publication age in days from explicit or relative date text."""
    if not text:
        return None
    lowered = text.lower()
    relative = re.search(
        r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?\s+ago", lowered
    )
    if relative:
        value = float(relative.group(1))
        multiplier = {"day": 1.0, "week": 7.0, "month": 30.4375, "year": 365.25}
        return value * multiplier[relative.group(2)]

    patterns = (
        r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})\b",
        r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b",
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b",
    )
    for pattern_index, pattern in enumerate(patterns):
        match = re.search(pattern, text)
        if not match:
            continue
        try:
            if pattern_index == 0:
                month = MONTHS[match.group(1).lower()]
                day, year = int(match.group(2)), int(match.group(3))
            elif pattern_index == 1:
                day = int(match.group(1))
                month = MONTHS[match.group(2).lower()]
                year = int(match.group(3))
            else:
                year, month, day = map(int, match.groups())
            parsed = datetime(year, month, day, tzinfo=timezone.utc)
            return max(0.0, (anchor - parsed).total_seconds() / 86400.0)
        except (KeyError, ValueError):
            continue
    return None


def _api_root(dataset_root: Path) -> Path:
    for name in ("api_responses", "api_responses_parsed"):
        candidate = dataset_root / "v9_api_pool" / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No parsed API response directory under {dataset_root}")


def _dataset_anchor(api_root: Path) -> datetime:
    timestamps: list[datetime] = []
    for path in api_root.glob("**/run_*.json"):
        match = re.search(r"_(\d{8}T\d{6})\.json$", path.name)
        if match:
            timestamps.append(
                datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(
                    tzinfo=timezone.utc
                )
            )
    if not timestamps:
        raise ValueError(f"Could not infer collection date from {api_root}")
    return max(timestamps)


def _load_web_targets(question_dir: Path) -> tuple[Counter[str], Counter[str]]:
    urls: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    for path in sorted(question_dir.glob("*_detail.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_urls: set[str] = set()
        run_domains: set[str] = set()
        for item in payload.get("all_urls") or []:
            if not item.get("cited"):
                continue
            normalized = deep_normalize(item.get("url") or "")
            domain = hostname(normalized)
            if normalized:
                run_urls.add(normalized)
            if domain:
                run_domains.add(domain)
        urls.update(run_urls)
        domains.update(run_domains)
    return urls, domains


def _load_google_urls(path: Path, anchor: datetime) -> tuple[dict[str, SourceObservation], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: dict[str, SourceObservation] = defaultdict(SourceObservation)
    for page in payload.get("pages") or []:
        start = int(page.get("start") or 0)
        for fallback_position, result in enumerate(page.get("results") or [], 1):
            normalized = deep_normalize(result.get("link") or "")
            if not normalized:
                continue
            position = start + int(result.get("position") or fallback_position)
            observation = output[normalized]
            observation.positions.append(float(position))
            observation.titles.append(result.get("title") or "")
            observation.snippets.append(result.get("snippet") or "")
            date_text = " ".join(
                str(result.get(key) or "") for key in ("date", "displayed_link", "snippet")
            )
            age = _parse_date_candidate(date_text, anchor)
            if age is not None:
                observation.recency_days.append(age)
    metadata = {
        "query": payload.get("query") or "",
        "language": (payload.get("language") or "unknown").lower(),
        "intent": (payload.get("primary_intent") or path.stem.split("__", 1)[0]).lower(),
    }
    return dict(output), metadata


def _load_api_urls(
    question_dir: Path, anchor: datetime
) -> tuple[dict[str, SourceObservation], int]:
    output: dict[str, SourceObservation] = defaultdict(SourceObservation)
    paths = sorted(question_dir.glob("run_*.json")) if question_dir.exists() else []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        search_results: dict[str, dict[str, object]] = {}
        for result in payload.get("search_results") or []:
            normalized = deep_normalize(result.get("url") or "")
            if normalized:
                search_results.setdefault(normalized, result)

        positions: dict[str, int] = {}
        for position, source in enumerate(payload.get("sources") or [], 1):
            raw_url = source.get("url") if isinstance(source, dict) else source
            normalized = deep_normalize(raw_url or "")
            if normalized:
                positions.setdefault(normalized, position)

        citations = {
            deep_normalize(item.get("url") or "")
            for item in payload.get("citations") or []
            if deep_normalize(item.get("url") or "")
        }
        for citation_offset, normalized in enumerate(sorted(citations), 1):
            positions.setdefault(normalized, len(positions) + citation_offset)

        for normalized, position in positions.items():
            observation = output[normalized]
            observation.appear_runs += 1
            observation.positions.append(float(position))
            observation.positions_by_run[path.name] = float(position)
            result = search_results.get(normalized) or {}
            title = str(result.get("title") or "")
            snippet = str(result.get("snippet") or "")
            observation.titles.append(title)
            observation.snippets.append(snippet)
            age = _parse_date_candidate(f"{title} {snippet}", anchor)
            if age is not None:
                observation.recency_days.append(age)
        for normalized in citations:
            output[normalized].citation_runs += 1
            output[normalized].citation_run_ids.add(path.name)
    return dict(output), len(paths)


def load_questions(dataset_root: Path) -> list[QuestionBundle]:
    api_root = _api_root(dataset_root)
    anchor = _dataset_anchor(api_root)
    questions: list[QuestionBundle] = []
    for subdataset in SUBDATASETS:
        search_root = dataset_root / subdataset / "search_detail"
        if not search_root.exists():
            continue
        for question_dir in sorted(search_root.iterdir()):
            if not question_dir.is_dir() or not question_dir.name.startswith("q"):
                continue
            qid = question_dir.name
            google_path = (
                dataset_root
                / "v9_google_pool"
                / "google_pools"
                / f"{subdataset}__{qid}.json"
            )
            google_urls, metadata = _load_google_urls(google_path, anchor)
            api_urls, n_api_runs = _load_api_urls(api_root / subdataset / qid, anchor)
            web_urls, web_domains = _load_web_targets(question_dir)
            entity_path = (
                dataset_root
                / "v9_trend_analysis"
                / "results"
                / "entity_llm_cache"
                / f"{subdataset}__{qid}.json"
            )
            entity_cache = json.loads(entity_path.read_text(encoding="utf-8"))
            questions.append(
                QuestionBundle(
                    dataset=dataset_root.name,
                    subdataset=subdataset,
                    qid=qid,
                    query=metadata["query"],
                    language=metadata["language"],
                    intent=metadata["intent"],
                    n_api_runs=n_api_runs,
                    web_url_relevance=web_urls,
                    web_domain_relevance=web_domains,
                    google_urls=google_urls,
                    api_urls=api_urls,
                    entity_cache=entity_cache,
                )
            )
    if not questions:
        raise ValueError(f"No questions found in {dataset_root}")
    return questions


def _categorical_features(intent: str, language: str, item: str) -> dict[str, float]:
    intent_key = re.sub(r"[^a-z0-9]+", "_", intent.lower()).strip("_")
    language_key = re.sub(r"[^a-z0-9]+", "_", language.lower()).strip("_")
    host = hostname(item) if "://" in item else item
    suffix = host.rsplit(".", 1)[-1] if "." in host else "none"
    if suffix in {"gov", "edu", "org", "com", "net"}:
        tld = suffix
    elif len(suffix) == 2:
        tld = "country"
    else:
        tld = "other"
    features = {
        f"intent_{name}": float(intent_key == name)
        for name in ("analysis", "factual_lookup", "info_synthesis", "recommendation")
    }
    features.update(
        {
            "language_english": float(language_key == "english"),
            "language_chinese": float(language_key == "chinese"),
        }
    )
    features.update(
        {
            f"tld_{name}": float(tld == name)
            for name in ("com", "org", "net", "gov", "edu", "country", "other", "none")
        }
    )
    return features


def _url_static_features(url: str) -> dict[str, float]:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split("/") if part]
    return {
        "path_depth": float(len(path_parts)),
        "query_parameter_count": float(len(parse_qs(parsed.query, keep_blank_values=True))),
        "is_homepage": float(not path_parts),
    }


def _url_features(question: QuestionBundle, url: str) -> tuple[dict[str, float], float, float]:
    google = question.google_urls.get(url)
    api = question.api_urls.get(url)
    in_google = float(google is not None)
    in_api = float(api is not None)
    google_rank = _safe_min(google.positions) if google else math.nan
    api_best_rank = _safe_min(api.positions) if api else math.nan
    api_mean_rank = _safe_mean(api.positions) if api else math.nan
    google_title = " ".join(google.titles) if google else ""
    google_snippet = " ".join(google.snippets) if google else ""
    api_title = " ".join(api.titles) if api else ""
    api_snippet = " ".join(api.snippets) if api else ""
    google_snippet_length_log = math.log1p(len(google_snippet))
    api_snippet_length_log = math.log1p(
        len(api_snippet) / max(1, api.appear_runs if api else 1)
    )
    api_appear_rate = (api.appear_runs / question.n_api_runs) if api and question.n_api_runs else 0.0
    api_citation_rate = (
        api.citation_runs / question.n_api_runs if api and question.n_api_runs else 0.0
    )
    citation_given_candidate = (
        api.citation_runs / api.appear_runs if api and api.appear_runs else 0.0
    )
    rank_gap = (
        reciprocal_rank(google_rank) - reciprocal_rank(api_mean_rank)
        if math.isfinite(google_rank) and math.isfinite(api_mean_rank)
        else 0.0
    )
    google_recency = _safe_min(google.recency_days) if google else math.nan
    api_recency = _safe_min(api.recency_days) if api else math.nan
    rrf = rrf_score(
        google_rank if math.isfinite(google_rank) else None,
        api_mean_rank if math.isfinite(api_mean_rank) else None,
    )
    features = {
        "in_google": in_google,
        "in_api_candidate": in_api,
        "in_api_citation": float(bool(api and api.citation_runs)),
        "source_consensus": in_google * in_api,
        "google_rank_log": _log_value(google_rank),
        "google_rank_reciprocal": reciprocal_rank(google_rank),
        "google_rank_missing": float(not math.isfinite(google_rank)),
        "api_best_rank_log": _log_value(api_best_rank),
        "api_mean_rank_log": _log_value(api_mean_rank),
        "api_rank_reciprocal": reciprocal_rank(api_mean_rank),
        "api_rank_missing": float(not math.isfinite(api_mean_rank)),
        "rank_reciprocal_gap": rank_gap,
        "api_appear_rate": api_appear_rate,
        "api_citation_rate": api_citation_rate,
        "api_citation_given_candidate": citation_given_candidate,
        "google_snippet_length_log": google_snippet_length_log,
        "api_snippet_length_log": api_snippet_length_log,
        "snippet_length_log": max(google_snippet_length_log, api_snippet_length_log),
        "google_recency_days_log": _log_value(google_recency),
        "api_recency_days_log": _log_value(api_recency),
        "google_recency_missing": float(not math.isfinite(google_recency)),
        "api_recency_missing": float(not math.isfinite(api_recency)),
        "title_query_overlap": max(
            query_overlap(question.query, google_title),
            query_overlap(question.query, api_title),
        ),
        "snippet_query_overlap": max(
            query_overlap(question.query, google_snippet),
            query_overlap(question.query, api_snippet),
        ),
        "rrf_score": rrf,
    }
    features.update(_url_static_features(url))
    features.update(_categorical_features(question.intent, question.language, url))
    return features, api_citation_rate, rrf


def build_url_rows(question: QuestionBundle) -> list[CandidateRow]:
    candidate_urls = set(question.google_urls) | set(question.api_urls)
    google_order = {
        item: rank
        for rank, item in enumerate(
            sorted(
                question.google_urls,
                key=lambda item: (_safe_min(question.google_urls[item].positions), item),
            ),
            1,
        )
    }
    api_order = {
        item: rank
        for rank, item in enumerate(
            sorted(
                question.api_urls,
                key=lambda item: (
                    -question.api_urls[item].appear_runs,
                    _safe_mean(question.api_urls[item].positions),
                    item,
                ),
            ),
            1,
        )
    }
    rows: list[CandidateRow] = []
    for url in sorted(candidate_urls):
        features, citation_score, _ = _url_features(question, url)
        google_aggregate_rank = float(google_order[url]) if url in google_order else math.nan
        api_aggregate_rank = float(api_order[url]) if url in api_order else math.nan
        rrf = rrf_score(
            google_aggregate_rank if math.isfinite(google_aggregate_rank) else None,
            api_aggregate_rank if math.isfinite(api_aggregate_rank) else None,
        )
        features.update(
            {
                "google_aggregate_rank_log": _log_value(google_aggregate_rank),
                "api_aggregate_rank_log": _log_value(api_aggregate_rank),
                "rrf_score": rrf,
            }
        )
        relevance = float(question.web_url_relevance.get(url, 0))
        rows.append(
            CandidateRow(
                group=question.group,
                item=url,
                label=int(relevance > 0),
                relevance=relevance,
                features=features,
                a_citation_score=citation_score,
                rrf_score=rrf,
            )
        )
    return rows


def _aggregate_domain_features(rows: list[CandidateRow], domain: str) -> dict[str, float]:
    feature_names = set().union(*(row.features for row in rows))
    features: dict[str, float] = {}
    min_names = {
        "google_rank_log",
        "api_best_rank_log",
        "api_mean_rank_log",
        "google_recency_days_log",
        "api_recency_days_log",
        "path_depth",
    }
    max_names = {
        "in_google",
        "in_api_candidate",
        "in_api_citation",
        "source_consensus",
        "google_rank_reciprocal",
        "api_rank_reciprocal",
        "api_appear_rate",
        "api_citation_rate",
        "api_citation_given_candidate",
        "google_snippet_length_log",
        "api_snippet_length_log",
        "snippet_length_log",
        "title_query_overlap",
        "snippet_query_overlap",
        "rrf_score",
        "is_homepage",
    }
    for name in feature_names:
        values = [row.features.get(name, math.nan) for row in rows]
        if name in min_names:
            features[name] = _safe_min(values)
        elif name in max_names or name.startswith(("intent_", "language_", "tld_")):
            features[name] = _safe_max(values)
        elif name.endswith("_missing"):
            features[name] = _safe_min(values)
        else:
            features[name] = _safe_mean(values)
    features["candidate_url_count_log"] = math.log1p(len(rows))
    features["google_url_count_log"] = math.log1p(
        sum(row.features["in_google"] > 0 for row in rows)
    )
    features["api_url_count_log"] = math.log1p(
        sum(row.features["in_api_candidate"] > 0 for row in rows)
    )
    features.update(_categorical_features("", "", domain))
    # Preserve query-level one-hot values overwritten by the host-only call.
    for row in rows[:1]:
        for name, value in row.features.items():
            if name.startswith(("intent_", "language_")):
                features[name] = value
    return features


def build_domain_rows(question: QuestionBundle) -> list[CandidateRow]:
    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in build_url_rows(question):
        domain = hostname(row.item)
        if domain:
            grouped[domain].append(row)
    google_native_rank: dict[str, float] = {}
    api_positions_by_domain: dict[str, dict[str, float]] = defaultdict(dict)
    api_citations_by_domain: dict[str, set[str]] = defaultdict(set)
    for domain, url_rows in grouped.items():
        google_native_rank[domain] = _safe_min(
            position
            for row in url_rows
            for position in (
                question.google_urls[row.item].positions
                if row.item in question.google_urls
                else []
            )
        )
        for row in url_rows:
            observation = question.api_urls.get(row.item)
            if observation is None:
                continue
            for run_id, position in observation.positions_by_run.items():
                api_positions_by_domain[domain][run_id] = min(
                    api_positions_by_domain[domain].get(run_id, math.inf), position
                )
            api_citations_by_domain[domain].update(observation.citation_run_ids)
    google_order = {
        item: rank
        for rank, item in enumerate(
            sorted(
                (item for item, rank in google_native_rank.items() if math.isfinite(rank)),
                key=lambda item: (google_native_rank[item], item),
            ),
            1,
        )
    }
    api_order = {
        item: rank
        for rank, item in enumerate(
            sorted(
                api_positions_by_domain,
                key=lambda item: (
                    -len(api_positions_by_domain[item]),
                    _safe_mean(api_positions_by_domain[item].values()),
                    item,
                ),
            ),
            1,
        )
    }
    rows: list[CandidateRow] = []
    for domain, url_rows in sorted(grouped.items()):
        relevance = float(question.web_domain_relevance.get(domain, 0))
        features = _aggregate_domain_features(url_rows, domain)
        google_rank = google_native_rank[domain]
        api_positions_by_run = api_positions_by_domain[domain]
        api_citation_runs = api_citations_by_domain[domain]
        api_positions = list(api_positions_by_run.values())
        api_best_rank = _safe_min(api_positions)
        api_mean_rank = _safe_mean(api_positions)
        api_appear_rate = len(api_positions_by_run) / max(1, question.n_api_runs)
        api_citation_rate = len(api_citation_runs) / max(1, question.n_api_runs)
        google_aggregate_rank = float(google_order[domain]) if domain in google_order else math.nan
        api_aggregate_rank = float(api_order[domain]) if domain in api_order else math.nan
        domain_rrf = rrf_score(
            google_aggregate_rank if math.isfinite(google_aggregate_rank) else None,
            api_aggregate_rank if math.isfinite(api_aggregate_rank) else None,
        )
        features.update(
            {
                "google_rank_log": _log_value(google_rank),
                "google_rank_reciprocal": reciprocal_rank(google_rank),
                "google_rank_missing": float(not math.isfinite(google_rank)),
                "api_best_rank_log": _log_value(api_best_rank),
                "api_mean_rank_log": _log_value(api_mean_rank),
                "api_rank_reciprocal": reciprocal_rank(api_mean_rank),
                "api_rank_missing": float(not math.isfinite(api_mean_rank)),
                "rank_reciprocal_gap": (
                    reciprocal_rank(google_rank) - reciprocal_rank(api_mean_rank)
                    if math.isfinite(google_rank) and math.isfinite(api_mean_rank)
                    else 0.0
                ),
                "api_appear_rate": api_appear_rate,
                "api_citation_rate": api_citation_rate,
                "api_citation_given_candidate": (
                    len(api_citation_runs) / len(api_positions_by_run)
                    if api_positions_by_run
                    else 0.0
                ),
                "google_aggregate_rank_log": _log_value(google_aggregate_rank),
                "api_aggregate_rank_log": _log_value(api_aggregate_rank),
                "rrf_score": domain_rrf,
            }
        )
        rows.append(
            CandidateRow(
                group=question.group,
                item=domain,
                label=int(relevance > 0),
                relevance=relevance,
                features=features,
                a_citation_score=api_citation_rate,
                rrf_score=domain_rrf,
            )
        )
    return rows


def _counter(payload: dict[str, object], key: str) -> Counter[str]:
    return Counter(
        {
            str(item): float(value)
            for item, value in (payload.get(key) or {}).items()
            if float(value) > 0
        }
    )


def _rank_map(primary: Counter[str], secondary: Counter[str] | None = None) -> dict[str, int]:
    secondary = secondary or Counter()
    ordered = sorted(primary, key=lambda item: (-primary[item], -secondary[item], item))
    return {item: index for index, item in enumerate(ordered, 1)}


def build_entity_rows(question: QuestionBundle) -> list[CandidateRow]:
    counts = question.entity_cache["counts"]
    w_cite = _counter(counts, "W_cite")
    g_exp = _counter(counts, "G_exp")
    g_n = _counter(counts, "G_n")
    g_pos = {str(item): float(value) for item, value in (counts.get("G_pos") or {}).items()}
    a_exp = _counter(counts, "A_exp")
    a_cite = _counter(counts, "A_cite")
    a_n = _counter(counts, "A_n")
    candidate_entities = set(g_exp) | set(a_exp)
    g_rank = _rank_map(g_exp, g_n)
    a_rank = _rank_map(a_exp, a_n)
    g_total = sum(g_exp.values()) or 1.0
    a_total = sum(a_exp.values()) or 1.0
    rows: list[CandidateRow] = []
    for entity in sorted(candidate_entities):
        g_rank_value = float(g_rank[entity]) if entity in g_rank else math.nan
        a_rank_value = float(a_rank[entity]) if entity in a_rank else math.nan
        g_value = float(g_exp[entity])
        a_value = float(a_exp[entity])
        a_cite_value = float(a_cite[entity])
        rrf = rrf_score(
            g_rank_value if math.isfinite(g_rank_value) else None,
            a_rank_value if math.isfinite(a_rank_value) else None,
        )
        features = {
            "in_google": float(g_value > 0),
            "in_api_candidate": float(a_value > 0),
            "in_api_citation": float(a_cite_value > 0),
            "source_consensus": float(g_value > 0 and a_value > 0),
            "google_rank_log": _log_value(g_rank_value),
            "google_rank_reciprocal": reciprocal_rank(g_rank_value),
            "google_rank_missing": float(not math.isfinite(g_rank_value)),
            "api_mean_rank_log": _log_value(a_rank_value),
            "api_rank_reciprocal": reciprocal_rank(a_rank_value),
            "api_rank_missing": float(not math.isfinite(a_rank_value)),
            "rank_reciprocal_gap": (
                reciprocal_rank(g_rank_value) - reciprocal_rank(a_rank_value)
                if math.isfinite(g_rank_value) and math.isfinite(a_rank_value)
                else 0.0
            ),
            "google_exposure_log": math.log1p(g_value),
            "google_exposure_share": g_value / g_total,
            "google_url_support_log": math.log1p(float(g_n[entity])),
            "google_best_position_log": _log_value(g_pos.get(entity)),
            "api_exposure_log": math.log1p(a_value),
            "api_exposure_share": a_value / a_total,
            "api_exposure_per_run": a_value / max(1, question.n_api_runs),
            "api_url_support_log": math.log1p(float(a_n[entity])),
            "api_citation_exposure_log": math.log1p(a_cite_value),
            "api_citation_rate": a_cite_value / a_value if a_value else 0.0,
            "entity_query_overlap": query_overlap(question.query, entity),
            "entity_length_log": math.log1p(len(entity)),
            "rrf_score": rrf,
        }
        features.update(_categorical_features(question.intent, question.language, ""))
        relevance = float(w_cite[entity])
        rows.append(
            CandidateRow(
                group=question.group,
                item=entity,
                label=int(relevance > 0),
                relevance=relevance,
                features=features,
                a_citation_score=a_cite_value,
                rrf_score=rrf,
            )
        )
    return rows


def build_rows(questions: list[QuestionBundle], granularity: str) -> list[CandidateRow]:
    builder = {
        "url": build_url_rows,
        "domain": build_domain_rows,
        "entity_brand": build_entity_rows,
    }[granularity]
    return [row for question in questions for row in builder(question)]


def make_fold_map(questions: list[QuestionBundle], n_folds: int = N_FOLDS) -> dict[str, int]:
    """Assign each intent-language stratum round-robin across query folds."""
    strata: dict[tuple[str, str], list[QuestionBundle]] = defaultdict(list)
    for question in questions:
        strata[(question.intent, question.language)].append(question)
    fold_map: dict[str, int] = {}
    fold_sizes = [0] * n_folds
    for stratum, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda item: item.group)
        start_fold = min(range(n_folds), key=lambda fold: (fold_sizes[fold], fold))
        for index, question in enumerate(ordered):
            fold = (start_fold + index) % n_folds
            fold_map[question.group] = fold
            fold_sizes[fold] += 1
    if set(fold_map) != {question.group for question in questions}:
        raise ValueError("Fold assignment does not cover every query")
    return fold_map


def _feature_matrix(
    rows: list[CandidateRow],
    granularity: str,
    feature_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    selected_features = list(feature_names or COMPACT_FEATURES[granularity])
    available = set().union(*(row.features for row in rows))
    missing = set(selected_features) - available
    if missing:
        raise ValueError(f"Missing compact features for {granularity}: {sorted(missing)}")
    matrix = pd.DataFrame(
        [
            {name: row.features.get(name, math.nan) for name in selected_features}
            for row in rows
        ],
        columns=selected_features,
        dtype=float,
    )
    labels = np.asarray([row.label for row in rows], dtype=int)
    return matrix, labels, selected_features


def fit_oof_xgboost(
    rows: list[CandidateRow],
    fold_map: dict[str, int],
    granularity: str,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    matrix, labels, selected_features = _feature_matrix(
        rows, granularity, feature_names
    )
    folds = np.asarray([fold_map[row.group] for row in rows], dtype=int)
    scores = np.full(len(rows), np.nan, dtype=float)
    gain_weighted: Counter[str] = Counter()
    shap_abs_weighted: Counter[str] = Counter()
    shap_signed_weighted: Counter[str] = Counter()
    shap_sample_count = 0
    fold_summaries: list[dict[str, object]] = []

    for fold in range(N_FOLDS):
        train_mask = folds != fold
        test_mask = folds == fold
        y_train = labels[train_mask]
        if len(set(y_train)) != 2:
            raise ValueError(f"Fold {fold} training labels do not contain both classes")
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=350,
            learning_rate=0.04,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=5.0,
            scale_pos_weight=negatives / max(1, positives),
            random_state=SEED + fold,
            n_jobs=4,
            tree_method="hist",
        )
        model.fit(matrix.loc[train_mask], y_train, verbose=False)
        fold_scores = model.predict_proba(matrix.loc[test_mask])[:, 1]
        scores[test_mask] = fold_scores

        gain = model.get_booster().get_score(importance_type="gain")
        gain_total = sum(gain.values()) or 1.0
        test_count = int(test_mask.sum())
        for name in selected_features:
            gain_weighted[name] += (gain.get(name, 0.0) / gain_total) * test_count

        sample_indices = np.flatnonzero(test_mask)
        if len(sample_indices) > 2000:
            rng = np.random.default_rng(SEED + fold)
            sample_indices = np.sort(rng.choice(sample_indices, 2000, replace=False))
        explainer = shap.TreeExplainer(model)
        explanation = explainer(matrix.iloc[sample_indices], check_additivity=False)
        values = np.asarray(explanation.values)
        shap_sample_count += len(sample_indices)
        for index, name in enumerate(selected_features):
            shap_abs_weighted[name] += float(np.abs(values[:, index]).mean()) * len(sample_indices)
            shap_signed_weighted[name] += float(values[:, index].mean()) * len(sample_indices)
        fold_summaries.append(
            {
                "fold": fold,
                "train_queries": len({row.group for row, keep in zip(rows, train_mask) if keep}),
                "test_queries": len({row.group for row, keep in zip(rows, test_mask) if keep}),
                "train_rows": int(train_mask.sum()),
                "test_rows": test_count,
                "train_positive_rate": positives / len(y_train),
                "test_positive_rate": float(labels[test_mask].mean()),
            }
        )

    if np.isnan(scores).any():
        raise ValueError("OOF predictions are incomplete")
    gain_denominator = len(rows)
    shap_denominator = shap_sample_count
    diagnostics = {
        "feature_names": selected_features,
        "n_features": len(selected_features),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "log_loss": float(log_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7))),
        "feature_importance_gain": sorted(
            (
                {"feature": name, "mean_normalized_gain": gain_weighted[name] / gain_denominator}
                for name in selected_features
            ),
            key=lambda row: (-row["mean_normalized_gain"], row["feature"]),
        ),
        "shap": sorted(
            (
                {
                    "feature": name,
                    "mean_abs_shap": shap_abs_weighted[name] / shap_denominator,
                    "mean_signed_shap": shap_signed_weighted[name] / shap_denominator,
                }
                for name in selected_features
            ),
            key=lambda row: (-row["mean_abs_shap"], row["feature"]),
        ),
        "folds": fold_summaries,
    }
    return scores, diagnostics


def classification_metrics(
    rows: list[CandidateRow], scores: Sequence[float]
) -> dict[str, float] | None:
    labels = np.asarray([row.label for row in rows], dtype=int)
    if len(set(labels)) != 2:
        return None
    score_array = np.asarray(scores, dtype=float)
    return {
        "roc_auc": float(roc_auc_score(labels, score_array)),
        "average_precision": float(average_precision_score(labels, score_array)),
    }


def top_k_metrics(
    rows: list[CandidateRow], scores: Sequence[float], k: int
) -> dict[str, float] | None:
    """Return tie-aware conditional P/R/F1@k for one query."""
    if k <= 0:
        raise ValueError("k must be positive")
    positives = sum(row.label for row in rows)
    if not positives:
        return None

    score_groups: dict[float, list[CandidateRow]] = defaultdict(list)
    for row, score in zip(rows, scores):
        score_groups[float(score)].append(row)

    remaining = min(k, len(rows))
    expected_hits = 0.0
    for score in sorted(score_groups, reverse=True):
        group_rows = score_groups[score]
        selected = min(remaining, len(group_rows))
        if selected:
            expected_hits += selected * sum(row.label for row in group_rows) / len(group_rows)
            remaining -= selected
        if remaining == 0:
            break

    precision = expected_hits / k
    recall = expected_hits / positives
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def query_prediction_metrics(
    rows: list[CandidateRow], scores: Sequence[float]
) -> dict[str, float] | None:
    metrics = classification_metrics(rows, scores)
    if metrics is None:
        return None
    for k in TOP_K_VALUES:
        top_k = top_k_metrics(rows, scores, k)
        if top_k is None:
            return None
        metrics.update(
            {
                f"precision@{k}": top_k["precision"],
                f"recall@{k}": top_k["recall"],
                f"f1@{k}": top_k["f1"],
            }
        )
    return metrics


def _bootstrap_summary(values: list[float], seed: int) -> dict[str, float | int]:
    if not values:
        return {"mean": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(BOOTSTRAP_REPEATS, len(array)), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n": len(values),
    }


def evaluate_method(
    rows: list[CandidateRow], scores: Sequence[float], method: str
) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    grouped: dict[str, list[tuple[CandidateRow, float]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[row.group].append((row, float(score)))
    per_query: dict[str, dict[str, float]] = {}
    for group, pairs in sorted(grouped.items()):
        metrics = query_prediction_metrics(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
        )
        if metrics is not None:
            per_query[group] = metrics
    metric_names = (
        "roc_auc",
        "average_precision",
        *(
            f"{metric}@{k}"
            for k in TOP_K_VALUES
            for metric in ("precision", "recall", "f1")
        ),
    )
    summary = {
        metric: _bootstrap_summary(
            [row[metric] for row in per_query.values()],
            stable_seed(SEED, method, metric),
        )
        for metric in metric_names
    }
    pooled = classification_metrics(rows, scores)
    if pooled is None:
        raise ValueError("Pooled prediction rows do not contain both labels")
    summary["pooled"] = pooled
    return summary, per_query


def evaluate_random(
    rows: list[CandidateRow], dataset: str, granularity: str
) -> tuple[dict[str, object], dict[str, dict[str, float]]]:
    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        grouped[row.group].append(row)
    per_query: dict[str, dict[str, float]] = {}
    for group, group_rows in sorted(grouped.items()):
        repeated: list[dict[str, float]] = []
        for repeat in range(RANDOM_REPEATS):
            rng = random.Random(stable_seed(SEED, dataset, granularity, group, repeat))
            metrics = query_prediction_metrics(
                group_rows, [rng.random() for _ in group_rows]
            )
            if metrics is not None:
                repeated.append(metrics)
        if repeated:
            per_query[group] = {
                metric: statistics.fmean(row[metric] for row in repeated)
                for metric in repeated[0]
            }
    summary: dict[str, object] = {
        metric: _bootstrap_summary(
            [row[metric] for row in per_query.values()],
            stable_seed(SEED, dataset, granularity, "Random", metric),
        )
        for metric in next(iter(per_query.values()))
    }
    pooled_repeats = []
    for repeat in range(RANDOM_REPEATS):
        rng = random.Random(stable_seed(SEED, dataset, granularity, "pooled", repeat))
        metrics = classification_metrics(rows, [rng.random() for _ in rows])
        if metrics is not None:
            pooled_repeats.append(metrics)
    summary["pooled"] = {
        metric: statistics.fmean(row[metric] for row in pooled_repeats)
        for metric in pooled_repeats[0]
    }
    return summary, per_query


def _paired_delta(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    metric: str,
    seed: int,
) -> dict[str, float | int]:
    groups = sorted(set(left) & set(right))
    values = np.asarray([left[group][metric] - right[group][metric] for group in groups])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_REPEATS, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "probability_gt_zero": float((draws > 0).mean()),
        "n": len(values),
    }


def _data_summary(
    questions: list[QuestionBundle], rows: list[CandidateRow], granularity: str
) -> dict[str, float | int]:
    grouped: dict[str, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        grouped[row.group].append(row)
    all_target_counts: dict[str, int] = {}
    for question in questions:
        if granularity == "url":
            all_target_counts[question.group] = len(question.web_url_relevance)
        elif granularity == "domain":
            all_target_counts[question.group] = len(question.web_domain_relevance)
        else:
            all_target_counts[question.group] = len(_counter(question.entity_cache["counts"], "W_cite"))
    coverages = []
    covered_micro = 0
    target_micro = 0
    eval_queries = 0
    for group, group_rows in grouped.items():
        covered = sum(row.label for row in group_rows)
        target = all_target_counts[group]
        if target:
            coverages.append(covered / target)
            covered_micro += covered
            target_micro += target
        if covered:
            eval_queries += 1
    positives = sum(row.label for row in rows)
    return {
        "n_rows": len(rows),
        "n_positive_rows": positives,
        "positive_rate": positives / len(rows),
        "mean_candidates_per_query": len(rows) / len(questions),
        "n_queries": len(questions),
        "n_eval_queries_with_covered_positive": eval_queries,
        "candidate_pool_coverage_macro": statistics.fmean(coverages),
        "candidate_pool_coverage_micro": covered_micro / target_micro,
    }


def evaluate_granularity(
    dataset: str,
    questions: list[QuestionBundle],
    granularity: str,
    fold_map: dict[str, int],
) -> dict[str, object]:
    rows = build_rows(questions, granularity)
    xgb_scores, diagnostics = fit_oof_xgboost(rows, fold_map, granularity)
    methods: dict[str, object] = {}
    per_query: dict[str, dict[str, dict[str, float]]] = {}

    methods["XGBoost"], per_query["XGBoost"] = evaluate_method(
        rows, xgb_scores, "XGBoost"
    )
    citation_scores = [row.a_citation_score for row in rows]
    methods["A_citation_only"], per_query["A_citation_only"] = evaluate_method(
        rows, citation_scores, "A_citation_only"
    )
    rrf_scores = [row.rrf_score for row in rows]
    methods["G_A_RRF"], per_query["G_A_RRF"] = evaluate_method(
        rows, rrf_scores, "G_A_RRF"
    )
    methods["Random"], per_query["Random"] = evaluate_random(
        rows, dataset, granularity
    )

    paired: dict[str, object] = {}
    for baseline in ("A_citation_only", "G_A_RRF"):
        paired[baseline] = {
            metric: _paired_delta(
                per_query["XGBoost"],
                per_query[baseline],
                metric,
                stable_seed(SEED, dataset, granularity, baseline, metric),
            )
            for metric in ("roc_auc", "average_precision")
        }
    return {
        "data": _data_summary(questions, rows, granularity),
        "methods": methods,
        "xgboost_diagnostics": diagnostics,
        "paired_xgboost_minus_baseline": paired,
        "per_query_metrics": per_query,
    }


def evaluate_dataset(dataset_root: Path) -> dict[str, object]:
    questions = load_questions(dataset_root)
    fold_map = make_fold_map(questions)
    fold_queries: dict[str, list[str]] = defaultdict(list)
    for group, fold in fold_map.items():
        fold_queries[str(fold)].append(group)
    return {
        "dataset": dataset_root.name,
        "n_questions": len(questions),
        "folds": {fold: sorted(groups) for fold, groups in sorted(fold_queries.items())},
        "granularities": {
            granularity: evaluate_granularity(
                dataset_root.name, questions, granularity, fold_map
            )
            for granularity in GRANULARITIES
        },
    }


def _metric_interval(result: dict[str, object], method: str, metric: str) -> str:
    summary = result["methods"][method][metric]
    return (
        f"{summary['mean']:.3f} "
        f"[{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]"
    )


def _pooled_metric(result: dict[str, object], method: str, metric: str) -> float:
    return float(result["methods"][method]["pooled"][metric])


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _dec(value: float) -> str:
    return f"{value:.3f}"


def _top_k_triplet(result: dict[str, object], method: str, k: int) -> str:
    metrics = result["methods"][method]
    return " / ".join(
        _dec(float(metrics[f"{metric}@{k}"]["mean"]))
        for metric in ("precision", "recall", "f1")
    )


def render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# Conditional Prediction of ChatGPT Web Citations",
        "",
        "Candidates are the per-query G union A_candidate set. Labels are whether a candidate "
        "is ever cited by W across 20 runs; compact XGBoost features use only G, A, and query "
        "metadata. Five query-grouped folds produce out-of-fold scores.",
        "",
        "- Query-macro ROC-AUC is the primary prediction metric; Average Precision is reported because positives are sparse.",
        "- AUC/AP and P/R/F1@1/3/5/10 are computed within each evaluable query and summarized with 2,000 query bootstrap resamples; pooled OOF values are secondary diagnostics.",
        "- Top-K recall is conditional on positives covered by G union A_candidate; open-world misses remain captured by the separate coverage ceiling.",
        "- Candidate-pool coverage is reported separately against all W target items.",
        "- G_A_RRF is a prediction-only reciprocal-rank-fusion score (k0=60); it does not redefine the unranked union used in Section 4 visibility analysis.",
        "- The compact feature set keeps one representative per previously validated signal family and removes redundant rank, citation, exposure, recency, TLD, and URL-shape variants.",
        "- llm_brand_score is excluded because the current design has no fixed, reproducible definition for it.",
        "",
    ]
    for model in result["models"]:
        lines.extend([f"## {model['dataset']}", ""])
        lines.extend(
            [
                "| Granularity | Rows | Positive rate | Macro coverage ceiling | Eval queries |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for granularity in GRANULARITIES:
            data = model["granularities"][granularity]["data"]
            lines.append(
                f"| {granularity} | {data['n_rows']} | {_pct(data['positive_rate'])} "
                f"| {_pct(data['candidate_pool_coverage_macro'])} "
                f"| {data['n_eval_queries_with_covered_positive']} |"
            )
        lines.append("")
        for granularity in GRANULARITIES:
            section = model["granularities"][granularity]
            lines.extend(
                [
                    f"### {granularity}",
                    "",
                    "| Method | Query-macro ROC-AUC [95% CI] | Query-macro AP [95% CI] | Pooled ROC-AUC | Pooled AP |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for method in METHODS:
                lines.append(
                    f"| {method} "
                    f"| {_metric_interval(section, method, 'roc_auc')} "
                    f"| {_metric_interval(section, method, 'average_precision')} "
                    f"| {_dec(_pooled_metric(section, method, 'roc_auc'))} "
                    f"| {_dec(_pooled_metric(section, method, 'average_precision'))} |"
                )
            lines.extend(
                [
                    "",
                    "Query-macro top-K means (each cell: Precision / Recall / F1):",
                    "",
                    "| Method | P/R/F1@1 | P/R/F1@3 | P/R/F1@5 | P/R/F1@10 |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for method in METHODS:
                lines.append(
                    f"| {method} "
                    + " ".join(
                        f"| {_top_k_triplet(section, method, k)}"
                        for k in TOP_K_VALUES
                    )
                    + " |"
                )
            diagnostics = section["xgboost_diagnostics"]
            top_gain = ", ".join(
                f"{row['feature']} ({row['mean_normalized_gain']:.3f})"
                for row in diagnostics["feature_importance_gain"][:5]
            )
            top_shap = ", ".join(
                f"{row['feature']} ({row['mean_abs_shap']:.3f})"
                for row in diagnostics["shap"][:5]
            )
            lines.extend(
                [
                    "",
                    f"OOF ROC-AUC={diagnostics['roc_auc']:.3f}; AP={diagnostics['average_precision']:.3f}; logloss={diagnostics['log_loss']:.3f}.",
                    f"Compact features ({diagnostics['n_features']}): {', '.join(diagnostics['feature_names'])}.",
                    f"Top gain: {top_gain}.",
                    f"Top mean |SHAP|: {top_shap}.",
                    "",
                    "| Paired XGBoost delta | ROC-AUC mean [95% CI] | AP mean [95% CI] |",
                    "|---|---:|---:|",
                ]
            )
            for baseline in ("A_citation_only", "G_A_RRF"):
                auc_delta = section["paired_xgboost_minus_baseline"][baseline]["roc_auc"]
                ap_delta = section["paired_xgboost_minus_baseline"][baseline]["average_precision"]
                lines.append(
                    f"| vs {baseline} "
                    f"| {auc_delta['mean']:+.3f} [{auc_delta['ci_low']:+.3f}, {auc_delta['ci_high']:+.3f}] "
                    f"| {ap_delta['mean']:+.3f} [{ap_delta['ci_low']:+.3f}, {ap_delta['ci_high']:+.3f}] |"
                )
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="Dataset roots"
    )
    parser.add_argument(
        "--output-prefix",
        default="result/prediction_experiment_1600run",
        help="Output path without extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = []
    for dataset in args.datasets:
        dataset_root = (ROOT / dataset).resolve()
        print(f"Running conditional prediction for {dataset_root.name}...")
        models.append(evaluate_dataset(dataset_root))
    result = {
        "method": {
            "task": "conditional candidate x query prediction within G union A_candidate",
            "label": "candidate appears in at least one ChatGPT Web citation across 20 runs",
            "features": "compact G/A evidence and query metadata only",
            "feature_sets": {name: list(values) for name, values in COMPACT_FEATURES.items()},
            "excluded_feature": "llm_brand_score (undefined in current design)",
            "model": "XGBoost binary logistic",
            "split": "five folds grouped by query and balanced by intent-language stratum",
            "evaluation": "query-macro ROC-AUC, average precision, and P/R/F1@1/3/5/10 with query bootstrap",
            "top_k_values": list(TOP_K_VALUES),
            "top_k_target": "positives covered by the per-query G union A_candidate pool",
            "precision_at_k_denominator": "fixed k",
            "score_ties": "expected hits under uniform ordering within each tied score group",
            "baselines": ["Random", "A_citation_only", "G_A_RRF(k0=60)"],
            "random_repeats": RANDOM_REPEATS,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "seed": SEED,
        },
        "models": models,
    }
    prefix = (ROOT / args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
