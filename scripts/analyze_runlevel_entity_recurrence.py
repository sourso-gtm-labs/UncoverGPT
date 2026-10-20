"""Extract run-level entities and evaluate recurrence-to-citation prediction.

This experiment reuses existing Web ``search_detail`` records. It calls the
official OpenAI API only for entity extraction, never for web search. Outputs
are written to a versioned directory that is separate from the existing
``entity_llm_cache`` and hierarchical-frequency results.

Entity extraction follows the existing v11-style pipeline:

* English: spaCy ``en_core_web_sm`` candidates followed by a GPT-4.1-mini
  quality filter.
* Chinese: GPT-4.1-mini extraction from title/snippet text.

Unlike the existing aggregate cache, this script preserves the normalized
URL-to-entity mapping. It can therefore reconstruct a deduplicated entity set
for every Web run and apply the same leave-one-run-out protocol used at URL and
hostname levels.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from openai import AsyncOpenAI

try:
    from analyze_hierarchical_frequency import (
        HELDOUT_FREQUENCY_BINS,
        FREQUENCY_BINS,
        SUBDATASETS,
        RunGroup,
        _bootstrap_mean,
        aggregate_web_frequency_rows,
        analyze_observations,
        build_leave_one_run_out_rows,
        build_per_appearance_rows,
        load_web_run_groups,
        paired_query_auc_delta,
        query_macro_auc,
        query_macro_bin_table,
        trial_macro_bin_table,
    )
    from evaluate_proxy_effectiveness import deep_normalize
except ModuleNotFoundError:
    from scripts.analyze_hierarchical_frequency import (
        HELDOUT_FREQUENCY_BINS,
        FREQUENCY_BINS,
        SUBDATASETS,
        RunGroup,
        _bootstrap_mean,
        aggregate_web_frequency_rows,
        analyze_observations,
        build_leave_one_run_out_rows,
        build_per_appearance_rows,
        load_web_run_groups,
        paired_query_auc_delta,
        query_macro_auc,
        query_macro_bin_table,
        trial_macro_bin_table,
    )
    from scripts.evaluate_proxy_effectiveness import deep_normalize


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("dataset-5.3-1600", "dataset-5.5-1600")
DEFAULT_OUTPUT = "result/entity_runlevel_official_gpt41mini_v1"
OFFICIAL_BASE_URL = "https://api.openai.com/v1"
METHOD_VERSION = "runlevel_spacy_gpt41mini_official_v1"
EN_SCORE_BATCH_SIZE = 150
ZH_SNIPPET_BATCH_SIZE = 15
MAX_TEXT_CHARS = 1_500
DEFAULT_CONCURRENCY = 10
API_MAX_ATTEMPTS = 7


@dataclass(frozen=True)
class QueryCorpus:
    """All unique Web URLs and original run files for one query."""

    dataset: str
    dataset_root: Path
    subdataset: str
    qid: str
    query: str
    language: str
    run_paths: tuple[Path, ...]
    url_records: dict[str, dict[str, str]]

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.subdataset}/{self.qid}"

    @property
    def file_stem(self) -> str:
        return f"{self.subdataset}__{self.qid}"


@dataclass(frozen=True)
class ApiJob:
    """One independently cached entity API request."""

    corpus_key: str
    kind: str
    index: int
    prompt: str
    records: tuple[dict[str, str], ...]
    cache_path: Path
    input_hash: str
    max_tokens: int

    @property
    def key(self) -> str:
        return f"{self.corpus_key}/{self.kind}_{self.index:03d}"


@dataclass
class PreparedCorpus:
    """Local preprocessing and API jobs needed for one query."""

    corpus: QueryCorpus
    input_hash: str
    jobs: list[ApiJob]
    raw_url_entities: dict[str, set[str]]


def _load_entity_helpers() -> Any:
    """Load the public artifact's dependency-free entity helpers."""

    path = ROOT / "scripts" / "entity_helpers.py"
    spec = importlib.util.spec_from_file_location("_entity_v11_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import entity helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENTITY_HELPERS = _load_entity_helpers()


def _stable_hash(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _load_official_api_key() -> str:
    """Load only OPENAI_API_KEY; explicitly ignore NUW/proxy credentials."""

    key = os.getenv("OPENAI_API_KEY", "").strip()
    env_path = ROOT / ".env"
    if not key and env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENAI_API_KEY":
                key = value.strip().strip('"').strip("'")
                break
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; this experiment will not fall back to "
            "OPENAI_API_KEY_NUW or NUW_URL"
        )
    return key


def _query_metadata(dataset_root: Path, subdataset: str) -> dict[str, dict[str, str]]:
    path = dataset_root / subdataset / "query_list.json"
    payload = _read_json(path)
    output: dict[str, dict[str, str]] = {}
    for row in payload.get("queries") or []:
        qid = f"q{str(row.get('id', '')).lstrip('q').zfill(3)}"
        output[qid] = {
            "query": str(row.get("query") or row.get("text") or ""),
            "language": str(row.get("language") or ""),
        }
    return output


def _representative_text(item: dict[str, Any]) -> tuple[str, str, str]:
    title = str(item.get("title") or "").strip()
    snippet = str(item.get("snippet") or "").strip()
    text = f"{title}. {snippet}".strip(". ")
    return title, snippet, text[:MAX_TEXT_CHARS]


def collect_corpora(dataset_names: Sequence[str]) -> list[QueryCorpus]:
    """Collect unique normalized URL records without changing source data."""

    corpora: list[QueryCorpus] = []
    for dataset_name in dataset_names:
        dataset_root = (ROOT / dataset_name).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(dataset_root)
        for subdataset in SUBDATASETS:
            search_root = dataset_root / subdataset / "search_detail"
            metadata = _query_metadata(dataset_root, subdataset)
            for question_dir in sorted(search_root.glob("q*")):
                run_paths = tuple(sorted(question_dir.glob("*_detail.json")))
                if not run_paths:
                    continue
                qid = question_dir.name
                first = _read_json(run_paths[0])
                meta = metadata.get(qid, {})
                query = str(
                    first.get("original_query")
                    or meta.get("query")
                    or ""
                ).strip()
                language = str(meta.get("language") or "").strip()
                if not language:
                    language = (
                        "Chinese" if ENTITY_HELPERS.has_chinese(query) else "English"
                    )

                url_records: dict[str, dict[str, str]] = {}
                for path in run_paths:
                    payload = _read_json(path)
                    for item in payload.get("all_urls") or []:
                        normalized = deep_normalize(str(item.get("url") or ""))
                        if not normalized:
                            continue
                        title, snippet, text = _representative_text(item)
                        current = url_records.get(normalized)
                        if current is None or len(text) > len(current["text"]):
                            url_records[normalized] = {
                                "url": normalized,
                                "title": title,
                                "snippet": snippet,
                                "text": text,
                            }
                corpora.append(
                    QueryCorpus(
                        dataset=dataset_name,
                        dataset_root=dataset_root,
                        subdataset=subdataset,
                        qid=qid,
                        query=query,
                        language=language,
                        run_paths=run_paths,
                        url_records=dict(sorted(url_records.items())),
                    )
                )
    return sorted(corpora, key=lambda corpus: corpus.key)


def _query_input_hash(corpus: QueryCorpus, model: str) -> str:
    return _stable_hash(
        {
            "method_version": METHOD_VERSION,
            "model": model,
            "dataset": corpus.dataset,
            "subdataset": corpus.subdataset,
            "qid": corpus.qid,
            "query": corpus.query,
            "language": corpus.language,
            "records": [
                {
                    "url": url,
                    "title": record["title"],
                    "snippet": record["snippet"],
                }
                for url, record in corpus.url_records.items()
            ],
        }
    )


def _job_cache_path(
    output_root: Path, corpus: QueryCorpus, kind: str, index: int
) -> Path:
    return (
        output_root
        / "api_batches"
        / corpus.dataset
        / corpus.file_stem
        / f"{kind}_{index:03d}.json"
    )


def _english_prompt(
    corpus: QueryCorpus, records: Sequence[dict[str, str]]
) -> str:
    rows = "\n".join(f"{row['id']}\t{row['candidate']}" for row in records)
    return (
        "You are an entity quality scorer. Score every candidate phrase from "
        "0.0 to 1.0 for the search query below.\n"
        "1.0 means a standalone named entity such as a person, organization, "
        "company, brand, product, AI model, event, or specific location.\n"
        "0.5 means ambiguous. 0.0 means a generic noun, metric, fragment, or "
        "a phrase that merely contains an entity name.\n"
        "Do not rename or merge candidates. Return JSON only in this form:\n"
        '{"items":[{"id":"e0000","score":1.0}]}\n'
        "Return exactly one item for every supplied id.\n\n"
        f"Query: {corpus.query}\nCandidates:\n{rows}"
    )


def _chinese_prompt(
    corpus: QueryCorpus, records: Sequence[dict[str, str]]
) -> str:
    rows = "\n".join(f"[{row['id']}] {row['text'][:400]}" for row in records)
    return (
        "Extract named entities from the search-result snippets for the query "
        "below. Include companies, organizations, brands, products, AI models, "
        "people, events, and specific locations. Exclude generic concepts and "
        "metrics. Translate Chinese entity names to canonical lowercase English "
        "names when possible and keep model/version numbers. Return JSON only "
        "in this form:\n"
        '{"items":[{"id":"u0000","entities":["entity one","entity two"]}]}\n'
        "Return one item for every supplied id; use an empty entities list when "
        "no valid entity is present.\n\n"
        f"Query: {corpus.query}\nSnippets:\n{rows}"
    )


def prepare_corpus(
    corpus: QueryCorpus, output_root: Path, model: str
) -> PreparedCorpus:
    """Run local spaCy preprocessing and create independently cached API jobs."""

    query_hash = _query_input_hash(corpus, model)
    jobs: list[ApiJob] = []
    raw_url_entities: dict[str, set[str]] = {}

    if corpus.language.lower().startswith("eng"):
        nlp = ENTITY_HELPERS.load_spacy()
        urls = list(corpus.url_records)
        texts = [corpus.url_records[url]["text"] for url in urls]
        for url, doc in zip(urls, nlp.pipe(texts, batch_size=64)):
            entities: set[str] = set()
            for entity in doc.ents:
                if entity.label_ not in ENTITY_HELPERS.KEEP_LABELS:
                    continue
                raw = entity.text.strip()
                normalized = ENTITY_HELPERS.normalize(raw)
                if (
                    normalized
                    and ENTITY_HELPERS.is_valid_en(raw, normalized)
                ):
                    entities.add(normalized)
            raw_url_entities[url] = entities

        candidates = sorted(set().union(*raw_url_entities.values()))
        for index, start in enumerate(range(0, len(candidates), EN_SCORE_BATCH_SIZE)):
            batch = candidates[start : start + EN_SCORE_BATCH_SIZE]
            records = tuple(
                {"id": f"e{start + offset:05d}", "candidate": candidate}
                for offset, candidate in enumerate(batch)
            )
            prompt = _english_prompt(corpus, records)
            input_hash = _stable_hash(
                {
                    "method": METHOD_VERSION,
                    "model": model,
                    "kind": "english_score",
                    "prompt": prompt,
                }
            )
            jobs.append(
                ApiJob(
                    corpus_key=corpus.key,
                    kind="english_score",
                    index=index,
                    prompt=prompt,
                    records=records,
                    cache_path=_job_cache_path(
                        output_root, corpus, "english_score", index
                    ),
                    input_hash=input_hash,
                    max_tokens=max(1_000, len(records) * 24),
                )
            )
    else:
        urls = list(corpus.url_records)
        for index, start in enumerate(range(0, len(urls), ZH_SNIPPET_BATCH_SIZE)):
            batch_urls = urls[start : start + ZH_SNIPPET_BATCH_SIZE]
            records = tuple(
                {
                    "id": f"u{start + offset:05d}",
                    "url": url,
                    "text": corpus.url_records[url]["text"],
                }
                for offset, url in enumerate(batch_urls)
            )
            prompt = _chinese_prompt(corpus, records)
            input_hash = _stable_hash(
                {
                    "method": METHOD_VERSION,
                    "model": model,
                    "kind": "chinese_extract",
                    "prompt": prompt,
                }
            )
            jobs.append(
                ApiJob(
                    corpus_key=corpus.key,
                    kind="chinese_extract",
                    index=index,
                    prompt=prompt,
                    records=records,
                    cache_path=_job_cache_path(
                        output_root, corpus, "chinese_extract", index
                    ),
                    input_hash=input_hash,
                    max_tokens=max(1_200, len(records) * 100),
                )
            )
    return PreparedCorpus(corpus, query_hash, jobs, raw_url_entities)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_job_output(
    kind: str,
    records: Sequence[dict[str, str]],
    text: str,
) -> tuple[dict[str, Any], set[str]]:
    payload = json.loads(_strip_code_fence(text))
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("Response JSON has no items list")
    expected = {record["id"] for record in records}
    if kind == "english_score":
        values: dict[str, float] = {}
        for item in items:
            if not isinstance(item, dict) or item.get("id") not in expected:
                continue
            try:
                score = float(item.get("score"))
            except (TypeError, ValueError):
                continue
            values[str(item["id"])] = min(1.0, max(0.0, score))
        missing = expected - set(values)
        return {"scores": values}, missing

    values_list: dict[str, list[str]] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("id") not in expected:
            continue
        raw_entities = item.get("entities")
        if not isinstance(raw_entities, list):
            raw_entities = []
        normalized = {
            ENTITY_HELPERS.normalize(str(entity))
            for entity in raw_entities
            if str(entity).strip()
        }
        values_list[str(item["id"])] = sorted(
            entity for entity in normalized if entity
        )
    missing = expected - set(values_list)
    return {"entities": values_list}, missing


def _repair_prompt(kind: str, records: Sequence[dict[str, str]]) -> str:
    if kind == "english_score":
        rows = "\n".join(
            f"{record['id']}\t{record['candidate']}" for record in records
        )
        return (
            "Score every named-entity candidate below from 0.0 to 1.0. "
            "Return JSON only as "
            '{"items":[{"id":"e00000","score":1.0}]}. '
            "Return exactly one item for every id and do not omit any id.\n\n"
            + rows
        )
    rows = "\n".join(
        f"[{record['id']}] {record['text'][:400]}" for record in records
    )
    return (
        "Extract named entities (company, organization, brand, product, AI "
        "model, person, event, or specific location) from every snippet. "
        "Use canonical lowercase English names where possible. Return JSON "
        "only as "
        '{"items":[{"id":"u00000","entities":["entity"]}]}. '
        "Return exactly one item for every id, using an empty list when needed.\n\n"
        + rows
    )


def _valid_cached_job(job: ApiJob, model: str) -> dict[str, Any] | None:
    if not job.cache_path.exists():
        return None
    try:
        payload = _read_json(job.cache_path)
    except Exception:
        return None
    if (
        payload.get("method_version") != METHOD_VERSION
        or payload.get("input_hash") != job.input_hash
        or payload.get("requested_model") != model
        or payload.get("base_url") != OFFICIAL_BASE_URL
    ):
        return None
    parsed = payload.get("parsed")
    return parsed if isinstance(parsed, dict) else None


async def _call_job(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    job: ApiJob,
    model: str,
    progress: dict[str, int],
    progress_lock: asyncio.Lock,
) -> dict[str, Any]:
    cached = _valid_cached_job(job, model)
    if cached is not None:
        async with progress_lock:
            progress["cached"] += 1
            progress["done"] += 1
        return cached

    last_error: Exception | None = None
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            pending_records = list(job.records)
            parsed: dict[str, Any] = (
                {"scores": {}} if job.kind == "english_score" else {"entities": {}}
            )
            response_records: list[dict[str, Any]] = []
            usage_totals: Counter[str] = Counter()
            for repair_round in range(5):
                prompt = (
                    job.prompt
                    if repair_round == 0
                    else _repair_prompt(job.kind, pending_records)
                )
                max_tokens = (
                    job.max_tokens
                    if repair_round == 0
                    else max(
                        600,
                        len(pending_records)
                        * (30 if job.kind == "english_score" else 120),
                    )
                )
                async with semaphore:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        response_format={"type": "json_object"},
                        max_tokens=max_tokens,
                    )
                content = response.choices[0].message.content or ""
                partial, missing = _parse_job_output(
                    job.kind, pending_records, content
                )
                result_key = (
                    "scores" if job.kind == "english_score" else "entities"
                )
                parsed[result_key].update(partial[result_key])
                usage = response.usage.model_dump() if response.usage else {}
                for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    usage_totals[name] += int(usage.get(name) or 0)
                response_records.append(
                    {
                        "response_id": response.id,
                        "response_model": response.model,
                        "created": response.created,
                        "system_fingerprint": response.system_fingerprint,
                        "usage": usage,
                        "repair_round": repair_round,
                    }
                )
                if not missing:
                    break
                pending_records = [
                    record for record in pending_records if record["id"] in missing
                ]
                print(
                    f"[entity-api] {job.key} repairing {len(pending_records)} "
                    f"missing ids (round {repair_round + 1})",
                    flush=True,
                )
            if pending_records and len(parsed[result_key]) != len(job.records):
                missing_count = len(job.records) - len(parsed[result_key])
                raise ValueError(
                    f"Missing {missing_count} outputs after repair requests"
                )
            response = response_records[-1]
            payload = {
                "method_version": METHOD_VERSION,
                "input_hash": job.input_hash,
                "base_url": OFFICIAL_BASE_URL,
                "requested_model": model,
                "response_model": response["response_model"],
                "response_id": response["response_id"],
                "created": response["created"],
                "system_fingerprint": response["system_fingerprint"],
                "usage": dict(usage_totals),
                "responses": response_records,
                "parsed": parsed,
            }
            _atomic_write_json(job.cache_path, payload)
            async with progress_lock:
                progress["api"] += 1
                progress["done"] += 1
                done = progress["done"]
                total = progress["total"]
                if done == total or done % 10 == 0:
                    print(
                        f"[entity-api] {done}/{total} jobs complete "
                        f"(api={progress['api']}, cached={progress['cached']})",
                        flush=True,
                    )
            return parsed
        except Exception as error:
            last_error = error
            if attempt + 1 >= API_MAX_ATTEMPTS:
                break
            delay = min(90.0, (2**attempt) * 2.0 + random.random())
            print(
                f"[entity-api] {job.key} attempt {attempt + 1} failed: "
                f"{type(error).__name__}: {str(error)[:240]}; retry in "
                f"{delay:.1f}s",
                flush=True,
            )
            await asyncio.sleep(delay)
    raise RuntimeError(f"{job.key} failed after retries") from last_error


async def execute_jobs(
    jobs: Sequence[ApiJob],
    model: str,
    concurrency: int,
) -> dict[str, dict[str, Any]]:
    """Execute missing jobs with an exact upper bound on concurrency."""

    client = AsyncOpenAI(
        api_key=_load_official_api_key(),
        base_url=OFFICIAL_BASE_URL,
        timeout=120.0,
        max_retries=0,
    )
    semaphore = asyncio.Semaphore(concurrency)
    progress = {"done": 0, "total": len(jobs), "api": 0, "cached": 0}
    progress_lock = asyncio.Lock()
    tasks = [
        asyncio.create_task(
            _call_job(client, semaphore, job, model, progress, progress_lock)
        )
        for job in jobs
    ]
    values = await asyncio.gather(*tasks)
    await client.close()
    return {job.key: value for job, value in zip(jobs, values)}


def _query_cache_path(output_root: Path, corpus: QueryCorpus) -> Path:
    return (
        output_root
        / "url_entity_maps"
        / corpus.dataset
        / f"{corpus.file_stem}.json"
    )


def _valid_query_cache(
    path: Path, prepared: PreparedCorpus, model: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = _read_json(path)
    except Exception:
        return None
    if (
        payload.get("method_version") != METHOD_VERSION
        or payload.get("input_hash") != prepared.input_hash
        or payload.get("requested_model") != model
        or payload.get("base_url") != OFFICIAL_BASE_URL
    ):
        return None
    mapping = payload.get("url_entities")
    return payload if isinstance(mapping, dict) else None


def materialize_query_map(
    prepared: PreparedCorpus,
    job_results: dict[str, dict[str, Any]],
    output_root: Path,
    model: str,
) -> dict[str, list[str]]:
    """Combine API batches into a normalized URL-to-entity map."""

    corpus = prepared.corpus
    cache_path = _query_cache_path(output_root, corpus)
    cached = _valid_query_cache(cache_path, prepared, model)
    if cached is not None:
        return {
            url: [str(entity) for entity in entities]
            for url, entities in cached["url_entities"].items()
        }

    if corpus.language.lower().startswith("eng"):
        scores: dict[str, float] = {}
        id_to_candidate = {
            record["id"]: record["candidate"]
            for job in prepared.jobs
            for record in job.records
        }
        for job in prepared.jobs:
            parsed = job_results[job.key]
            for item_id, score in parsed["scores"].items():
                scores[id_to_candidate[item_id]] = float(score)
        mapping = {
            url: sorted(
                ENTITY_HELPERS.merge_aliases(
                    {
                        entity
                        for entity in entities
                        if scores.get(entity, 0.0)
                        >= ENTITY_HELPERS.LLM_SCORE_THRESHOLD
                    }
                )
            )
            for url, entities in prepared.raw_url_entities.items()
        }
    else:
        mapping_sets: dict[str, set[str]] = {
            url: set() for url in corpus.url_records
        }
        for job in prepared.jobs:
            parsed = job_results[job.key]
            id_to_url = {record["id"]: record["url"] for record in job.records}
            for item_id, entities in parsed["entities"].items():
                mapping_sets[id_to_url[item_id]].update(entities)
        mapping = {
            url: sorted(ENTITY_HELPERS.merge_aliases(entities))
            for url, entities in mapping_sets.items()
        }

    payload = {
        "method_version": METHOD_VERSION,
        "method": (
            "English spaCy en_core_web_sm candidates plus official "
            "GPT-4.1-mini quality filter; Chinese official GPT-4.1-mini "
            "title/snippet extraction; normalized URL-to-entity mapping "
            "preserved for run-level reconstruction"
        ),
        "input_hash": prepared.input_hash,
        "base_url": OFFICIAL_BASE_URL,
        "requested_model": model,
        "dataset": corpus.dataset,
        "subdataset": corpus.subdataset,
        "qid": corpus.qid,
        "query": corpus.query,
        "language": corpus.language,
        "n_urls": len(mapping),
        "n_urls_with_entities": sum(bool(entities) for entities in mapping.values()),
        "n_distinct_entities": len(
            set().union(*(set(entities) for entities in mapping.values()))
            if mapping
            else set()
        ),
        "url_entities": mapping,
    }
    _atomic_write_json(cache_path, payload)
    return mapping


def _run_sets_path(output_root: Path, corpus: QueryCorpus) -> Path:
    return (
        output_root
        / "run_entity_sets"
        / corpus.dataset
        / f"{corpus.file_stem}.json"
    )


def build_run_group(
    corpus: QueryCorpus,
    url_entities: dict[str, list[str]],
    output_root: Path,
) -> RunGroup:
    """Reconstruct within-run deduplicated appeared and cited entity sets."""

    appeared_runs: list[frozenset[str]] = []
    cited_runs: list[frozenset[str]] = []
    run_rows: list[dict[str, Any]] = []
    for path in corpus.run_paths:
        payload = _read_json(path)
        appeared: set[str] = set()
        cited: set[str] = set()
        for item in payload.get("all_urls") or []:
            normalized = deep_normalize(str(item.get("url") or ""))
            entities = set(url_entities.get(normalized, []))
            appeared.update(entities)
            if item.get("cited"):
                cited.update(entities)
        appeared_runs.append(frozenset(appeared))
        cited_runs.append(frozenset(cited))
        run_rows.append(
            {
                "run_file": path.name,
                "appeared_entities": sorted(appeared),
                "cited_entities": sorted(cited),
            }
        )

    _atomic_write_json(
        _run_sets_path(output_root, corpus),
        {
            "method_version": METHOD_VERSION,
            "dataset": corpus.dataset,
            "subdataset": corpus.subdataset,
            "qid": corpus.qid,
            "n_runs": len(run_rows),
            "runs": run_rows,
        },
    )
    return RunGroup(
        group=f"{corpus.subdataset}/{corpus.qid}",
        appeared=tuple(appeared_runs),
        cited=tuple(cited_runs),
    )


def _intent_summary(
    observations: Sequence[Any], dataset: str, signal: str
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for intent in SUBDATASETS:
        rows = [row for row in observations if row.group.startswith(intent + "/")]
        output[intent] = query_macro_auc(
            rows, dataset, signal, intent
        )
    return output


def summarize_entity_stability(entity_groups: Sequence[RunGroup]) -> dict[str, float]:
    """Return query-macro recurrence statistics for run-level entity sets."""

    per_query: list[tuple[float, float, float, float]] = []
    for group in entity_groups:
        union = set().union(*group.appeared)
        if not union:
            continue
        counts = {
            entity: sum(entity in run for run in group.appeared)
            for entity in union
        }
        pairwise_jaccard: list[float] = []
        for left_index, left in enumerate(group.appeared):
            for right in group.appeared[left_index + 1 :]:
                combined = left | right
                pairwise_jaccard.append(
                    len(left & right) / len(combined) if combined else 1.0
                )
        per_query.append(
            (
                float(len(union)),
                sum(count == 1 for count in counts.values()) / len(counts),
                sum(count >= 0.7 * len(group.appeared) for count in counts.values())
                / len(counts),
                sum(pairwise_jaccard) / len(pairwise_jaccard),
            )
        )
    labels = (
        "mean_unique_per_query",
        "mean_one_run_share",
        "mean_stable_core_share",
        "mean_pairwise_jaccard",
    )
    return {
        label: sum(row[index] for row in per_query) / len(per_query)
        for index, label in enumerate(labels)
    }


def analyze_dataset_run_groups(
    dataset: str,
    dataset_root: Path,
    entity_groups: list[RunGroup],
) -> dict[str, Any]:
    """Compare strict entity recurrence with URL and hostname recurrence."""

    web_groups = load_web_run_groups(dataset_root)
    all_groups: dict[str, list[RunGroup]] = {
        "url": web_groups["url"],
        "hostname": web_groups["domain"],
        "entity": entity_groups,
    }
    helper_keys = {"url": "url", "hostname": "domain", "entity": "url"}
    heldout = {}
    legacy = {}
    per_appearance = {}
    for level, groups in all_groups.items():
        helper_key = helper_keys[level]
        heldout[level] = build_leave_one_run_out_rows(
            {helper_key: groups}
        )[helper_key]
        legacy[level] = aggregate_web_frequency_rows(
            {helper_key: groups}
        )[helper_key]
        per_appearance[level] = build_per_appearance_rows(
            {helper_key: groups}
        )[helper_key]
    heldout_analysis = {
        level: {
            "query_macro_auc": query_macro_auc(
                rows, dataset, "heldout", level
            ),
            "query_macro_bins": query_macro_bin_table(
                rows,
                HELDOUT_FREQUENCY_BINS,
                dataset,
                "heldout_bins",
                level,
            ),
            "by_intent": _intent_summary(rows, dataset, f"heldout_{level}"),
        }
        for level, rows in heldout.items()
    }
    heldout_analysis["entity_minus_url_auc"] = paired_query_auc_delta(
        heldout["url"],
        heldout["entity"],
        dataset,
        "entity_minus_url",
    )
    heldout_analysis["hostname_minus_url_auc"] = paired_query_auc_delta(
        heldout["url"],
        heldout["hostname"],
        dataset,
        "hostname_minus_url",
    )
    heldout_analysis["entity_minus_hostname_auc"] = paired_query_auc_delta(
        heldout["hostname"],
        heldout["entity"],
        dataset,
        "entity_minus_hostname",
    )
    return {
        "dataset": dataset,
        "n_queries": len(entity_groups),
        "n_runs": sum(len(group.appeared) for group in entity_groups),
        "entity_run_stats": {
            "mean_appeared_entities": (
                sum(len(run) for group in entity_groups for run in group.appeared)
                / sum(len(group.appeared) for group in entity_groups)
            ),
            "mean_cited_entities": (
                sum(len(run) for group in entity_groups for run in group.cited)
                / sum(len(group.cited) for group in entity_groups)
            ),
            "runs_without_appeared_entities": sum(
                not run for group in entity_groups for run in group.appeared
            ),
            "runs_without_cited_entities": sum(
                not run for group in entity_groups for run in group.cited
            ),
        },
        "entity_candidate_stability": summarize_entity_stability(entity_groups),
        "same_batch_ever_cited": {
            level: analyze_observations(rows, FREQUENCY_BINS)
            for level, rows in legacy.items()
        },
        "full_sample_per_appearance": {
            level: {
                "query_macro_bins": trial_macro_bin_table(
                    rows,
                    FREQUENCY_BINS,
                    dataset,
                    "full_sample",
                    level,
                )
            }
            for level, rows in per_appearance.items()
        },
        "leave_one_run_out": heldout_analysis,
    }


def paired_cross_model_auc_delta(
    left: dict[str, Any],
    right: dict[str, Any],
    level: str,
) -> dict[str, float | int]:
    """Return right-minus-left query-macro AUC with paired query bootstrap."""

    left_auc = left["leave_one_run_out"][level]["query_macro_auc"]["per_query"]
    right_auc = right["leave_one_run_out"][level]["query_macro_auc"]["per_query"]
    groups = sorted(set(left_auc) & set(right_auc))
    deltas = [right_auc[group] - left_auc[group] for group in groups]
    return _bootstrap_mean(
        deltas,
        METHOD_VERSION,
        left["dataset"],
        right["dataset"],
        level,
        "paired_cross_model_auc",
    )


def _bootstrap_summary(result: dict[str, Any]) -> dict[str, Any]:
    return result["query_macro_auc"]["summary"]


def _format_auc(result: dict[str, Any]) -> str:
    summary = _bootstrap_summary(result)
    return (
        f"{summary['mean']:.3f} [{summary['ci_low']:.3f}, "
        f"{summary['ci_high']:.3f}]"
    )


def _format_delta(result: dict[str, Any]) -> str:
    return (
        f"{result['mean']:+.3f} [{result['ci_low']:+.3f}, "
        f"{result['ci_high']:+.3f}]"
    )


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Run-Level Entity Recurrence via Official GPT-4.1-mini",
        "",
        "This is a new, non-overwriting experiment. Existing Web runs are reused; "
        "the official OpenAI API is called only for entity extraction.",
        "",
        f"- Method version: `{METHOD_VERSION}`",
        f"- Requested model: `{result['manifest']['requested_model']}`",
        f"- API base URL: `{result['manifest']['base_url']}`",
        f"- Maximum concurrency: {result['manifest']['concurrency']}",
        "",
    ]
    for model in result["models"]:
        loo = model["leave_one_run_out"]
        stats = model["entity_run_stats"]
        lines.extend(
            [
                f"## {model['dataset']}",
                "",
                f"- Queries/runs: {model['n_queries']} / {model['n_runs']}",
                f"- Mean appeared/cited entities per run: "
                f"{stats['mean_appeared_entities']:.2f} / "
                f"{stats['mean_cited_entities']:.2f}",
                f"- Runs with no extracted appeared/cited entity: "
                f"{stats['runs_without_appeared_entities']} / "
                f"{stats['runs_without_cited_entities']}",
                "",
                "### Leave-one-run-out citation propensity",
                "",
                "| Level | Query-macro AUC (95% CI) |",
                "|---|---:|",
                f"| URL | {_format_auc(loo['url'])} |",
                f"| Hostname | {_format_auc(loo['hostname'])} |",
                f"| Entity | {_format_auc(loo['entity'])} |",
                "",
                f"- Entity minus URL: "
                f"{_format_delta(loo['entity_minus_url_auc'])}",
                f"- Entity minus hostname: "
                f"{_format_delta(loo['entity_minus_hostname_auc'])}",
                "",
                "### Entity held-out recurrence bins",
                "",
                "| Prior frequency | Queries | Held-out appearances | Citations | "
                "Query-macro rate (95% CI) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in loo["entity"]["query_macro_bins"]:
            if math.isfinite(row["macro_rate"]):
                rate = (
                    f"{row['macro_rate'] * 100:.1f}% "
                    f"[{row['macro_ci_low'] * 100:.1f}%, "
                    f"{row['macro_ci_high'] * 100:.1f}%]"
                )
            else:
                rate = "N/A"
            lines.append(
                f"| {row['bin']} | {row['n_queries']} | "
                f"{row['n_occurrences']} | {row['n_cited']} | {rate} |"
            )
        lines.extend(
            [
                "",
                "### Entity leave-one-run-out AUC by intent",
                "",
                "| Intent | Evaluated queries | AUC (95% CI) |",
                "|---|---:|---:|",
            ]
        )
        for intent in SUBDATASETS:
            summary = loo["entity"]["by_intent"][intent]["summary"]
            lines.append(
                f"| {intent} | {summary['n']} | {summary['mean']:.3f} "
                f"[{summary['ci_low']:.3f}, {summary['ci_high']:.3f}] |"
            )
        lines.append("")
    if "cross_model_auc_delta" in result:
        lines.extend(
            [
                "## Cross-model paired AUC differences",
                "",
                "| Level | GPT-5.5 minus GPT-5.3 AUC (95% CI) |",
                "|---|---:|",
            ]
        )
        for level in ("url", "hostname", "entity"):
            lines.append(
                f"| {level.title()} | "
                f"{_format_delta(result['cross_model_auc_delta'][level])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _stable_numbers(value: object) -> object:
    if isinstance(value, float):
        return value if not math.isfinite(value) else float(f"{value:.10g}")
    if isinstance(value, dict):
        return {key: _stable_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stable_numbers(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching caches in an existing output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare local inputs and report API job counts without writing or calling",
    )
    parser.add_argument(
        "--max-api-jobs",
        type=int,
        default=None,
        help="Run only the first N missing API jobs, cache them, and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency != DEFAULT_CONCURRENCY:
        raise ValueError(
            f"This experiment is fixed to {DEFAULT_CONCURRENCY} concurrent calls"
        )
    output_root = (ROOT / args.output_dir).resolve()
    if output_root.exists() and not (args.resume or args.dry_run):
        raise FileExistsError(
            f"{output_root} already exists; pass --resume to reuse matching caches"
        )

    corpora = collect_corpora(args.datasets)
    print(
        f"Collected {len(corpora)} query corpora and "
        f"{sum(len(corpus.url_records) for corpus in corpora):,} "
        "normalized URL-query records",
        flush=True,
    )
    prepared = [
        prepare_corpus(corpus, output_root, args.model) for corpus in corpora
    ]
    jobs = [job for item in prepared for job in item.jobs]
    language_counts = Counter(corpus.language for corpus in corpora)
    kind_counts = Counter(job.kind for job in jobs)
    print(f"Languages: {dict(language_counts)}", flush=True)
    print(f"API jobs: {len(jobs)} {dict(kind_counts)}", flush=True)
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    created_unix = time.time()
    if args.resume and manifest_path.exists():
        existing_manifest = _read_json(manifest_path)
        expected = {
            "method_version": METHOD_VERSION,
            "datasets": list(args.datasets),
            "requested_model": args.model,
            "base_url": OFFICIAL_BASE_URL,
            "concurrency": args.concurrency,
        }
        mismatches = {
            key: (existing_manifest.get(key), value)
            for key, value in expected.items()
            if existing_manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Existing experiment manifest is incompatible with this resume: "
                f"{mismatches}"
            )
        created_unix = float(existing_manifest.get("created_unix", created_unix))

    manifest = {
        "method_version": METHOD_VERSION,
        "created_unix": created_unix,
        "datasets": list(args.datasets),
        "requested_model": args.model,
        "base_url": OFFICIAL_BASE_URL,
        "concurrency": args.concurrency,
        "n_corpora": len(corpora),
        "n_url_query_records": sum(
            len(corpus.url_records) for corpus in corpora
        ),
        "n_api_jobs": len(jobs),
        "language_counts": dict(language_counts),
        "job_kind_counts": dict(kind_counts),
        "source_data_modified": False,
        "existing_entity_cache_modified": False,
    }
    _atomic_write_json(manifest_path, manifest)

    cached_results: dict[str, dict[str, Any]] = {}
    missing_jobs: list[ApiJob] = []
    for job in jobs:
        cached = _valid_cached_job(job, args.model)
        if cached is None:
            missing_jobs.append(job)
        else:
            cached_results[job.key] = cached
    print(
        f"Job cache status: {len(cached_results)} cached, "
        f"{len(missing_jobs)} missing",
        flush=True,
    )

    jobs_to_run = missing_jobs
    if args.max_api_jobs is not None:
        jobs_to_run = missing_jobs[: args.max_api_jobs]
    if jobs_to_run:
        new_results = asyncio.run(
            execute_jobs(jobs_to_run, args.model, args.concurrency)
        )
        cached_results.update(new_results)
    if args.max_api_jobs is not None and len(jobs_to_run) < len(missing_jobs):
        print(
            f"Smoke run complete; {len(missing_jobs) - len(jobs_to_run)} "
            "API jobs remain. Resume without --max-api-jobs.",
            flush=True,
        )
        return

    for job in jobs:
        if job.key not in cached_results:
            cached = _valid_cached_job(job, args.model)
            if cached is None:
                raise RuntimeError(f"Missing completed cache for {job.key}")
            cached_results[job.key] = cached

    maps: dict[str, dict[str, list[str]]] = {}
    for item in prepared:
        maps[item.corpus.key] = materialize_query_map(
            item, cached_results, output_root, args.model
        )

    groups_by_dataset: dict[str, list[RunGroup]] = {
        dataset: [] for dataset in args.datasets
    }
    for corpus in corpora:
        groups_by_dataset[corpus.dataset].append(
            build_run_group(corpus, maps[corpus.key], output_root)
        )

    models = [
        analyze_dataset_run_groups(
            dataset,
            (ROOT / dataset).resolve(),
            groups_by_dataset[dataset],
        )
        for dataset in args.datasets
    ]
    result = {
        "manifest": manifest,
        "method": {
            "entity_membership": (
                "entities are mapped to normalized URLs, then deduplicated "
                "within each Web run"
            ),
            "citation_label": (
                "an entity is cited in a run when at least one mapped URL has "
                "the project cited=true label"
            ),
            "leave_one_run_out": (
                "for every held-out entity appearance, use its frequency in "
                "the other 19 runs to predict held-out citation"
            ),
            "inference": (
                "per-query AUC, macro average, and 2000 deterministic "
                "query-bootstrap resamples"
            ),
        },
        "entity_mapping": {
            "n_url_query_records": sum(
                len(mapping) for mapping in maps.values()
            ),
            "n_url_query_records_with_entities": sum(
                bool(entities)
                for mapping in maps.values()
                for entities in mapping.values()
            ),
        },
        "models": models,
    }
    result["entity_mapping"]["coverage"] = (
        result["entity_mapping"]["n_url_query_records_with_entities"]
        / result["entity_mapping"]["n_url_query_records"]
    )
    if len(models) == 2:
        result["cross_model_auc_delta"] = {
            level: paired_cross_model_auc_delta(models[0], models[1], level)
            for level in ("url", "hostname", "entity")
        }
    result = _stable_numbers(result)
    _atomic_write_json(output_root / "results.json", result)
    (output_root / "results.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(f"Wrote {output_root / 'results.json'}", flush=True)
    print(f"Wrote {output_root / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
