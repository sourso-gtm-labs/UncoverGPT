# Anonymous Citation-Prediction Analysis Code

This branch distributes analysis, parsing, evaluation, and plotting scripts
for studying retrieval and citation behavior across four ChatGPT Web settings.
It is a code-only release. Collected SSE traces, parsed Web and proxy records,
generated answers, webpage titles and snippets, entity caches, experiment
outputs, and generated figures are not distributed through this branch.

## Scope

The scripts analyze the following datasets when the required inputs are
available separately:

- GPT-5.2 Instant: 32 queries and 547 Web runs;
- GPT-5.3 Instant: 80 queries and 1,600 Web runs;
- GPT-5.4 Thinking: 30 queries and 491 Web runs; and
- GPT-5.5 Instant: 80 queries and 1,600 Web runs.

The GPT-5.3 and GPT-5.5 analyses additionally use 1,600 parsed Responses API
runs and one Google Search snapshot per query. These inputs are not included.

## Repository Layout

```text
chapter_code/              Paper-facing statistics and figure preparation
scripts/                   Proxy, recurrence, and prediction experiments
tests/                     Tests for analysis and parsing logic
dataset-5.2/               Parser entry points only
dataset-5.3-1600/          Parser entry points only
dataset-5.4-thinking/      Thinking-mode parser and parser tests only
dataset-5.5-1600/          Parser entry points only
tools/                     Validation and data-processing utilities
```

The Instant-model traces use `scripts/web_sse_parser.py`. GPT-5.4 Thinking uses
`dataset-5.4-thinking/parse_sse.py`, which reconstructs interleaved multi-round
web-tool calls.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Validation

Run the source-code checks from the repository root:

```bash
python -m unittest discover -s tests
python dataset-5.4-thinking/test_parse_sse.py
python tools/audit_anonymity.py
```

Tests that require withheld trace files are skipped when those inputs are
absent. Synthetic-fixture tests do not require collected data.

## Running the Analyses

Full reproduction requires the original input files and, where applicable,
entity-extraction caches. They are not bundled with this code-only release.
The scripts retain their original relative input paths. After independently
obtaining the required inputs, the main analysis entry points are:

```bash
python chapter_code/query_rewriting_patterns.py
python chapter_code/thinking_query_rewriting_patterns.py
python chapter_code/snippet_citation.py
python chapter_code/other_candidate_signals.py
python scripts/analyze_hierarchical_frequency.py
python scripts/analyze_runlevel_entity_recurrence.py --resume
python scripts/evaluate_proxy_effectiveness.py
python scripts/run_prediction_experiment.py
python scripts/run_prediction_rrf_ablation.py
python chapter_code/cross_setting_significance.py
```

Do not run entity extraction without reviewing its API configuration and
costs. The `--resume` option only reuses caches that are already available;
this branch does not distribute those caches.

## Responsible Release

The release excludes collected data and data-derived outputs, including
de-identified copies. Withholding them here does not delete the privately
retained research records. Only the source code and its documentation are
distributed. No browser automation, CDP tooling, account setup, or collection
orchestration is included.
