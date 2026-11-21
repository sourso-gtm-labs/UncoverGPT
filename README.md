# UncoverGPT

Analysis code for the WSDM 2027 paper
**Understanding ChatGPT Citations: Evidence from Production Web Search Across Model Generations**.

Yunxuan Jiang, Shiqi Zhang, Anqi Xing, Xiangyu Chang, and Xiaokui Xiao.
In *Proceedings of the Twentieth ACM International Conference on Web Search and
Data Mining (WSDM '27)*, February 15--19, 2027, Hong Kong.
DOI: [10.1145/3862322.3863469](https://doi.org/10.1145/3862322.3863469)

## Overview

Generative engines such as ChatGPT Web retrieve webpages to answer a query, but
cite only some of them. We collect server-sent event (SSE) traces from ChatGPT
Web and observe, for every response, the executed (rewritten) queries, the
retrieved candidate webpages, and the references selected as citations. We then
evaluate Google Search and the OpenAI Responses API as external proxies for
ChatGPT Web citations, and test whether external signals can predict which
candidates are cited at the URL, hostname, and entity levels.

The study covers four ChatGPT Web settings:

| Setting          | Queries | Web runs |
|------------------|--------:|---------:|
| GPT-5.2 Instant  |      32 |      547 |
| GPT-5.3 Instant  |      80 |    1,600 |
| GPT-5.4 Thinking |      30 |      491 |
| GPT-5.5 Instant  |      80 |    1,600 |

The GPT-5.3 and GPT-5.5 analyses additionally use 1,600 Responses API runs per
model and one Google Search snapshot per query.

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

## Data Availability

This repository currently contains source code and documentation only. The
collected data, including SSE traces, parsed ChatGPT Web, Responses API, and
Google Search records, entity caches, and experiment outputs, is not included.
We are preparing a browsable gallery that shows, for individual queries, how
ChatGPT Web rewrites the query, which webpages it retrieves, and which of them
it cites. Updates on data access will be posted here. See
[DATA_CARD.md](DATA_CARD.md) for details.

No browser automation, account setup, or collection tooling is included.

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
python tools/audit_anonymity.py   # no local paths, credentials, or e-mail addresses
```

Tests that require trace files are skipped when those inputs are absent.
Synthetic-fixture tests do not require collected data.

## Running the Analyses

The analyses require the collected input files and, where applicable,
entity-extraction caches, which are not part of this repository. The scripts
keep their original relative input paths. With the inputs in place, the main
entry points are:

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

Entity extraction calls a paid LLM API. Review its configuration and costs
before running it. The `--resume` option only reuses caches that already exist.

## Citation

```bibtex
@inproceedings{jiang2027understanding,
  title     = {Understanding {ChatGPT} Citations: Evidence from Production Web Search Across Model Generations},
  author    = {Jiang, Yunxuan and Zhang, Shiqi and Xing, Anqi and Chang, Xiangyu and Xiao, Xiaokui},
  booktitle = {Proceedings of the Twentieth ACM International Conference on Web Search and Data Mining},
  series    = {WSDM '27},
  year      = {2027},
  address   = {Hong Kong},
  publisher = {ACM},
  doi       = {10.1145/3862322.3863469}
}
```

## License

The source code and documentation are released under the [MIT License](LICENSE).
