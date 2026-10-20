# Data Availability

## Data Sources

The analysis scripts use ChatGPT Web traces, Google Search snapshots, and
OpenAI Responses API web-search outputs. The query set originates from a
public research dataset and spans four intents and two languages.

## Public Release

This branch contains source code and documentation only. It does not
distribute the collected datasets or generated experiment outputs.

## Excluded Data

- all captured SSE traces, including de-identified copies;
- parsed Web, Google, and Responses API records;
- generated answers, source titles, snippets, and query-level measurements;
- entity mappings, extraction caches, and intermediate data;
- generated statistics, result reports, tables, and figures; and
- credentials, local environment configuration, and collection tooling.

The original research records remain privately retained. Their absence from
this code-only branch is not a deletion of those records. Running the
data-dependent analyses requires the corresponding inputs separately.

## Limitations

The measurements are observational and cover a finite set of queries, dates,
languages, and model settings. Cross-setting differences may also reflect
changes in search indexes or serving backends. The data should not be treated
as evidence of source quality, credibility, or causal ranking effects.
