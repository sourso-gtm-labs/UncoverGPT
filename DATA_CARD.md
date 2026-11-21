# Data Availability

## Data Sources

The analysis scripts use ChatGPT Web traces, Google Search snapshots, and
OpenAI Responses API web-search outputs. The query set is drawn from Search
Arena, a public research dataset, and spans four intents and two languages.

## Current Release

This repository contains source code and documentation only. It does not
distribute the collected datasets or generated experiment outputs.

## Not Included

- captured SSE traces, including de-identified copies;
- parsed Web, Google, and Responses API records;
- generated answers, source titles, snippets, and query-level measurements;
- entity mappings, extraction caches, and intermediate data;
- generated statistics, result reports, tables, and figures; and
- credentials, local environment configuration, and collection tooling.

The collected records are retained by the authors. We plan to make example
traces available through a browsable gallery (query, rewritten queries,
retrieved webpages, response, and citations). Updates will be posted in the
README.

## Limitations

The measurements are observational and cover a finite set of queries, dates,
languages, and model settings. Cross-setting differences may also reflect
changes in search indexes or serving backends. The data should not be treated
as evidence of source quality, credibility, or causal ranking effects.
