# Chapter code

This directory contains paper-facing analysis and figure-generation scripts.
Input data and generated JSON, LaTeX, and figure files are not distributed in
this code-only branch. The commands below require the relevant private inputs.

## Data Collection table

Regenerate the dataset audit and Table `tab:dataset` from the repository root:

```bash
python3 chapter_code/data_collection_stats.py
```

With previously generated outputs available, check that they still match the
datasets without rewriting them:

```bash
python3 chapter_code/data_collection_stats.py --check
```

The generated JSON records both target and observed coverage, along with
per-response OpenAI candidate/citation counts and per-query Google organic-result
counts when those proxy caches are available. The table generator requires
complete 20-run coverage for GPT-5.3 and GPT-5.5, while GPT-5.2 and GPT-5.4
report their finalized reduced cohorts.
