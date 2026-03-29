# legal-measurement

Code release for the paper project on auditable measurement of contested legal
constructs with large language models.

This initial public release is intentionally conservative:

- it includes the core scoring pipeline
- it includes manuscript-side analysis and prompting scripts
- it does **not yet** include the paper source
- it does **not yet** include the research dataset release

The goal of this first step is to establish a clean public code repository
before adding the manuscript and curated data package in follow-up commits.

## Current Contents

```text
legal-measurement/
├── code/
│   ├── xunxinzishi_qwen_boundary_benchmark.py
│   └── scripts/
├── LICENSE
├── DATA_NOTICE.md
└── README.md
```

## Included Code

- `code/xunxinzishi_qwen_boundary_benchmark.py`
  - main extraction and four-dimension scoring pipeline
- `code/scripts/war1_prompt_rerun_api.py`
  - single-case prompt-based AI rerun script
- `code/scripts/run_group20_model.py`
  - grouped-20 prompt-based AI rerun script
- `code/scripts/recompute_api_rerun_metrics.py`
  - recomputes single-case AI-vs-human / AI-vs-index metrics
- `code/scripts/compute_group20_metrics.py`
  - recomputes grouped-20 metrics
- `code/scripts/war1_make_figures.py`
  - figure generation helper
- `code/scripts/war1_threshold_robustness.py`
  - threshold robustness helper
- `code/scripts/war1_ieee_extra_analysis.py`
  - extra analysis helper used during manuscript preparation

The prompt-based rerun scripts do not contain any hard-coded API key. They
expect the environment variable `VOLCENGINE_CODING_API_KEY` when rerunning the
prompt-based comparison experiments.

## Planned Additions

Later releases can add:

- the paper source and arXiv package
- the curated 9000-case corpus release used in the paper
- the 100-case human and AI blind-audit release package

## License

Code in this repository is released under the MIT license; see
[`LICENSE`](./LICENSE).

## Data Notice

See [`DATA_NOTICE.md`](./DATA_NOTICE.md) for the repository-level note on data
release boundaries.
