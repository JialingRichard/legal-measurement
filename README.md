# legal-measurement

Code release for the paper project on auditable measurement of contested legal
constructs with large language models.

This public repository is intentionally conservative:

- it includes the core scoring pipeline
- it includes manuscript-side analysis and prompting scripts
- it does **not** include the full LaTeX paper source tree
- it does **not yet** include the research dataset release

The goal is to keep the public repository centered on code and reproducible
analysis utilities. The paper is maintained separately in a local working
directory, and future public updates can add the compiled manuscript PDF
without mirroring the full private LaTeX working tree.

## Current Contents

```text
legal-measurement/
├── code/
│   ├── xunxinzishi_qwen_boundary_benchmark.py
│   └── scripts/
├── data/
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

- the compiled paper PDF
- the curated 9000-case corpus release used in the paper
- the 100-case human and AI blind-audit release package

## License

Code in this repository is released under the MIT license; see
[`LICENSE`](./LICENSE).

## Data Notice

See [`DATA_NOTICE.md`](./DATA_NOTICE.md) for the repository-level note on data
release boundaries.
