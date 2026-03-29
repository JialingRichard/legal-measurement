# legal-measurement

Public code-and-data companion repository for the paper project on auditable
measurement of contested legal constructs with large language models.

This repository includes:

- the core scoring and analysis code
- the compiled manuscript PDF
- the paper's 9000-case master research table, which includes case text and
  framework outputs
- the 100-case human and AI blind-audit materials used in the validation

The full private LaTeX working tree is not mirrored here. The paper is released
as a compiled PDF in `paper/`.

## Repository Layout

```text
legal-measurement/
├── code/
│   ├── xunxinzishi_qwen_boundary_benchmark.py
│   └── scripts/
├── data/
│   ├── audit_100/
│   └── corpus_9000/
├── paper/
│   └── legal-measurement-paper.pdf
├── LICENSE
├── DATA_NOTICE.md
└── README.md
```

## Paper

- `paper/legal-measurement-paper.pdf`
  - compiled manuscript corresponding to the current public release

## Code

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

## Data

### `data/corpus_9000/`

- `xunxinzishi_qwen_boundary_2013_2021_1000x9.csv.xz`
  - compressed public release of the paper's 9000-case master research table
  - this file contains case metadata, full text, extracted semantic fields,
    quotes, dimension scores, and final framework outputs

### `data/audit_100/`

- human blind-audit pack and completed audit labels
- single-case prompt-based AI comparison outputs
- grouped-20 prompt-based AI comparison outputs
- recomputed metric tables used in the manuscript

See [`data/README.md`](./data/README.md) for file-level guidance.
Checksums for the released paper and data files are listed in
[`SHA256SUMS.txt`](./SHA256SUMS.txt).

## License

Code in this repository is released under the MIT license; see
[`LICENSE`](./LICENSE). Data release conditions are described separately in
[`DATA_NOTICE.md`](./DATA_NOTICE.md).
