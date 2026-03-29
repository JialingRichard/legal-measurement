# Data Release Notes

This directory contains the research data materials used in the paper.

## `corpus_9000/`

- `xunxinzishi_qwen_boundary_2013_2021_1000x9.csv.xz`
  - compressed release of the 9000-case master research table used in the paper
  - includes case metadata, full text, extracted semantic fields, quotes,
    four dimension scores, and final framework outputs

To decompress the prediction output:

```bash
xz -d xunxinzishi_qwen_boundary_2013_2021_1000x9.csv.xz
```

## `audit_100/`

- `human/`
  - blind-audit prompt pack and completed human labels
- `ai_single_case/`
  - merged single-case prompt-based AI outputs
- `ai_grouped20/`
  - grouped-20 label outputs and summaries for each model
- `metrics/`
  - metric tables used in the manuscript
