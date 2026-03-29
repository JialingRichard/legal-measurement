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

- `human_audit_100sample_quantile_pack_minimal_judgment_only.md`
  - human blind-audit pack
- `human_audit_100sample_quantile_prefilled.csv`
  - completed human blind-audit labels
- `ai_blind_audit_pack_minimal.md`
  - AI blind-audit prompt pack
- `ai_grouped20/*.csv`
  - grouped-20 label outputs for each model
- `group20_merged_frame.csv`
  - merged grouped-20 comparison frame used in the manuscript
- `ai_single_case_human_metrics.csv`
  - single-case AI versus human summary metrics
- `ai_single_case_metrics_summary.json`
  - single-case rerun summary metadata
