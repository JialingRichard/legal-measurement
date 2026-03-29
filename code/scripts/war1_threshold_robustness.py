from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize_100(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for threshold in thresholds:
        sub = df[df["expansion_index"] >= threshold].copy()
        drop = sub[sub["human_label"].isin([0, 1])].copy()
        rows.append(
            {
                "threshold": threshold,
                "count": int(len(sub)),
                "share": float(len(sub) / total),
                "mean_index": float(sub["expansion_index"].mean()) if len(sub) else None,
                "human_positive_rate_merged12": float((sub["human_label"] != 0).mean()) if len(sub) else None,
                "human_clear_risk_rate": float((sub["human_label"] == 1).mean()) if len(sub) else None,
                "human_gray_rate": float((sub["human_label"] == 2).mean()) if len(sub) else None,
                "human_ordinary_rate": float((sub["human_label"] == 0).mean()) if len(sub) else None,
                "drop2_count": int(len(drop)),
                "drop2_clear_risk_rate": float((drop["human_label"] == 1).mean()) if len(drop) else None,
            }
        )
    return pd.DataFrame(rows)


def summarize_9000(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for threshold in thresholds:
        sub = df[df["expansion_index"] >= threshold].copy()
        vc = sub["dominant_expansion_path"].value_counts(normalize=True) if len(sub) else pd.Series(dtype=float)
        rows.append(
            {
                "threshold": threshold,
                "count": int(len(sub)),
                "share": float(len(sub) / total),
                "mean_index": float(sub["expansion_index"].mean()) if len(sub) else None,
                "median_index": float(sub["expansion_index"].median()) if len(sub) else None,
                "top_path": vc.index[0] if len(vc) else None,
                "top_path_share": float(vc.iloc[0]) if len(vc) else None,
                "share_private_plus_order": float(vc.get("private+order", 0.0)),
                "share_order": float(vc.get("order", 0.0)),
                "share_overcriminalization": float(vc.get("overcriminalization", 0.0)),
                "share_order_plus_private": float(vc.get("order+private", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def representative_cases(df: pd.DataFrame, path_order: list[str]) -> pd.DataFrame:
    rows = []
    for path in path_order:
        sub = df[df["dominant_expansion_path"] == path].sort_values("expansion_index", ascending=False).head(1)
        if len(sub):
            rows.append(sub)
    if not rows:
        return pd.DataFrame()
    cols = [
        "source_year",
        "案号",
        "案件名称",
        "法院",
        "裁判日期",
        "expansion_index",
        "expansion_review_band",
        "dominant_expansion_path",
        "boundary_reason",
        "evidence_excerpt",
    ]
    return pd.concat(rows, ignore_index=True)[cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--full-corpus-csv", required=True, type=Path)
    parser.add_argument("--top-path-cases-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audit_df = pd.read_csv(args.audit_csv)
    full_df = pd.read_csv(args.full_corpus_csv)
    top_path_df = pd.read_csv(args.top_path_cases_csv)

    thresholds = [35, 45, 50, 55, 60, 65, 70]
    audit_summary = summarize_100(audit_df, thresholds)
    full_summary = summarize_9000(full_df, thresholds)
    reps = representative_cases(
        top_path_df,
        ["order", "private+order", "private", "overcriminalization", "order+private"],
    )

    audit_summary.to_csv(args.output_dir / "threshold_sensitivity_100.csv", index=False, encoding="utf-8-sig")
    full_summary.to_csv(args.output_dir / "threshold_sensitivity_9000.csv", index=False, encoding="utf-8-sig")
    reps.to_csv(args.output_dir / "representative_cases.csv", index=False, encoding="utf-8-sig")

    summary = {
        "thresholds": thresholds,
        "audit_rows": int(len(audit_df)),
        "full_rows": int(len(full_df)),
        "representative_paths": reps["dominant_expansion_path"].tolist() if len(reps) else [],
    }
    (args.output_dir / "threshold_robustness_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
