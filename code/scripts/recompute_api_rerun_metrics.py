from __future__ import annotations

import json
import math
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "audit_100"
API_RESULTS = BASE / "ai_single_case" / "rerun_20260328_130149_merged_best.csv"
OUT_DIR = BASE / "metrics"

MODEL_MAP = {
    "doubao-seed-2.0-pro": "doubao20pro",
    "deepseek-v3.2": "deepseekv32_api",
    "glm-4.7": "glm47",
    "kimi-k2.5": "kimik25",
}


def parse_human_labels(path: Path) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    current_sample_no: int | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        sample_match = re.match(r"^##\s*(\d+)\.", line)
        if sample_match:
            current_sample_no = int(sample_match.group(1))
            continue
        if current_sample_no is None:
            continue
        label_match = re.search(r"人类终局定性：\s*`?\s*([012])", line)
        if label_match:
            rows.append({"sample_no": current_sample_no, "human_label": int(label_match.group(1))})
            current_sample_no = None
    if not rows:
        raise ValueError(f"No human labels parsed from {path}")
    return pd.DataFrame(rows).sort_values("sample_no").drop_duplicates("sample_no", keep="last")


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


def assign_quintiles(values: np.ndarray) -> pd.Series:
    try:
        return pd.qcut(values, 5, labels=False, duplicates="drop") + 1
    except ValueError:
        ranks = pd.Series(values).rank(method="average")
        return pd.qcut(ranks, 5, labels=False, duplicates="drop") + 1


def quantile_edges(values: np.ndarray, q: int = 5) -> list[float]:
    return [float(x) for x in np.quantile(values, np.linspace(0, 1, q + 1))]


def triangular_probs_from_index(index_values: np.ndarray) -> np.ndarray:
    lo = float(np.min(index_values))
    hi = float(np.max(index_values))
    if math.isclose(lo, hi):
        s = np.full_like(index_values, 0.5, dtype=float)
    else:
        s = (index_values - lo) / (hi - lo)

    probs = np.zeros((len(index_values), 3), dtype=float)
    left = s <= 0.5
    right = ~left

    probs[left, 0] = 1.0 - 2.0 * s[left]
    probs[left, 1] = 2.0 * s[left]
    probs[left, 2] = 0.0

    probs[right, 0] = 0.0
    probs[right, 1] = 2.0 * (1.0 - s[right])
    probs[right, 2] = 2.0 * s[right] - 1.0
    return probs


def binary_index_metrics(df: pd.DataFrame, label_col: str, scheme_name: str, positive_rule: str) -> dict:
    work = df[["sample_no", "expansion_index", label_col]].rename(columns={label_col: "label"}).dropna().copy()
    work["label"] = work["label"].astype(int)
    if positive_rule == "merged_12":
        work["target"] = (work["label"] != 0).astype(int)
    elif positive_rule == "drop_2_keep_01":
        work = work[work["label"].isin([0, 1])].copy()
        work["target"] = work["label"].astype(int)
    else:
        raise ValueError(positive_rule)

    x = work["expansion_index"].to_numpy(dtype=float)
    y = work["target"].to_numpy(dtype=int)
    neg = work.loc[work["target"] == 0, "expansion_index"].to_numpy(dtype=float)
    pos = work.loc[work["target"] == 1, "expansion_index"].to_numpy(dtype=float)

    spearman = stats.spearmanr(x, y)
    kendall = stats.kendalltau(x, y)
    auc = roc_auc_score(y, x)
    mann_whitney = stats.mannwhitneyu(neg, pos, alternative="two-sided")

    work["quintile"] = assign_quintiles(x)
    trend_rows = []
    for q in sorted(work["quintile"].dropna().unique()):
        part = work[work["quintile"] == q]
        trend_rows.append(
            {
                "quintile": int(q),
                "n": int(len(part)),
                "index_mean": float(part["expansion_index"].mean()),
                "positive_rate": float(part["target"].mean()),
            }
        )

    return {
        "label_col": label_col,
        "scheme": scheme_name,
        "matched_rows": int(len(work)),
        "count_0": int((y == 0).sum()),
        "count_1": int((y == 1).sum()),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau_b": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "auc": float(auc),
        "mannwhitney_u": float(mann_whitney.statistic),
        "mannwhitney_p": float(mann_whitney.pvalue),
        "cliffs_delta": float(cliffs_delta(pos, neg)),
        "quintile_cuts": quantile_edges(x, 5),
        "trend": trend_rows,
    }


def multiclass_index_metrics(df: pd.DataFrame, label_col: str) -> dict:
    work = df[["sample_no", "expansion_index", label_col]].rename(columns={label_col: "label"}).dropna().copy()
    work["label"] = work["label"].astype(int)
    x = work["expansion_index"].to_numpy(dtype=float)
    y = work["label"].to_numpy(dtype=int)
    mapped = work["label"].map({0: 0.0, 1: 1.0, 2: 0.5}).to_numpy(dtype=float)

    spearman = stats.spearmanr(x, mapped)
    kendall = stats.kendalltau(x, mapped)
    probs = triangular_probs_from_index(x)
    groups = [work.loc[work["label"] == c, "expansion_index"].to_numpy(dtype=float) for c in [0, 1, 2]]
    non_empty_groups = [g for g in groups if len(g) > 0]
    kruskal = stats.kruskal(*non_empty_groups)

    return {
        "label_col": label_col,
        "scheme": "keep_012",
        "matched_rows": int(len(work)),
        "count_0": int((y == 0).sum()),
        "count_1": int((y == 1).sum()),
        "count_2": int((y == 2).sum()),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau_b": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "auc_ovr_macro": float(roc_auc_score(y, probs, labels=[0, 1, 2], multi_class="ovr", average="macro")),
        "auc_ovo_macro": float(roc_auc_score(y, probs, labels=[0, 1, 2], multi_class="ovo", average="macro")),
        "kruskal_h": float(kruskal.statistic),
        "kruskal_p": float(kruskal.pvalue),
    }


def continuous_score_metrics(df: pd.DataFrame, score_col: str) -> dict:
    work = df[["sample_no", "expansion_index", score_col]].dropna().copy()
    x = work["expansion_index"].to_numpy(dtype=float)
    s = work[score_col].to_numpy(dtype=float)
    spearman = stats.spearmanr(x, s)
    kendall = stats.kendalltau(x, s)
    pearson = stats.pearsonr(x, s)
    lin = stats.linregress(x, s)
    return {
        "metric": score_col,
        "matched_rows": int(len(work)),
        "spearman": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau_b": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
        "pearson": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "linregress_slope": float(lin.slope),
        "linregress_p": float(lin.pvalue),
    }


def binary_human_comparison(df: pd.DataFrame, pred_col: str, positive_rule: str) -> dict:
    work = df[["sample_no", "human_label", pred_col]].rename(columns={pred_col: "pred"}).dropna().copy()
    work["human_label"] = work["human_label"].astype(int)
    work["pred"] = work["pred"].astype(int)

    if positive_rule == "merged_12":
        y_true = (work["human_label"] != 0).astype(int)
        y_pred = (work["pred"] != 0).astype(int)
    elif positive_rule == "drop_2_keep_01":
        work = work[work["human_label"].isin([0, 1]) & work["pred"].isin([0, 1])].copy()
        y_true = work["human_label"].astype(int)
        y_pred = work["pred"].astype(int)
    else:
        raise ValueError(positive_rule)

    return {
        "rater": pred_col,
        "scheme": positive_rule,
        "matched_rows": int(len(work)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def agreement_metrics(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    work = df[["sample_no", col_a, col_b]].dropna().copy()
    work[col_a] = work[col_a].astype(int)
    work[col_b] = work[col_b].astype(int)
    a = work[col_a].to_numpy()
    b = work[col_b].to_numpy()
    return {
        "left": col_a,
        "right": col_b,
        "matched_rows": int(len(work)),
        "agreement": float(np.mean(a == b)),
        "kappa": float(cohen_kappa_score(a, b)),
        "weighted_kappa": float(cohen_kappa_score(a, b, weights="quadratic")),
    }


def build_frame() -> pd.DataFrame:
    feature_df = pd.read_csv(BASE / "human" / "human_audit_100sample_quantile_prefilled.csv")
    feature_df = feature_df.rename(columns={"audit_sample_id": "sample_no"})
    feature_df["sample_no"] = feature_df["sample_no"].astype(int)

    human_df = parse_human_labels(BASE / "human" / "human_audit_100sample_quantile_pack_minimal_judgment_only.md")
    merged = feature_df.merge(human_df, on="sample_no", how="left")

    api_df = pd.read_csv(API_RESULTS)
    api_df = api_df[(api_df["status"] == "ok") & (api_df["parse_status"].isin(["ok", "ok_fallback_label_only"]))].copy()
    api_df["sample_id"] = api_df["sample_id"].astype(int)
    api_df["label"] = api_df["label"].astype(int)

    for model_requested, short_name in MODEL_MAP.items():
        part = api_df[api_df["model_requested"] == model_requested][["sample_id", "label"]].rename(
            columns={"sample_id": "sample_no", "label": short_name}
        )
        merged = merged.merge(part, on="sample_no", how="left")

    ai_cols = list(MODEL_MAP.values())
    merged["avg_binary12_mean"] = merged[ai_cols].replace({2: 1}).mean(axis=1)
    merged["avg_strength_mean"] = merged[ai_cols].replace({2: 0.5}).mean(axis=1)
    merged["majority4"] = (merged[ai_cols].replace({2: 1}).mean(axis=1) >= 0.5).astype(int)
    return merged.sort_values("sample_no")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_frame()
    df.to_csv(OUT_DIR / "api_rerun_merged_frame.csv", index=False, encoding="utf-8-sig")

    ai_cols = list(MODEL_MAP.values())

    index_rows = []
    for col in ["human_label", *ai_cols]:
        index_rows.append(binary_index_metrics(df, col, "merged_12", "merged_12"))
        index_rows.append(binary_index_metrics(df, col, "drop_2_keep_01", "drop_2_keep_01"))
        index_rows.append(multiclass_index_metrics(df, col))

    continuous_rows = [
        continuous_score_metrics(df, "avg_binary12_mean"),
        continuous_score_metrics(df, "avg_strength_mean"),
    ]

    majority_rows = [
        binary_index_metrics(df.rename(columns={"majority4": "label"}), "label", "majority4_merged_12", "merged_12"),
        binary_index_metrics(df.rename(columns={"majority4": "label"}), "label", "majority4_drop_2_keep_01", "drop_2_keep_01"),
    ]

    human_rows = []
    for col in [*ai_cols, "majority4"]:
        human_rows.append(agreement_metrics(df, "human_label", col))
        human_rows.append(binary_human_comparison(df, col, "merged_12"))
        human_rows.append(binary_human_comparison(df, col, "drop_2_keep_01"))

    pairwise_rows = [agreement_metrics(df, a, b) for a, b in combinations(ai_cols, 2)]

    pd.DataFrame(index_rows).to_csv(OUT_DIR / "api_rerun_index_metrics_labels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(continuous_rows).to_csv(OUT_DIR / "api_rerun_index_metrics_continuous.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(majority_rows).to_csv(OUT_DIR / "api_rerun_index_metrics_majority.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(human_rows).to_csv(OUT_DIR / "api_rerun_human_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pairwise_rows).to_csv(OUT_DIR / "api_rerun_pairwise_ai_agreement.csv", index=False, encoding="utf-8-sig")

    summary = {
        "api_results": str(API_RESULTS),
        "output_dir": str(OUT_DIR),
        "matched_rows": int(len(df)),
        "models": MODEL_MAP,
        "files": {
            "merged_frame": "api_rerun_merged_frame.csv",
            "index_metrics_labels": "api_rerun_index_metrics_labels.csv",
            "index_metrics_continuous": "api_rerun_index_metrics_continuous.csv",
            "index_metrics_majority": "api_rerun_index_metrics_majority.csv",
            "human_comparison_metrics": "api_rerun_human_comparison_metrics.csv",
            "pairwise_ai_agreement": "api_rerun_pairwise_ai_agreement.csv",
        },
    }
    (OUT_DIR / "api_rerun_metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
