from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "paper" / "analysis" / "metrics_100"
MINE_DIR = ROOT / "paper" / "analysis" / "mine_9000"
IEEE_DIR = ROOT / "paper"
OUT_ANALYSIS = IEEE_DIR / "analysis"
OUT_FIGURES = IEEE_DIR / "figures"

MERGED_100 = METRICS_DIR / "war1_100_merged_frame.csv"
RAW_9000 = MINE_DIR / "full_9000_raw.csv"

WEIGHTS = {
    "order_thinness_score": 0.45,
    "private_dispute_score": 0.30,
    "contestation_score": 0.10,
    "overcriminalization_score": 0.15,
}

CAPS = {
    "order_thinness_score": 4.0,
    "private_dispute_score": 5.0,
    "contestation_score": 4.0,
    "overcriminalization_score": 4.0,
}


def rankdata(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def auc_score(scores: list[float], labels: list[int]) -> float:
    ranks = rankdata(scores)
    pos = [r for r, l in zip(ranks, labels) if l == 1]
    n_pos = len(pos)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = float(sum(pos))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(10**8)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def compute_index(row: dict[str, str], active_dims: list[str]) -> float:
    weight_sum = sum(WEIGHTS[d] for d in active_dims)
    total = 0.0
    for dim in active_dims:
        raw = float(row[dim])
        normalized = min(max(raw, 0.0), CAPS[dim]) / CAPS[dim]
        total += (WEIGHTS[dim] / weight_sum) * normalized
    return 100.0 * total


def main() -> None:
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    rows_100 = load_csv(MERGED_100)
    rows_9000 = load_csv(RAW_9000)

    dims = list(WEIGHTS.keys())
    variants = {
        "baseline": dims,
        "drop_order_thinness": [d for d in dims if d != "order_thinness_score"],
        "drop_private_dispute": [d for d in dims if d != "private_dispute_score"],
        "drop_contestation": [d for d in dims if d != "contestation_score"],
        "drop_overcriminalization": [d for d in dims if d != "overcriminalization_score"],
    }

    merged12_labels = [1 if int(r["human_label"]) in (1, 2) else 0 for r in rows_100]
    drop2_rows = [r for r in rows_100 if int(r["human_label"]) in (0, 1)]
    drop2_labels = [int(r["human_label"]) for r in drop2_rows]

    ablation_records = []
    baseline_scores_9000 = [compute_index(r, variants["baseline"]) for r in rows_9000]

    for name, active_dims in variants.items():
        scores_100 = [compute_index(r, active_dims) for r in rows_100]
        scores_drop2 = [compute_index(r, active_dims) for r in drop2_rows]
        scores_9000 = [compute_index(r, active_dims) for r in rows_9000]
        ablation_records.append(
            {
                "variant": name,
                "dims_used": ",".join(active_dims),
                "merged12_spearman": spearman(scores_100, merged12_labels),
                "merged12_auc": auc_score(scores_100, merged12_labels),
                "drop2_spearman": spearman(scores_drop2, drop2_labels),
                "drop2_auc": auc_score(scores_drop2, drop2_labels),
                "corr_with_baseline_9000": spearman(scores_9000, baseline_scores_9000),
            }
        )

    out_csv = OUT_ANALYSIS / "dimension_ablation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "dims_used",
                "merged12_spearman",
                "merged12_auc",
                "drop2_spearman",
                "drop2_auc",
                "corr_with_baseline_9000",
            ],
        )
        writer.writeheader()
        writer.writerows(ablation_records)

    scores = np.array([float(r["expansion_index"]) for r in rows_9000], dtype=float)
    quantiles = {
        "Median": np.quantile(scores, 0.50),
        "P75": np.quantile(scores, 0.75),
        "P90": np.quantile(scores, 0.90),
        "P95": np.quantile(scores, 0.95),
    }
    mean_val = float(np.mean(scores))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    bins = np.arange(0, 82, 2)
    ax.hist(scores, bins=bins, color="#d95f02", alpha=0.78, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Expansion index")
    ax.set_ylabel("Number of cases")
    ax.set_title("Continuous distribution of the expansion index on the 9000-case corpus")
    ax.axvline(mean_val, color="#1b9e77", linestyle="-", linewidth=2, label=f"Mean {mean_val:.1f}")
    colors = ["#7570b3", "#66a61e", "#e7298a", "#1f78b4"]
    for (label, value), color in zip(quantiles.items(), colors):
        ax.axvline(value, color=color, linestyle="--", linewidth=1.8, label=f"{label} {value:.1f}")
    ax.legend(frameon=True, fontsize=9, ncol=3, loc="upper right")
    fig.savefig(OUT_FIGURES / "continuous_index_distribution.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
