from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def plot_review_bands(df: pd.DataFrame, out_path: Path) -> None:
    order = ["low_pool", "gray", "focused_gray", "auto_candidate", "high_risk"]
    labels = ["Low pool", "Gray", "Focused gray", "Auto-candidate", "High risk"]
    work = df.set_index("expansion_review_band").reindex(order).reset_index()
    shares = work["share"].astype(float).tolist()

    plt.figure(figsize=(8, 4.6))
    colors = ["#566573", "#7FB3D5", "#F8C471", "#E67E22", "#C0392B"]
    bars = plt.bar(labels, shares, color=colors, edgecolor="black", linewidth=0.6)
    plt.ylabel("Share of 9000-case corpus")
    plt.ylim(0, max(shares) * 1.18)
    plt.title("Deployment review-band distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, share in zip(bars, shares):
        plt.text(bar.get_x() + bar.get_width() / 2, share + 0.008, pct(share), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_dominant_paths(df: pd.DataFrame, out_path: Path) -> None:
    work = df.head(6).copy()
    labels = [
        "Order",
        "Private+order",
        "Private",
        "Overcriminalization",
        "Order+private",
        "Contestation",
    ]
    shares = work["share"].astype(float).tolist()

    plt.figure(figsize=(8.2, 4.8))
    bars = plt.barh(labels, shares, color="#4C78A8", edgecolor="black", linewidth=0.6)
    plt.xlabel("Share of 9000-case corpus")
    plt.title("Dominant expansion pathways")
    plt.xlim(0, max(shares) * 1.18)
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    plt.gca().invert_yaxis()
    for bar, share in zip(bars, shares):
        plt.text(share + 0.006, bar.get_y() + bar.get_height() / 2, pct(share), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_threshold_compression(df: pd.DataFrame, out_path: Path) -> None:
    work = df.copy()
    thresholds = work["threshold"].astype(str).tolist()
    shares = work["share"].astype(float).tolist()
    counts = work["count"].astype(int).tolist()

    plt.figure(figsize=(7.8, 4.6))
    bars = plt.bar(thresholds, shares, color="#59A14F", edgecolor="black", linewidth=0.6)
    plt.ylabel("Share of 9000-case corpus")
    plt.xlabel("Expansion-index threshold")
    plt.title("Threshold compression of the review tail")
    plt.ylim(0, max(shares) * 1.22)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, share, count in zip(bars, shares, counts):
        label = f"{pct(share)}\\n(n={count})"
        plt.text(bar.get_x() + bar.get_width() / 2, share + 0.006, label, ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_framework_pipeline(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    boxes = [
        (0.5, 4.15, 2.2, 1.0, "#D4E6F1", "Judgment text\nand excerpt"),
        (3.2, 4.15, 2.3, 1.0, "#D5F5E3", "LLM semantic\nextraction"),
        (6.0, 4.15, 2.4, 1.0, "#FCF3CF", "Four dimensions\nand score caps"),
        (8.9, 4.15, 2.1, 1.0, "#FADBD8", "Continuous\nindex"),
        (3.2, 1.55, 2.3, 1.0, "#E8DAEF", "Audit layer\n(human + AI)"),
        (6.0, 1.55, 2.4, 1.0, "#D6EAF8", "Structural checks\nand robustness"),
        (8.9, 1.55, 2.1, 1.0, "#D5F5E3", "Operational\nviews"),
    ]
    for x, y, w, h, color, text in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)

    ax.annotate("", xy=(3.2, 4.65), xytext=(2.7, 4.65), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(6.0, 4.65), xytext=(5.5, 4.65), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(8.9, 4.65), xytext=(8.4, 4.65), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.annotate("", xy=(4.35, 3.95), xytext=(4.35, 2.55), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(7.2, 3.95), xytext=(7.2, 2.55), arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.annotate("", xy=(9.95, 3.95), xytext=(9.95, 2.55), arrowprops=dict(arrowstyle="->", lw=1.4))

    ax.text(1.6, 3.15, "16 semantic fields with short evidence quotes", ha="left", va="center", fontsize=10.5)
    ax.text(6.15, 3.15, "Dimensions: order, private, contestation, overcrime", ha="center", va="center", fontsize=10.5)
    ax.text(9.95, 3.15, "Index is the primary measurement object", ha="center", va="center", fontsize=10.5)

    ax.text(4.35, 0.7, "Convergent validation", ha="center", va="center", fontsize=10.5)
    ax.text(7.2, 0.7, "Distribution, yearly movement, sensitivity", ha="center", va="center", fontsize=10.5)
    ax.text(9.95, 0.7, "Optional cut-points and case review", ha="center", va="center", fontsize=10.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def compute_yearly_quantiles(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_df.groupby("source_year")["expansion_index"]
    out = pd.DataFrame(
        {
            "source_year": sorted(raw_df["source_year"].dropna().astype(int).unique()),
        }
    )
    q = grouped.quantile([0.5, 0.75, 0.9, 0.95]).unstack().reset_index()
    q.columns = ["source_year", "median", "p75", "p90", "p95"]
    return out.merge(q, on="source_year", how="left").sort_values("source_year")


def plot_yearly_quantile_lines(df: pd.DataFrame, out_path: Path) -> None:
    years = df["source_year"].astype(int).tolist()

    plt.figure(figsize=(8.4, 4.8))
    plt.plot(years, df["median"].astype(float), marker="o", linewidth=2.0, label="Median", color="#4C78A8")
    plt.plot(years, df["p75"].astype(float), marker="s", linewidth=2.0, label="P75", color="#59A14F")
    plt.plot(years, df["p90"].astype(float), marker="^", linewidth=2.0, label="P90", color="#F28E2B")
    plt.plot(years, df["p95"].astype(float), marker="D", linewidth=2.0, label="P95", color="#E15759")

    plt.ylabel("Expansion index")
    plt.xlabel("Year")
    plt.title("Yearly quantile movement")
    plt.xticks(years)
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_yearly_signal_lines(
    core_df: pd.DataFrame,
    upstream_df: pd.DataFrame,
    path_df: pd.DataFrame,
    out_path: Path,
) -> None:
    core_work = core_df.sort_values("source_year").copy()
    upstream_work = upstream_df.sort_values("source_year").copy()
    path_work = path_df.sort_values("source_year").copy()
    years = core_work["source_year"].astype(int).tolist()

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(
        years,
        core_work["expansion_index"].astype(float),
        marker="o",
        linewidth=2.4,
        color="#4C78A8",
        label="Expansion index",
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean index")
    ax1.set_title("Overall movement of the continuous index")
    ax1.grid(linestyle="--", alpha=0.25)
    ax1.legend(frameon=False, loc="upper left")

    core_specs = [
        ("order_thinness_score", "Order thinness", "#59A14F", "o"),
        ("private_dispute_score", "Private dispute", "#F28E2B", "s"),
        ("contestation_score", "Contestation", "#E15759", "^"),
        ("overcriminalization_score", "Overcriminalization", "#AF7AA1", "D"),
    ]
    for col, label, color, marker in core_specs:
        ax2.plot(years, core_work[col].astype(float), marker=marker, linewidth=2.0, color=color, label=label)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Yearly mean raw dimension score")
    ax2.set_title("Core doctrinal dimensions")
    ax2.grid(linestyle="--", alpha=0.25)
    ax2.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)

    upstream_specs = [
        ("qualification_dispute_present", "Qualification dispute", "#4C78A8", "o"),
        ("defense_strategy", "Defense strategy", "#59A14F", "s"),
        ("prosecutor_sentence_suggestion", "Sentence suggestion", "#F28E2B", "^"),
        ("plead_guilty_status", "Plea-related signal", "#E15759", "D"),
    ]
    for col, label, color, marker in upstream_specs:
        ax3.plot(years, upstream_work[col].astype(float), marker=marker, linewidth=2.0, color=color, label=label)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Yearly mean upstream indicator")
    ax3.set_title("Procedural and contestation-related signals")
    ax3.grid(linestyle="--", alpha=0.25)
    ax3.legend(frameon=False, loc="upper left", ncol=2, fontsize=9)

    path_specs = [
        ("order", "Order", "#59A14F", "o"),
        ("private+order", "Private+order", "#9C755F", "s"),
        ("private", "Private", "#4C78A8", "D"),
        ("overcriminalization", "Overcriminalization", "#AF7AA1", "^"),
    ]
    for col, label, color, marker in path_specs:
        if col in path_work.columns:
            ax4.plot(years, path_work[col].astype(float), marker=marker, linewidth=2.0, color=color, label=label)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Share within yearly slice")
    ax4.set_title("Dominant pathway composition")
    ax4.grid(linestyle="--", alpha=0.25)
    ax4.legend(frameon=False, loc="upper right", ncol=2, fontsize=9)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xticks(years)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_path_by_band_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    order = ["low_pool", "gray", "focused_gray", "auto_candidate", "high_risk"]
    path_cols = [
        "order",
        "private+order",
        "order+private",
        "order+private+overcriminalization",
        "private+order+overcriminalization",
        "private",
        "overcriminalization",
        "contestation",
    ]
    label_map = {
        "low_pool": "Low pool",
        "gray": "Gray",
        "focused_gray": "Focused gray",
        "auto_candidate": "Auto-candidate",
        "high_risk": "High risk",
        "order": "Order",
        "private+order": "Private+order",
        "order+private": "Order+private",
        "order+private+overcriminalization": "Order+private+overcrime",
        "private+order+overcriminalization": "Private+order+overcrime",
        "private": "Private",
        "overcriminalization": "Overcrime",
        "contestation": "Contestation",
    }
    work = df.set_index("expansion_review_band").reindex(order)
    available_cols = [c for c in path_cols if c in work.columns]
    mat = work[available_cols].fillna(0.0).to_numpy(dtype=float)

    plt.figure(figsize=(9.2, 4.6))
    im = plt.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(0.2, float(mat.max())))
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Within-band share")
    plt.xticks(range(len(available_cols)), [label_map[c] for c in available_cols], rotation=25, ha="right")
    plt.yticks(range(len(order)), [label_map[r] for r in order])
    plt.title("Dominant-path composition by review band")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= 0.08:
                plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    review_bands = pd.read_csv(args.analysis_dir / "review_band_counts.csv", encoding="utf-8-sig")
    paths = pd.read_csv(args.analysis_dir / "dominant_path_counts.csv", encoding="utf-8-sig")
    thresholds = pd.read_csv(args.analysis_dir / "threshold_summary.csv", encoding="utf-8-sig")
    path_by_band = pd.read_csv(args.analysis_dir / "path_by_review_band_share.csv", encoding="utf-8-sig")
    yearly_core = pd.read_csv(args.analysis_dir / "yearly_core_means.csv", encoding="utf-8-sig")
    yearly_paths = pd.read_csv(args.analysis_dir / "yearly_path_share.csv", encoding="utf-8-sig")
    yearly_upstream = pd.read_csv(args.analysis_dir / "yearly_upstream_means.csv", encoding="utf-8-sig")
    raw_df = pd.read_csv(args.analysis_dir / "full_9000_raw.csv", encoding="utf-8-sig", low_memory=False)
    yearly_quantiles = compute_yearly_quantiles(raw_df)
    yearly_quantiles.to_csv(args.analysis_dir / "yearly_quantiles.csv", index=False, encoding="utf-8-sig")

    plot_review_bands(review_bands, args.output_dir / "review_band_distribution.png")
    plot_dominant_paths(paths, args.output_dir / "dominant_path_distribution.png")
    plot_threshold_compression(thresholds, args.output_dir / "threshold_compression.png")
    plot_framework_pipeline(args.output_dir / "framework_pipeline.png")
    plot_yearly_quantile_lines(yearly_quantiles, args.output_dir / "yearly_quantile_lines.png")
    plot_yearly_signal_lines(yearly_core, yearly_upstream, yearly_paths, args.output_dir / "yearly_signal_lines.png")
    plot_path_by_band_heatmap(path_by_band, args.output_dir / "path_by_review_band_heatmap.png")


if __name__ == "__main__":
    main()
