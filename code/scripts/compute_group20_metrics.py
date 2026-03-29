from __future__ import annotations

import json
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
OUT = BASE / "metrics"
OUT.mkdir(parents=True, exist_ok=True)

GROUPED_FILES = {
    "deepseekv32_group20": BASE / "ai_grouped20" / "group20_deepseekv32_20260328" / "labels.csv",
    "doubao20pro_group20": BASE / "ai_grouped20" / "group20_doubaoseed20pro_20260328" / "labels.csv",
    "glm47_group20": BASE / "ai_grouped20" / "group20_glm47_20260328" / "labels.csv",
    "kimik25_group20": BASE / "ai_grouped20" / "group20_kimik25_20260328" / "labels.csv",
}
SINGLE_FILE = BASE / "ai_single_case" / "rerun_20260328_130149_merged_best.csv"
INDEX_FILE = BASE / "human" / "human_audit_100sample_quantile_prefilled.csv"
HUMAN_MD = BASE / "human" / "human_audit_100sample_quantile_pack_minimal_judgment_only.md"


def parse_human_labels(path: Path) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    current = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("## "):
            try:
                current = int(line.split()[1].rstrip("."))
            except Exception:
                current = None
            continue
        if current is None:
            continue
        if "人类终局定性" in line:
            import re

            m = re.search(r"([012])", line)
            if m:
                rows.append({"sample_no": current, "human_label": int(m.group(1))})
                current = None
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


def load_grouped() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key, path in GROUPED_FILES.items():
        df = pd.read_csv(path)
        df = df[["sample_id", "label"]].rename(columns={"sample_id": "sample_no", "label": key})
        frames.append(df)
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on="sample_no", how="outer")
    return out.sort_values("sample_no")


def load_single() -> pd.DataFrame:
    df = pd.read_csv(SINGLE_FILE)
    keep = df[["sample_id", "model_requested", "label"]].copy()
    keep = keep.rename(columns={"sample_id": "sample_no"})
    pivot = keep.pivot(index="sample_no", columns="model_requested", values="label").reset_index()
    return pivot.rename(
        columns={
            "doubao-seed-2.0-pro": "doubao20pro_single",
            "deepseek-v3.2": "deepseekv32_single",
            "glm-4.7": "glm47_single",
            "kimi-k2.5": "kimik25_single",
        }
    ).sort_values("sample_no")


def human_metrics(series: pd.Series, human: pd.Series) -> dict:
    s = series.astype(int).to_numpy()
    h = human.astype(int).to_numpy()
    out = {
        "agreement": float(np.mean(s == h)),
        "kappa": float(cohen_kappa_score(h, s)),
        "weighted_kappa": float(cohen_kappa_score(h, s, weights="quadratic")),
    }

    y_true = (h != 0).astype(int)
    y_pred = (s != 0).astype(int)
    out["merged_12_balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["merged_12_mcc"] = float(matthews_corrcoef(y_true, y_pred))

    mask = (h != 2) & (s != 2)
    hs = h[mask]
    ss = s[mask]
    if len(hs) > 0:
        out["drop_2_keep_01_balanced_accuracy"] = float(balanced_accuracy_score(hs, ss))
        out["drop_2_keep_01_mcc"] = float(matthews_corrcoef(hs, ss))
    else:
        out["drop_2_keep_01_balanced_accuracy"] = float("nan")
        out["drop_2_keep_01_mcc"] = float("nan")

    raw_sp = stats.spearmanr(s, h)
    out["spearman_raw012"] = float(raw_sp.statistic)
    out["spearman_raw012_p"] = float(raw_sp.pvalue)

    mapf = {0: 0.0, 1: 1.0, 2: 0.5}
    sx = np.array([mapf[int(x)] for x in s], dtype=float)
    hx = np.array([mapf[int(x)] for x in h], dtype=float)
    sp = stats.spearmanr(sx, hx)
    kd = stats.kendalltau(sx, hx)
    out["spearman_midband"] = float(sp.statistic)
    out["spearman_midband_p"] = float(sp.pvalue)
    out["kendall_midband"] = float(kd.statistic)
    out["kendall_midband_p"] = float(kd.pvalue)
    return out


def index_binary_metrics(index_vals: pd.Series, labels: pd.Series, rule: str) -> dict:
    work = pd.DataFrame({"index": index_vals, "label": labels}).dropna().copy()
    work["label"] = work["label"].astype(int)
    if rule == "merged_12":
        work["target"] = (work["label"] != 0).astype(int)
    elif rule == "drop_2_keep_01":
        work = work[work["label"].isin([0, 1])].copy()
        work["target"] = work["label"].astype(int)
    else:
        raise ValueError(rule)

    x = work["index"].to_numpy(dtype=float)
    y = work["target"].to_numpy(dtype=int)
    sp = stats.spearmanr(x, y)
    kd = stats.kendalltau(x, y)
    auc = float(roc_auc_score(y, x))
    neg = work.loc[work["target"] == 0, "index"].to_numpy(dtype=float)
    pos = work.loc[work["target"] == 1, "index"].to_numpy(dtype=float)
    mw = stats.mannwhitneyu(neg, pos, alternative="two-sided")
    return {
        "n": int(len(work)),
        "count_0": int((work["target"] == 0).sum()),
        "count_1": int((work["target"] == 1).sum()),
        "spearman": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall": float(kd.statistic),
        "kendall_p": float(kd.pvalue),
        "auc": auc,
        "mannwhitney_p": float(mw.pvalue),
        "cliffs_delta": cliffs_delta(pos, neg),
    }


def index_multiclass_metrics(index_vals: pd.Series, labels: pd.Series) -> dict:
    work = pd.DataFrame({"index": index_vals, "label": labels}).dropna().copy()
    work["label"] = work["label"].astype(int)
    mapf = {0: 0.0, 1: 1.0, 2: 0.5}
    y = np.array([mapf[int(v)] for v in work["label"]], dtype=float)
    sp = stats.spearmanr(work["index"], y)
    kd = stats.kendalltau(work["index"], y)
    kr = stats.kruskal(*[grp["index"].to_numpy(dtype=float) for _, grp in work.groupby("label")])
    labels_arr = work["label"].to_numpy(dtype=int)
    x = work["index"].to_numpy(dtype=float)
    uniq = sorted(work["label"].unique())
    ovr: list[float] = []
    ovo: list[float] = []
    if len(uniq) >= 2:
        for cls in uniq:
            y_bin = (labels_arr == cls).astype(int)
            if len(np.unique(y_bin)) == 2:
                ovr.append(roc_auc_score(y_bin, x))
        for i, a in enumerate(uniq):
            for b in uniq[i + 1 :]:
                mask = np.isin(labels_arr, [a, b])
                y_bin = (labels_arr[mask] == b).astype(int)
                if len(np.unique(y_bin)) == 2:
                    ovo.append(roc_auc_score(y_bin, x[mask]))
    return {
        "n": int(len(work)),
        "count_0": int((work["label"] == 0).sum()),
        "count_1": int((work["label"] == 1).sum()),
        "count_2": int((work["label"] == 2).sum()),
        "spearman": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "kendall": float(kd.statistic),
        "kendall_p": float(kd.pvalue),
        "auc_ovr_macro": float(np.mean(ovr)) if ovr else float("nan"),
        "auc_ovo_macro": float(np.mean(ovo)) if ovo else float("nan"),
        "kruskal_p": float(kr.pvalue),
    }


def main() -> None:
    human = parse_human_labels(HUMAN_MD)
    index_df = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")[["audit_sample_id", "expansion_index"]].rename(
        columns={"audit_sample_id": "sample_no"}
    )
    base = human.merge(index_df, on="sample_no", how="left")
    grouped = load_grouped()
    single = load_single()
    frame = base.merge(grouped, on="sample_no", how="left").merge(single, on="sample_no", how="left")
    frame.to_csv(OUT / "group20_merged_frame.csv", index=False, encoding="utf-8-sig")

    human_rows = []
    for col in GROUPED_FILES.keys():
        human_rows.append({"model": col, **human_metrics(frame[col], frame["human_label"])})
    pd.DataFrame(human_rows).to_csv(OUT / "group20_human_metrics.csv", index=False, encoding="utf-8-sig")

    index_rows = []
    for col in GROUPED_FILES.keys():
        for scheme in ["merged_12", "drop_2_keep_01"]:
            index_rows.append({"model": col, "scheme": scheme, **index_binary_metrics(frame["expansion_index"], frame[col], scheme)})
        index_rows.append({"model": col, "scheme": "keep_012", **index_multiclass_metrics(frame["expansion_index"], frame[col])})
    pd.DataFrame(index_rows).to_csv(OUT / "group20_index_metrics.csv", index=False, encoding="utf-8-sig")

    pair_rows = []
    cols = list(GROUPED_FILES.keys())
    for a, b in combinations(cols, 2):
        aa = frame[a].astype(int)
        bb = frame[b].astype(int)
        pair_rows.append(
            {
                "model_a": a,
                "model_b": b,
                "agreement": float(np.mean(aa == bb)),
                "kappa": float(cohen_kappa_score(aa, bb)),
                "weighted_kappa": float(cohen_kappa_score(aa, bb, weights="quadratic")),
            }
        )
    pd.DataFrame(pair_rows).to_csv(OUT / "group20_pairwise_ai_agreement.csv", index=False, encoding="utf-8-sig")

    cmp_rows = []
    map_group_single = {
        "deepseekv32_group20": "deepseekv32_single",
        "doubao20pro_group20": "doubao20pro_single",
        "glm47_group20": "glm47_single",
        "kimik25_group20": "kimik25_single",
    }
    for g, s in map_group_single.items():
        a = frame[g].astype(int)
        b = frame[s].astype(int)
        changed = a != b
        from collections import Counter

        cmp_rows.append(
            {
                "model": g,
                "changed_count": int(changed.sum()),
                "changed_rate": float(changed.mean()),
                "group_counts": json.dumps(dict(Counter(a))),
                "single_counts": json.dumps(dict(Counter(b))),
                "agreement_with_single": float(np.mean(a == b)),
                "kappa_with_single": float(cohen_kappa_score(a, b)),
            }
        )
    pd.DataFrame(cmp_rows).to_csv(OUT / "group20_vs_single_comparison.csv", index=False, encoding="utf-8-sig")

    summary = {
        "grouped_label_counts": {col: frame[col].value_counts(dropna=False).to_dict() for col in cols},
        "human_metrics_file": str(OUT / "group20_human_metrics.csv"),
        "index_metrics_file": str(OUT / "group20_index_metrics.csv"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
