from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_URL_DEFAULT = "https://YOUR_API_BASE_URL/v1/chat/completions"

SECTION_RE = re.compile(r"^##\s+(\d+)\.\s*$", re.M)
TEXT_BLOCK_RE = re.compile(r"### 全文\s+```text\s*(.*?)\s*```", re.S)
JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--input-md",
        required=True,
        help="Path to a user-prepared markdown prompt pack for grouped evaluation",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL_DEFAULT,
        help="Provider-compatible chat completions endpoint placeholder",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def load_samples(path: Path) -> dict[int, str]:
    md = path.read_text(encoding="utf-8")
    matches = list(SECTION_RE.finditer(md))
    samples: dict[int, str] = {}
    for i, match in enumerate(matches):
        sid = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        block = md[start:end]
        text_match = TEXT_BLOCK_RE.search(block)
        if text_match:
            samples[sid] = text_match.group(1).strip()
    return samples


SYSTEM_PROMPT = (
    "你是一名审慎的中文刑事法研究助理。你将看到一组裁判样本。"
    "请把它们作为同一批盲审任务统一校准后再判断，不要逐案机械二元化。"
    "允许充分使用“2（灰区）”作为中间带。最终只输出JSON。"
)


def build_user(group: list[int], samples: dict[int, str]) -> str:
    parts = [f"## {sid}\n{samples[sid]}" for sid in group]
    joined = "\n\n".join(parts)
    return f"""下面是一批共{len(group)}个裁判样本。请在通读后，基于整组样本形成相对一致的内部标尺，再分别给每个样本做三分类判断。

标签定义：
- 0 = 正常定罪，无明显风险
- 1 = 确属扩大化风险/不当入罪
- 2 = 灰区，人类也难以决断

要求：
1. 请先整体理解这一整组样本，再逐个给标签。
2. 不要因为单案判断不确定就强行二元化；如果它处在中间带，可以标记为2。
3. 最终只输出一个JSON对象，不要输出任何额外文字。
4. JSON格式必须为：
{{"results":[{{"sample_id":1,"label":0,"reason":"不超过20字"}}, ... ]}}
5. results中必须覆盖这组样本编号，且每个样本仅出现一次。

样本全文如下：

{joined}
"""


def parse_results(content: str) -> list[dict] | None:
    content = content.strip()
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        match = JSON_RE.search(content)
        if match:
            try:
                obj = json.loads(match.group(0))
            except Exception:
                obj = None
    if not isinstance(obj, dict) or not isinstance(obj.get("results"), list):
        return None
    rows = []
    for item in obj["results"]:
        try:
            sid = int(item.get("sample_id"))
            label = int(item.get("label"))
            reason = str(item.get("reason", "")).strip()
        except Exception:
            continue
        if label in {0, 1, 2}:
            rows.append({"group_index": None, "sample_id": sid, "label": label, "reason": reason})
    return rows


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise SystemExit("Missing MODEL_API_KEY")

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    samples = load_samples(Path(args.input_md))
    all_ids = sorted(samples)
    groups = [all_ids[i : i + args.batch_size] for i in range(0, len(all_ids), args.batch_size)]

    results = []
    print(f"START model={args.model} groups={len(groups)} outdir={root}", flush=True)
    for idx, group in enumerate(groups, start=1):
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user(group, samples)},
            ],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "disabled"},
        }
        req = urllib.request.Request(
            args.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        body = None
        err = ""
        t0 = time.perf_counter()
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="ignore")
                err = f"HTTPError {exc.code}: {detail[:300]}"
                print(f"group={idx} attempt={attempt + 1} http_error {err[:180]}", flush=True)
                time.sleep(3 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                print(f"group={idx} attempt={attempt + 1} exc {err[:180]}", flush=True)
                time.sleep(3 * (attempt + 1))

        elapsed = round(time.perf_counter() - t0, 3)
        if body is None:
            print(f"group={idx} FAILED elapsed={elapsed}s err={err}", flush=True)
            continue

        raw = root / f"group_{idx}_raw.json"
        raw.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
        parsed = parse_results(body["choices"][0]["message"]["content"])
        print(
            f"group={idx} finish={body['choices'][0].get('finish_reason')} "
            f"elapsed={elapsed}s parsed={len(parsed) if parsed else 0}/{len(group)}",
            flush=True,
        )
        results.append(
            {
                "group_index": idx,
                "group_ids": group,
                "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
                "completion_tokens": body.get("usage", {}).get("completion_tokens"),
                "total_tokens": body.get("usage", {}).get("total_tokens"),
                "reasoning_tokens": body.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens"),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "raw_path": str(raw),
                "parsed": parsed,
            }
        )
        time.sleep(1.5)

    summary = {
        "model": args.model,
        "batch_size": args.batch_size,
        "groups": [
            {k: v for k, v in result.items() if k != "parsed"} | {"parsed_count": len(result["parsed"]) if result["parsed"] is not None else 0}
            for result in results
        ],
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for result in results:
        for item in (result["parsed"] or []):
            item["group_index"] = result["group_index"]
            rows.append(item)
    rows = sorted(rows, key=lambda x: x["sample_id"])

    with open(root / "labels.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_index", "sample_id", "label", "reason"])
        writer.writeheader()
        writer.writerows(rows)

    print("FINAL label_counts", dict(Counter(r["label"] for r in rows)), flush=True)
    print("WROTE", root / "labels.csv", flush=True)


if __name__ == "__main__":
    main()
