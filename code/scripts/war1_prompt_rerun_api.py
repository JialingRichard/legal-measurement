from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]


SECTION_RE = re.compile(r"^##\s+(\d+)\.\s*$", re.M)
TEXT_BLOCK_RE = re.compile(r"### 全文\s+```text\s*(.*?)\s*```", re.S)
JSON_RE = re.compile(r"\{.*\}", re.S)


SYSTEM_PROMPT = (
    "你是一名审慎的中文刑事法研究助理。"
    "你的任务是只根据给定裁判全文，对该案是否属于“扩大化风险/不当入罪”进行盲审判断。"
    "不要猜测本研究框架，不要补充外部事实，不要输出法律普及。"
)


USER_PROMPT_TEMPLATE = """请阅读以下裁判全文，并只根据文本内容做三分类判断：

标签定义：
- 0 = 正常定罪，无明显风险
- 1 = 确属扩大化风险/不当入罪
- 2 = 灰区，人类也难以决断

输出要求：
1. 只输出一个 JSON 对象，不要输出任何额外文字。
2. JSON 必须严格包含以下键：
   - "label": 只能是 0、1、2 之一
   - "reason": 用一句中文简短说明主要理由，控制在 40 字以内
3. 不要使用 markdown 代码块。

样本编号：{sample_id}

裁判全文：
{judgment_text}
"""


@dataclass
class Sample:
    sample_id: int
    judgment_text: str


def load_samples(input_md: Path) -> list[Sample]:
    text = input_md.read_text(encoding="utf-8")
    matches = list(SECTION_RE.finditer(text))
    samples: list[Sample] = []
    for idx, match in enumerate(matches):
        sample_id = int(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        m = TEXT_BLOCK_RE.search(block)
        if not m:
            raise ValueError(f"Failed to extract judgment text for sample {sample_id}")
        samples.append(Sample(sample_id=sample_id, judgment_text=m.group(1).strip()))
    return samples


def parse_sample_ids(spec: str | None, all_ids: Iterable[int]) -> set[int]:
    if not spec:
        return set(all_ids)
    chosen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            chosen.update(range(int(start), int(end) + 1))
        else:
            chosen.add(int(part))
    return chosen


def extract_json_obj(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def call_api(
    *,
    api_key: str,
    base_url: str,
    model: str,
    sample: Sample,
    max_tokens: int,
    timeout: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    sample_id=sample.sample_id,
                    judgment_text=sample.judgment_text,
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        # Volcengine Coding Plan supports this shape and it reduces reasoning-token overhead.
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_one(
    *,
    api_key: str,
    base_url: str,
    model: str,
    sample: Sample,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> dict:
    last_err = None
    for attempt in range(retries + 1):
        try:
            body = call_api(
                api_key=api_key,
                base_url=base_url,
                model=model,
                sample=sample,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = extract_json_obj(content)
            label = ""
            reason = ""
            parse_status = "parse_failed"
            if parsed is not None and str(parsed.get("label", "")).strip() in {"0", "1", "2"}:
                label = str(parsed["label"]).strip()
                reason = str(parsed.get("reason", "")).strip()
                parse_status = "ok"
            return {
                "sample_id": str(sample.sample_id),
                "model_requested": model,
                "model_returned": body.get("model", ""),
                "status": "ok",
                "parse_status": parse_status,
                "label": label,
                "reason": reason,
                "raw_content": content,
                "raw_json": json.dumps(body, ensure_ascii=False),
                "error": "",
                "attempts": str(attempt + 1),
                "total_tokens": str(body.get("usage", {}).get("total_tokens", "")),
                "completion_tokens": str(body.get("usage", {}).get("completion_tokens", "")),
                "reasoning_tokens": str(
                    body.get("usage", {})
                    .get("completion_tokens_details", {})
                    .get("reasoning_tokens", "")
                ),
            }
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            last_err = f"HTTPError {e.code}: {detail}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.2 * (attempt + 1))
    return {
        "sample_id": str(sample.sample_id),
        "model_requested": model,
        "model_returned": "",
        "status": "error",
        "parse_status": "",
        "label": "",
        "reason": "",
        "raw_content": "",
        "raw_json": "",
        "error": last_err or "unknown_error",
        "attempts": str(retries + 1),
        "total_tokens": "",
        "completion_tokens": "",
        "reasoning_tokens": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-md",
        required=True,
        help="Path to a user-prepared markdown prompt pack for single-case evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "audit_100" / "ai_single_case"),
    )
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--sample-ids", default=None, help="e.g. 1,2,5-8")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--base-url",
        default="https://YOUR_API_BASE_URL/v1",
        help="Provider-compatible chat completions base URL placeholder",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise SystemExit("Missing MODEL_API_KEY")

    samples = load_samples(Path(args.input_md))
    selected_ids = parse_sample_ids(args.sample_ids, (s.sample_id for s in samples))
    selected_samples = [s for s in samples if s.sample_id in selected_ids]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models provided")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_csv = output_dir / f"rerun_{run_tag}.csv"

    rows: list[dict] = []
    lock = threading.Lock()

    futures = []
    total_jobs = len(models) * len(selected_samples)
    completed = 0
    print(
        f"Starting rerun: models={len(models)} samples={len(selected_samples)} total_jobs={total_jobs} max_workers={args.max_workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for model in models:
            for sample in selected_samples:
                futures.append(
                    ex.submit(
                        run_one,
                        api_key=api_key,
                        base_url=args.base_url,
                        model=model,
                        sample=sample,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                )
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                rows.append(row)
                completed += 1
                print(
                    f"[{completed}/{total_jobs}] [{row['status']}/{row.get('parse_status','')}] "
                    f"model={row['model_requested']} sample={row['sample_id']} "
                    f"label={row['label']} error={row['error'][:80]}",
                    flush=True,
                )

    rows.sort(key=lambda r: (r["model_requested"], int(r["sample_id"])))
    fieldnames = [
        "sample_id",
        "model_requested",
        "model_returned",
        "status",
        "parse_status",
        "label",
        "reason",
        "raw_content",
        "raw_json",
        "error",
        "attempts",
        "total_tokens",
        "completion_tokens",
        "reasoning_tokens",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "input_md": str(args.input_md),
        "output_csv": str(out_csv),
        "models": models,
        "sample_ids": sorted(selected_ids),
        "max_workers": args.max_workers,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "retries": args.retries,
        "base_url": args.base_url,
        "run_tag": run_tag,
    }
    (output_dir / f"rerun_{run_tag}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote results to {out_csv}", flush=True)


if __name__ == "__main__":
    main()
