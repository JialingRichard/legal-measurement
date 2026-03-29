import argparse
import csv
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TEXT_CANDIDATES = ["全文", "正文", "文书内容", "text", "content"]

KEYWORDS = [
    "民间纠纷",
    "邻里纠纷",
    "经济纠纷",
    "感情纠纷",
    "家庭纠纷",
    "工作纠纷",
    "口角",
    "争执",
    "纠纷",
    "公共场所",
    "市场",
    "街道",
    "KTV",
    "饭店",
    "店门",
    "玻璃",
    "打砸",
    "追逐",
    "辱骂",
    "恐吓",
    "随意殴打",
    "情节恶劣",
    "秩序严重混乱",
    "轻伤",
    "轻微伤",
    "不予采纳",
    "辩护意见",
    "认罪认罚",
    "具结书",
    "量刑建议",
    "速裁程序",
    "简易程序",
    "从犯",
    "故意伤害",
    "故意毁坏财物",
    "不构成寻衅滋事",
    "定性不当",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--year-start", type=int, default=2013)
    parser.add_argument("--year-end", type=int, default=2021)
    parser.add_argument("--sample-per-year", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interleave-years", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--excerpt-max-sentences", type=int, default=20)
    parser.add_argument("--excerpt-max-chars", type=int, default=2800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def detect_text_column(columns):
    for candidate in TEXT_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"未找到正文列，候选列：{TEXT_CANDIDATES}，实际列：{columns}")


def split_sentences(text):
    parts = re.split(r"[\n。；;！？!?]", str(text).replace("\r", "\n"))
    return [p.strip() for p in parts if p and p.strip()]


def build_excerpt(text, max_sentences=20, max_chars=2800):
    sentences = split_sentences(text)
    selected = []
    seen = set()
    for sentence in sentences:
        if any(keyword in sentence for keyword in KEYWORDS):
            norm = sentence.strip()
            if norm not in seen:
                selected.append(norm)
                seen.add(norm)
        if len(selected) >= max_sentences:
            break
    if not selected:
        selected = sentences[: min(18, len(sentences))]
    return "。\n".join(selected)[:max_chars]


def build_prompt(row, text_col, excerpt_max_sentences=20, excerpt_max_chars=2800):
    excerpt = build_excerpt(
        row[text_col],
        max_sentences=excerpt_max_sentences,
        max_chars=excerpt_max_chars,
    )
    return [
        {
            "role": "system",
            "content": (
                "你是中文刑事裁判文书信息抽取器。"
                "你只能依据文书内容抽取结构化字段，禁止补充常识、禁止输出思考过程。"
                "只允许输出一个 JSON 对象。"
            ),
        },
        {
            "role": "user",
            "content": f"""
/no_think

请阅读下面的寻衅滋事一审判决书片段，抽取与“法条边界扩大化”相关的结构化字段。
你的任务仅限于抽取客观事实与诉讼行为，禁止直接判断案件是否属于“边界扩大化”。

字段定义：
1. victim_relationship_type
- 0 = 完全陌生人/无差别攻击对象
- 1 = 偶发琐事摩擦，如走错门、敬酒、排队、临时口角等临时起意冲突
- 2 = 实质性特定纠纷，如债务、劳资、土地、产权、工程、婚恋等明确纠葛

2. location_type
- 0 = 开放的室外公共空间，如街道、广场、路边、市场外部
- 1 = 真正私密/偏僻空间，如私人住宅、偏僻荒郊、不对外营业的封闭厂区、封闭施工工地
- 2 = 营业性公共场所，如KTV、饭店、医院、大排档、网吧、游戏厅、商场等

3. harm_severity_text
- 只从以下选一项：无明显伤情 / 轻微伤 / 轻伤二级 / 轻伤一级 / 重伤 / 仅财产损失 / 混合后果 / 看不清

4. qualification_dispute_present
- 0 = 未见针对“是否构成寻衅滋事”的定性异议
- 1 = 明确出现“更像民间纠纷/故意伤害/故意毁坏财物/不构成寻衅滋事”等定性异议

5. court_response_to_qualification_dispute
- 0 = 未见定性异议，或看不清法院是否回应
- 1 = 法院简短驳回，缺少具体要件分析
- 2 = 法院具体回应了为何仍构成寻衅滋事

6. property_damage_present
- 0 = 未见明确财产损失或打砸
- 1 = 明确出现打砸/毁损财物/财产损失

7. public_order_fact_level
- 0 = 未见明确公共秩序受扰事实
- 1 = 有一定公共性事实，例如公共场所、围观、追逐、多人受扰，但不算特别强
- 2 = 公共秩序受扰事实很强，例如多人围观、持续追逐、市场/门店秩序明显混乱

8. plead_guilty_status
- 0 = 明确拒不认罪，或对核心事实/定性有强烈异议
- 1 = 一般性认罪，如“当庭认罪”“如实供述”，但未明确提认罪认罚制度或签字具结
- 2 = 明确适用认罪认罚从宽制度，如“自愿认罪认罚”“签署具结书”“同意适用速裁/简易程序”

9. defense_strategy
- 0 = 无律师，或仅做极为简略的罪轻辩护，如初犯、偶犯、态度好
- 1 = 积极量刑辩护，如主张从犯、赔偿谅解、被害人有过错，但不挑战寻衅滋事定性
- 2 = 无罪辩护，或明确提出定性异议，如认为系普通纠纷、应定故意伤害、故意毁坏财物等

10. prosecutor_sentence_suggestion
- 0 = 未见检察院的具体量刑建议
- 1 = 文书明确记载检察院提出了量刑建议，如建议判处有期徒刑X个月或建议适用缓刑

11. extreme_violence_indicators
- 0 = 未见明显极端暴力形态
- 1 = 明确存在纠集多人、持刀/持枪/钢管/砍刀/洋镐把等器械、大规模追砍打砸、明显报复性暴力

12. evil_force_indicators
- 0 = 未见明显涉恶/黑产打击特征
- 1 = 明确出现村霸/涉村干、暴力讨债、插手工程、垄断市场、持器械在偏僻处恐吓、断水断电/剪电缆等软暴力、扫黑除恶式表述或近似特征

13. public_order_conclusion_present
- 0 = 文书未明确使用“破坏社会秩序/严重影响社会秩序/公共场所秩序严重混乱”等结论性表述
- 1 = 文书明确使用了上述公共秩序结论性表述

14. public_order_specific_fact_score
- 0 = 未见具体可核实的秩序扰乱事实
- 1 = 仅见一般公共场所发生、轻微围观、一般追逐等较弱事实
- 2 = 明确出现营业受阻、交通受阻、多人恐慌、持续失控、多人持续受扰等较强事实
- 3 = 明确出现大范围功能中断、公共场所秩序明显瘫痪等非常强事实

15. target_specificity
- 0 = 随机/无特定对象
- 1 = 临时冲突中形成的特定对象
- 2 = 明确既有特定对象或既有纠纷对象

16. violence_escalation_pattern
- 0 = 单次即时冲突
- 1 = 短暂追打或局部升级
- 2 = 持续追逐、反复攻击、明显扩大化滋扰
- 3 = 结伙围殴、持续性打砸追逐、明显波及周边秩序

同时输出：
- relationship_quote
- location_quote
- harm_quote
- qualification_quote
- court_response_quote
- property_damage_quote
- public_order_quote
- plead_guilty_quote
- defense_strategy_quote
- prosecutor_sentence_quote
- extreme_violence_quote
- evil_force_quote
- public_order_conclusion_quote
- public_order_specific_fact_quote
- target_specificity_quote
- violence_escalation_quote
- confidence: low / medium / high

特别要求：
- 你不能直接输出“边界扩大化候选”结论
- 你只能抽取客观字段，不要帮代码做最终法律评价
- 如果辩护意见只谈从轻处罚、不谈定性，qualification_dispute_present 必须记为0
- 营业性公共场所（KTV、饭店、医院、游戏厅、网吧等）一律优先记为 location_type=2，不能因为“在屋里”就记成私密空间
- 只有私人住宅、偏僻荒地、不对外营业的封闭厂区、封闭施工工地等，才适合记为 location_type=1
- 如果文书核心冲突发生在封闭施工工地、封闭作业区、非顾客开放区域，即使整体地点带有“园区/垂钓园/饭店”字样，也优先考虑 location_type=1
- 偶发口角、敬酒摩擦、走错门、排队冲突等临时冲突，应记为 victim_relationship_type=1，不是实质性特定纠纷
- 债务、劳资、土地、产权、工程、婚恋等明确纠葛，才记为 victim_relationship_type=2
- 如果文书明确出现讨要货款、土地施工阻拦、工程争议、索要物品/财物下落等持续性利益冲突，应优先记为 victim_relationship_type=2
- 如果文书表现为无差别殴打陌生人或随机路人，victim_relationship_type 才记为0
- 如果只是“认罪”“如实供述”“当庭无异议”，而没有认罪认罚制度、具结书、速裁/简易程序等字样，plead_guilty_status 通常记为1
- 如果辩护意见只说自首、坦白、初犯、偶犯、态度好，defense_strategy 记为0
- 如果辩护意见主张从犯、赔偿谅解、被害人过错、量刑过重，但不挑战罪名，defense_strategy 记为1
- 如果辩护意见明确主张不构成寻衅滋事或应改定其他罪名，defense_strategy 记为2
- 若看不清，就保守填 0 或较低等级
- 所有 quote 字段都必须极短，只保留最关键的半句或一句，尽量控制在 30 个汉字以内
- 如果原文相关摘录很长，必须主动压缩，不能整段照抄

请只输出如下 JSON：
{{
  "victim_relationship_type": 0,
  "location_type": 0,
  "harm_severity_text": "看不清",
  "qualification_dispute_present": 0,
  "court_response_to_qualification_dispute": 0,
  "property_damage_present": 0,
  "public_order_fact_level": 0,
  "plead_guilty_status": 0,
  "defense_strategy": 0,
  "prosecutor_sentence_suggestion": 0,
  "extreme_violence_indicators": 0,
  "evil_force_indicators": 0,
  "public_order_conclusion_present": 0,
  "public_order_specific_fact_score": 0,
  "target_specificity": 0,
  "violence_escalation_pattern": 0,
  "relationship_quote": "",
  "location_quote": "",
  "harm_quote": "",
  "qualification_quote": "",
  "court_response_quote": "",
  "property_damage_quote": "",
  "public_order_quote": "",
  "plead_guilty_quote": "",
  "defense_strategy_quote": "",
  "prosecutor_sentence_quote": "",
  "extreme_violence_quote": "",
  "evil_force_quote": "",
  "public_order_conclusion_quote": "",
  "public_order_specific_fact_quote": "",
  "target_specificity_quote": "",
  "violence_escalation_quote": "",
  "confidence": "low"
}}

案件信息：
- 案号：{row.get('案号', '')}
- 裁判日期：{row.get('裁判日期', '')}
- 来源年月：{row.get('source_month', '')}

判决书片段：
{excerpt}
""".strip(),
        },
    ]


def build_prompt_text(row, text_col, tokenizer, excerpt_max_sentences=20, excerpt_max_chars=2800):
    messages = build_prompt(
        row,
        text_col,
        excerpt_max_sentences=excerpt_max_sentences,
        excerpt_max_chars=excerpt_max_chars,
    )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def extract_json(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        stripped = str(text).strip()
        if stripped.startswith("{") and not stripped.endswith("}"):
            return json.loads(stripped + "}")
        raise ValueError("模型输出中未找到 JSON 对象")
    payload = match.group(0).strip()
    if payload.startswith("{") and not payload.endswith("}"):
        payload = payload + "}"
    return json.loads(payload)


def build_selected_df(df: pd.DataFrame, args) -> tuple[pd.DataFrame, list[dict]]:
    df = df.copy().reset_index().rename(columns={"index": "source_row_id"})
    sampled = []
    per_year_counts = []
    for year in range(args.year_start, args.year_end + 1):
        year_df = df[df["source_year"] == year].copy()
        keep_n = min(args.sample_per_year, len(year_df))
        if keep_n:
            sampled.append(year_df.sample(n=keep_n, random_state=args.seed + year).reset_index(drop=True))
        per_year_counts.append(
            {"year": year, "available_rows": int(len(year_df)), "sampled_rows": int(keep_n)}
        )
    if not sampled:
        selected = df.iloc[0:0].copy()
    elif args.interleave_years:
        interleaved_rows = []
        max_len = max(len(part) for part in sampled)
        for row_idx in range(max_len):
            for part in sampled:
                if row_idx < len(part):
                    interleaved_rows.append(part.iloc[row_idx].to_dict())
        selected = pd.DataFrame(interleaved_rows, columns=sampled[0].columns)
    else:
        selected = pd.concat(sampled, ignore_index=True)
    return selected.reset_index(drop=True), per_year_counts


def output_fieldnames(df_columns):
    extra = [
        "inference_sec",
        "text_length",
        "evidence_excerpt",
        "model_raw_output",
        "parse_ok",
        "parse_error",
        "victim_relationship_type",
        "location_type",
        "harm_severity_text",
        "qualification_dispute_present",
        "court_response_to_qualification_dispute",
        "property_damage_present",
        "public_order_fact_level",
        "plead_guilty_status",
        "defense_strategy",
        "prosecutor_sentence_suggestion",
        "extreme_violence_indicators",
        "evil_force_indicators",
        "public_order_conclusion_present",
        "public_order_specific_fact_score",
        "target_specificity",
        "violence_escalation_pattern",
        "relationship_quote",
        "location_quote",
        "harm_quote",
        "qualification_quote",
        "court_response_quote",
        "property_damage_quote",
        "public_order_quote",
        "plead_guilty_quote",
        "defense_strategy_quote",
        "prosecutor_sentence_quote",
        "extreme_violence_quote",
        "evil_force_quote",
        "public_order_conclusion_quote",
        "public_order_specific_fact_quote",
        "target_specificity_quote",
        "violence_escalation_quote",
        "confidence",
        "order_thinness_score",
        "private_dispute_score",
        "contestation_score",
        "overcriminalization_score",
        "expansion_index",
        "expansion_risk_level",
        "expansion_review_band",
        "dominant_expansion_path",
        "boundary_expansion_candidate",
        "boundary_score",
        "boundary_reason",
    ]
    names = []
    for col in list(df_columns) + extra:
        if col not in names:
            names.append(col)
    return names


def normalize_record_defaults(record):
    defaults = {
        "victim_relationship_type": 0,
        "location_type": 0,
        "harm_severity_text": "看不清",
        "qualification_dispute_present": 0,
        "court_response_to_qualification_dispute": 0,
        "property_damage_present": 0,
        "public_order_fact_level": 0,
        "plead_guilty_status": 0,
        "defense_strategy": 0,
        "prosecutor_sentence_suggestion": 0,
        "extreme_violence_indicators": 0,
        "evil_force_indicators": 0,
        "public_order_conclusion_present": 0,
        "public_order_specific_fact_score": 0,
        "target_specificity": 0,
        "violence_escalation_pattern": 0,
        "relationship_quote": "",
        "location_quote": "",
        "harm_quote": "",
        "qualification_quote": "",
        "court_response_quote": "",
        "property_damage_quote": "",
        "public_order_quote": "",
        "plead_guilty_quote": "",
        "defense_strategy_quote": "",
        "prosecutor_sentence_quote": "",
        "extreme_violence_quote": "",
        "evil_force_quote": "",
        "public_order_conclusion_quote": "",
        "public_order_specific_fact_quote": "",
        "target_specificity_quote": "",
        "violence_escalation_quote": "",
        "confidence": "low",
        "order_thinness_score": 0,
        "private_dispute_score": 0,
        "contestation_score": 0,
        "overcriminalization_score": 0,
        "expansion_index": 0.0,
        "expansion_risk_level": "low",
        "expansion_review_band": "low_pool",
        "dominant_expansion_path": "mixed",
        "boundary_expansion_candidate": 0,
        "boundary_score": 0,
        "boundary_reason": "",
    }
    for key, value in defaults.items():
        if key not in record or pd.isna(record.get(key)):
            record[key] = value
    return record


def normalize_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def derive_boundary_candidate(record):
    relationship = normalize_int(record.get("victim_relationship_type"))
    location = normalize_int(record.get("location_type"))
    qualification = normalize_int(record.get("qualification_dispute_present"))
    court_response = normalize_int(record.get("court_response_to_qualification_dispute"))
    property_damage = normalize_int(record.get("property_damage_present"))
    public_order = normalize_int(record.get("public_order_fact_level"))
    plead_guilty = normalize_int(record.get("plead_guilty_status"))
    defense_strategy = normalize_int(record.get("defense_strategy"))
    prosecutor_sentence = normalize_int(record.get("prosecutor_sentence_suggestion"))
    extreme_violence = normalize_int(record.get("extreme_violence_indicators"))
    public_order_conclusion = normalize_int(record.get("public_order_conclusion_present"))
    public_order_specific = normalize_int(record.get("public_order_specific_fact_score"))
    target_specificity = normalize_int(record.get("target_specificity"))
    violence_pattern = normalize_int(record.get("violence_escalation_pattern"))
    harm_text = str(record.get("harm_severity_text", "")).strip()
    reasons = []

    order_score = 0
    private_score = 0
    contestation_score = 0
    overcrime_score = 0

    if public_order_conclusion == 1 and public_order_specific == 0:
        order_score += 2
        reasons.append("存在公共秩序结论但缺少具体秩序事实")
    if location == 1:
        order_score += 1
        reasons.append("地点为真私密/偏僻空间")
    elif location == 2 and public_order_specific <= 1:
        order_score += 1
        reasons.append("营业性公共场所但秩序具体事实偏弱")
    if relationship == 2 or target_specificity == 2:
        order_score += 1
        reasons.append("对象具有明确特定性")
    elif relationship == 0 and target_specificity == 0:
        order_score -= 1
        reasons.append("对象更接近陌生/随机")
    if location == 1 and public_order_specific == 0:
        order_score += 1
        reasons.append("私密/偏僻空间且缺少具体秩序事实")

    if relationship == 2:
        private_score += 2
        reasons.append("存在实质性特定纠纷")
    elif relationship == 1:
        private_score += 1
        reasons.append("属于偶发琐事摩擦")
    if target_specificity == 2:
        private_score += 1
        reasons.append("存在既有特定对象")
    if relationship == 2 and target_specificity == 2:
        private_score += 1
        reasons.append("纠纷对象高度特定")

    if qualification == 1:
        contestation_score += 1
        reasons.append("存在定性异议")
    if qualification == 1 and court_response <= 1:
        contestation_score += 1
        reasons.append("法院对定性异议回应偏弱")
    if defense_strategy == 2:
        contestation_score += 2
        reasons.append("存在无罪/改定性辩护")
    elif defense_strategy == 1:
        contestation_score += 1
        reasons.append("存在积极量刑/事实辩护")

    if harm_text in {"轻微伤", "轻伤二级", "仅财产损失"}:
        overcrime_score += 2
        reasons.append(f"后果偏轻（{harm_text}）")
    elif harm_text == "轻伤一级":
        overcrime_score += 1
        reasons.append("后果为轻伤一级")
    elif "重伤" in harm_text or "死亡" in harm_text:
        overcrime_score -= 2
        reasons.append("存在重伤/死亡后果")
    if extreme_violence == 1:
        overcrime_score -= 1
        reasons.append("存在极端暴力形态")
    if violence_pattern >= 2:
        overcrime_score -= 1
        reasons.append("暴力升级模式较强")
    elif violence_pattern == 1:
        overcrime_score += 0
        reasons.append("存在短暂追打或局部升级")

    if property_damage == 1 and harm_text in {"仅财产损失", "看不清"}:
        overcrime_score += 1
        reasons.append("更像财产侵害路径")

    order_norm = max(min(order_score, 4), 0) / 4
    private_norm = max(min(private_score, 5), 0) / 5
    contestation_norm = max(min(contestation_score, 4), 0) / 4
    overcrime_norm = max(min(overcrime_score, 4), 0) / 4
    order_component = 0.45 * order_norm
    private_component = 0.30 * private_norm
    contestation_component = 0.10 * contestation_norm
    overcrime_component = 0.15 * overcrime_norm
    expansion_index = round(
        (order_component + private_component + contestation_component + overcrime_component) * 100,
        1,
    )

    if expansion_index >= 70:
        risk_level = "high"
    elif expansion_index >= 35:
        risk_level = "medium"
    else:
        risk_level = "low"

    if expansion_index >= 70:
        review_band = "high_risk"
    elif expansion_index >= 55:
        review_band = "auto_candidate"
    elif expansion_index >= 50:
        review_band = "focused_gray"
    elif expansion_index >= 35:
        review_band = "gray"
    else:
        review_band = "low_pool"

    candidate = 1 if expansion_index >= 55 else 0
    boundary_score = int(round(expansion_index))
    reason = "；".join(reasons)

    dominant_components = {
        "order": order_component,
        "private": private_component,
        "contestation": contestation_component,
        "overcriminalization": overcrime_component,
    }
    ranked_components = sorted(dominant_components.items(), key=lambda item: item[1], reverse=True)
    active_components = [name for name, value in ranked_components if value >= 0.08]
    if not active_components:
        dominant_path = ranked_components[0][0]
    elif len(active_components) >= 3:
        dominant_path = "+".join(active_components[:3])
    elif len(active_components) == 2:
        dominant_path = "+".join(active_components)
    else:
        dominant_path = active_components[0]

    record["order_thinness_score"] = order_score
    record["private_dispute_score"] = private_score
    record["contestation_score"] = contestation_score
    record["overcriminalization_score"] = overcrime_score
    record["expansion_index"] = expansion_index
    record["expansion_risk_level"] = risk_level
    record["expansion_review_band"] = review_band
    record["dominant_expansion_path"] = dominant_path

    return candidate, boundary_score, reason


def load_resume_count(out_csv: Path) -> int:
    if not out_csv.exists():
        return 0
    try:
        done_df = pd.read_csv(out_csv, encoding="utf-8-sig", usecols=["source_row_id"])
        return int(len(done_df))
    except Exception:
        done_df = pd.read_csv(out_csv, encoding="utf-8-sig")
        return int(len(done_df))


def write_progress(progress_path: Path, payload: dict) -> None:
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_results(out_csv, args, per_year_counts):
    df = pd.read_csv(out_csv, encoding="utf-8-sig")
    total_inference_sec = round(float(df["inference_sec"].fillna(0).sum()), 4) if len(df) else 0.0
    return {
        "model_id": args.model_id,
        "input_csv": args.input_csv,
        "year_start": args.year_start,
        "year_end": args.year_end,
        "sample_per_year": args.sample_per_year,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "excerpt_max_sentences": args.excerpt_max_sentences,
        "excerpt_max_chars": args.excerpt_max_chars,
        "sample_size_actual": int(len(df)),
        "avg_inference_sec": round(total_inference_sec / max(len(df), 1), 4),
        "total_inference_sec": total_inference_sec,
        "parse_ok_rate": round(float(df["parse_ok"].mean()), 4) if len(df) else 0.0,
        "boundary_candidate_count": int(df["boundary_expansion_candidate"].fillna(0).astype(int).sum())
        if len(df)
        else 0,
        "confidence_counts": df["confidence"].value_counts(dropna=False).to_dict() if len(df) else {},
        "per_year_counts": per_year_counts,
        "output_csv": str(out_csv),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"xunxinzishi_qwen_boundary_{args.year_start}_{args.year_end}_{args.sample_per_year}x{args.year_end-args.year_start+1}"
    out_csv = output_dir / f"{stem}.csv"
    summary_path = output_dir / f"{stem}_summary.json"
    log_path = output_dir / f"{stem}.log"
    progress_path = output_dir / f"{stem}_progress.json"

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    text_col = detect_text_column(df.columns.tolist())
    df, per_year_counts = build_selected_df(df, args)
    if len(df) == 0:
        raise ValueError("筛样后没有样本")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.eval()

    fieldnames = output_fieldnames(df.columns.tolist())
    completed = load_resume_count(out_csv) if args.resume else 0
    if not args.resume and out_csv.exists():
        out_csv.unlink()
    if not args.resume and summary_path.exists():
        summary_path.unlink()

    def append_log(msg):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    append_log(
        f"[RUN] input={args.input_csv} selected={len(df)} model={args.model_id} "
        f"sample_per_year={args.sample_per_year} batch_size={args.batch_size} "
        f"max_new_tokens={args.max_new_tokens} excerpt_max_sentences={args.excerpt_max_sentences} "
        f"excerpt_max_chars={args.excerpt_max_chars} "
        f"resume={args.resume} completed={completed}"
    )
    write_progress(
        progress_path,
        {
            "status": "running",
            "input_csv": args.input_csv,
            "output_csv": str(out_csv),
            "summary_json": str(summary_path),
            "log_path": str(log_path),
            "total_selected": len(df),
            "completed": completed,
            "remaining": len(df) - completed,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    warmup_text = build_prompt_text(
        df.iloc[min(completed, len(df) - 1)],
        text_col,
        tokenizer,
        excerpt_max_sentences=args.excerpt_max_sentences,
        excerpt_max_chars=args.excerpt_max_chars,
    )
    warmup_inputs = tokenizer([warmup_text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        _ = model.generate(**warmup_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    run_started = time.perf_counter()
    for batch_start in range(completed, len(df), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        batch_records = []
        prompt_texts = []
        for _, row in batch_df.iterrows():
            record = row.to_dict()
            record["text_length"] = int(len(str(row[text_col])))
            record["evidence_excerpt"] = build_excerpt(
                row[text_col],
                max_sentences=args.excerpt_max_sentences,
                max_chars=args.excerpt_max_chars,
            )
            batch_records.append(record)
            prompt_texts.append(
                build_prompt_text(
                    row,
                    text_col,
                    tokenizer,
                    excerpt_max_sentences=args.excerpt_max_sentences,
                    excerpt_max_chars=args.excerpt_max_chars,
                )
            )
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
        prompt_token_len = inputs["input_ids"].shape[1]
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_elapsed = time.perf_counter() - start

        decoded_responses = []
        for i in range(len(batch_records)):
            generated = outputs[i][prompt_token_len:]
            decoded_responses.append(tokenizer.decode(generated, skip_special_tokens=True))

        write_header = not out_csv.exists()
        with out_csv.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for i, record in enumerate(batch_records):
                row = batch_df.iloc[i]
                response = decoded_responses[i]
                record["inference_sec"] = round(batch_elapsed / max(len(batch_records), 1), 4)
                record["model_raw_output"] = response
                try:
                    parsed = extract_json(response)
                    record.update(parsed)
                    record["parse_ok"] = 1
                    record["parse_error"] = ""
                except Exception as exc:
                    record["parse_ok"] = 0
                    record["parse_error"] = str(exc)
                normalize_record_defaults(record)
                candidate, score, reason = derive_boundary_candidate(record)
                record["boundary_expansion_candidate"] = candidate
                record["boundary_score"] = score
                record["boundary_reason"] = reason
                writer.writerow({k: record.get(k, "") for k in fieldnames})

                processed = batch_start + i + 1
                avg_sec = (time.perf_counter() - run_started) / max(processed - completed, 1)
                remaining = len(df) - processed
                append_log(
                    f"[{processed}/{len(df)}] year={row.get('source_year', '')} case={row.get('案号', '')} "
                    f"batch_elapsed={batch_elapsed:.3f}s per_item={record['inference_sec']:.3f}s "
                    f"parse_ok={record['parse_ok']} boundary_candidate={record.get('boundary_expansion_candidate', '')} "
                    f"remaining={remaining} eta_sec={round(avg_sec * remaining, 2)}"
                )
                if processed % args.progress_every == 0 or processed == len(df):
                    write_progress(
                        progress_path,
                        {
                            "status": "running" if processed < len(df) else "completed",
                            "input_csv": args.input_csv,
                            "output_csv": str(out_csv),
                            "summary_json": str(summary_path),
                            "log_path": str(log_path),
                            "total_selected": len(df),
                            "completed": processed,
                            "remaining": remaining,
                            "last_case_no": row.get("案号", ""),
                            "avg_sec_this_run": round(avg_sec, 4),
                            "eta_sec": round(avg_sec * remaining, 2),
                            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                    )

    summary = summarize_results(out_csv, args, per_year_counts)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress(
        progress_path,
        {
            "status": "completed",
            "input_csv": args.input_csv,
            "output_csv": str(out_csv),
            "summary_json": str(summary_path),
            "log_path": str(log_path),
            "total_selected": len(df),
            "completed": len(df),
            "remaining": 0,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
        },
    )
    append_log(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
