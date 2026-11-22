"""CDS-Hint 核心推理引擎.

纯标准库实现, 不依赖任何第三方包, 便于在离线环境做可重复的单元测试.

输入: 结构化的病例主诉 (自由文本 + 症状 + 生命体征 + 年龄性别)
输出: 匹配到的临床情景、鉴别诊断提示排序、红旗征、建议检查、需补充的问诊信息
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "presentations"
KB_VERSION = "0.1.0"

DISCLAIMER = (
    "本结果为规则化知识库生成的思路提示，不是诊断结论，也不构成处方、治疗方案或替代医师判断。"
    "所有临床决策必须由具备资质的医师结合完整病史、查体与检查结果作出；"
    "出现红旗征时请按院内急诊/急救流程处置。"
)

BANDS = [
    (5.0, "匹配度较高"),
    (2.5, "部分匹配"),
    (0.0, "匹配度低，仅作排除思路"),
]

GENERIC_QUESTIONS = [
    "症状的起病时间、诱因与演变过程？",
    "是否伴随发热、出汗、恶心呕吐、意识改变？",
    "既往病史、长期用药与近期手术/外伤史？",
    "生命体征（体温、脉搏、呼吸、血压、血氧）当前数值？",
]


def normalize(text: str | None) -> str:
    """全角转半角、转小写, 并去掉空白与常见标点, 便于做子串匹配."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\s，。；、,.!?！？:：;'\"“”‘’()（）\[\]【】/\\-]+", "", text)


def _hit(terms: Iterable[str], text: str) -> list[str]:
    return [t for t in terms if normalize(t) and normalize(t) in text]


def _item_matched(item: dict, text: str) -> bool:
    """条目命中: 主词或 synonyms(any) 任一命中即可."""
    candidates = [item.get("term")] + list(item.get("any") or [])
    return any(normalize(t) and normalize(t) in text for t in candidates)


def _band(score: float) -> str:
    """按证据权重总分分档, 避免不同疾病条目数量差异导致的百分比失真."""
    for threshold, label in BANDS:
        if score >= threshold:
            return label
    return BANDS[-1][1]


def _vital_triggered(rules: list[dict], vitals: dict) -> list[dict]:
    """检查生命体征规则, 返回命中的规则(附原因)."""
    hits = []
    for rule in rules or []:
        key, op, value = rule.get("key"), rule.get("op"), rule.get("value")
        current = vitals.get(key)
        if current is None or value is None:
            continue
        try:
            current = float(current)
            value = float(value)
        except (TypeError, ValueError):
            continue
        matched = {
            "<": current < value,
            "<=": current <= value,
            ">": current > value,
            ">=": current >= value,
        }.get(str(op), False)
        if matched:
            hits.append(
                {
                    "reason": rule.get("reason") or f"{key} {op} {value}",
                    "vital": key,
                    "value": current,
                }
            )
    return hits


def _modifier_hits(modifiers: list[dict], case: dict) -> list[dict]:
    hits = []
    age, sex = case.get("age"), case.get("sex")
    for mod in modifiers or []:
        ok = True
        reason = mod.get("reason") or "风险修饰因素"
        if "age_min" in mod:
            ok = ok and age is not None and int(age) >= int(mod["age_min"])
        if "age_max" in mod:
            ok = ok and age is not None and int(age) <= int(mod["age_max"])
        if "sex" in mod:
            ok = ok and sex == mod["sex"]
        if ok:
            hits.append({"reason": reason, "weight": float(mod.get("weight", 0.5))})
    return hits


def load_kb(path: str | Path | None = None) -> dict:
    """加载知识库: 支持单个 JSON 文件或按情景拆分的目录(便于分病种维护)."""
    target = Path(path) if path else KB_DIR
    if target.is_dir():
        presentations = []
        for f in sorted(target.glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                presentations.append(json.load(fh))
        return {"version": KB_VERSION, "presentations": presentations}
    with open(target, encoding="utf-8") as fh:
        return json.load(fh)


def _case_text(case: dict) -> str:
    parts = [case.get("complaint") or ""]
    parts.extend(case.get("symptoms") or [])
    history = case.get("history") or []
    if isinstance(history, str):
        history = [history]
    parts.extend(history)
    return normalize(" ".join(str(p) for p in parts))


def _select_presentation(kb: dict, text: str) -> tuple[dict | None, float]:
    best, best_score = None, 0.0
    for pres in kb.get("presentations", []):
        score = 0.0
        for entry in pres.get("entry_terms", []):
            if normalize(entry["term"]) in text:
                score += float(entry.get("weight", 1))
        if score > best_score:
            best, best_score = pres, score
    return best, best_score


def _score_differential(diff: dict, text: str, case: dict) -> dict:
    support_hits = [i for i in diff.get("support", []) if _item_matched(i, text)]
    against_hits = [i for i in diff.get("against", []) if _item_matched(i, text)]
    modifier_hits = _modifier_hits(diff.get("modifiers", []), case)

    raw = sum(float(i.get("weight", 1)) for i in support_hits)
    raw += sum(m["weight"] for m in modifier_hits)
    raw -= sum(float(i.get("weight", 1)) for i in against_hits)
    raw = max(raw, 0.0)

    max_score = sum(float(i.get("weight", 1)) for i in diff.get("support", []))
    max_score += sum(float(m.get("weight", 0.5)) for m in diff.get("modifiers", []))
    ratio = raw / max_score if max_score else 0.0

    missing = [
        i["term"] for i in diff.get("support", []) if not _item_matched(i, text)
    ]

    return {
        "name": diff["name"],
        "critical": bool(diff.get("critical")),
        "score": round(raw, 2),
        "match_ratio": round(ratio, 2),  # 该疾病典型特征覆盖度, 仅供参考
        "match_level": _band(raw),
        "supporting_findings": [i["term"] for i in support_hits]
        + [m["reason"] for m in modifier_hits],
        "against_findings": [i["term"] for i in against_hits],
        "missing_findings": missing[:3],
        "recommended_exams": list(diff.get("exams", [])),
        "caveat": diff.get("caveat", ""),
    }


def assess(case: dict, kb: dict | None = None, max_differentials: int = 3) -> dict:
    """对一例结构化病例给出鉴别诊断提示.

    case 字段: complaint(str) / symptoms(list[str]) / history(list[str]|str) /
               age(int|None) / sex("male"|"female"|None) / vitals(dict)
    """
    kb = kb or load_kb()
    text = _case_text(case)
    vitals = case.get("vitals") or {}

    pres, pres_score = _select_presentation(kb, text)
    if pres is None or pres_score == 0:
        return {
            "status": "no_match",
            "matched_presentation": None,
            "message": "未匹配到内置的临床情景（胸痛、腹痛、头痛、发热、呼吸困难）。可补充主诉关键词，或按通用问诊清单继续采集。",
            "differentials": [],
            "red_flags": [],
            "next_questions": GENERIC_QUESTIONS,
            "disclaimer": DISCLAIMER,
            "kb_version": kb.get("version"),
        }

    red_flags = []
    for flag in pres.get("red_flags", []):
        term_hits = _hit(flag.get("terms", []), text)
        vital_hits = _vital_triggered(flag.get("vitals", []), vitals)
        # min_hits 用于降低单一非特异线索导致的误报
        if len(term_hits) + len(vital_hits) >= int(flag.get("min_hits", 1)):
            red_flags.append(
                {
                    "label": flag["label"],
                    "urgency": flag.get("urgency", "尽快"),
                    "evidence": term_hits + [h["reason"] for h in vital_hits],
                    "action": flag.get("action", ""),
                    "exams": list(flag.get("exams", [])),
                }
            )

    scored = [_score_differential(d, text, case) for d in pres.get("differentials", [])]
    scored.sort(key=lambda x: (x["score"], x["critical"]), reverse=True)
    top = [d for d in scored if d["score"] > 0][:max_differentials]

    # 红旗命中且对应的危重鉴别未进入前三时, 强制补位, 避免漏项
    if red_flags:
        top_names = {d["name"] for d in top}
        for d in scored:
            if d["critical"] and d["name"] not in top_names and any(
                k in d["name"] for k in _flag_keywords(red_flags)
            ):
                top.append(d)
                break

    next_questions = []
    for d in top:
        if d["match_ratio"] < 0.25:
            continue
        for item in d["missing_findings"]:
            q = f"是否{_as_question(item)}？"
            if q not in next_questions:
                next_questions.append(q)
    if not next_questions:
        next_questions = GENERIC_QUESTIONS[:2]

    return {
        "status": "ok",
        "matched_presentation": {
            "id": pres["id"],
            "name": pres["name"],
            "entry_score": round(pres_score, 2),
        },
        "red_flags": red_flags,
        "differentials": top,
        "next_questions": next_questions[:5],
        "recommended_exams": _aggregate_exams(red_flags, top),
        "disclaimer": DISCLAIMER,
        "kb_version": kb.get("version"),
    }


def _as_question(finding: str) -> str:
    """把体征/病史条目转成问诊句式, 避免机械拼接."""
    if finding.endswith(("史", "炎", "病")):
        return f"有{finding}"
    return f"存在{finding}"


def _flag_keywords(red_flags: list[dict]) -> list[str]:
    mapping = {
        "冠脉": ["冠脉", "心肌缺血"],
        "主动脉夹层": ["主动脉"],
        "肺栓塞": ["肺栓塞"],
        "气胸": ["气胸"],
        "穿孔": ["穿孔", "腹膜炎"],
        "消化道出血": ["出血"],
        "肠系膜": ["肠系膜"],
        "异位妊娠": ["妇科", "妊娠"],
        "胰腺炎": ["胰腺"],
        "蛛网膜下腔": ["蛛网膜"],
        "颅内感染": ["颅内感染", "脑膜炎"],
        "颅内占位": ["占位", "颅高压"],
        "血压": ["高血压"],
        "脓毒症": ["脓毒症", "感染"],
        "中枢神经系统": ["中枢", "颅内感染"],
        "呼吸衰竭": ["呼吸衰竭", "心力衰竭", "肺栓塞", "气胸"],
        "心力衰竭": ["心力衰竭"],
    }
    out: list[str] = []
    for flag in red_flags:
        for key, kws in mapping.items():
            if key in flag["label"]:
                out.extend(kws)
    return out


def _aggregate_exams(red_flags: list[dict], top: list[dict]) -> list[dict]:
    urgent, routine = [], []
    for flag in red_flags:
        for exam in flag["exams"]:
            if exam not in urgent:
                urgent.append(exam)
    # 只汇总达到“部分匹配”以上的鉴别, 避免低相关鉴别把检查清单撑得过长
    for d in [x for x in top if x["score"] >= 2.5]:
        for exam in d["recommended_exams"]:
            if exam not in urgent and exam not in routine:
                routine.append(exam)
    groups = []
    if urgent:
        groups.append({"level": "优先完成", "items": urgent})
    if routine:
        groups.append({"level": "按需完善", "items": routine})
    return groups
