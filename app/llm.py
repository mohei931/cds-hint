"""可选的 LLM 解释层.

默认关闭. 设置 CDS_LLM_BASE_URL / CDS_LLM_MODEL / CDS_LLM_API_KEY 后,
把引擎的结构化结果交给模型生成一段自然语言解释, 让提示更易读.

约束: 模型只能复述结构化结果, 不得新增诊断、不得给出用药剂量或治疗处方.
任何 LLM 失败都会静默降级为规则模板, 不影响主流程.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "你是一名临床辅助提示的解释器。只能依据给定的结构化结果做解释与归纳，"
    "禁止新增诊断、禁止给出药物剂量或具体治疗方案、禁止编造检查结果。"
    "输出 120 字以内的中文要点，先说明需要优先排除的危重情况，再说明需补充的信息。"
    "结尾必须保留：仅供思路提示，不替代医师判断。"
)


def enabled() -> bool:
    return bool(os.environ.get("CDS_LLM_BASE_URL") and os.environ.get("CDS_LLM_MODEL"))


def explain(assessment: dict, timeout: float = 20.0) -> str | None:
    if not enabled():
        return None
    base = os.environ["CDS_LLM_BASE_URL"].rstrip("/")
    payload = {
        "model": os.environ["CDS_LLM_MODEL"],
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "matched_presentation": assessment.get("matched_presentation"),
                        "red_flags": [f["label"] for f in assessment.get("red_flags", [])],
                        "differentials": [
                            {
                                "name": d["name"],
                                "match_level": d["match_level"],
                                "supporting_findings": d["supporting_findings"],
                            }
                            for d in assessment.get("differentials", [])
                        ],
                        "next_questions": assessment.get("next_questions", []),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('CDS_LLM_API_KEY', '')}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError, TimeoutError):
        return None
