"""命令行演示: 不需要启动 Web 服务即可查看推理结果.

用法:
    python -m app.cli chest_pain_acs
    python -m app.cli --list
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .engine import assess, load_kb

CASES = Path(__file__).resolve().parent.parent / "data" / "sample_cases.json"


def main(argv: list[str]) -> int:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    names = [k for k in cases if not k.startswith("_")]
    if "--list" in argv or not argv:
        print("可用示例病例:\n  " + "\n  ".join(names))
        return 0

    key = argv[0]
    if key not in cases:
        print(f"未找到病例: {key}")
        return 1

    result = assess(cases[key], load_kb())
    print(f"主诉: {cases[key]['complaint']}")
    if result["status"] == "no_match":
        print(result["message"])
        return 0
    print(f"情景: {result['matched_presentation']['name']}\n")

    if result["red_flags"]:
        print("红旗征:")
        for f in result["red_flags"]:
            print(f"  [{f['urgency']}] {f['label']}  依据: {'、'.join(f['evidence'])}")
            print(f"    → {f['action']}")
    else:
        print("红旗征: 未触发\n")

    print("鉴别诊断提示:")
    for d in result["differentials"]:
        tag = "★" if d["critical"] else " "
        print(f" {tag} {d['match_level']}  {d['name']}  (证据分 {d['score']})")
        print(f"    支持: {'、'.join(d['supporting_findings']) or '暂无'}")
        print(f"    待补充: {'、'.join(d['missing_findings']) or '已覆盖'}")

    print("\n建议检查:")
    for g in result["recommended_exams"]:
        print(f"  [{g['level']}] {'、'.join(g['items'])}")
    print("\n需补充问诊:")
    for q in result["next_questions"]:
        print(f"  - {q}")
    print(f"\n{result['disclaimer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
