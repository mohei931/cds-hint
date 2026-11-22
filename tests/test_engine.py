"""引擎单元测试: 只依赖标准库, 可直接 python -m unittest 运行."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engine import DISCLAIMER, assess, load_kb, normalize  # noqa: E402

KB = load_kb()
CASES = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "sample_cases.json").read_text(encoding="utf-8")
)


class NormalizeTest(unittest.TestCase):
    def test_fullwidth_and_english_case(self):
        self.assertEqual(normalize("ＡＣＳ，胸痛"), normalize("acs胸痛"))

    def test_punctuation_removed(self):
        self.assertEqual(normalize("胸痛、出汗；气短"), "胸痛出汗气短")


class ChestPainTest(unittest.TestCase):
    def setUp(self):
        self.result = assess(CASES["chest_pain_acs"], KB)

    def test_matches_chest_pain(self):
        self.assertEqual(self.result["status"], "ok")
        self.assertEqual(self.result["matched_presentation"]["id"], "chest_pain")

    def test_acs_red_flag_triggered_first(self):
        labels = [f["label"] for f in self.result["red_flags"]]
        self.assertIn("疑似急性冠脉综合征（ACS）", labels)

    def test_acs_ranked_first_and_flagged_critical(self):
        top = self.result["differentials"][0]
        self.assertIn("冠脉", top["name"])
        self.assertTrue(top["critical"])
        self.assertIn("12导联心电图", " ".join(top["recommended_exams"]))

    def test_exam_priority_group_first(self):
        self.assertEqual(self.result["recommended_exams"][0]["level"], "优先完成")

    def test_next_questions_bounded(self):
        self.assertLessEqual(len(self.result["next_questions"]), 5)

    def test_disclaimer_present(self):
        self.assertEqual(self.result["disclaimer"], DISCLAIMER)


class AbdominalPainTest(unittest.TestCase):
    def test_appendicitis_ranks_high(self):
        r = assess(CASES["abdominal_pain_appendicitis"], KB)
        self.assertEqual(r["matched_presentation"]["id"], "abdominal_pain")
        self.assertIn("阑尾炎", r["differentials"][0]["name"])
        self.assertGreaterEqual(r["differentials"][0]["match_ratio"], 0.55)

    def test_perforation_red_flag(self):
        r = assess(CASES["abdominal_pain_perforation"], KB)
        labels = [f["label"] for f in r["red_flags"]]
        self.assertTrue(any("穿孔" in l for l in labels))
        self.assertTrue(any("外科" in f["action"] for f in r["red_flags"]))


class HeadacheTest(unittest.TestCase):
    def test_thunderclap_warns_sah(self):
        r = assess(CASES["headache_thunderclap"], KB)
        labels = [f["label"] for f in r["red_flags"]]
        self.assertTrue(any("蛛网膜" in l for l in labels))
        self.assertTrue(any("头颅CT" in e for f in r["red_flags"] for e in f["exams"]))

    def test_migraine_profile(self):
        r = assess(CASES["headache_migraine"], KB)
        self.assertEqual(r["differentials"][0]["name"], "偏头痛")
        self.assertEqual(r["red_flags"], [])


class NegativeCaseTest(unittest.TestCase):
    def test_no_match_returns_generic_questions(self):
        r = assess({"complaint": "左膝关节扭伤后疼痛"}, KB)
        self.assertEqual(r["status"], "no_match")
        self.assertEqual(r["differentials"], [])
        self.assertGreaterEqual(len(r["next_questions"]), 3)
        self.assertIn("不是诊断结论", r["disclaimer"])

    def test_empty_case(self):
        r = assess({}, KB)
        self.assertEqual(r["status"], "no_match")


class ScoringTest(unittest.TestCase):
    def test_against_terms_lower_score(self):
        base = assess(CASES["chest_pain_acs"], KB)["differentials"][0]["score"]
        with_against = assess(
            {
                **CASES["chest_pain_acs"],
                "complaint": CASES["chest_pain_acs"]["complaint"] + "，深呼吸加重，按压痛",
            },
            KB,
        )
        target = [d for d in with_against["differentials"] if "冠脉" in d["name"]]
        self.assertTrue(target)
        self.assertLess(target[0]["score"], base)

    def test_scores_never_negative(self):
        for name, case in CASES.items():
            if name.startswith("_") or not isinstance(case, dict):
                continue
            for d in assess(case, KB)["differentials"]:
                self.assertGreaterEqual(d["score"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
