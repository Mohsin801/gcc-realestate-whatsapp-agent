"""scripts/calibrate_kb.py: the threshold judging logic, and the labelled questions stay in sync with the facts."""
import json
from pathlib import Path

from core.config import Settings
from scripts.calibrate_kb import CALIBRATION_FILE, judge, recommend, sweep
from tools.rag import load_facts

FACTS_DIR = Path(__file__).resolve().parents[1] / "knowledge_base" / "facts"

POS = {"q": "DLD fee?", "expect": ["dld"]}
NEG = {"q": "weather?", "expect": []}
RELATED = {"q": "golden visa duration?", "expect": [], "ok": ["golden"]}


def test_judge_positive_answer_wrong_defer():
    assert judge(POS, {"dld": 0.85, "other": 0.70}, 0.8, top_k=3) == "ANSWER"
    assert judge(POS, {"dld": 0.75, "other": 0.82}, 0.8, top_k=3) == "WRONG"  # only the wrong fact passes
    assert judge(POS, {"dld": 0.75}, 0.8, top_k=3) == "DEFER"
    # the expected fact passes but is pushed out of the top k by better-scoring wrong ones
    assert judge(POS, {"a": 0.9, "b": 0.89, "dld": 0.81}, 0.8, top_k=2) == "WRONG"


def test_judge_negative_false_unless_only_ok_facts_pass():
    assert judge(NEG, {"freehold": 0.61}, 0.6, top_k=3) == "FALSE"
    assert judge(NEG, {"freehold": 0.61}, 0.79, top_k=3) == "DEFER"
    assert judge(RELATED, {"golden": 0.75}, 0.6, top_k=3) == "DEFER"  # a fair partial answer
    assert judge(RELATED, {"golden": 0.75, "freehold": 0.7}, 0.6, top_k=3) == "FALSE"


def test_sweep_and_recommend_lowest_safe_threshold():
    items = [POS, NEG]
    scores = [{"dld": 0.85}, {"freehold": 0.77}]
    outcomes = sweep(items, scores, [0.70, 0.78, 0.90], top_k=3)
    assert [(o.threshold, o.answered, o.false, o.safe) for o in outcomes] == [
        (0.70, 1, 1, False), (0.78, 1, 0, True), (0.90, 0, 0, True)]
    assert recommend(outcomes).threshold == 0.78
    assert recommend(outcomes[:1]) is None


def test_calibration_questions_reference_existing_facts():
    items = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    ids = {f.id for f in load_facts(FACTS_DIR)}
    for item in items:
        unknown = set(item["expect"]) | set(item.get("ok") or [])
        assert unknown <= ids, f"{item['q']!r} names unknown facts: {unknown - ids}"
        assert item["lang"] in ("en", "ar") and item.get("country") in (None, "DUBAI", "SAUDI", "QATAR", "KUWAIT")
    positives = [i for i in items if i["expect"]]
    assert {d for i in positives for d in i["expect"]} == ids  # every fact has at least one question
    assert {i["lang"] for i in positives} == {"en", "ar"} and len(items) - len(positives) >= 10


def test_default_threshold_is_the_calibrated_one():
    assert Settings(_env_file=None).rag_score_threshold == 0.79
