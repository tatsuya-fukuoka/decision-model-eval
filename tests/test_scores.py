import numpy as np
import pytest

from src.baselines import majority_rows, sokudan_random_rows, uniform_rows
from src.data.types import Target
from src.metrics import bench_ja_scores
from src.metrics.s1mb_scores import aggregate, task_scores
from src.results import predictions_index


def T(task, i, qtype, ids, probs, values=None):
    return Target(task, f"c{i}", "q", qtype, tuple(ids), tuple(probs), values)


def preds(rows):
    return {(f"c{i}", "q"): {"probs": p, "error": None, "abstention": None} for i, p in enumerate(rows)}


def test_choice_adjusted_against_fixed_answer_baseline():
    # 4件・3択、正解は a,a,b,c -> 固定ID 'a' の基準値 0.5
    gold = ["a", "a", "b", "c"]
    targets = [T("t", i, "choice", "abc", [1.0 if o == g else 0.0 for o in "abc"]) for i, g in enumerate(gold)]
    perfect = preds([{o: (1.0 if o == g else 0.0) for o in "abc"} for g in gold])
    r = task_scores("choice", targets, perfect)
    assert r["primary"] == 1.0 and r["fixed_answer_accuracy_baseline"] == 0.5
    assert r["baseline_adjusted_score"] == 1.0
    always_a = preds([{"a": 0.9, "b": 0.05, "c": 0.05}] * 4)
    r = task_scores("choice", targets, always_a)
    assert r["primary"] == 0.5 and r["baseline_adjusted_skill"] == pytest.approx(0.0)


def test_noul_balanced_accuracy_skill():
    # SCORING.md の例：TP=1, FN=10, FP=0, TN=89 -> BA=0.5455, skill=0.0909
    targets, rows = [], []
    spec = [(1, 0.9)] * 1 + [(1, 0.1)] * 10 + [(0, 0.1)] * 89
    for i, (y, p) in enumerate(spec):
        targets.append(T("t", i, "noul", ["false", "true"], [1 - y, y]))
        rows.append({"false": 1 - p, "true": p})
    r = task_scores("noul", targets, preds(rows))
    assert r["accuracy"] == pytest.approx(0.90)
    assert r["balanced_accuracy"] == pytest.approx((1 / 11 + 1) / 2)
    assert r["baseline_adjusted_skill"] == pytest.approx(1 / 11)


def test_score_mae_and_median_baseline():
    vals = (0.0, 1.0, 2.0)
    targets = [T("t", i, "score", "012", np.eye(3)[g], vals) for i, g in enumerate([0, 1, 2])]
    # 定数予測（中央値=0.5）の MAE = (0.5+0+0.5)/3 = 1/3
    r = task_scores("score", targets, preds([{"0": 0, "1": 1.0, "2": 0}] * 3))
    assert r["constant_mae_baseline"] == pytest.approx(1 / 3)
    assert r["primary"] == pytest.approx(1 / 3) and r["baseline_adjusted_skill"] == pytest.approx(0.0)


def test_failures_make_task_incomplete_and_block_task_avg():
    targets = [T("t", i, "choice", "ab", [1, 0]) for i in range(3)]
    p = preds([{"a": 1.0, "b": 0.0}] * 3)
    p[("c1", "q")] = {"probs": None, "error": "OOM", "abstention": None}
    r = task_scores("choice", targets, p)
    assert not r["complete"] and r["n_failed"] == 1
    agg = aggregate({"t": r}, {"t": "choice"}, ["t"])
    assert agg["task_avg"] is None and agg["incomplete_tasks"] == ["t"]


def test_aggregate_equal_type_weights():
    per_task = {
        "n1": {"complete": True, "eligible": True, "baseline_adjusted_score": 0.2},
        "c1": {"complete": True, "eligible": True, "baseline_adjusted_score": 0.4},
        "c2": {"complete": True, "eligible": True, "baseline_adjusted_score": 0.8},
        "s1": {"complete": True, "eligible": True, "baseline_adjusted_score": 0.3},
    }
    types = {"n1": "noul", "c1": "choice", "c2": "choice", "s1": "score"}
    agg = aggregate(per_task, types, list(per_task))
    assert agg["choice"] == pytest.approx(0.6)
    assert agg["task_avg"] == pytest.approx((0.2 + 0.6 + 0.3) / 3)


def test_aggregate_skips_ineligible_but_requires_completion():
    per_task = {
        "c1": {"complete": True, "eligible": True, "baseline_adjusted_score": 0.4},
        "c2": {"complete": True, "eligible": False, "baseline_adjusted_score": None},
    }
    types = {"c1": "choice", "c2": "choice"}
    assert aggregate(per_task, types, ["c1", "c2"])["choice"] == pytest.approx(0.4)
    per_task["c2"]["complete"] = False
    assert aggregate(per_task, types, ["c1", "c2"])["choice"] is None


def test_noul_all_one_class_is_ineligible():
    targets = [T("t", i, "noul", ["false", "true"], [0, 1]) for i in range(3)]
    preds = {(t.case_id, t.question_id): {"probs": {"false": 0.5, "true": 0.5}} for t in targets}
    assert task_scores("noul", targets, preds)["eligible"] is False


def test_reference_keeps_baseline_fixed():
    targets = [T("t", i, "choice", "ab", [1, 0] if i < 3 else [0, 1]) for i in range(4)]
    sub = targets[:2]
    r = task_scores("choice", sub, preds([{"a": 1, "b": 0}] * 2), reference=targets)
    assert r["fixed_answer_accuracy_baseline"] == 0.75


def test_baselines_are_normalized():
    targets = [T("t", i, "choice", "abc", np.eye(3)[i % 3]) for i in range(6)]
    for rows in (majority_rows(targets), uniform_rows(targets)):
        assert all(abs(sum(r.probs) - 1) < 1e-9 for r in rows)
    varying = [T("t", 0, "choice", "ab", [1, 0]), T("t", 1, "choice", "xyz", [0, 1, 0])]
    assert all(abs(sum(r.probs) - 1) < 1e-9 for r in majority_rows(varying))


def test_sokudan_random_is_deterministic():
    tasks = {k: [T(k, i, "choice", ids, np.eye(len(ids))[0]) for i in range(5)]
             for k, ids in [("bench_ja-churn", ["false", "true"]), ("bench_ja-department", list("abcd")),
                            ("bench_ja-urgency", list("012"))]}
    a, b = sokudan_random_rows(tasks), sokudan_random_rows(tasks)
    assert a["bench_ja-department"][0].probs == b["bench_ja-department"][0].probs


def test_bench_ja_scores():
    labels = np.array([0, 1, 1, 0])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [1.0, 0.0]])
    r = bench_ja_scores.score_task("noul", probs, labels)
    assert r["accuracy"] == 0.75 and r["auroc"] == 1.0
    s = bench_ja_scores.score_task("score", np.eye(3)[[0, 1, 2]], np.array([0, 1, 2]))
    assert s["rps"] == pytest.approx(0.0, abs=1e-6) and s["mae_argmax"] == 0.0


def test_predictions_index_rejects_duplicates():
    from src.results import RawRow
    rows = [RawRow("c", "q", ["a", "b"], [1, 0]), RawRow("c", "q", ["a", "b"], [1, 0])]
    with pytest.raises(ValueError):
        predictions_index(rows)
