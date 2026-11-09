import numpy as np

from src import kernel_parity as kp


def _p(probs: dict) -> dict:
    return {"probs": probs, "error": None, "abstention": None}


def test_pair_stats_identical():
    a = {("c1", "q"): _p({"x": 0.7, "y": 0.3}), ("c2", "q"): _p({"x": 0.2, "y": 0.8})}
    st = kp._pair_stats(a, dict(a))
    assert st["s1"] == 0 and st["s2"] == 0 and st["same_failures"]


def test_pair_stats_disagreement_and_failures():
    a = {("c1", "q"): _p({"x": 0.6, "y": 0.4}), ("c2", "q"): _p({"x": 0.5, "y": 0.5}),
         ("c3", "q"): {"probs": None, "error": "E", "abstention": None}}
    b = {("c1", "q"): _p({"x": 0.4, "y": 0.6}), ("c2", "q"): _p({"x": 0.5, "y": 0.5}),
         ("c3", "q"): _p({"x": 1.0, "y": 0.0})}
    st = kp._pair_stats(a, b)
    assert st["n_both"] == 2 and st["argmax_disagree"] == 1 and st["s1"] == 0.5
    assert np.isclose(st["max_abs_diff"], 0.2)
    assert not st["same_failures"]


def test_as_user_clears_environment():
    cmd = kp.as_user(["python", "-c", "1"], {"A": "1"})
    assert cmd[0] == "setpriv" and "--no-new-privs" in cmd and "--bounding-set=-all" in cmd
    i = cmd.index("env")
    assert cmd[i + 1] == "-i" and cmd[i + 2] == "A=1" and cmd[-3:] == ["python", "-c", "1"]


def test_thresholds_match_criteria_doc():
    doc = (kp.ROOT / "docs" / "kernel_parity_criteria.md").read_text(encoding="utf-8")
    assert "s1 ≤ 0.005" in doc and "s2 ≤ 0.01" in doc and "s3 ≤ 0.005" in doc and "2 倍以下" in doc
    assert kp.THRESHOLDS == {"s1": 0.005, "s2": 0.01, "s3": 0.005} and kp.C4_FACTOR == 2.0
