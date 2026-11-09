import numpy as np
import pytest

from src.calibration.temperature import apply_temperature, fit_temperature, split_groups


def test_split_does_not_cross_groups_and_is_deterministic():
    groups = [f"g{i // 3}" for i in range(300)]  # 100 groups x 3 rows
    split = split_groups(groups, "taskA", seed=1)
    assert set(split) == set(groups)
    assert set(split.values()) == {"calibration", "test"}
    assert sum(v == "calibration" for v in split.values()) == 30
    # 行に割り当てたとき、同じ group の行は必ず同じ側
    assignment = [split[g] for g in groups]
    for g in set(groups):
        sides = {a for gg, a in zip(groups, assignment) if gg == g}
        assert len(sides) == 1
    assert split == split_groups(list(reversed(groups)), "taskA", seed=1)
    assert split != split_groups(groups, "taskA", seed=2)


def test_split_small_task_has_both_sides():
    split = split_groups(["a", "b"], "t", seed=0)
    assert sorted(split.values()) == ["calibration", "test"]


def test_apply_temperature():
    p = [0.7, 0.2, 0.1]
    assert np.allclose(apply_temperature(p, 1.0), p)
    assert apply_temperature(p, 1e3).max() < 0.34
    assert apply_temperature(p, 0.1).max() > 0.99


def test_fit_recovers_overconfidence():
    rng = np.random.default_rng(0)
    rows, targets = [], []
    for _ in range(2000):
        logits = rng.normal(size=3) * 2
        true_p = np.exp(logits) / np.exp(logits).sum()
        y = rng.choice(3, p=true_p)
        # 過信したモデル：真の確率を T=0.5 で尖らせたもの
        rows.append(apply_temperature(true_p, 0.5))
        t = np.zeros(3)
        t[y] = 1
        targets.append(t)
    fit = fit_temperature(rows, targets)
    assert fit["temperature"] == pytest.approx(2.0, rel=0.15)
    assert fit["nll_after"] < fit["nll_before"]
