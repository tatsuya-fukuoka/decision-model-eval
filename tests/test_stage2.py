import numpy as np

from src.metrics import core
from src.stage2_calibration import ece_from, max_coverage, row_stats


def _sample(seed=0, n=400):
    rng = np.random.default_rng(seed)
    probs, targets = [], []
    for i in range(n):
        k = 2 + i % 4
        p = rng.dirichlet(np.ones(k) * 0.5)
        if i % 7 == 0:  # 同値の信頼度・ビン境界ちょうどを混ぜる
            p = np.full(k, 1.0 / k)
        t = np.zeros(k)
        t[rng.integers(k)] = 1.0
        if i % 5 == 0:
            t = rng.dirichlet(np.ones(k))  # ソフトターゲット
        probs.append(p)
        targets.append(t)
    return probs, targets


def test_row_stats_match_core():
    p, t = _sample()
    st = row_stats(p, t)
    assert np.isclose(st["correct"].mean(), core.accuracy(p, t))
    assert np.isclose(st["nll"].mean(), core.nll(p, t))
    assert np.isclose(st["brier"].mean(), core.brier(p, t))
    assert np.isclose(ece_from(st["confidence"], st["correct"]), core.ece(p, t))


def test_max_coverage_matches_core():
    p, t = _sample(1)
    st = row_stats(p, t)
    for target in (0.3, 0.5, 0.9, 0.95, 1.0):
        assert np.isclose(max_coverage(st["confidence"], st["correct"], target), core.max_coverage_at(p, t, target))
