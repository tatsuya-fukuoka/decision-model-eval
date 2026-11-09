import numpy as np
import pytest

from src.metrics import core as m


def onehot(k, i):
    v = [0.0] * k
    v[i] = 1.0
    return v


def test_perfect_prediction():
    T = [onehot(3, 0), onehot(3, 2), onehot(3, 1)]
    assert m.accuracy(T, T) == 1.0
    assert m.brier(T, T) == 0.0
    assert m.nll(T, T) == pytest.approx(0.0, abs=1e-9)
    assert m.ece(T, T) == 0.0
    assert m.rps(T, T) == 0.0
    assert m.macro_f1(T, T) == 1.0


def test_uniform_prediction():
    k = 4
    P = [[0.25] * k] * 8
    T = [onehot(k, i % k) for i in range(8)]
    assert m.nll(P, T) == pytest.approx(np.log(4))
    assert m.brier(P, T) == pytest.approx(0.75)  # 3*(1/16) + (3/4)^2
    # 同点は先頭列 -> 予測は常にクラス0、正解率 1/4
    assert m.accuracy(P, T) == pytest.approx(0.25)
    # 信頼度 0.25 のビン1つに全件、正解率 0.25 -> ECE 0
    assert m.ece(P, T) == pytest.approx(0.0)


def test_known_ece():
    # 信頼度 0.9 で 2件中1件正解 -> |0.5-0.9| = 0.4、信頼度 0.6 で1件正解 -> |1-0.6|=0.4
    P = [[0.9, 0.1], [0.9, 0.1], [0.6, 0.4]]
    T = [onehot(2, 0), onehot(2, 1), onehot(2, 0)]
    assert m.ece(P, T) == pytest.approx((2 / 3) * 0.4 + (1 / 3) * 0.4)


def test_soft_target_accuracy_is_mass_at_argmax():
    P = [[0.7, 0.3]]
    T = [[0.6, 0.4]]
    assert m.accuracy(P, T) == pytest.approx(0.6)


def test_rps_known():
    # 3段階、予測 [0.5,0.5,0]、正解 2 -> CDF差 (0.5-0)^2 + (1-0)^2 + 0 = 1.25 / 2
    assert m.rps([[0.5, 0.5, 0.0]], [onehot(3, 2)]) == pytest.approx(0.625)


def test_score_mae_normalized():
    # 値 [1,3,5]、予測期待値 3、正解 5 -> |3-5|/4 = 0.5
    assert m.score_mae([[0.0, 1.0, 0.0]], [[0, 0, 1.0]], [[1, 3, 5]]) == pytest.approx(0.5)


def test_auroc():
    assert m.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert m.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert m.auroc([0.9, 0.1], [0, 1]) == 0.0
    assert m.auroc([0.1, 0.2], [1, 1]) is None


def test_macro_f1_undefined_for_variable_k():
    assert m.macro_f1([[0.5, 0.5], [0.2, 0.3, 0.5]], [[1, 0], [0, 0, 1]]) is None


def test_flip_rate():
    assert m.flip_rate(["a", "b"], [["a", "a"], ["b", "c"]]) == pytest.approx(0.25)


def test_coverage():
    P = [[0.99, 0.01], [0.9, 0.1], [0.6, 0.4], [0.55, 0.45]]
    T = [onehot(2, 0), onehot(2, 0), onehot(2, 1), onehot(2, 0)]
    curve = m.coverage_curve(P, T)
    assert list(curve["coverage"]) == [0.25, 0.5, 0.75, 1.0]
    assert curve["accuracy"][1] == 1.0
    assert m.max_coverage_at(P, T, 0.95) == 0.5
    assert m.max_coverage_at(P, T, 0.75) == 1.0


def test_coverage_ties_are_not_split():
    P = [[0.9, 0.1], [0.9, 0.1]]
    T = [onehot(2, 0), onehot(2, 1)]
    assert m.max_coverage_at(P, T, 0.95) == 0.0


def test_invalid_probabilities_raise():
    with pytest.raises(ValueError):
        m.brier([[0.5, 0.6]], [[1, 0]])


def test_bootstrap_ci_contains_point():
    vals = np.array([0, 1] * 50, dtype=float)
    point, lo, hi = m.bootstrap_ci([list(range(100))], lambda idx: float(vals[idx].mean()), n_boot=200, seed=1)
    assert point == 0.5 and lo < 0.5 < hi


def test_bench_ja_vectorized_matches_core():
    from src.metrics import bench_ja_scores as bj

    rng = np.random.default_rng(3)
    for k, qtype in ((4, "choice"), (3, "score"), (2, "noul")):
        P = rng.dirichlet(np.full(k, 0.7), size=200)
        P[:5] = 1 / k  # 同点と ECE のビン境界
        P[5] = np.eye(k)[0]
        y = rng.integers(0, k, 200)
        T = np.eye(k)[y]
        got = bj.summarize(P, y)
        assert got["accuracy"] == pytest.approx(m.accuracy(P, T))
        assert got["macro_f1"] == pytest.approx(m.macro_f1(P, T))
        assert got["ece"] == pytest.approx(m.ece(P, T))
        assert got["brier"] == pytest.approx(m.brier(P, T))
        assert got["nll"] == pytest.approx(m.nll(P, T))
        r = bj.score_task(qtype, P, y)
        if qtype == "score":
            assert r["rps"] == pytest.approx(m.rps(bj.floor_and_renormalise(P), T))
        if qtype == "noul":
            assert r["auroc"] == pytest.approx(m.auroc(P[:, 1], y))
