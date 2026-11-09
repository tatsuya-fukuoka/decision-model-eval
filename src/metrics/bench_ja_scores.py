"""bench_ja の指標（指示書 §4-2）。

sokudan @ eccded37 の scripts/run_baseline_ja.py → sokudan/eval/report.py::score_baseline と同じ定義：
- 全モデルに同じ確率下限 5e-5 を適用して再正規化してから採点する（PROB_FLOOR、floor_and_renormalise）
- Choice（部署）：accuracy, macro_f1, ECE(15), Brier, NLL
- Score（緊急度）：上記 + RPS（K-1 で正規化）, MAE（argmax / 期待値）
- Bool（解約示唆）：上記（2クラス）+ AUROC（下限適用前の P(true) で計算）
sokudan のコードは実行せず、式を src/metrics/core.py で実装している（tests で既知例を検証）。
"""

from __future__ import annotations

import numpy as np

from . import core

PROB_FLOOR = 5e-5


def floor_and_renormalise(probs, floor: float = PROB_FLOOR) -> np.ndarray:
    out = np.clip(np.asarray(probs, dtype=np.float64), floor, None)
    return out / out.sum(axis=1, keepdims=True)


def summarize(probs, labels: np.ndarray) -> dict:
    """固定 K・ハードラベル用のベクトル化版。core の行ごと実装との一致は tests/test_metrics.py で確認。"""
    P = np.asarray(probs, dtype=np.float64)
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probabilities must sum to 1")
    n, k = P.shape
    y = np.asarray(labels, dtype=np.int64)
    T = np.eye(k)[y]
    pred = P.argmax(axis=1)
    corr = (pred == y).astype(np.float64)
    conf = P.max(axis=1)
    # ECE：core.reliability_bins と同じビン（(lo, hi]、最初のビンのみ [0, hi]）
    edges = np.linspace(0.0, 1.0, core.ECE_BINS + 1)
    b = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, core.ECE_BINS - 1)
    sc = np.bincount(b, weights=conf, minlength=core.ECE_BINS)
    sa = np.bincount(b, weights=corr, minlength=core.ECE_BINS)
    ece = float(np.abs(sa - sc).sum() / n)
    f1s = []
    for c in range(k):
        g = y == c
        if not g.any():
            continue
        pc = pred == c
        tp = float((pc & g).sum())
        f1s.append(2 * tp / (2 * tp + float((pc & ~g).sum()) + float((~pc & g).sum())))
    return {
        "accuracy": float(corr.mean()),
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "ece": ece,
        "brier": float(((P - T) ** 2).sum(axis=1).mean()),
        "nll": float(-np.log(np.clip(P[np.arange(n), y], core.EPS, 1.0)).mean()),
        "mean_confidence": float(conf.mean()),
        "n": int(n),
    }


def _rps(P: np.ndarray, y: np.ndarray) -> float:
    k = P.shape[1]
    return float(((np.cumsum(P, axis=1) - np.cumsum(np.eye(k)[y], axis=1)) ** 2).sum(axis=1).mean() / (k - 1))


def _auroc(s: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import rankdata

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rankdata(s, method="average")
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def score_task(qtype: str, probs: np.ndarray, labels: np.ndarray) -> dict:
    """probs: (N, K) 作者の選択肢順。labels: 正解インデックス。Noul は列 [false, true]。"""
    raw = np.asarray(probs, dtype=np.float64)
    floored = floor_and_renormalise(raw)
    out = summarize(floored, labels)
    if qtype == "score":
        k = floored.shape[1]
        out["rps"] = _rps(floored, labels)
        levels = np.arange(k, dtype=np.float64)
        out["mae_argmax"] = float(np.abs(floored.argmax(axis=1) - labels).mean())
        out["mae_expectation"] = float(np.abs(floored @ levels - labels).mean())
    if qtype == "noul":
        out["auroc"] = _auroc(raw[:, 1], np.asarray(labels))
        out["mean_p_true"] = float(raw[:, 1].mean())
        out["gold_true_rate"] = float(labels.mean())
    return out
