"""指標関数（指示書 §5-1、§5-2、§5-3、§5-5）。

約束事（report.md にも記載する）:
- 入力は「1行=1判断」の確率行列 P (N, K) とソフトターゲット T (N, K)。
  行ごとに K が異なるタスクは、呼び出し側で同じ K ごとにまとめるか、
  `per_row` 系の関数（リスト入力）を使う。
- ソフトターゲットは丸めない。正解率は「予測（argmax）に載った正解確率質量」の平均
  （S1MB の Choice primary metric と同じ定義。ハードラベルなら通常の正解率に一致）。
- argmax の同点は列の若い順（＝作者が定めた元の選択肢順）で解決する。
- NLL はソフトターゲットとの交差エントロピー -Σ t log p（p は EPS で下限クリップ）。
- Brier は多クラス版 Σ_k (p_k - t_k)^2 の平均。
- ECE はトップラベル、15 等幅ビン（(lo, hi]、最初のビンのみ [0, hi]）、
  「正解」は argmax に載った正解確率質量。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

EPS = 1e-12
ECE_BINS = 15


def _rows(probs: Sequence[Sequence[float]]) -> list[np.ndarray]:
    return [np.asarray(p, dtype=np.float64) for p in probs]


def _check_pair(p: np.ndarray, t: np.ndarray) -> None:
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch {p.shape} vs {t.shape}")
    if p.size == 0:
        raise ValueError("empty row")
    if not np.isclose(p.sum(), 1.0, atol=1e-4):
        raise ValueError(f"probabilities must sum to 1 (got {p.sum()})")
    if not np.isclose(t.sum(), 1.0, atol=1e-6):
        raise ValueError(f"targets must sum to 1 (got {t.sum()})")


def correctness(probs: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> np.ndarray:
    """各判断の「argmax に載った正解確率質量」。ハードラベルなら 0/1。"""
    out = []
    for p, t in zip(_rows(probs), _rows(targets), strict=True):
        _check_pair(p, t)
        out.append(t[int(np.argmax(p))])  # np.argmax は同点で最初の列を返す
    return np.asarray(out)


def confidence(probs: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray([float(np.max(p)) for p in _rows(probs)])


def accuracy(probs, targets) -> float:
    return float(correctness(probs, targets).mean())


def nll(probs, targets) -> float:
    vals = []
    for p, t in zip(_rows(probs), _rows(targets), strict=True):
        _check_pair(p, t)
        vals.append(float(-(t * np.log(np.clip(p, EPS, 1.0))).sum()))
    return float(np.mean(vals))


def brier(probs, targets) -> float:
    vals = []
    for p, t in zip(_rows(probs), _rows(targets), strict=True):
        _check_pair(p, t)
        vals.append(float(((p - t) ** 2).sum()))
    return float(np.mean(vals))


def reliability_bins(probs, targets, n_bins: int = ECE_BINS) -> dict[str, np.ndarray]:
    conf = confidence(probs)
    corr = correctness(probs, targets)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, mean_conf, mean_acc = [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (conf <= hi) & ((conf > lo) if i > 0 else (conf >= lo))
        n = int(mask.sum())
        counts.append(n)
        mean_conf.append(float(conf[mask].mean()) if n else np.nan)
        mean_acc.append(float(corr[mask].mean()) if n else np.nan)
    return {
        "edges": edges,
        "counts": np.asarray(counts),
        "mean_confidence": np.asarray(mean_conf),
        "mean_accuracy": np.asarray(mean_acc),
    }


def ece(probs, targets, n_bins: int = ECE_BINS) -> float:
    bins = reliability_bins(probs, targets, n_bins)
    total = bins["counts"].sum()
    if total == 0:
        raise ValueError("empty input")
    gaps = np.abs(bins["mean_accuracy"] - bins["mean_confidence"])
    weights = bins["counts"] / total
    return float(np.nansum(weights * gaps))


def macro_f1(probs, targets, labels: Sequence[str] | None = None) -> float | None:
    """固定ラベル集合（全行で列の意味が同じ）のときだけ定義する。

    ハードな正解（ターゲットの argmax）に対する one-vs-rest F1 の平均。
    正解に現れないクラスは除外する（sokudan の macro_f1 と同じ規則）。全行の列数が揃わない場合は None。
    """
    P, T = _rows(probs), _rows(targets)
    if not P or len({len(p) for p in P}) != 1:
        return None
    pred = np.asarray([int(np.argmax(p)) for p in P])
    gold = np.asarray([int(np.argmax(t)) for t in T])
    scores = []
    for k in range(len(P[0])):
        tp = float(((pred == k) & (gold == k)).sum())
        fp = float(((pred == k) & (gold != k)).sum())
        fn = float(((pred != k) & (gold == k)).sum())
        if (gold == k).sum() == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else None


def rps(probs, targets) -> float:
    """順序カテゴリの Ranked Probability Score（K-1 で正規化、小さいほど良い）。"""
    vals = []
    for p, t in zip(_rows(probs), _rows(targets), strict=True):
        _check_pair(p, t)
        k = len(p)
        if k < 2:
            raise ValueError("RPS needs at least 2 levels")
        vals.append(float(((np.cumsum(p) - np.cumsum(t)) ** 2).sum() / (k - 1)))
    return float(np.mean(vals))


def score_mae(probs, targets, values: Sequence[Sequence[float]]) -> float:
    """S1MB の Score primary metric：(value-min)/(max-min) に正規化した期待値の絶対誤差。"""
    vals = []
    for p, t, v in zip(_rows(probs), _rows(targets), _rows(values), strict=True):
        _check_pair(p, t)
        span = v.max() - v.min()
        if span <= 0:
            raise ValueError("Score values must not be constant")
        vals.append(abs(float(((p - t) * v).sum())) / span)
    return float(np.mean(vals))


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """平均順位による AUROC（Mann-Whitney U と同じ同点処理）。片方のクラスが無ければ None。"""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def flip_rate(original_argmax: Sequence[str], permuted_argmax: Sequence[Sequence[str]]) -> float:
    """並べ替え後の予測（元 ID で比較）が元の予測と変わった割合。

    original_argmax[i]：元の順序での予測 ID。permuted_argmax[i]：その件の各並べ替えでの予測 ID。
    件ごとに「1回でも変わったか」ではなく、(件, 並べ替え) 単位の変化率を返す。
    """
    changed, total = 0, 0
    for orig, perms in zip(original_argmax, permuted_argmax, strict=True):
        for p in perms:
            total += 1
            changed += int(p != orig)
    if total == 0:
        raise ValueError("no permutations")
    return changed / total


def coverage_curve(probs, targets) -> dict[str, np.ndarray]:
    """信頼度の高い順に採用したときの coverage と、その中の正解率。"""
    conf = confidence(probs)
    corr = correctness(probs, targets)
    order = np.argsort(-conf, kind="mergesort")
    cum = np.cumsum(corr[order])
    n = np.arange(1, len(conf) + 1)
    return {"threshold": conf[order], "coverage": n / len(conf), "accuracy": cum / n}


def max_coverage_at(probs, targets, target_accuracy: float) -> float:
    """正解率 >= target_accuracy を満たす最大 coverage（閾値は信頼度の値で切る）。

    同じ信頼度の判断は分割できないため、同値のかたまりの末尾でのみ判定する。
    満たす点が無ければ 0.0。
    """
    curve = coverage_curve(probs, targets)
    thr = curve["threshold"]
    best = 0.0
    for i in range(len(thr)):
        if i + 1 < len(thr) and thr[i + 1] == thr[i]:
            continue
        if curve["accuracy"][i] >= target_accuracy - 1e-12:
            best = float(curve["coverage"][i])
    return best


def bootstrap_ci(
    clusters: Sequence[Sequence[int]],
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """層化ブートストラップ（パーセンタイル法）。

    clusters：層（例：タスク）ごとの行インデックスのリスト。各層の中で復元抽出し、
    statistic(選ばれた行インデックス) を再計算する。戻り値は (点推定, 下限, 上限)。
    """
    rng = np.random.default_rng(seed)
    full = np.concatenate([np.asarray(c, dtype=np.int64) for c in clusters])
    point = statistic(full)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(np.asarray(c), size=len(c), replace=True) for c in clusters if len(c)]
        )
        stats.append(statistic(idx))
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)
