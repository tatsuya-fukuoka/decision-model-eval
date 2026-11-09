"""多数派クラスとランダムのベースライン（指示書 §4-2、§5-1）。

どちらも評価データの正解だけから作る記述的な基準で、学習したモデルではない
（多数派は評価セット自身の事前分布を使うので、実モデルより有利な「硬い下限」）。

- baseline-majority：
  S1MB …選択肢 ID が全件で同じタスクは ID ごとの平均正解確率（評価セットの事前分布）、
         ID が件ごとに異なるタスクは位置ごとの平均正解確率を各件の選択肢数で再正規化。
  bench_ja … sokudan の MajorityClassBaseline と同じ（評価セットの経験的事前分布）。
- baseline-random：
  S1MB …一様分布。
  bench_ja … sokudan の RandomBaseline と同じ（一様 + 1e-6 のゆらぎ、seed 20260920）。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data.types import Target
from .results import RawRow

SOKUDAN_RANDOM_SEED = 20260920
JITTER = 1e-6


def majority_rows(targets: list[Target]) -> list[RawRow]:
    fixed = len({t.option_ids for t in targets}) == 1
    if fixed:
        mean = np.mean([t.probs for t in targets], axis=0)
        return [RawRow(t.case_id, t.question_id, list(t.option_ids), list(mean / mean.sum())) for t in targets]
    by_pos = defaultdict(list)
    for t in targets:
        for i, p in enumerate(t.probs):
            by_pos[i].append(p)
    pos_mean = {i: float(np.mean(v)) for i, v in by_pos.items()}
    rows = []
    for t in targets:
        v = np.array([pos_mean[i] for i in range(len(t.option_ids))]) + 1e-12
        rows.append(RawRow(t.case_id, t.question_id, list(t.option_ids), list(v / v.sum())))
    return rows


def uniform_rows(targets: list[Target]) -> list[RawRow]:
    return [RawRow(t.case_id, t.question_id, list(t.option_ids), [1 / len(t.option_ids)] * len(t.option_ids))
            for t in targets]


def sokudan_random_rows(task_targets: dict[str, list[Target]]) -> dict[str, list[RawRow]]:
    """sokudan RandomBaseline と同じ乱数の消費順（bool → choice → score）で作る。"""
    rng = np.random.default_rng(SOKUDAN_RANDOM_SEED)
    out = {}
    for task in ("bench_ja-churn", "bench_ja-department", "bench_ja-urgency"):
        ts = task_targets[task]
        k = len(ts[0].option_ids)
        p = np.full((len(ts), k), 1.0 / k) + JITTER * rng.random((len(ts), k))
        p = p / p.sum(axis=1, keepdims=True)
        out[task] = [RawRow(t.case_id, t.question_id, list(t.option_ids), list(row)) for t, row in zip(ts, p, strict=True)]
    return out
