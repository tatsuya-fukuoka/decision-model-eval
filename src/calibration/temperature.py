"""温度合わせ（指示書 §5-2）。

- 分割：各タスクを group_id 単位で calibration 30% / test 70% に分ける（seed 固定、保存する）。
- 温度：型（choice / noul / score）ごとに 1 つのスカラー T を、
  学習データと重ならない 17 タスクの calibration 側だけで、ソフトターゲットとの NLL 最小化で求める。
- 変換：p^(1/T) を再正規化する（logit を T で割るのと同値）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ..metrics.core import EPS

T_BOUNDS = (0.05, 20.0)


def apply_temperature(probs: Sequence[float], temperature: float) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), EPS, 1.0)
    logp = np.log(p) / temperature
    logp -= logp.max()
    q = np.exp(logp)
    return q / q.sum()


def _nll(rows: list[np.ndarray], targets: list[np.ndarray], temperature: float) -> float:
    total = 0.0
    for p, t in zip(rows, targets, strict=True):
        q = apply_temperature(p, temperature)
        total += float(-(t * np.log(np.clip(q, EPS, 1.0))).sum())
    return total / len(rows)


def fit_temperature(probs: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> dict:
    """log T 上で有界な 1 次元最適化。戻り値に最適化前後の NLL も入れる。"""
    rows = [np.asarray(p, dtype=np.float64) for p in probs]
    tgts = [np.asarray(t, dtype=np.float64) for t in targets]
    if not rows:
        raise ValueError("no calibration rows")
    lo, hi = np.log(T_BOUNDS[0]), np.log(T_BOUNDS[1])
    res = minimize_scalar(lambda x: _nll(rows, tgts, float(np.exp(x))), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    t = float(np.exp(res.x))
    return {
        "temperature": t,
        "nll_before": _nll(rows, tgts, 1.0),
        "nll_after": float(res.fun),
        "n": len(rows),
        "at_bound": bool(np.isclose(t, T_BOUNDS[0], rtol=1e-3) or np.isclose(t, T_BOUNDS[1], rtol=1e-3)),
    }


def split_groups(group_ids: Iterable[str], task_id: str, seed: int, calib_fraction: float = 0.3) -> dict[str, str]:
    """group_id を calibration / test に振り分ける。group をまたがない。

    group ごとに (seed, task_id, group_id) のハッシュで順位を付け、上位 calib_fraction を calibration にする。
    ハッシュなのでデータの並び順や件数の増減に左右されず、同じ入力から常に同じ分割になる。
    """
    unique = sorted(set(group_ids))
    if not unique:
        return {}

    def rank(g: str) -> str:
        return hashlib.sha256(f"{seed}\0{task_id}\0{g}".encode()).hexdigest()

    ordered = sorted(unique, key=rank)
    n_calib = int(round(calib_fraction * len(ordered)))
    if len(ordered) >= 2:
        n_calib = min(max(n_calib, 1), len(ordered) - 1)
    calib = set(ordered[:n_calib])
    return {g: ("calibration" if g in calib else "test") for g in unique}
