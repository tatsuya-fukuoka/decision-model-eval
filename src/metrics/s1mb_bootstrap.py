"""S1MB 基準値調整済みスコアの高速ブートストラップ。

task_scores（公式式の移植）を反復ごとに呼ぶと遅いため、件ごとの寄与を配列にしておき、
復元抽出のインデックス行列 (n_boot, n) でまとめて計算する。基準値・有病率・適格性は
データセット全体で固定（S1MB の規則、task_scores の reference と同じ）。
完了していないタスクはブートストラップしない（Task Avg 自体が未定義のため）。

一致は tests/test_bootstrap_fast.py で task_scores と照合している。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.types import Target
from .s1mb_scores import EPSILON, task_scores


@dataclass
class TaskArrays:
    task: str
    eligible: bool
    n: int
    # choice: 正解質量（棄権 0）／score: |誤差|（棄権 1）／noul: 下の4成分
    value: np.ndarray | None = None
    baseline: float | None = None
    tp: np.ndarray | None = None
    fp: np.ndarray | None = None
    fn: np.ndarray | None = None
    tn: np.ndarray | None = None


def task_arrays(task: str, targets: list[Target], predictions: dict[tuple[str, str], dict]) -> TaskArrays:
    full = task_scores(task, targets, predictions)
    if not full["complete"]:
        raise ValueError("bootstrap requires a complete task")
    vals, tp, fp, fn, tn = [], [], [], [], []
    for t in targets:
        pred = predictions[(t.case_id, t.question_id)]
        tgt = dict(zip(t.option_ids, t.probs, strict=True))
        abst = bool(pred.get("abstention"))
        p = pred.get("probs")
        if task == "choice":
            vals.append(0.0 if abst else tgt[max(t.option_ids, key=lambda o: p[o])])
        elif task == "score":
            if abst:
                vals.append(1.0)
            else:
                assert t.values is not None
                e = sum(v * (p[k] - tgt[k]) for k, v in zip(t.option_ids, t.values, strict=True))
                vals.append(abs(e / (max(t.values) - min(t.values))))
        else:
            pos = (not abst) and p["true"] >= 0.5
            neg = (not abst) and p["true"] < 0.5
            tp.append(tgt["true"] if pos else 0.0)
            fp.append(tgt["false"] if (pos or abst) else 0.0)
            fn.append(tgt["true"] if (neg or abst) else 0.0)
            tn.append(tgt["false"] if neg else 0.0)
    arr = TaskArrays(task=task, eligible=full["eligible"], n=len(targets))
    if task == "choice":
        arr.value, arr.baseline = np.asarray(vals), full["fixed_answer_accuracy_baseline"]
    elif task == "score":
        arr.value, arr.baseline = np.asarray(vals), full["constant_mae_baseline"]
    else:
        arr.tp, arr.fp, arr.fn, arr.tn = map(np.asarray, (tp, fp, fn, tn))
    return arr


def adjusted_scores(a: TaskArrays, idx: np.ndarray) -> np.ndarray:
    """idx (n_boot, n) の各行について clip 済み調整スコアを返す。未定義は nan。"""
    if a.task == "choice":
        acc = a.value[idx].mean(axis=1)
        skill = (acc - a.baseline) / (1 - a.baseline) if a.baseline < 1 - EPSILON else np.full(len(idx), np.nan)
    elif a.task == "score":
        mae = a.value[idx].mean(axis=1)
        skill = 1 - mae / a.baseline if a.baseline > EPSILON else np.full(len(idx), np.nan)
    else:
        tp, fp, fn, tn = (x[idx].sum(axis=1) for x in (a.tp, a.fp, a.fn, a.tn))
        with np.errstate(invalid="ignore", divide="ignore"):
            rec = np.where(tp + fn > EPSILON, tp / (tp + fn), np.nan)
            spec = np.where(tn + fp > EPSILON, tn / (tn + fp), np.nan)
        skill = 2 * ((rec + spec) / 2) - 1
    return np.clip(skill, 0.0, 1.0)


def aggregate_boot(
    arrays: dict[str, TaskArrays], scores: dict[str, np.ndarray], tasks: list[str]
) -> dict[str, np.ndarray]:
    """aggregate() と同じ規則（不適格タスクを除外、型の等重み平均）を反復ごとに適用する。"""
    out: dict[str, np.ndarray] = {}
    present = []
    for k in ("noul", "choice", "score"):
        ts = [t for t in tasks if arrays[t].task == k]
        if not ts:
            continue
        present.append(k)
        el = [scores[t] for t in ts if arrays[t].eligible]
        out[k] = np.mean(el, axis=0) if el else np.full(len(next(iter(scores.values()))), np.nan)
    out["task_avg"] = np.mean([out[k] for k in present], axis=0)
    return out
