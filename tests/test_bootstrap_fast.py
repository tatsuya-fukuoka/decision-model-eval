import numpy as np
import pytest

from src.data.types import Target
from src.metrics.s1mb_bootstrap import adjusted_scores, aggregate_boot, task_arrays
from src.metrics.s1mb_scores import aggregate, task_scores


def _case(task, rng, n):
    targets, preds = [], {}
    for i in range(n):
        if task == "noul":
            ids, vals = ["false", "true"], None
        elif task == "choice":
            ids, vals = [f"o{j}" for j in range(rng.integers(2, 5))], None
        else:
            ids, vals = ["a", "b", "c"], [1.0, 2.0, 5.0]
        t = rng.dirichlet(np.ones(len(ids))) if rng.random() < 0.5 else np.eye(len(ids))[rng.integers(len(ids))]
        x = Target("t", f"c{i}", "q", task, tuple(ids), tuple(float(v) for v in t), vals)
        targets.append(x)
        u = rng.random()
        if u < 0.05:
            preds[(x.case_id, "q")] = {"probs": None, "abstention": "refusal"}
        else:
            p = np.full(len(ids), 1 / len(ids)) if u < 0.1 else rng.dirichlet(np.ones(len(ids)))
            preds[(x.case_id, "q")] = {"probs": dict(zip(ids, p, strict=True))}
    return targets, preds


def _slow(task, targets, preds, idx):
    sample = [targets[i] for i in idx]
    rel = [Target(x.task_id, f"{x.case_id}#{j}", x.question_id, x.qtype, x.option_ids, x.probs, x.values)
           for j, x in enumerate(sample)]
    p = {(f"{x.case_id}#{j}", x.question_id): preds[(x.case_id, x.question_id)] for j, x in enumerate(sample)}
    return task_scores(task, rel, p, reference=targets)


@pytest.mark.parametrize("task", ["noul", "choice", "score"])
def test_fast_matches_task_scores(task):
    rng = np.random.default_rng(1)
    targets, preds = _case(task, rng, 60)
    a = task_arrays(task, targets, preds)
    idx = np.vstack([np.arange(60), rng.integers(0, 60, (20, 60))])
    fast = adjusted_scores(a, idx)
    for row, v in zip(idx, fast, strict=True):
        slow = _slow(task, targets, preds, row)["baseline_adjusted_score"]
        assert (np.isnan(v) and slow is None) or v == pytest.approx(slow, abs=1e-12)


def test_aggregate_boot_matches_aggregate():
    rng = np.random.default_rng(2)
    data = {f"{k}{i}": (k, *_case(k, rng, 40)) for k in ("noul", "choice", "score") for i in range(2)}
    arrays = {t: task_arrays(k, ts, ps) for t, (k, ts, ps) in data.items()}
    full = {t: adjusted_scores(a, np.arange(a.n)[None, :]) for t, a in arrays.items()}
    got = aggregate_boot(arrays, full, list(data))
    per_task = {t: task_scores(k, ts, ps) for t, (k, ts, ps) in data.items()}
    want = aggregate(per_task, {t: v[0] for t, v in data.items()}, list(data))
    for k in ("noul", "choice", "score", "task_avg"):
        assert got[k][0] == pytest.approx(want[k], abs=1e-12)
