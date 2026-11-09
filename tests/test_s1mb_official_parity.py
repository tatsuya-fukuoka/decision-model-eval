"""移植版の採点（src/metrics/s1mb_scores.py）が S1MB 公式評価器と一致することの確認。

results/parity/<model>.official.json は scripts/s1mb_official_score.py が公式コードで計算したもの。
ファイルか評価データが無い環境ではスキップする（README の再現手順を参照）。
"""

import json
import math
from pathlib import Path

import pytest

from src.data import s1mb
from src.metrics.s1mb_scores import aggregate, task_scores
from src.results import path_for, predictions_index, read_rows

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = sorted((ROOT / "results" / "parity").glob("*.official.json"))
SHARED = [
    "accuracy", "balanced_accuracy", "f1", "positive_prevalence", "uniform_accuracy_baseline",
    "fixed_answer_accuracy_baseline", "constant_mae_baseline", "normalized_score_rmse",
    "baseline_adjusted_skill", "baseline_adjusted_score",
]


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


@pytest.fixture(scope="module")
def data():
    if not (s1mb.DEFAULT_ROOT / "categories").is_dir():
        pytest.skip("S1MB data not fetched")
    return {t: s1mb.load_benchmark(t) for t in s1mb.load_category()}


@pytest.mark.parametrize("official_path", OFFICIAL, ids=lambda p: p.name)
def test_matches_official(official_path, data):
    from src.stage1_tables import scopes

    official = json.loads(official_path.read_text(encoding="utf-8"))
    model = official["model"]
    if not path_for(model, "s1mb", next(iter(data))).is_file():
        pytest.skip(f"raw predictions for {model} not present")
    ours, types = {}, {}
    for t, (bench, _, targets) in data.items():
        preds = predictions_index(read_rows(path_for(model, "s1mb", t)))
        r = task_scores(bench.task, targets, preds)
        ours[t], types[t] = r, bench.task
        om = official["per_task"][t]["metrics"]
        assert r["complete"] == official["per_task"][t]["complete"], t
        assert _close(r["primary"], om[s1mb.PRIMARY_METRICS[bench.task]]), (t, "primary")
        for k in SHARED:
            if k in om and k in r:
                assert _close(r[k], om[k]), (t, k, r[k], om[k])
    for scope, tasks in scopes().items():
        agg = aggregate(ours, types, tasks)
        for k, v in official["summary"][scope].items():
            assert _close(agg[k], v), (scope, k, agg[k], v)
