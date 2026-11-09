"""ステージ2：選択肢順序の入れ替えに対する頑健性（指示書 §5-3）。

    uv run python -m src.stage2_flip [--n-boot 1000]

- 元の順序の予測：results/raw/<model>/{s1mb,bench_ja}/<task>.jsonl（ステージ1の既定設定の結果）。
- 並べ替えの予測：results/raw/<model>/flip/<task>.jsonl（scripts/flip_export.py の順序。1件あたり最大5通り）。
- 予測は元の選択肢 ID で比べる（argmax。同点は作者の順序で先の選択肢）。
- flip rate：(件, 並べ替え) の組のうち、予測が元と変わった割合（core.flip_rate と同じ定義）。
  any_flip_rate：1回でも変わった件の割合。mean_tv：元と並べ替え後の確率分布の全変動距離の平均。
- 元または並べ替え後のどちらかが失敗した組は除外し、n_pairs_excluded に数える（補完しない）。
- 95% 区間：件を単位に復元抽出（その件の並べ替えをまとめて抽出）1,000回、範囲内のタスクで層化。
- 並べ替えの選び方の乱数は1回（seed 固定）。そのばらつきは、件ごとに並べ替えを復元抽出した区間
  （perm_ci_*）で別に示す。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import bench_ja, s1mb
from .results import RAW, path_for, predictions_index, read_rows
from .stage1_tables import SEED, scopes
from .stage2_calibration import write_csv

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def task_pairs(model: str, task: str):
    """件ごとに (変化フラグの配列, TV距離の配列)、除外した組の数、件数。"""
    dataset = "bench_ja" if task.startswith("bench_ja-") else "s1mb"
    fp, op = path_for(model, "flip", task), path_for(model, dataset, task)
    if not fp.is_file() or not op.is_file():
        return None
    orig = predictions_index(read_rows(op))
    perms: dict[tuple[str, str], list] = {}
    for r in read_rows(fp):
        perms.setdefault((r.case_id, r.question_id), []).append(r)
    cases, excluded = [], 0
    for key, rows in perms.items():
        o = orig.get(key)
        if not o or o.get("probs") is None:
            excluded += len(rows)
            continue
        po = np.asarray([o["probs"][i] for i in rows[0].option_ids])
        flips, tvs = [], []
        for r in rows:
            if r.probs is None:
                excluded += 1
                continue
            pp = np.asarray(r.probs)
            flips.append(int(np.argmax(pp)) != int(np.argmax(po)))
            tvs.append(0.5 * float(np.abs(pp - po).sum()))
        if flips:
            cases.append((np.asarray(flips, float), np.asarray(tvs)))
    return cases, excluded, len(perms)


def summarize(per_task: dict[str, tuple], n_boot: int, rng) -> dict:
    cases = [c for t in per_task.values() for c in t[0]]
    if not cases:
        return {"n_cases": 0}
    strata = []
    off = 0
    for t in per_task.values():
        strata.append(np.arange(off, off + len(t[0])))
        off += len(t[0])
    strata = [s for s in strata if len(s)]
    sums = np.array([c[0].sum() for c in cases])
    cnts = np.array([len(c[0]) for c in cases], float)
    anyf = np.array([c[0].max() for c in cases])
    tvs = np.array([c[1].sum() for c in cases])
    out = {"n_cases": len(cases), "n_pairs": int(cnts.sum()), "flip_rate": float(sums.sum() / cnts.sum()),
           "any_flip_rate": float(anyf.mean()), "mean_tv": float(tvs.sum() / cnts.sum())}
    if n_boot:
        fr, af = [], []
        for _ in range(n_boot):
            idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])
            fr.append(sums[idx].sum() / cnts[idx].sum())
            af.append(anyf[idx].mean())
        out["flip_rate_ci_low"], out["flip_rate_ci_high"] = map(float, np.percentile(fr, [2.5, 97.5]))
        out["any_flip_rate_ci_low"], out["any_flip_rate_ci_high"] = map(float, np.percentile(af, [2.5, 97.5]))
        # 並べ替えの選び方によるばらつき：件は固定し、各件の並べ替えを復元抽出
        pr = []
        for _ in range(n_boot):
            s = c = 0.0
            for f, _tv in cases:
                pick = f[rng.integers(0, len(f), len(f))]
                s += pick.sum()
                c += len(pick)
            pr.append(s / c)
        out["perm_ci_low"], out["perm_ci_high"] = map(float, np.percentile(pr, [2.5, 97.5]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    choice_tasks = [t for t in s1mb.load_category() if s1mb.load_benchmark_descriptor(s1mb.DEFAULT_ROOT, t).task == "choice"]
    scope_map = {k: [t for t in v if t in choice_tasks] for k, v in scopes().items()}
    scope_map["bench_ja-department"] = ["bench_ja-department"]
    models = sorted(p.name for p in RAW.iterdir() if (p / "flip").is_dir())
    rows = []
    for model in models:
        rng = np.random.default_rng(SEED)
        per = {}
        for t in [*choice_tasks, "bench_ja-department"]:
            r = task_pairs(model, t)
            if r is not None:
                per[t] = r
        for t, (cases, excl, n_cases) in per.items():
            rows.append({"model": model, "scope": "task", "task_id": t, "n_cases_expected": n_cases,
                         "n_pairs_excluded": excl, **summarize({t: (cases, excl, n_cases)}, 0, rng)})
        for scope, tasks in scope_map.items():
            sub = {t: per[t] for t in tasks if t in per}
            missing = [t for t in tasks if t not in per]
            rows.append({"model": model, "scope": scope, "task_id": "", "n_tasks": len(tasks),
                         "missing_tasks": ";".join(missing), "n_cases_expected": sum(v[2] for v in sub.values()),
                         "n_pairs_excluded": sum(v[1] for v in sub.values()), **summarize(sub, args.n_boot, rng)})
        print(model, flush=True)
    write_csv(TABLES / "stage2_flip.csv", rows)


if __name__ == "__main__":
    main()
