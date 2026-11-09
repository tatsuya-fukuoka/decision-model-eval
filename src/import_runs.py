"""公式評価器の結果（results/s1mb_runs/<run_id>/*.json）と bench_ja の出力を results/raw/ に取り込む。

- 予測（ID・確率・失敗・棄権）だけを取り込み、入力テキストや正解は持ち込まない。
- 選択肢の順序は本リポジトリの Target（作者の順序）に合わせる。
- 公式結果の要約（status・件数・model 情報・環境）は results/runs/<model>/<run_id>.json に残す
  （予測本体は含めない）。
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import bench_ja, s1mb
from .results import RawRow, path_for, write_rows

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "s1mb_runs"
SUMMARIES = ROOT / "results" / "runs"


def import_s1mb(model: str, run_id: str, *, write: bool = True, runs_dir: Path = RUNS, root: Path | None = None) -> dict:
    run_dir = runs_dir / run_id
    summary: dict = {"run_id": run_id, "benchmarks": {}}
    for t in s1mb.load_category():
        f = run_dir / f"{t}.json"
        if not f.is_file():
            summary["benchmarks"][t] = {"status": "missing"}
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        _, _, targets = s1mb.load_benchmark(t)
        order = {(x.case_id, x.question_id): list(x.option_ids) for x in targets}
        rows = []
        for p in res["predictions"]:
            ids = order[(p["case_id"], p["question_id"])]
            probs = None if p.get("probabilities") is None else [float(p["probabilities"][o]) for o in ids]
            rows.append(RawRow(p["case_id"], p["question_id"], ids, probs, error=p.get("error"),
                               abstention=p.get("abstention")))
        if write:
            write_rows(path_for(model, "s1mb", t) if root is None else path_for(model, "s1mb", t, root), rows)
        summary["benchmarks"][t] = {"status": res["status"], "counts": res["counts"],
                                    "elapsed_seconds": res.get("elapsed_seconds")}
        summary.setdefault("model", res["model"])
        summary.setdefault("evaluator_version", res.get("evaluator_version"))
        env = {k: v for k, v in res.get("environment", {}).items() if k != "input_hashes"}
        summary.setdefault("environment", env)
    return summary


def import_bench_ja(model: str, out_file: Path, *, write: bool = True, root: Path | None = None) -> dict:
    by_task: dict[str, list[dict]] = {}
    for line in out_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_task.setdefault(r["task_id"], []).append(r)
    summary = {}
    for task, recs in by_task.items():
        _, targets = bench_ja.load_task(task)
        ids = {(x.case_id, x.question_id): list(x.option_ids) for x in targets}
        rows = []
        for r in recs:
            oid = ids[(r["case_id"], r["question_id"])]
            probs = None if r.get("probabilities") is None else [float(r["probabilities"][o]) for o in oid]
            rows.append(RawRow(r["case_id"], r["question_id"], oid, probs, error=r.get("error"),
                               abstention=r.get("abstention")))
        if write:
            write_rows(path_for(model, "bench_ja", task) if root is None else path_for(model, "bench_ja", task, root),
                       rows)
        summary[task] = {"n": len(rows), "failed": sum(r.error is not None for r in rows),
                         "expected": len(targets)}
    return summary


def write_summary(model: str, run_id: str, summary: dict) -> Path:
    path = SUMMARIES / model / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def import_flip(model: str, out_file: Path, *, write: bool = True) -> dict:
    """選択肢を並べ替えた推論（scripts/flip_export.py → s1mb_bridge --requests-only）を
    results/raw/<model>/flip/<task_id>.jsonl に取り込む。condition="perm:<i>"、order にモデルへ見せた順序。"""
    by_task: dict[str, list[dict]] = {}
    for line in out_file.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            by_task.setdefault(r["task_id"], []).append(r)
    summary = {}
    for task, recs in by_task.items():
        targets = bench_ja.load_task(task)[1] if task.startswith("bench_ja-") else s1mb.load_benchmark(task)[2]
        ids = {(x.case_id, x.question_id): list(x.option_ids) for x in targets}
        rows = []
        for r in recs:
            oid = ids[(r["case_id"], r["question_id"])]
            probs = None if r.get("probabilities") is None else [float(r["probabilities"][o]) for o in oid]
            rows.append(RawRow(r["case_id"], r["question_id"], oid, probs, error=r.get("error"),
                               abstention=r.get("abstention"), condition=f"perm:{r['meta']['perm']}",
                               order=r["meta"]["order"]))
        if write:
            write_rows(path_for(model, "flip", task), rows)
        summary[task] = {"n": len(rows), "failed": sum(r.error is not None for r in rows)}
    return summary
