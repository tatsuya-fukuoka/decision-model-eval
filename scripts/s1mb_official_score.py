"""results/raw/ の件ごとの予測を、S1MB 公式評価器のコードそのもので採点する。

評価器の環境（envs/s1mb-base）で実行し、本リポジトリの移植版（src/metrics/s1mb_scores.py）と
比較するための JSON を results/parity/<model>.official.json に書く。

    cd third_party/S1MB/evaluator
    HF_HUB_OFFLINE=1 ../../../envs/s1mb-base/.venv/bin/python -I ../../../scripts/s1mb_official_score.py MODEL...

- タスク：s1mb.metrics.calculate（primary・診断・baseline_adjusted_score）
- 型平均：s1mb.submission_report.task_score（完了要求・適格性による除外・0–100）
本スクリプトは results/raw の JSON を読むだけで、本リポジトリの src は import しない。
"""

import json
import sys
from pathlib import Path

from s1mb.data import DATA_DIR, Prediction, load_benchmark, load_cases, load_category
from s1mb.metrics import calculate
from s1mb.submission_report import Measurement, task_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
OUT = ROOT / "results" / "parity"


def predictions(model: str, task_id: str) -> list[Prediction]:
    path = RAW / model / "s1mb" / f"{task_id}.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("condition", "original") != "original":
            continue
        probs = None if r["probs"] is None else dict(zip(r["option_ids"], r["probs"], strict=True))
        out.append(Prediction(case_id=r["case_id"], question_id=r["question_id"], probabilities=probs,
                              error=r.get("error"), abstention=r.get("abstention")))
    return out


def main(models: list[str]) -> None:
    overlap = json.loads((ROOT / "results" / "overlap_tasks.json").read_text(encoding="utf-8"))
    category = load_category(DATA_DIR, "japanese-v1")
    benches = {b: load_benchmark(DATA_DIR, b) for b in category.benchmarks}
    cases = {b: load_cases(DATA_DIR, bench) for b, bench in benches.items()}
    ov = [t["task_id"] for t in overlap["overlap_tasks"]]
    scopes = {
        "non_overlap_17": overlap["non_overlap_tasks"],
        "all_27": list(category.benchmarks),
        "overlap_10": ov,
        "synthetic_8": [b for b in category.benchmarks if b.startswith("generalization-")],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for model in models:
        per_task, measurements = {}, {}
        for b, bench in benches.items():
            preds = predictions(model, b)
            metrics = calculate(cases[b], preds, bench.task)
            n_expected = bench.decision_count
            failed = sum(p.error is not None for p in preds)
            complete = len(preds) == n_expected and failed == 0
            measurements[b] = Measurement(complete, metrics, sum(p.abstention is not None for p in preds), failed)
            per_task[b] = {"task": bench.task, "complete": complete, "metrics": metrics}
        summary = {}
        for s, tasks in scopes.items():
            row = {}
            for task in ("noul", "choice", "score"):
                req = [benches[b] for b in tasks if benches[b].task == task]
                v = task_score(req, measurements) if req else None
                row[task] = None if v is None else v / 100
            present = [k for k in ("noul", "choice", "score") if any(benches[b].task == k for b in tasks)]
            row["task_avg"] = (None if any(row[k] is None for k in present)
                               else sum(row[k] for k in present) / len(present))
            summary[s] = row
        out = {"model": model, "evaluator": "S1MB evaluator (third_party/S1MB, see manifest)",
               "per_task": per_task, "summary": summary}
        (OUT / f"{model}.official.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True))
        print(model, summary["non_overlap_17"], summary["all_27"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
