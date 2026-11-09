"""実行コスト計測（指示書 §5-4）用の入力を作る。評価器の隔離環境で third_party/S1MB/evaluator を作業ディレクトリにして実行:

    HF_HUB_OFFLINE=1 <env>/bin/python -I scripts/cost_export.py --out ../../../data/bridge/cost_requests.jsonl

- S1MB japanese-v1 の重ならない17タスクから、型（noul / choice / score）ごとに 40 件ずつ（計120件）を
  seed 固定で抽出する。加えてウォームアップ用に別の 12 件（型ごと4件）を先頭に置く（meta.warmup=true）。
- 全モデルで同じ120件を使う。入力の長さでレイテンシが変わるため、件の選び方は全モデル共通で固定する。
- 出力は data/bridge/（git に含めない）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SEED = 20261107
PER_TYPE, WARMUP_PER_TYPE = 40, 4
NON_OVERLAP = Path(__file__).resolve().parents[1] / "results" / "overlap_tasks.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from s1mb.data import DATA_DIR, InferenceCase, load_benchmark, load_cases

    tasks = json.loads(NON_OVERLAP.read_text(encoding="utf-8"))["non_overlap_tasks"]
    pool: dict[str, list[tuple[str, dict]]] = {"noul": [], "choice": [], "score": []}
    for name in sorted(tasks):
        bench = load_benchmark(DATA_DIR, name)
        for case in load_cases(DATA_DIR, bench):
            inf = InferenceCase.model_validate(case.model_dump(include=set(InferenceCase.model_fields)))
            pool[bench.task].append((bench.id, inf.model_dump(mode="json")))
    rng = random.Random(SEED)
    warm, timed = [], []
    for qtype, items in pool.items():
        pick = rng.sample(items, PER_TYPE + WARMUP_PER_TYPE)
        warm += [(qtype, *x) for x in pick[:WARMUP_PER_TYPE]]
        timed += [(qtype, *x) for x in pick[WARMUP_PER_TYPE:]]
    rng.shuffle(timed)
    with args.out.open("w", encoding="utf-8") as fh:
        for flag, rows in ((True, warm), (False, timed)):
            for qtype, task, case in rows:
                fh.write(json.dumps({"task_id": task, "case": case, "meta": {"warmup": flag, "type": qtype}},
                                    ensure_ascii=False) + "\n")
    print(f"warmup={len(warm)} timed={len(timed)} -> {args.out}")


if __name__ == "__main__":
    main()
