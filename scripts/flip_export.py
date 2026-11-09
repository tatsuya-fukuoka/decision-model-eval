"""選択肢順序の入れ替え（指示書 §5-3）用の入力を作る。評価器の隔離環境で、third_party/S1MB/evaluator を作業
ディレクトリにして実行する（公式の load_cases で S1MB の InferenceCase を作るため。本リポジトリの src は import しない）:

    HF_HUB_OFFLINE=1 <env>/bin/python -I scripts/flip_export.py --bench-ja ../../../data/bridge/bench_ja_requests.jsonl \
        --out ../../../data/bridge/flip_requests.jsonl

- 対象：S1MB japanese-v1 の Choice 型 16 タスクの全件と、bench_ja の部署（Choice）300件。
- 各件で、恒等でない並べ替えを最大5通り、重複なしで一様に選ぶ（K=2 は 1 通り、K=3 は 5 通りすべて）。
  乱数は (seed, task_id, case_id, question_id) から決まるので、件の並びや件数に左右されない。
- 並べ替えるのは options の並びだけ。選択肢 ID・説明文・状態・指示文は変えない（予測は元の ID で比較できる）。
- 出力は data/bridge/（git に含めない。評価データの再配布になるため）。行には meta.order（モデルに見せた順序を
  元のインデックスで表したもの）と meta.perm を入れる。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

N_PERMS = 5
SEED = 20261107


def permutations_for(k: int, key: str) -> list[list[int]]:
    rng = random.Random(int(hashlib.sha256(f"{SEED}\0{key}".encode()).hexdigest()[:16], 16))
    identity = tuple(range(k))
    if k <= 4:  # 4! = 24 通りまでは全列挙から選ぶ
        pool = [list(p) for p in itertools.permutations(range(k)) if p != identity]
        return rng.sample(pool, min(N_PERMS, len(pool)))
    out: list[list[int]] = []
    while len(out) < N_PERMS:
        p = list(range(k))
        rng.shuffle(p)
        if tuple(p) != identity and p not in out:
            out.append(p)
    return out


def permuted_rows(task_id: str, case: dict) -> list[dict]:
    choice = [i for i, q in enumerate(case["questions"]) if q["task"] == "choice"]
    if len(choice) != 1:
        raise ValueError(f"{task_id} {case['case_id']}: expected exactly one choice question")
    qi = choice[0]
    q = case["questions"][qi]
    rows = []
    for n, order in enumerate(permutations_for(len(q["options"]), f"{task_id}\0{case['case_id']}\0{q['id']}")):
        c = json.loads(json.dumps(case))
        c["questions"][qi]["options"] = [q["options"][i] for i in order]
        rows.append({"task_id": task_id, "case": c, "meta": {"perm": n, "order": order}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-ja", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from s1mb.data import DATA_DIR, InferenceCase, load_benchmark, load_cases, load_category

    rows = []
    for name in load_category(DATA_DIR, "japanese-v1").benchmarks:
        bench = load_benchmark(DATA_DIR, name)
        if bench.task != "choice":
            continue
        for case in load_cases(DATA_DIR, bench):
            inf = InferenceCase.model_validate(case.model_dump(include=set(InferenceCase.model_fields)))
            rows += permuted_rows(bench.id, inf.model_dump(mode="json"))
    for line in args.bench_ja.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["task_id"] == "bench_ja-department":
            rows += permuted_rows(r["task_id"], r["case"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tasks = {}
    for r in rows:
        tasks[r["task_id"]] = tasks.get(r["task_id"], 0) + 1
    print(json.dumps({"rows": len(rows), "per_task": tasks}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
