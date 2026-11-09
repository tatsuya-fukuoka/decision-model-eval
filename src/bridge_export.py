"""公式評価器の環境（scripts/s1mb_bridge.py）に渡す bench_ja の InferenceCase を書き出す。

    uv run python -m src.bridge_export

出力 data/bridge/bench_ja_requests.jsonl は本文テキストを含むので gitignore 対象（/data/）。
正解・属性・生成条件は含めない。1件1設問（設問ごとに独立した case）として渡す。
"""

import json
from pathlib import Path

from .data import bench_ja
from .data.types import DecisionRequest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bridge" / "bench_ja_requests.jsonl"


def inference_case(req: DecisionRequest) -> dict:
    return {
        "case_id": req.case_id,
        "state": req.state,
        "questions": [
            {
                "id": q.id, "task": q.type, "instructions": q.instructions, "system_prompt": q.system_prompt,
                "options": [{"id": o.id, "description": o.description, "value": o.value} for o in q.options],
            }
            for q in req.questions
        ],
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for task in [*bench_ja.TASKS, *bench_ja.VARIANT_TASKS]:
            reqs, _ = bench_ja.load_task(task)
            for r in reqs:
                fh.write(json.dumps({"task_id": task, "case": inference_case(r)}, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} cases -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
