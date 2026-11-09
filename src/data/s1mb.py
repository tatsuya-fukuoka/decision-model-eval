"""S1MB-japanese-v1 のローダ（指示書 §4-1）。

データセットカードの推論ルールに従う：
- input.state_json / decisions[].instructions_json / criteria[].description_json を JSON として復号し、
  それだけをリクエストに入れる。ラベル・group_id・provenance_json・legacy_aux_json はリクエストに入れない。
- 選択肢の意味と順序は変えない（criteria の順序をそのまま保つ）。
- Score は criteria[].value（数値）を保持する。
- 正解（targets）はソフトターゲットのまま、criteria の順序に揃えて Target に入れる。

データは `src.data.fetch` で固定リビジョンを data/hf/S1MB-japanese-v1 に取得したものを読む。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import DecisionRequest, Option, Question, Target

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data" / "hf" / "S1MB-japanese-v1"
CATEGORY = "jdmb-v1"
PRIMARY_METRICS = {
    "choice": "target_mass_at_prediction",
    "noul": "binary_brier",
    "score": "normalized_expected_score_mae",
}


@dataclass(frozen=True)
class Benchmark:
    id: str
    task: str
    dataset: str
    case_count: int
    decision_count: int
    primary_metric: str

    @property
    def synthetic(self) -> bool:
        return self.id.startswith("generalization-")


def load_category(root: Path = DEFAULT_ROOT, name: str = CATEGORY) -> list[str]:
    return json.loads((root / "categories" / f"{name}.json").read_text(encoding="utf-8"))["benchmarks"]


def load_benchmark_descriptor(root: Path, task_id: str) -> Benchmark:
    d = json.loads((root / "benchmarks" / f"{task_id}.json").read_text(encoding="utf-8"))
    if d["id"] != task_id:
        raise ValueError(f"descriptor id mismatch: {d['id']} != {task_id}")
    if d["primary_metric"] != PRIMARY_METRICS[d["task"]]:
        raise ValueError(f"unexpected primary metric for {task_id}: {d['primary_metric']}")
    return Benchmark(
        id=d["id"], task=d["task"], dataset=d["dataset"], case_count=d["case_count"],
        decision_count=d["decision_count"], primary_metric=d["primary_metric"],
    )


def _decode(value: str | None):
    return None if value is None else json.loads(value)


def _text(value) -> str:
    """構造化された JSON 値を、テキストしか受け付けないモデル向けに文字列化する（意味は変えない）。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def parse_row(row: dict, task_id: str, task: str) -> tuple[DecisionRequest, list[Target]]:
    inp = row["input"]
    questions, targets = [], []
    target_by_id = {t["decision_id"]: t for t in row["targets"]}
    for d in inp["decisions"]:
        if d["type"] != task:
            continue  # ベンチマーク記述子の task と同じ型の判断だけを評価対象にする
        options = tuple(
            Option(
                id=c["id"],
                description=_text(_decode(c["description_json"])),
                description_json=c["description_json"],
                value=None if c.get("value") is None else float(c["value"]),
            )
            for c in d["criteria"]
        )
        q = Question(
            id=d["id"], type=d["type"], instructions=_text(_decode(d["instructions_json"])),
            instructions_json=d["instructions_json"], system_prompt=d.get("system_prompt") or "",
            options=options,
        )
        questions.append(q)
        t = target_by_id[d["id"]]
        tp = dict(zip(t["ids"], t["probabilities"], strict=True))
        if set(tp) != {o.id for o in options}:
            raise ValueError(f"target ids differ from criteria in {row['case_id']}/{d['id']}")
        targets.append(
            Target(
                task_id=task_id, case_id=row["case_id"], question_id=d["id"], qtype=d["type"],
                option_ids=tuple(o.id for o in options), probs=tuple(float(tp[o.id]) for o in options),
                values=tuple(float(o.value) for o in options) if d["type"] == "score" else None,
                group_id=row["group_id"],
            )
        )
    state_json = inp["state_json"]
    request = DecisionRequest(
        case_id=row["case_id"], state=_decode(state_json), state_json=state_json, questions=tuple(questions)
    )
    return request, targets


def load_benchmark(task_id: str, root: Path = DEFAULT_ROOT) -> tuple[Benchmark, list[DecisionRequest], list[Target]]:
    import pyarrow.parquet as pq

    bench = load_benchmark_descriptor(root, task_id)
    path = root / "data" / Path(bench.dataset).name / "test.parquet"
    rows = pq.read_table(path).to_pylist()
    requests, targets = [], []
    for row in rows:
        if row.get("split") not in (None, "test"):
            raise ValueError(f"unexpected split {row.get('split')}")
        req, tg = parse_row(row, task_id, bench.task)
        if req.questions:
            requests.append(req)
            targets.extend(tg)
    if len({r.case_id for r in requests}) != len(requests):
        raise ValueError(f"duplicate case ids in {task_id}")
    if len(requests) != bench.case_count or len(targets) != bench.decision_count:
        raise ValueError(
            f"{task_id}: counts {len(requests)}/{len(targets)} differ from descriptor "
            f"{bench.case_count}/{bench.decision_count}"
        )
    return bench, requests, targets
