"""件ごとの予測確率（results/raw/）の入出力。

results/raw/<model>/<dataset>/<task_id>.jsonl に 1行=1判断 で保存する。
保存するのは ID・選択肢ID（作者の順序）・確率・失敗/棄権・条件（並べ替え）だけで、
入力テキストと正解ラベルは保存しない（評価データを再配布しないため）。指標は正解を
data/ から読み直して再計算する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"


@dataclass
class RawRow:
    case_id: str
    question_id: str
    option_ids: list[str]  # 作者が定めた元の順序
    probs: list[float] | None  # option_ids と同じ順序
    error: str | None = None
    abstention: str | None = None
    condition: str = "original"  # "original" または "perm:<i>"
    order: list[int] | None = None  # 並べ替えのとき、モデルに見せた順序（元のインデックス）
    extra: dict = field(default_factory=dict)

    def probs_dict(self) -> dict[str, float] | None:
        return None if self.probs is None else dict(zip(self.option_ids, self.probs, strict=True))


def path_for(model: str, dataset: str, task_id: str, root: Path = RAW) -> Path:
    return root / model / dataset / f"{task_id}.jsonl"


def write_rows(path: Path, rows: list[RawRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_rows(path: Path) -> list[RawRow]:
    with path.open(encoding="utf-8") as fh:
        return [RawRow(**json.loads(line)) for line in fh if line.strip()]


def predictions_index(rows: list[RawRow], condition: str = "original") -> dict[tuple[str, str], dict]:
    out = {}
    for r in rows:
        if r.condition != condition:
            continue
        key = (r.case_id, r.question_id)
        if key in out:
            raise ValueError(f"duplicate prediction {key}")
        out[key] = {"probs": r.probs_dict(), "error": r.error, "abstention": r.abstention}
    return out
