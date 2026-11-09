"""共通の入出力型（指示書 §6-2）。

S1MB 公式評価器の InferenceCase / Question / Option と同じ情報を持つが、
メイン環境が s1mb パッケージに依存しないよう、素の dataclass で定義する。
ラベル・group_id・provenance はこの型に含めない（モデルに渡らないことを型で保証する）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QType = Literal["choice", "noul", "score"]


@dataclass(frozen=True)
class Option:
    id: str
    description: str
    description_json: str | None = None
    value: float | None = None  # Score のみ。criteria[].value（数値）


@dataclass(frozen=True)
class Question:
    id: str
    type: QType
    instructions: str
    options: tuple[Option, ...]
    instructions_json: str | None = None
    system_prompt: str = ""


@dataclass(frozen=True)
class DecisionRequest:
    case_id: str
    state: Any
    questions: tuple[Question, ...]
    state_json: str | None = None


@dataclass
class DecisionResult:
    case_id: str
    question_id: str
    probs: dict[str, float] | None = None  # option_id -> 確率
    error: str | None = None
    abstention: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # expected_value, latency_ms など


@dataclass(frozen=True)
class Target:
    """評価側だけが持つ正解。モデル入力には渡さない。"""

    task_id: str
    case_id: str
    question_id: str
    qtype: QType
    option_ids: tuple[str, ...]  # 作者が定めた順序
    probs: tuple[float, ...]  # ソフトターゲット（丸めない）
    values: tuple[float, ...] | None = None  # Score の数値
    group_id: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.task_id, self.case_id, self.question_id)
