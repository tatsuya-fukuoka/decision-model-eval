"""選択肢 ID の匿名化と、選択肢順序の並べ替え・復元（指示書 §4-1、§5-3、§6-3）。

- 匿名化：選択肢の意味（description）と順序は保ったまま、ID だけを中立 ID に置き換える。
- 並べ替え：頑健性評価（flip rate）用。並べ替えても ID は元のまま保持し、
  予測は ID で照合するので、元の順序への復元は ID の対応だけで決まる。
"""

from __future__ import annotations

import random
from dataclasses import replace

from .types import DecisionRequest, Option, Question


def neutral_ids(n: int) -> list[str]:
    width = len(str(max(n - 1, 0)))
    return [f"option_{i:0{width}d}" for i in range(n)]


def anonymize(question: Question) -> tuple[Question, dict[str, str]]:
    """Choice の ID を中立 ID に置き換える。戻り値の dict は 中立ID -> 元ID。

    Noul（false/true）と Score（数値レベル）は意味を持つ ID なので変えない。
    """
    if question.type != "choice":
        return question, {o.id: o.id for o in question.options}
    ids = neutral_ids(len(question.options))
    mapping = dict(zip(ids, (o.id for o in question.options), strict=True))
    options = tuple(replace(o, id=new) for new, o in zip(ids, question.options, strict=True))
    return replace(question, options=options), mapping


def restore(probs: dict[str, float], mapping: dict[str, str]) -> dict[str, float]:
    """中立 ID の確率を元 ID に戻す。キーの過不足はエラー（黙って捨てない）。"""
    if set(probs) != set(mapping):
        raise ValueError(f"returned keys {sorted(probs)} differ from request {sorted(mapping)}")
    return {mapping[k]: float(v) for k, v in probs.items()}


def permutations_for(question: Question, n: int, seed: int) -> list[tuple[int, ...]]:
    """元の順序以外の並べ替えを最大 n 通り、seed 固定で選ぶ（重複なし）。

    選択肢数 k に対して k! - 1 通りしかない場合はその数までに減る。
    """
    k = len(question.options)
    identity = tuple(range(k))
    rng = random.Random(f"{seed}:{question.id}:{k}")
    limit = 1
    for i in range(2, k + 1):
        limit *= i
    limit -= 1
    out: list[tuple[int, ...]] = []
    seen = {identity}
    while len(out) < min(n, limit):
        order = list(identity)
        rng.shuffle(order)
        perm = tuple(order)
        if perm not in seen:
            seen.add(perm)
            out.append(perm)
    return out


def permute(question: Question, order: tuple[int, ...]) -> Question:
    """選択肢を order の順に並べ替える。ID と description は変えない。"""
    if sorted(order) != list(range(len(question.options))):
        raise ValueError("order must be a permutation of option indices")
    return replace(question, options=tuple(question.options[i] for i in order))


def permute_request(request: DecisionRequest, orders: dict[str, tuple[int, ...]]) -> DecisionRequest:
    questions = tuple(permute(q, orders[q.id]) if q.id in orders else q for q in request.questions)
    return replace(request, questions=questions)


def argmax_id(probs: dict[str, float], option_order: list[str]) -> str:
    """同点は作者が定めた元の順序で解決する（S1MB の規則と同じ）。"""
    return max(option_order, key=lambda o: probs[o])


def make_option(id: str, description: str, value: float | None = None) -> Option:
    return Option(id=id, description=description, value=value)
