"""sokudan の bench_ja ローダ（指示書 §4-2）。

- データ：third_party/sokudan/data/bench_ja.jsonl（固定コミット。CC BY 4.0、評価専用。再配布しない）
- 設問：sokudan/eval/bench_ja.py の bench_questions() と同じ文言・選択肢・順序を使う
  （部署ルーティング 4択 Choice、緊急度 3段階 Score、解約示唆 Noul）。
- Noul の選択肢定義：S1MB の Noul は false/true それぞれに定義文を持つ形式なので、
  設問文から機械的に「〜している／〜していない」を作る（sokudan 側は定義文なしで Noul を出す）。
  これは本評価の入力条件として report.md に記録する。
- Score の値：sokudan の順序インデックス 0,1,2 をそのまま value にする。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .types import DecisionRequest, Option, Question, Target

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "third_party" / "sokudan" / "data" / "bench_ja.jsonl"

# sokudan @ eccded37 の sokudan/eval/bench_ja.py から転記（順序も同じ）
DEPARTMENTS = {
    "請求": "支払い・返金・請求書・料金の二重引き落としなど金銭処理に関するもの",
    "技術": "不具合・障害・エラー・ログイン不能など製品が動かないことに関するもの",
    "営業": "料金プラン・新規契約・見積もり・増席など購入判断に関するもの",
    "その他": "上のどれにも当てはまらない一般的な連絡",
}
URGENCY_LEVELS = ["急がない", "早めに", "業務が止まっている"]
CHURN_INSTRUCTIONS = "送信者は解約・契約終了を示唆しているか"
CHURN_DEFINITIONS = {
    "false": "送信者は解約・契約終了を示唆していない",
    "true": "送信者は解約・契約終了を示唆している",
}

# 言い換え1パターン（ユーザー指示：差の確認用。主結果は CHURN_DEFINITIONS で固定し全モデル同一）
CHURN_DEFINITIONS_PARA1 = {
    "false": "解約や契約の終了をほのめかす内容は含まれていない",
    "true": "解約や契約の終了をほのめかす内容が含まれている",
}
TASKS = {
    "bench_ja-department": "choice",
    "bench_ja-urgency": "score",
    "bench_ja-churn": "noul",
}
# 主結果には含めない入力条件の変種（report.md で別表）
VARIANT_TASKS = {"bench_ja-churn-para1": "noul"}


def questions() -> dict[str, Question]:
    return {
        "bench_ja-department": Question(
            id="department", type="choice", instructions="この問い合わせはどの部署が担当すべきか",
            options=tuple(Option(id=k, description=v) for k, v in DEPARTMENTS.items()),
        ),
        "bench_ja-urgency": Question(
            id="urgency", type="score", instructions="この依頼の緊急度は",
            options=tuple(Option(id=str(i), description=d, value=float(i)) for i, d in enumerate(URGENCY_LEVELS)),
        ),
        "bench_ja-churn": Question(
            id="churn", type="noul", instructions=CHURN_INSTRUCTIONS,
            options=(Option("false", CHURN_DEFINITIONS["false"]), Option("true", CHURN_DEFINITIONS["true"])),
        ),
        "bench_ja-churn-para1": Question(
            id="churn", type="noul", instructions=CHURN_INSTRUCTIONS,
            options=(Option("false", CHURN_DEFINITIONS_PARA1["false"]), Option("true", CHURN_DEFINITIONS_PARA1["true"])),
        ),
    }


def sha256(path: Path = DEFAULT_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items(path: Path = DEFAULT_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _onehot(ids: list[str], gold: str) -> tuple[float, ...]:
    if gold not in ids:
        raise ValueError(f"gold {gold!r} not in {ids}")
    return tuple(1.0 if i == gold else 0.0 for i in ids)


def load_task(task_id: str, path: Path = DEFAULT_PATH) -> tuple[list[DecisionRequest], list[Target]]:
    """1つの設問を1タスクとして返す。state は本文テキストのみ（属性・生成条件は渡さない）。"""
    q = questions()[task_id]
    ids = [o.id for o in q.options]
    requests, targets = [], []
    for item in load_items(path):
        if task_id == "bench_ja-department":
            gold = item["department"]
        elif task_id == "bench_ja-urgency":
            gold = str(int(item["urgency"]))
        elif task_id in ("bench_ja-churn", "bench_ja-churn-para1"):
            gold = "true" if item["churn"] else "false"
        else:
            raise KeyError(task_id)
        requests.append(DecisionRequest(case_id=item["item_id"], state=item["state"], questions=(q,)))
        targets.append(
            Target(
                task_id=task_id, case_id=item["item_id"], question_id=q.id, qtype=q.type,
                option_ids=tuple(ids), probs=_onehot(ids, gold),
                values=tuple(float(o.value) for o in q.options) if q.type == "score" else None,
                group_id=item["item_id"],
            )
        )
    return requests, targets
