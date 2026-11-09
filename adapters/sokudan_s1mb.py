"""sokudan-ja-310m を S1MB の ModelAdapter 形式で動かす自作アダプタ（S1MB に公式アダプタが無いため）。

評価器の隔離環境で scripts/s1mb_bridge.py --custom-adapter から読み込まれる。本リポジトリの src は import しない。

読み出しは作者の既定に合わせる（sokudan @ eccded37 の sokudan/predict.py::Agent.predict と同じ計算）：
- sokudan.load(スナップショット, backend="torch")：既定のキャリブレーション（同梱 calibration.json の bool 温度のみ）。
- Agent.predict は確率を小数4桁に丸めて返すので、丸める前の値を同じ手順（Agent._forward → 温度）で取り出す。
- 状態は 1024 トークンを超えると作者の既定どおり切り詰められる。切り詰めた件数は metadata.settings に記録する。
- 拡張設定（既定では使わない）：max_tokens を渡すと Agent.max_state_tokens（joint 符号化では状態＋質問の
  合計上限、作者の既定 1024）をその値に引き上げる。超過時の扱い（状態の切り詰め／質問だけで超える場合の
  QuestionTooLongError）は作者のコードのまま。作者の既定と異なるため「拡張設定」として既定の結果と分ける。

S1MB の質問から sokudan の質問への変換（入力条件として report.md に記録）：
- Choice：ラベルは匿名の連番 "1".."K"（作者の順序）。説明文は sokudan の criteria の説明として渡す。
- Score：作者の順序を値の昇順に並べ（sokudan の score は低い方から）、各レベル名を "値: 説明" にする。
- Noul：sokudan の bool の false_label / true_label に S1MB の false / true の定義文を入れる。
- 指示文：system_prompt があれば指示文の前に改行で連結する。
- 状態：文字列はそのまま、構造化データは JSON（ensure_ascii=False）で渡す。
"""

from __future__ import annotations

import json
from typing import Any

from s1mb.data import InferenceCase, ModelInfo, Prediction


class SokudanAdapter:
    def __init__(self, checkpoint: str, revision: str, device: str = "cuda:0", max_tokens: int | None = None):
        import sokudan
        from sokudan.calibration.temperature import apply_temperature

        self.agent = sokudan.load(checkpoint, backend="torch", device=device)
        self.author_max_tokens = self.agent.max_state_tokens
        if max_tokens:
            self.agent.max_state_tokens = int(max_tokens)
        self.apply_temperature = apply_temperature
        self.revision = revision
        self.device = device
        self.truncated_total = 0
        self.truncated_since_metadata = 0

    @staticmethod
    def _state(case: InferenceCase) -> str:
        if isinstance(case.state, str):
            return case.state
        return json.dumps(case.state, ensure_ascii=False)

    @staticmethod
    def _question(q) -> tuple[dict[str, Any], list[str]]:
        """sokudan の質問 dict と、sokudan のスロット順に並んだ S1MB の選択肢 ID。"""
        instructions = f"{q.system_prompt}\n{q.instructions}" if q.system_prompt.strip() else q.instructions
        if q.task == "choice":
            ids = [o.id for o in q.options]
            criteria = {str(i + 1): o.description for i, o in enumerate(q.options)}
            return {"type": "choice", "instructions": instructions, "criteria": criteria}, ids
        if q.task == "score":
            opts = sorted(q.options, key=lambda o: float(o.value))  # 安定ソート：同値は作者の順序
            return ({"type": "score", "instructions": instructions,
                     "criteria": [f"{o.value:g}: {o.description}" for o in opts]}, [o.id for o in opts])
        if q.task == "noul":
            d = {o.id: o.description for o in q.options}
            return ({"type": "bool", "instructions": instructions, "false_label": d["false"],
                     "true_label": d["true"]}, ["false", "true"])
        raise ValueError(f"unsupported task {q.task}")

    def predict(self, case: InferenceCase) -> list[Prediction]:
        from sokudan.predict import _Prepared
        from sokudan.schema.question import parse_questions

        specs = {q.id: self._question(q) for q in case.questions}
        parsed = parse_questions({qid: s for qid, (s, _) in specs.items()})
        prepared = {qid: _Prepared(question, question.type, list(question.labels)) for qid, question in parsed.items()}
        raw, _, _, truncated, _ = self.agent._forward(self._state(case), prepared)
        if truncated:
            self.truncated_total += 1
            self.truncated_since_metadata += 1
        out = []
        for qid, (_, ids) in specs.items():
            item = prepared[qid]
            row = raw[qid]
            t = self.agent.temperatures.get((item.kind, len(item.labels)), 1.0)
            if t != 1.0:
                row = self.apply_temperature(row.reshape(1, -1), t)[0]
            row = [float(x) for x in row]
            s = sum(row)
            out.append(Prediction(case_id=case.case_id, question_id=qid,
                                  probabilities={oid: p / s for oid, p in zip(ids, row, strict=True)}))
        return out

    def metadata(self) -> ModelInfo:
        n, self.truncated_since_metadata = self.truncated_since_metadata, 0
        return ModelInfo(
            id="GeneLab/sokudan-ja-310m", adapter="custom:sokudan", revision=self.revision,
            settings={
                "readout": "sokudan.predict.Agent._forward + default calibration (bool temperatures only), unrounded",
                "temperatures": {f"{k[0]}/{k[1]}": v for k, v in self.agent.temperatures.items()},
                "encoding": self.agent.encoding, "max_state_tokens": self.agent.max_state_tokens,
                "author_default_max_state_tokens": self.author_max_tokens,
                "state_truncated_cases": n, "state_truncated_cases_total": self.truncated_total,
                "device": self.device, "backbone": getattr(self.agent.backend, "backbone_id", None),
            },
        )

    def close(self) -> None:
        self.agent = None
