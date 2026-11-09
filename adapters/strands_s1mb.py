"""Strands Decider 2B を S1MB の ModelAdapter 形式で動かす自作アダプタ（S1MB に公式アダプタが無いため）。

評価器の隔離環境で scripts/s1mb_bridge.py --custom-adapter から読み込まれる。本リポジトリの src は import しない。

読み出しは作者の既定に合わせる（strands-decider==0.3.0 の infer.load_engine → SystemOneEngine.evaluate）：
- HTTP サーバ（strands-decider serve）は使わず、同じエンジンをプロセス内で呼ぶ。
- 温度（プリミティブごと1つ）は作者の既定どおりエンジン内で適用される。
- prompting.to_answer は確率を小数4桁に丸めるため、prompting モジュール内の round だけを恒等関数に差し替えて
  丸める前の値を受け取る（計算経路は変えない）。
- 拡張設定（既定では使わない）：max_length を渡すと model.config.max_length（作者の既定 4096、学習時の長さ）を
  その値に引き上げる。超過時に例外を出す作者の規則（切り詰めない）は変えない。
- 状態がウィンドウを超えると作者の fit_window の規則で切り詰められる（選択肢まで切れた場合は作者のコードが例外を出し、その件は失敗として記録される）。

S1MB の質問からの変換（入力条件として report.md に記録）：
- Choice：選択肢名は匿名の連番 "1".."K"（作者の順序）、説明文はそのまま。
- Score：値の昇順に並べ（Strands の score は低い方から）、各レベルを "値: 説明" にする。11段階以上は作者の上限（10）超で失敗。
- Noul：Strands の noul criteria {"false": 定義, "true": 定義} に S1MB の定義文を入れる。
- 指示文：system_prompt があれば前に改行で連結。状態は文字列・構造化データのまま渡す（作者の render_content）。
"""

from __future__ import annotations

from s1mb.data import InferenceCase, ModelInfo, Prediction


class StrandsAdapter:
    def __init__(self, checkpoint: str, revision: str, device: str = "cuda:0", max_length: int | None = None):
        import strands_decider
        from strands_decider import prompting
        from strands_decider.infer import load_engine

        prompting.round = lambda x, ndigits=None: x  # 表示用の丸めだけを外す
        self.engine = load_engine(checkpoint, device=device)
        self.author_max_length = self.engine.model.config.max_length
        if max_length:
            self.engine.model.config.max_length = int(max_length)
        self.version = strands_decider.__version__
        self.revision = revision
        self.device = device

    def _question(self, q):
        from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion

        instructions = f"{q.system_prompt}\n{q.instructions}" if q.system_prompt.strip() else q.instructions
        if q.task == "choice":
            names = [str(i + 1) for i in range(len(q.options))]
            return (ChoiceQuestion(instructions=instructions,
                                   criteria={n: o.description for n, o in zip(names, q.options, strict=True)}),
                    dict(zip(names, [o.id for o in q.options], strict=True)))
        if q.task == "score":
            opts = sorted(q.options, key=lambda o: float(o.value))
            return (ScoreQuestion(instructions=instructions, criteria=[f"{o.value:g}: {o.description}" for o in opts]),
                    {str(i): o.id for i, o in enumerate(opts)})
        if q.task == "noul":
            d = {o.id: o.description for o in q.options}
            return NoulQuestion(instructions=instructions, criteria={"false": d["false"], "true": d["true"]}), None
        raise ValueError(f"unsupported task {q.task}")

    def predict(self, case: InferenceCase) -> list[Prediction]:
        specs = {q.id: self._question(q) for q in case.questions}
        resp = self.engine.ask(case.state, {qid: s for qid, (s, _) in specs.items()})
        out = []
        for qid, (_, mapping) in specs.items():
            a = resp.answers[qid]
            if mapping is None:
                p = float(a.noul)
                probs = {"false": 1.0 - p, "true": p}
            else:
                raw = {mapping[k]: float(v) for k, v in a.probabilities.items()}
                s = sum(raw.values())
                probs = {k: v / s for k, v in raw.items()}
            out.append(Prediction(case_id=case.case_id, question_id=qid, probabilities=probs))
        return out

    def metadata(self) -> ModelInfo:
        cfg = self.engine.model.config
        return ModelInfo(
            id="StrandsAgents/strands-decider-2B-hobson-v19", adapter="custom:strands", revision=self.revision,
            settings={
                "package": f"strands-decider=={self.version}",
                "readout": "infer.SystemOneEngine.evaluate in process (no HTTP server); prompting.round disabled",
                "max_length": getattr(cfg, "max_length", None),
                "author_default_max_length": self.author_max_length, "base_model": getattr(cfg, "base_model", None),
                "base_revision": getattr(cfg, "base_revision", None), "device": self.device,
            },
        )

    def close(self) -> None:
        self.engine = None
