"""公式評価器との一致確認（parity）用の、正解を見ない人工予測を results/raw/parity-dirichlet/ に書く。

    uv run python -m src.parity_predictions

一様・多数派だけでは調整スコアが 0 付近に張り付き、式の違いを検出しにくい。そこで
- Dirichlet(0.5) の乱数確率（seed 固定、正解は使わない）
- 約 3% の一様予測（argmax の同点処理を検査）
- 約 1% の棄権（最大誤差の扱いを検査）
を混ぜる。モデルの結果ではないので、報告書の比較表には載せない。
"""

import numpy as np

from .data import s1mb
from .results import RawRow, path_for, write_rows

SEED = 20261107
MODEL = "parity-dirichlet"
# 正解とノイズの混合。正解ラベルを含むので .gitignore 対象（評価データを再配布しない）。比較表には載せない。
ORACLE = "parity-oracle-noisy"


def main() -> None:
    rng = np.random.default_rng(SEED)
    for t in s1mb.load_category():
        _, _, targets = s1mb.load_benchmark(t)
        rows = []
        for x in targets:
            u = rng.random()
            k = len(x.option_ids)
            if u < 0.01:
                rows.append(RawRow(x.case_id, x.question_id, list(x.option_ids), None, abstention="cannot_determine"))
                continue
            p = np.full(k, 1 / k) if u < 0.04 else rng.dirichlet(np.full(k, 0.5))
            rows.append(RawRow(x.case_id, x.question_id, list(x.option_ids), [float(v) for v in p]))
        write_rows(path_for(MODEL, "s1mb", t), rows)
        orows = []
        for x in targets:
            k = len(x.option_ids)
            w = rng.random()
            p = w * np.asarray(x.probs) + (1 - w) * rng.dirichlet(np.full(k, 0.5))
            orows.append(RawRow(x.case_id, x.question_id, list(x.option_ids), [float(v) for v in p / p.sum()]))
        write_rows(path_for(ORACLE, "s1mb", t), orows)
    print(f"wrote results/raw/{MODEL}/ and results/raw/{ORACLE}/ (gitignored)")


if __name__ == "__main__":
    main()
