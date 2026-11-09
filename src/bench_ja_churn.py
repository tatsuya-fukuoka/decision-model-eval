"""bench_ja 解約示唆（Noul）の P(true) の分布（ユーザー依頼 2026-11-07：多数派と同じ正解率のモデルがあるため、
AUROC と確率分布を併記する）。

    uv run python -m src.bench_ja_churn

出力：
- results/tables/bench_ja_churn_distribution.csv：モデル × 説明文（機械生成版 / 言い換え版）× 正解（全体 / true / false）
  ごとの件数、P(true) の平均と分位点（10/25/50/75/90%）、0.5 以上と判定した割合、AUROC（stage1 と同じ定義）。
- results/figures/bench_ja_churn_ptrue.png：正解ごとの P(true) のヒストグラム（機械生成版）。
温度合わせ前の値（作者の既定の読み出しのまま）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import bench_ja
from .metrics import bench_ja_scores
from .results import RAW, path_for, predictions_index, read_rows
from .stage2_calibration import write_csv

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"bench_ja-churn": "machine-generated", "bench_ja-churn-para1": "paraphrase-1"}
QS = (10, 25, 50, 75, 90)


def p_true(model: str, task: str):
    path = path_for(model, "bench_ja", task)
    if not path.is_file():
        return None
    _, targets = bench_ja.load_task(task)
    preds = predictions_index(read_rows(path))
    p, y, failed = [], [], 0
    for t in targets:
        r = preds.get((t.case_id, t.question_id))
        if not r or r.get("probs") is None:
            failed += 1
            continue
        p.append(r["probs"]["true"])
        y.append(int(t.probs[t.option_ids.index("true")] > 0.5))
    return np.asarray(p), np.asarray(y), failed


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(p.name for p in RAW.iterdir() if (p / "bench_ja").is_dir() and not p.name.startswith("parity-"))
    rows, hist = [], {}
    for model in models:
        for task, label in TASKS.items():
            r = p_true(model, task)
            if r is None:
                continue
            p, y, failed = r
            if task == "bench_ja-churn":
                hist[model] = (p, y)
            P = np.stack([1 - p, p], axis=1)
            auroc = bench_ja_scores.score_task("noul", P, y).get("auroc") if len(p) else None
            for gold, mask in (("all", np.ones(len(p), bool)), ("true", y == 1), ("false", y == 0)):
                v = p[mask]
                row = {"model": model, "task_id": task, "definitions": label, "gold": gold, "n": int(mask.sum()),
                       "n_excluded": failed, "auroc": auroc if gold == "all" else None}
                if len(v):
                    row.update({"mean_p_true": float(v.mean()), "share_p_true_ge_0.5": float((v >= 0.5).mean()),
                                **{f"p{q}": float(np.percentile(v, q)) for q in QS}})
                rows.append(row)
    write_csv(ROOT / "results" / "tables" / "bench_ja_churn_distribution.csv", rows)
    if hist:
        n = len(hist)
        cols = 4
        fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(3.2 * cols, 2.6 * ((n + cols - 1) // cols)),
                                 squeeze=False)
        bins = np.linspace(0, 1, 21)
        for ax, (model, (p, y)) in zip(axes.flat, hist.items()):
            ax.hist(p[y == 0], bins=bins, alpha=0.6, label="gold false")
            ax.hist(p[y == 1], bins=bins, alpha=0.6, label="gold true")
            ax.axvline(0.5, color="0.4", lw=0.8, ls="--")
            ax.set_title(model, fontsize=8)
            ax.set_xlabel("P(true)", fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(hist):]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=6)
        fig.suptitle("bench_ja churn (machine-generated definitions), P(true) by gold label, before temperature",
                     fontsize=9)
        fig.tight_layout()
        out = ROOT / "results" / "figures" / "bench_ja_churn_ptrue.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
