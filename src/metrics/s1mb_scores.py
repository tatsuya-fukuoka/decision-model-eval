"""S1MB の primary metric と基準値調整済みスコア（baseline-adjusted-v1）。

S1MB 公式評価器 evaluator/src/s1mb/metrics.py と diagnostics.py（MIT License,
Copyright (c) Yuichi Tateno, commit dbfc74699b814cadc8e3384ffa8705a39958cc91）の式を、
本リポジトリの Target / 予測 dict に合わせて移植したもの。式は変えていない。
公式実装と同じ値になることは tests/test_s1mb_official_parity.py（公式評価器が実行できる環境のみ）で確認する。

- Choice：primary = argmax に載った正解確率質量。基準値 = max(一様, 固定ID, 固定位置)。
- Noul：primary = binary Brier。調整スキル = 2*BA - 1（閾値 p(true) >= 0.5 固定）。
- Score：primary = 正規化期待値 MAE。基準値 = 正解期待値の中央値を出す定数予測の MAE。
- 調整スコア = スキルを [0, 1] にクリップ。タスク内平均 → 型ごとの等重み平均 → 3型の等重み平均（Task Avg）。
- 推論失敗（error）はスコアから除くが、そのタスクは incomplete とし、Task Avg は出さない。
- 棄権（abstention）は最大誤差として分母に残す。
"""

from __future__ import annotations

from statistics import median

from ..data.types import Target

EPSILON = 1e-12


def _value(t: Target, probs: dict[str, float]) -> float:
    assert t.values is not None
    low, high = min(t.values), max(t.values)
    return sum((v - low) / (high - low) * probs[k] for k, v in zip(t.option_ids, t.values, strict=True))


def task_scores(
    task: str,
    targets: list[Target],
    predictions: dict[tuple[str, str], dict],
    reference: list[Target] | None = None,
) -> dict:
    """predictions: (case_id, question_id) -> {"probs": {id: p} | None, "error": str|None, "abstention": str|None}

    reference：基準値・有病率を計算する母集団（既定は targets）。ブートストラップではデータセット全体を渡し、
    基準値を固定したまま採点対象だけを再標本化する（S1MB：基準値はデータセット全体で固定）。
    """
    entries = [(t, dict(zip(t.option_ids, t.probs, strict=True))) for t in targets]
    ref = entries if reference is None else [(t, dict(zip(t.option_ids, t.probs, strict=True))) for t in reference]
    rows, abstentions, failed, missing = [], [], 0, 0
    for t, tp in entries:
        pred = predictions.get((t.case_id, t.question_id))
        if pred is None:
            missing += 1
        elif pred.get("abstention"):
            abstentions.append((t, tp))
        elif pred.get("probs") is None:
            failed += 1
        else:
            rows.append((t, tp, pred["probs"]))
    n = len(rows) + len(abstentions)
    out: dict = {
        "n_expected": len(entries), "n_scored": n, "n_failed": failed, "n_missing": missing,
        "n_abstained": len(abstentions), "complete": failed == 0 and missing == 0,
    }
    skill = None
    if task == "noul":
        prevalence = sum(tp["true"] for _, tp in ref) / len(ref)
        tp_ = sum(t_["true"] for _, t_, p in rows if p["true"] >= 0.5)
        fp_ = sum(t_["false"] for _, t_, p in rows if p["true"] >= 0.5)
        fn_ = sum(t_["true"] for _, t_, p in rows if p["true"] < 0.5)
        tn_ = sum(t_["false"] for _, t_, p in rows if p["true"] < 0.5)
        fn_ += sum(t_["true"] for _, t_ in abstentions)
        fp_ += sum(t_["false"] for _, t_ in abstentions)
        recall = tp_ / (tp_ + fn_) if tp_ + fn_ > EPSILON else None
        spec = tn_ / (tn_ + fp_) if tn_ + fp_ > EPSILON else None
        ba = (recall + spec) / 2 if recall is not None and spec is not None else None
        brier = (sum((p["true"] - t_["true"]) ** 2 for _, t_, p in rows) + len(abstentions)) / n if n else None
        skill = 2 * ba - 1 if ba is not None else None
        out.update({
            "primary_metric": "binary_brier", "primary": brier, "accuracy": (tp_ + tn_) / n if n else None,
            "balanced_accuracy": ba,
            "f1": 2 * tp_ / (2 * tp_ + fp_ + fn_) if 2 * tp_ + fp_ + fn_ > EPSILON else None,
            "positive_prevalence": prevalence,
        })
        eligible = EPSILON < prevalence < 1 - EPSILON
    elif task == "choice":
        uniform = sum(1 / len(t.option_ids) for t, _ in ref) / len(ref)
        ids = {o for t, _ in ref for o in t.option_ids}
        fixed_id = max(sum(tp.get(k, 1 / len(t.option_ids)) for t, tp in ref) / len(ref) for k in ids)
        max_k = max(len(t.option_ids) for t, _ in ref)
        fixed_pos = max(
            sum(tp[t.option_ids[i]] if i < len(t.option_ids) else 1 / len(t.option_ids) for t, tp in ref)
            / len(ref)
            for i in range(max_k)
        )
        baseline = max(uniform, fixed_id, fixed_pos)
        acc = sum(tp[max(t.option_ids, key=lambda o: p[o])] for t, tp, p in rows) / n if n else None
        skill = (acc - baseline) / (1 - baseline) if acc is not None and baseline < 1 - EPSILON else None
        out.update({
            "primary_metric": "target_mass_at_prediction", "primary": acc, "accuracy": acc,
            "uniform_accuracy_baseline": uniform, "fixed_answer_accuracy_baseline": baseline,
        })
        eligible = baseline < 1 - EPSILON
    else:
        tvals = [_value(t, tp) for t, tp in ref]
        constant = median(tvals)
        baseline = sum(abs(v - constant) for v in tvals) / len(tvals)
        errors = []
        for t, tp, p in rows:
            assert t.values is not None
            errors.append(
                sum(v * (p[k] - tp[k]) for k, v in zip(t.option_ids, t.values, strict=True))
                / (max(t.values) - min(t.values))
            )
        errors.extend([1.0] * len(abstentions))
        mae = sum(abs(e) for e in errors) / n if n else None
        skill = 1 - mae / baseline if baseline > EPSILON and mae is not None else None
        out.update({
            "primary_metric": "normalized_expected_score_mae", "primary": mae,
            "normalized_score_rmse": (sum(e * e for e in errors) / n) ** 0.5 if n else None,
            "constant_mae_baseline": baseline,
        })
        eligible = baseline > EPSILON
    # 適格性（SCORING.md）：データセット全体の正解で決まり、全モデル共通。不適格タスクは調整平均から除くが、
    # 完了していることは引き続き必要（submission_report.task_score と同じ）。
    out["eligible"] = eligible
    out["baseline_adjusted_skill"] = skill
    out["baseline_adjusted_score"] = max(0.0, min(1.0, skill)) if skill is not None else None
    return out


def aggregate(per_task: dict[str, dict], task_types: dict[str, str], tasks: list[str]) -> dict:
    """指定タスク集合について、型ごとの平均と Task Avg（3型の等重み平均）を返す。

    1タスクでも incomplete なら、その型と Task Avg は None（分母を減らして良く見せない）。
    不適格（target-degenerate）なタスクは、完了を要求したうえで型平均から除く（公式 task_score と同じ）。
    適格タスクの調整スコアが未定義なら、その型は None。
    """
    by_type: dict[str, list] = {"noul": [], "choice": [], "score": []}
    incomplete = []
    for t in tasks:
        r = per_task.get(t)
        by_type[task_types[t]].append(t)
        if r is None or not r["complete"]:
            incomplete.append(t)
    means = {}
    for k, ts in by_type.items():
        if not ts or any(t in incomplete for t in ts):
            means[k] = None
            continue
        vals = [per_task[t]["baseline_adjusted_score"] for t in ts if per_task[t]["eligible"]]
        means[k] = None if (not vals or any(v is None for v in vals)) else sum(vals) / len(vals)
    present = [k for k, v in by_type.items() if v]
    task_avg = None if any(means[k] is None for k in present) else sum(means[k] for k in present) / len(present)
    return {
        "n_tasks": len(tasks), "types_present": present,
        "noul": means["noul"], "choice": means["choice"], "score": means["score"],
        "task_avg": task_avg, "incomplete_tasks": incomplete,
    }
