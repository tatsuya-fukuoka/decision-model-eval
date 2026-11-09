"""ステージ1の集計（指示書 §4、§5-5）。results/raw/ から再計算し results/tables/ に出力する。

    uv run python -m src.stage1_tables [--n-boot 1000]

S1MB：
- タスクごとの primary metric、基準値調整済みスコア（S1MB 公式 baseline-adjusted-v1 の式）
- 範囲：全27タスク / 重ならない17タスク / 重なる10タスク / 合成8タスク
- 型（Noul/Choice/Score）ごとの平均と Task Avg（3型の等重み平均）
- 95%信頼区間：タスク内で case を復元抽出する層化ブートストラップ（基準値はデータセット全体で固定）
bench_ja：
- sokudan と同じ定義の指標（確率下限 5e-5）と、件の復元抽出による 95% 信頼区間
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import bench_ja, s1mb
from .metrics import bench_ja_scores
from .metrics.s1mb_bootstrap import adjusted_scores, aggregate_boot, task_arrays
from .metrics.s1mb_scores import aggregate, task_scores
from .results import RAW, path_for, predictions_index, read_rows

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
SEED = 20261107


# 補助の範囲（common_complete_*）を決める基準のモデル：既定設定・公式カーネル問題の無い Tier 1 の6モデル。
# 参考値（d3 / Clef）や拡張設定を後から加えても、補助の範囲のタスク集合が変わらないよう固定する。
COMMON_BASIS = ["saihai-v1-310m", "saihai-v1-8b", "sokudan-ja-310m", "strands-decider-2b", "julia-1",
                "laya-multilingual"]


def model_labels() -> dict[str, dict]:
    """configs/models.yaml から、モデルごとの tier・設定（default / extended）・参考値の注記。"""
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    out = {}
    for tier in ("tier1", "tier2", "tier3"):
        for m in cfg.get(tier) or []:
            out[m["name"]] = {"tier": tier, "setting": "default", "base_model": m["name"],
                              "reference": m.get("reference", "")}
    for m in cfg.get("extended") or []:
        out[m["name"]] = {**out[m["base"]], "setting": "extended", "base_model": m["base"],
                          "extended": m.get("extended", "")}
    return out


def _label(model: str, labels: dict) -> dict:
    lab = labels.get(model, {"tier": "baseline" if model.startswith("baseline-") else "", "setting": "default",
                             "base_model": model, "reference": ""})
    return {"tier": lab["tier"], "setting": lab["setting"], "reference": lab.get("reference", "")}


def scopes() -> dict[str, list[str]]:
    overlap = json.loads((ROOT / "results" / "overlap_tasks.json").read_text(encoding="utf-8"))
    all27 = s1mb.load_category()
    ov = [t["task_id"] for t in overlap["overlap_tasks"]]
    return {
        "non_overlap_17": overlap["non_overlap_tasks"],
        "all_27": all27,
        "overlap_10": ov,
        "synthetic_8": [t for t in all27 if t.startswith("generalization-")],
    }


def load_s1mb():
    data = {}
    for t in s1mb.load_category():
        bench, _, targets = s1mb.load_benchmark(t)
        data[t] = (bench, targets)
    return data


def s1mb_model(model: str, data, scope_map, n_boot: int) -> tuple[list[dict], list[dict]]:
    types = {t: b.task for t, (b, _) in data.items()}
    per_task, task_rows, preds_by_task = {}, [], {}
    for t, (bench, targets) in data.items():
        path = path_for(model, "s1mb", t)
        preds = predictions_index(read_rows(path)) if path.is_file() else {}
        preds_by_task[t] = preds
        r = task_scores(bench.task, targets, preds)
        per_task[t] = r
        task_rows.append({"model": model, "task_id": t, "type": bench.task, "synthetic": bench.synthetic,
                          **{k: v for k, v in r.items() if not isinstance(v, (list, dict))}})
    # ブートストラップ：各反復でタスクごとに case を復元抽出し、全範囲の集計を同じ標本で計算する
    # （件ごとの寄与を配列化した高速版。task_scores との一致は tests/test_bootstrap_fast.py）
    rng = np.random.default_rng(SEED)
    boot: dict[str, dict[str, np.ndarray]] = {}
    # 完了したタスクだけを再標本化し、範囲内の全タスクが完了している範囲についてだけ区間を出す
    done = [t for t, r in per_task.items() if r["complete"]]
    if done and n_boot:
        arrays = {t: task_arrays(data[t][0].task, data[t][1], preds_by_task[t]) for t in done}
        scores = {t: adjusted_scores(a, rng.integers(0, a.n, (n_boot, a.n))) for t, a in arrays.items()}
        boot = {s: aggregate_boot(arrays, scores, tasks) for s, tasks in scope_map.items()
                if tasks and all(t in arrays for t in tasks)}
    summary = []
    for s, tasks in scope_map.items():
        agg = aggregate(per_task, types, tasks)
        row = {"model": model, "scope": s, "n_tasks": len(tasks), "incomplete_tasks": ";".join(agg["incomplete_tasks"])}
        for k in ("task_avg", "noul", "choice", "score"):
            row[k] = agg[k]
            vals = boot.get(s, {}).get(k)
            vals = [] if vals is None else vals[~np.isnan(vals)]
            if agg[k] is not None and len(vals):
                lo, hi = np.percentile(vals, [2.5, 97.5])
                row[f"{k}_ci_low"], row[f"{k}_ci_high"] = float(lo), float(hi)
            else:
                row[f"{k}_ci_low"] = row[f"{k}_ci_high"] = None
        summary.append(row)
    return task_rows, summary


def bench_ja_model(model: str, n_boot: int) -> list[dict]:
    out = []
    rng = np.random.default_rng(SEED)
    # 変種（VARIANT_TASKS）は主結果ではなく、説明文の言い換えによる差の確認用（variant 列で区別）
    for task, qtype in {**bench_ja.TASKS, **bench_ja.VARIANT_TASKS}.items():
        _, targets = bench_ja.load_task(task)
        path = path_for(model, "bench_ja", task)
        if not path.is_file():
            continue
        preds = predictions_index(read_rows(path))
        missing = [t for t in targets if (preds.get((t.case_id, t.question_id)) or {}).get("probs") is None]
        row = {"model": model, "task_id": task, "type": qtype, "variant": task in bench_ja.VARIANT_TASKS,
               "n_expected": len(targets), "n_failed": len(missing)}
        if missing:
            row["status"] = "incomplete"
            out.append(row)
            continue
        P = np.array([[preds[(t.case_id, t.question_id)]["probs"][o] for o in t.option_ids] for t in targets])
        y = np.array([int(np.argmax(t.probs)) for t in targets])
        r = bench_ja_scores.score_task(qtype, P, y)
        row.update({"status": "complete", **r})
        keys = ["accuracy", "macro_f1", "ece", "brier", "nll"] + (["rps"] if qtype == "score" else []) + (
            ["auroc"] if qtype == "noul" else [])
        samples = {k: [] for k in keys}
        for _ in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            rb = bench_ja_scores.score_task(qtype, P[idx], y[idx])
            for k in keys:
                if rb.get(k) is not None:
                    samples[k].append(rb[k])
        for k in keys:
            if samples[k]:
                lo, hi = np.percentile(samples[k], [2.5, 97.5])
                row[f"{k}_ci_low"], row[f"{k}_ci_high"] = float(lo), float(hi)
        out.append(row)
    return out


def _complete(model: str, task: str, targets) -> bool:
    path = path_for(model, "s1mb", task)
    if not path.is_file():
        return False
    preds = predictions_index(read_rows(path))
    return all((p := preds.get((t.case_id, t.question_id))) and (p["probs"] is not None or p["abstention"])
               for t in targets)


def extended_compare(models: list[str], labels: dict, data) -> list[dict]:
    """拡張設定と既定設定の件ごとの比較：既定で失敗し拡張で成功した件（入力上限の変更で結果が出た件）など。"""
    rows = []
    for ext in models:
        lab = labels.get(ext, {})
        if lab.get("setting") != "extended":
            continue
        base = lab["base_model"]
        for t, (bench, targets) in data.items():
            pb, pe = path_for(base, "s1mb", t), path_for(ext, "s1mb", t)
            if not (pb.is_file() and pe.is_file()):
                continue
            b, e = predictions_index(read_rows(pb)), predictions_index(read_rows(pe))
            ok = lambda d, x: (p := d.get((x.case_id, x.question_id))) is not None and p["probs"] is not None  # noqa: E731
            fb = {x.key for x in targets if not ok(b, x)}
            fe = {x.key for x in targets if not ok(e, x)}
            both = [x for x in targets if ok(b, x) and ok(e, x)]
            changed = sum(int(np.argmax([b[(x.case_id, x.question_id)]["probs"][o] for o in x.option_ids])
                              != np.argmax([e[(x.case_id, x.question_id)]["probs"][o] for o in x.option_ids]))
                          for x in both)
            rows.append({"model": ext, "base_model": base, "task_id": t, "type": bench.task, "n": len(targets),
                         "failed_default": len(fb), "failed_extended": len(fe), "recovered": len(fb - fe),
                         "newly_failed": len(fe - fb), "argmax_changed_on_common": changed,
                         "n_common": len(both), "extended": lab.get("extended", "")})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--models", nargs="*", help="省略時は results/raw/ の全モデル")
    args = ap.parse_args()
    # parity-* は公式評価器との一致確認用の人工予測で、比較表には載せない
    models = args.models or sorted(p.name for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("parity-"))
    data, scope_map = load_s1mb(), scopes()
    # 補助の範囲（公式の Task Avg ではない）：評価した全モデル（ベースライン除く）が完了したタスクだけ。
    # 入力上限による失敗でタスクが欠けるモデルどうしを、同じタスク集合で並べるため。
    labels = model_labels()
    basis = [m for m in COMMON_BASIS if (RAW / m / "s1mb").is_dir()]
    if basis:
        common = [t for t in data if all(_complete(m, t, data[t][1]) for m in basis)]
        scope_map["common_complete_non_overlap"] = [t for t in scope_map["non_overlap_17"] if t in common]
        scope_map["common_complete_all"] = [t for t in scope_map["all_27"] if t in common]
        print("common complete tasks:", len(scope_map["common_complete_non_overlap"]), "/ 17,",
              len(scope_map["common_complete_all"]), "/ 27", flush=True)
    out = {k: {"tasks": [], "summary": [], "bench": []} for k in ("default", "extended")}
    for m in models:
        lab = _label(m, labels)
        o = out[lab["setting"]]
        if (RAW / m / "s1mb").is_dir():
            t, s = s1mb_model(m, data, scope_map, args.n_boot)
            o["tasks"] += [{**r, **lab} for r in t]
            o["summary"] += [{**r, **lab, "common_basis": ";".join(basis)} for r in s]
        if (RAW / m / "bench_ja").is_dir():
            o["bench"] += [{**r, **lab} for r in bench_ja_model(m, args.n_boot)]
        print(f"done {m}", flush=True)
    # 既定設定（主表）。参考値（公式カーネル未使用）は reference 列で区別する
    write_csv(TABLES / "stage1_s1mb_tasks.csv", out["default"]["tasks"])
    write_csv(TABLES / "stage1_s1mb_summary.csv", out["default"]["summary"])
    write_csv(TABLES / "stage1_bench_ja.csv", out["default"]["bench"])
    # 拡張設定（入力上限を変えた追加実行。別表）
    if out["extended"]["tasks"]:
        write_csv(TABLES / "stage1_extended_s1mb_tasks.csv", out["extended"]["tasks"])
        write_csv(TABLES / "stage1_extended_s1mb_summary.csv", out["extended"]["summary"])
        write_csv(TABLES / "stage1_extended_bench_ja.csv", out["extended"]["bench"])
        write_csv(TABLES / "stage1_extended_compare.csv", extended_compare(models, labels, data))
    print(f"wrote {TABLES.relative_to(ROOT)}/stage1_*.csv")


if __name__ == "__main__":
    main()
