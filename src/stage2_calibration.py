"""ステージ2：確率の質（指示書 §5-2、§5-5）。results/raw/ から計算し results/tables/、results/figures/ に出力する。

    uv run python -m src.stage2_calibration [--n-boot 1000]

手順：
1. S1MB の各タスクを group_id 単位で calibration 30% / test 70% に分ける（split_groups、seed 固定）。
   分割は results/splits/s1mb_calibration_split.json に保存する（group_id と側だけ）。
2. 型（choice / noul / score）ごとに 1 つのスカラー温度 T を、重ならない17タスクの calibration 側だけで
   NLL 最小化により求める（p^(1/T) を再正規化。ソフトターゲット）。test には触れない。
3. 温度合わせ前の指標は全データ、合わせた後の指標は test 70%。比較用に「合わせる前・test 70%」も出す。
   母集団は population 列に書く。
4. 失敗（確率が無い件、abstention）は補完しない。除外件数を n_excluded 列に書く。
5. bench_ja には S1MB で求めた型ごとの温度をそのまま適用する（bench_ja では温度を求めない）。母集団は全件。

指標：Accuracy（argmax に載った正解確率質量）、NLL、多クラス Brier、ECE（top-label 信頼度、15等幅ビン、
最初のビンだけ 0 を含み他は (lo, hi]）、選択的予測（信頼度の高い順に採用したときの coverage と正解率、
目標正解率 0.90 / 0.95 を満たす最大 coverage）。95% 区間は件の復元抽出 1,000 回（範囲内のタスクで層化）。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration.temperature import apply_temperature, fit_temperature, split_groups
from .data import bench_ja, s1mb
from .metrics.core import ECE_BINS, EPS
from .results import RAW, path_for, predictions_index, read_rows
from .stage1_tables import SEED, scopes

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures" / "reliability"
SPLIT_PATH = ROOT / "results" / "splits" / "s1mb_calibration_split.json"
TYPES = ("noul", "choice", "score")
TARGET_ACCS = (0.90, 0.95)


@dataclass
class Rows:
    """1 モデル・1 タスクの、確率がある件だけの配列（K が件ごとに違うので行ごとのリスト）。"""

    task_id: str
    qtype: str
    probs: list[np.ndarray]
    targets: list[np.ndarray]
    side: np.ndarray  # "calibration" / "test"
    n_expected: int
    n_excluded: int


def row_stats(probs: list[np.ndarray], targets: list[np.ndarray]) -> dict[str, np.ndarray]:
    """件ごとの寄与（平均すると core.accuracy / nll / brier と一致する）と top-label 信頼度。"""
    conf, corr, nll, brier = [], [], [], []
    for p, t in zip(probs, targets, strict=True):
        k = int(np.argmax(p))
        conf.append(float(p[k]))
        corr.append(float(t[k]))
        nll.append(float(-(t * np.log(np.clip(p, EPS, 1.0))).sum()))
        brier.append(float(((p - t) ** 2).sum()))
    return {k: np.asarray(v) for k, v in
            {"confidence": conf, "correct": corr, "nll": nll, "brier": brier}.items()}


def ece_from(conf: np.ndarray, corr: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """core.ece と同じビン（最初のビンは [0, 1/15]、以降 (lo, hi]）を searchsorted で求める。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges[1:], conf, side="left"), 0, n_bins - 1)
    n = np.bincount(idx, minlength=n_bins)
    sc = np.bincount(idx, weights=conf, minlength=n_bins)
    sa = np.bincount(idx, weights=corr, minlength=n_bins)
    nz = n > 0
    return float(np.sum(np.abs(sa[nz] - sc[nz])) / n.sum())


def max_coverage(conf: np.ndarray, corr: np.ndarray, target: float) -> float:
    """core.max_coverage_at と同じ（同値の信頼度は分割しない）。"""
    order = np.argsort(-conf, kind="mergesort")
    c, a = conf[order], np.cumsum(corr[order]) / np.arange(1, len(conf) + 1)
    end = np.r_[c[1:] != c[:-1], True]
    ok = end & (a >= target - 1e-12)
    return float((np.nonzero(ok)[0].max() + 1) / len(conf)) if ok.any() else 0.0


def summarize(stats: dict[str, np.ndarray], groups: list[np.ndarray], n_boot: int, rng) -> dict:
    """stats：連結した件ごとの値。groups：層（タスク）ごとの添字。"""
    conf, corr = stats["confidence"], stats["correct"]
    out = {"n": int(len(conf)), "accuracy": float(corr.mean()), "nll": float(stats["nll"].mean()),
           "brier": float(stats["brier"].mean()), "ece": ece_from(conf, corr),
           "mean_confidence": float(conf.mean())}
    for t in TARGET_ACCS:
        out[f"max_coverage_at_{t:.2f}"] = max_coverage(conf, corr, t)
    if n_boot:
        samples = {k: [] for k in ("accuracy", "nll", "brier", "ece")}
        for _ in range(n_boot):
            idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
            samples["accuracy"].append(corr[idx].mean())
            samples["nll"].append(stats["nll"][idx].mean())
            samples["brier"].append(stats["brier"][idx].mean())
            samples["ece"].append(ece_from(conf[idx], corr[idx]))
        for k, v in samples.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            out[f"{k}_ci_low"], out[f"{k}_ci_high"] = float(lo), float(hi)
    return out


def build_split(data) -> dict[str, dict[str, str]]:
    split = {t: split_groups([x.group_id for x in targets], t, seed=SEED) for t, (_, targets) in data.items()}
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps({"seed": SEED, "calib_fraction": 0.3, "unit": "group_id",
                                      "method": "src.calibration.temperature.split_groups", "tasks": split},
                                     ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return split


def model_rows(model: str, data, split) -> dict[str, Rows]:
    out = {}
    for t, (bench, targets) in data.items():
        path = path_for(model, "s1mb", t)
        preds = predictions_index(read_rows(path)) if path.is_file() else {}
        probs, tg, side = [], [], []
        for x in targets:
            p = preds.get((x.case_id, x.question_id))
            if not p or p.get("probs") is None:
                continue
            probs.append(np.array([p["probs"][o] for o in x.option_ids], dtype=np.float64))
            tg.append(np.asarray(x.probs, dtype=np.float64))
            side.append(split[t][x.group_id])
        out[t] = Rows(t, bench.task, probs, tg, np.asarray(side), len(targets), len(targets) - len(probs))
    return out


def fit_model_temperatures(rows: dict[str, Rows], fit_tasks: list[str]) -> dict[str, dict]:
    temps = {}
    for qt in TYPES:
        P, T = [], []
        for t in fit_tasks:
            r = rows[t]
            if r.qtype != qt:
                continue
            m = r.side == "calibration"
            P += [p for p, keep in zip(r.probs, m, strict=True) if keep]
            T += [y for y, keep in zip(r.targets, m, strict=True) if keep]
        temps[qt] = fit_temperature(P, T) if P else {"temperature": None, "n": 0}
        temps[qt]["fit_tasks"] = sorted(t for t in fit_tasks if rows[t].qtype == qt)
        temps[qt]["n_excluded_in_fit_tasks"] = sum(rows[t].n_excluded for t in fit_tasks if rows[t].qtype == qt)
    return temps


def _select(rows: dict[str, Rows], tasks: list[str], qtype: str | None, side: str | None, temp: dict | None):
    probs, targets, groups, n_excl, n_exp, offset = [], [], [], 0, 0, 0
    for t in tasks:
        r = rows[t]
        if qtype and r.qtype != qtype:
            continue
        n_excl += r.n_excluded
        n_exp += r.n_expected
        keep = np.ones(len(r.probs), bool) if side is None else (r.side == side)
        ps = [p for p, k in zip(r.probs, keep, strict=True) if k]
        if temp is not None:
            ps = [apply_temperature(p, temp[r.qtype]["temperature"]) for p in ps]
        probs += ps
        targets += [y for y, k in zip(r.targets, keep, strict=True) if k]
        groups.append(np.arange(offset, offset + len(ps)))
        offset += len(ps)
    return probs, targets, [g for g in groups if len(g)], n_excl, n_exp


POPULATIONS = (
    ("before", "all", None),          # 温度合わせ前・全データ
    ("before", "test70", "test"),     # 温度合わせ前・test 70%（合わせた後との比較用）
    ("after", "test70", "test"),      # 温度合わせ後・test 70%
)


def s1mb_tables(model: str, rows, temps, scope_map, n_boot, curves) -> list[dict]:
    out = []
    rng = np.random.default_rng(SEED)
    for scope, tasks in scope_map.items():
        for qt in (None, *TYPES):
            for phase, pop, side in POPULATIONS:
                temp = temps if phase == "after" else None
                if temp is not None and any(temps[q]["temperature"] is None for q in TYPES
                                            if any(rows[t].qtype == q for t in tasks)):
                    continue
                probs, targets, groups, n_excl, n_exp = _select(rows, tasks, qt, side, temp)
                row = {"model": model, "dataset": "s1mb", "scope": scope, "type": qt or "all", "phase": phase,
                       "population": pop, "n_tasks": len({t for t in tasks if qt in (None, rows[t].qtype)}),
                       "n_expected": n_exp, "n_excluded": n_excl}
                if probs:
                    st = row_stats(probs, targets)
                    row.update(summarize(st, groups, n_boot, rng))
                    if scope == "non_overlap_17" and pop == "test70":
                        curves.append((model, "s1mb", scope, qt or "all", phase, pop, st))
                out.append(row)
    return out


def bench_ja_tables(model: str, temps, n_boot, curves) -> list[dict]:
    out = []
    rng = np.random.default_rng(SEED)
    for task, qtype in {**bench_ja.TASKS, **bench_ja.VARIANT_TASKS}.items():
        path = path_for(model, "bench_ja", task)
        if not path.is_file():
            continue
        _, targets = bench_ja.load_task(task)
        preds = predictions_index(read_rows(path))
        P, T = [], []
        for x in targets:
            p = preds.get((x.case_id, x.question_id))
            if p and p.get("probs") is not None:
                P.append(np.array([p["probs"][o] for o in x.option_ids], dtype=np.float64))
                T.append(np.asarray(x.probs, dtype=np.float64))
        for phase in ("before", "after"):
            row = {"model": model, "dataset": "bench_ja", "scope": task, "type": qtype, "phase": phase,
                   "population": "all", "variant": task in bench_ja.VARIANT_TASKS, "n_tasks": 1,
                   "n_expected": len(targets), "n_excluded": len(targets) - len(P)}
            t = temps.get(qtype, {}).get("temperature")
            if phase == "after":
                if t is None:
                    out.append(row)
                    continue
                row["temperature_from"] = "s1mb non_overlap_17 calibration side"
            ps = [apply_temperature(p, t) for p in P] if phase == "after" else P
            if ps:
                st = row_stats(ps, T)
                row.update(summarize(st, [np.arange(len(ps))], n_boot, rng))
                if task in bench_ja.TASKS:
                    curves.append((model, "bench_ja", task, qtype, phase, "all", st))
            out.append(row)
    return out


def reliability_figure(model: str, curves) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [c for c in curves if c[0] == model and c[1] == "s1mb" and c[3] in TYPES]
    if not sel:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    edges = np.linspace(0, 1, ECE_BINS + 1)
    for ax, qt in zip(axes, TYPES, strict=True):
        ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
        for _, _, _, q, phase, _, st in sel:
            if q != qt:
                continue
            idx = np.clip(np.searchsorted(edges[1:], st["confidence"], side="left"), 0, ECE_BINS - 1)
            n = np.bincount(idx, minlength=ECE_BINS)
            nz = n > 0
            mc = np.bincount(idx, st["confidence"], ECE_BINS)[nz] / n[nz]
            ma = np.bincount(idx, st["correct"], ECE_BINS)[nz] / n[nz]
            ax.plot(mc, ma, marker="o", ms=3, label=f"{'before' if phase == 'before' else 'after'} T "
                    f"(ECE {ece_from(st['confidence'], st['correct']):.3f})")
        ax.set_title(f"{qt} (S1MB non-overlap 17, test 70%)", fontsize=9)
        ax.set_xlabel("confidence (top label)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("accuracy (target mass at argmax)")
    fig.suptitle(model, fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{model}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def coverage_rows(curves) -> list[dict]:
    """coverage–accuracy 曲線を 0.05 刻みの coverage で間引いて保存する。"""
    out = []
    for model, ds, scope, qt, phase, pop, st in curves:
        conf, corr = st["confidence"], st["correct"]
        order = np.argsort(-conf, kind="mergesort")
        acc = np.cumsum(corr[order]) / np.arange(1, len(conf) + 1)
        for cov in np.round(np.arange(0.05, 1.0001, 0.05), 2):
            k = max(int(np.ceil(cov * len(conf))), 1)
            out.append({"model": model, "dataset": ds, "scope": scope, "type": qt, "phase": phase,
                        "population": pop, "coverage": float(cov), "n_accepted": k,
                        "threshold": float(conf[order][k - 1]), "accuracy": float(acc[k - 1])})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--models", nargs="*")
    args = ap.parse_args()
    data = {t: s1mb.load_benchmark(t)[::2] for t in s1mb.load_category()}  # (bench, targets)
    split = build_split(data)
    scope_map = scopes()
    fit_tasks = scope_map["non_overlap_17"]
    models = args.models or sorted(p.name for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("parity-")
                                   and (p / "s1mb").is_dir())
    cal, temps_out, curves = [], {}, []
    for model in models:
        rows = model_rows(model, data, split)
        temps = fit_model_temperatures(rows, fit_tasks)
        temps_out[model] = temps
        cal += s1mb_tables(model, rows, temps, scope_map, args.n_boot, curves)
        cal += bench_ja_tables(model, temps, args.n_boot, curves)
        fig = reliability_figure(model, curves)
        print(f"{model}: T=" + ", ".join(f"{q}={temps[q]['temperature']:.3f}" if temps[q]["temperature"] else f"{q}=-"
                                        for q in TYPES) + (f"  fig={fig.relative_to(ROOT)}" if fig else ""), flush=True)
    write_csv(TABLES / "stage2_calibration.csv", cal)
    write_csv(TABLES / "stage2_coverage_curves.csv", coverage_rows(curves))
    (TABLES / "stage2_temperatures.json").write_text(
        json.dumps(temps_out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
