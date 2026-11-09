"""report.md の数値表を results/ の CSV・JSON から生成する（指示書 §8）。

    uv run python -m src.make_report

- 本文（事実と解釈の文章）は docs/report_template.md に書き、表は `{{table:<名前>}}` の位置に差し込む。
  数値はすべてこのスクリプトが results/ から読む。手で数値を書き写さない。
- 値が無いもの（測れなかった、未定義）は「未測定」「未定義」と書く。
- 95% 区間は各 CSV の *_ci_low / *_ci_high（ブートストラップ 1,000 回）。
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TABLES = RES / "tables"
TEMPLATE = ROOT / "docs" / "report_template.md"
OUT = ROOT / "report.md"
HF = ROOT / "hf_cache" / "hub"

# 表に載せる順（既定設定）。ベースラインは最後。
ORDER = ["saihai-v1-8b", "saihai-v1-310m", "sokudan-ja-310m", "strands-decider-2b", "julia-1", "laya-multilingual",
         "d3-nano", "d3-mini", "clef-flash", "d3-lite", "decision-2.0-nox-4b", "d3-flash", "decision-2.0-lux-9b",
         "baseline-majority", "baseline-random"]
EXT = ["saihai-v1-310m__ext", "sokudan-ja-310m__ext", "strands-decider-2b__ext", "julia-1__ext", "laya-multilingual__ext"]
REF = {"d3-nano", "d3-mini", "clef-flash", "d3-lite", "d3-flash"}  # Hub カーネルなし（参考値）
TIER = {"d3-lite": "2", "decision-2.0-nox-4b": "2", "d3-flash": "3", "decision-2.0-lux-9b": "3"}

# パラメータ数を数える safetensors（固定リビジョンのスナップショット内）。既定は全 *.safetensors。
PARAM_FILES = {
    # 推論用（0_BekkoInference）と checkpoint/ に同じ重みが2組入っているため、推論用だけを数える
    "saihai-v1-310m": [("hotchpotch/saihai-v1-310m", None, ["0_BekkoInference/model.safetensors"])],
    # LoRA アダプタ＋基盤（provenance.json が指す Qwen3.5-2B-Base のリビジョン）
    "strands-decider-2b": [("StrandsAgents/strands-decider-2B-hobson-v19", None, None), ("Qwen/Qwen3.5-2B-Base", "*", None)],
}


def label(m: str) -> str:
    s = m.replace("__ext", "（拡張）")
    if m in REF:
        s += " ※"
    return s


def f3(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "未定義"
    return f"{x:.{nd}f}"


def ci(row, col, nd=3) -> str:
    v = row.get(col)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "未定義"
    lo, hi = row.get(f"{col}_ci_low"), row.get(f"{col}_ci_high")
    if lo is None or hi is None or any(isinstance(z, float) and math.isnan(z) for z in (lo, hi)):
        return f3(v, nd)
    return f"{v:.{nd}f}（{lo:.{nd}f}–{hi:.{nd}f}）"


def md(header: list[str], rows: list[list], align: str | None = None) -> str:
    al = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else "---" for a in al]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def read(name: str) -> pd.DataFrame | None:
    p = TABLES / f"{name}.csv"
    return pd.read_csv(p) if p.is_file() else None


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()}
            for r in df.to_dict("records")]


# ---------------------------------------------------------------- 実施概要・環境

def load_meta():
    manifest = json.loads((RES / "manifest.json").read_text())
    status = json.loads((RES / "model_status.json").read_text())
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    return manifest, status, cfg


def snapshot(hf_id: str, rev: str | None) -> Path | None:
    base = HF / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None
    if rev and rev != "*":
        p = base / rev
        return p if p.is_dir() else None
    snaps = sorted(base.iterdir())
    return snaps[0] if len(snaps) == 1 else None


def count_params(files: list[Path]) -> int:
    n = 0
    for f in files:
        with f.open("rb") as fh:
            h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, t in h.items():
            if k != "__metadata__":
                n += math.prod(t["shape"])
    return n


def params_for(name: str, hf_id: str, rev: str) -> int | None:
    specs = PARAM_FILES.get(name, [(hf_id, None, None)])
    total = 0
    for repo, r, files in specs:
        snap = snapshot(repo, rev if r is None else r)
        if snap is None:
            return None
        paths = [snap / f for f in files] if files else sorted(snap.rglob("*.safetensors"))
        if not paths or not all(p.is_file() for p in paths):
            return None
        total += count_params(paths)
    return total


def t_models(manifest, status, cfg) -> str:
    rows = []
    for name in ORDER[:-2] + ["h2o-lightning-4b"]:
        m = manifest["models"].get(name, {})
        hub = m.get("hub") or {}
        st = status.get(name, {})
        rev = hub.get("revision") or ""
        p = params_for(name, m.get("hf_id", ""), rev) if rev else None
        state = {"ok": "完了", "partial": "一部失敗", "skipped": "スキップ"}.get(st.get("status"), st.get("status") or "—")
        how = "Hub カーネルなし（参考値）" if name in REF else ("自作アダプタ" if m.get("runner") == "custom" else "公式アダプタ")
        if name == "h2o-lightning-4b":
            how = "—"
        rows.append([label(name), f"`{m.get('hf_id', '')}`", TIER.get(name, "1"), f"`{rev[:12]}`", hub.get("license") or "—",
                     how, state,
                     st.get("s1mb_failed_decisions", "—") if st.get("status") != "skipped" else "—",
                     st.get("bench_ja_failed", "—") if st.get("status") != "skipped" else "—",
                     f"{hub['weight_file_bytes'] / 1e9:.2f}" if hub.get("weight_file_bytes") else "—",
                     f"{p / 1e9:.2f}" if p else "未測定"])
    return md(["モデル", "HF ID", "Tier", "リビジョン", "ライセンス（Hub）", "実行経路", "状態", "S1MB 失敗", "bench_ja 失敗",
               "重みファイル（GB）", "パラメータ数（B）"], rows, "lllllllrrrr")


def t_excluded(manifest, status, cfg) -> str:
    rows = [[e["name"], e["reason"]] for e in manifest.get("excluded_models", [])]
    rows.append(["H2O-Lightning-4B", status.get("h2o-lightning-4b", {}).get("reason", "スキップ")])
    for tier in ("tier2", "tier3"):
        for m in cfg.get(tier, []):
            if m.get("status") == "not_run":
                rows.append([m["name"], m.get("reason", "")])
    return md(["モデル", "除外・未実行の理由"], rows, "ll")


def t_env(manifest) -> str:
    e = manifest["environment"]
    rows = [["OS / カーネル", f"{e['os']} / {e['kernel']}"], ["CPU", f"{e['cpu_model']}（{e['cpu_count']} スレッド）"],
            ["メモリ", e["memory_total"]], ["GPU / ドライバ", f"{e['gpu']}（CUDA {e['cuda_driver_version']}）"],
            ["GPU の共有", f"別ジョブと共有（記録時 {e['gpu_usage_at_record_time']}）。精度の評価だけを行い、レイテンシ・VRAM は測っていない"],
            ["Python / uv", f"{e['python']} / {e['uv']}"], ["乱数の seed", str(e["seed"])],
            ["HF トークン", "使用していない" if not e.get("hf_token_used") else "使用"]]
    for k, v in manifest.get("third_party", {}).items():
        rows.append([f"外部コード：{k}", f"`{v.get('commit', v.get('revision', ''))}` {v.get('url', '')}".strip()])
    for k, v in manifest["datasets"].items():
        rows.append([f"データ：{k}", f"`{(v.get('revision') or '')[:12]}`、ライセンス {v.get('license_name') or v.get('license')}"])
    return md(["項目", "値"], rows, "ll")


def t_envs(manifest) -> str:
    rows = []
    for name, v in manifest["environment"].get("envs", {}).items():
        pk = {p.split("==")[0].split(" @")[0]: p for p in v.get("packages", [])}
        keys = [k for k in ("torch", "transformers", "s1mb", "laya", "sokudan", "strands-decider", "vllm") if k in pk]
        rows.append([f"`envs/{name}`", ", ".join(pk[k].replace(" @ file://", " (local) ") for k in keys) or "—"])
    return md(["隔離環境", "主なパッケージ"], rows, "ll")


# ---------------------------------------------------------------- ステージ1

SCOPE_LABEL = {"non_overlap_17": "重ならない17タスク（主比較）", "all_27": "全27タスク", "overlap_10": "重なる10タスク（参考）",
               "synthetic_8": "合成8タスク", "common_complete_non_overlap": "17タスクのうち共通完了タスク（補助）",
               "common_complete_all": "27タスクのうち共通完了タスク（補助）"}


def t_s1mb(scope: str, ext: bool = False) -> str:
    d = read("stage1_extended_s1mb_summary" if ext else "stage1_s1mb_summary")
    if d is None:
        return "（表なし）"
    d = d[d.scope == scope]
    by = {r["model"]: r for r in records(d)}
    rows = []
    for m in (EXT if ext else ORDER):
        r = by.get(m)
        if not r:
            continue
        inc = r.get("incomplete_tasks")
        n_inc = len(str(inc).split(";")) if inc else 0
        rows.append([label(m), r["n_tasks"], n_inc, ci(r, "task_avg"), ci(r, "noul"), ci(r, "choice"), ci(r, "score")])
    return md(["モデル", "タスク数", "未完了タスク", "Task Avg（95%区間）", "Noul", "Choice", "Score"], rows)


def t_common_tasks() -> str:
    d = read("stage1_s1mb_summary")
    out = []
    for s in ("common_complete_non_overlap", "common_complete_all"):
        r = d[d.scope == s].iloc[0]
        out.append(f"- {SCOPE_LABEL[s]}：{int(r['n_tasks'])} タスク（基準の6モデル：{r['common_basis'].replace(';', '、')}）")
    return "\n".join(out)


def t_failed_tasks() -> str:
    d = read("stage1_s1mb_tasks")
    d = d[(d.n_failed > 0) & d.model.isin(ORDER)]
    piv = d.pivot_table(index="task_id", columns="model", values="n_failed", aggfunc="sum").fillna(0).astype(int)
    cols = [m for m in ORDER if m in piv.columns]
    rows = [[t, *[(piv.loc[t, c] or "") for c in cols]] for t in piv.index]
    return md(["タスク", *[label(c) for c in cols]], rows)


BJ = [("bench_ja-department", "accuracy", "部署（Choice）正解率"), ("bench_ja-department", "macro_f1", "部署 macro-F1"),
      ("bench_ja-urgency", "accuracy", "緊急度（Score）正解率"), ("bench_ja-urgency", "rps", "緊急度 RPS（低いほど良い）"),
      ("bench_ja-churn", "accuracy", "解約示唆（Noul）正解率"), ("bench_ja-churn", "auroc", "解約示唆 AUROC"),
      ("bench_ja-churn-para1", "auroc", "解約示唆（言い換え版）AUROC")]


def t_bench_ja(ext: bool = False) -> str:
    d = read("stage1_extended_bench_ja" if ext else "stage1_bench_ja")
    by = {(r["model"], r["task_id"]): r for r in records(d)}
    rows = []
    for m in (EXT if ext else ORDER):
        if not any(k[0] == m for k in by):
            continue
        row = [label(m)]
        for task, col, _ in BJ:
            r = by.get((m, task))
            row.append(ci(r, col) if r else "—")
        rows.append(row)
    return md(["モデル", *[h for _, _, h in BJ]], rows)


def t_bench_ja_failed() -> str:
    d = read("stage1_bench_ja")
    d = d[d.model.isin(ORDER)]
    n = int(d.n_failed.fillna(0).sum())
    return f"bench_ja の失敗件数（全モデル・全設問の合計）：{n} 件。"


def t_churn() -> str:
    d = read("bench_ja_churn_distribution")
    d = d[d.task_id == "bench_ja-churn"]
    by = {(r["model"], r["gold"]): r for r in records(d)}
    rows = []
    for m in ORDER:
        a = by.get((m, "all"))
        if not a:
            continue
        t, f = by.get((m, "true"), {}), by.get((m, "false"), {})
        rows.append([label(m), a["n"], a["n_excluded"], f3(a.get("auroc")), f3(a.get("mean_p_true")),
                     f3(a.get("share_p_true_ge_0.5")),
                     f"{t.get('n', 0)} / {f3(t.get('p50'))} / {f3(t.get('share_p_true_ge_0.5'))}",
                     f"{f.get('n', 0)} / {f3(f.get('p50'))} / {f3(f.get('share_p_true_ge_0.5'))}"])
    return md(["モデル", "件数", "除外", "AUROC", "平均 P(true)", "P(true)≥0.5 の割合",
               "正解 true：件数 / 中央値 / ≥0.5 の割合", "正解 false：件数 / 中央値 / ≥0.5 の割合"], rows)


def t_ext_compare() -> str:
    d = read("stage1_extended_compare")
    g = d.groupby("model")[["failed_default", "failed_extended", "recovered", "newly_failed",
                             "argmax_changed_on_common", "n_common"]].sum()
    setting = d.groupby("model")["extended"].first()
    rows = [[label(m), setting[m], *[int(g.loc[m, c]) for c in g.columns]] for m in EXT if m in g.index]
    return md(["拡張設定", "設定値", "既定で失敗", "拡張で失敗", "回復", "新たな失敗", "共通完了件で argmax が変化",
               "共通完了件数"], rows, "llrrrrrr")


# ---------------------------------------------------------------- ステージ2

def t_temperatures() -> str:
    t = json.loads((TABLES / "stage2_temperatures.json").read_text())
    rows = []
    for m in ORDER:
        if m not in t:
            continue
        row = [label(m)]
        for typ in ("noul", "choice", "score"):
            v = t[m].get(typ)
            row.append("—" if not v else f"{v['temperature']:.3f}{'（上限）' if v.get('at_bound') else ''}"
                                         f"（n={v['n']}, 除外 {v.get('n_excluded_in_fit_tasks', 0)}）")
        rows.append(row)
    return md(["モデル", "T（Noul）", "T（Choice）", "T（Score）"], rows)


def t_calib(dataset: str, scope: str, typ: str) -> str:
    d = read("stage2_calibration")
    d = d[(d.dataset == dataset) & (d.scope == scope) & (d.type == typ)]
    by = {(r["model"], r["phase"], r["population"]): r for r in records(d)}
    rows = []
    pops = [("before", "test70"), ("after", "test70")] if dataset == "s1mb" else [("before", "all"), ("after", "all")]
    for m in ORDER:
        b, a = by.get((m, *pops[0])), by.get((m, *pops[1]))
        if not b:
            continue
        rows.append([label(m), b["n"], b["n_excluded"], ci(b, "nll"), ci(a, "nll") if a else "—", ci(b, "brier"),
                     ci(a, "brier") if a else "—", ci(b, "ece"), ci(a, "ece") if a else "—"])
    return md(["モデル", "件数", "除外", "NLL 前", "NLL 後", "Brier 前", "Brier 後", "ECE 前", "ECE 後"], rows)


def t_coverage(dataset: str, scope: str, typ: str) -> str:
    d = read("stage2_calibration")
    d = d[(d.dataset == dataset) & (d.scope == scope) & (d.type == typ)]
    by = {(r["model"], r["phase"], r["population"]): r for r in records(d)}
    rows = []
    pops = [("before", "test70"), ("after", "test70")] if dataset == "s1mb" else [("before", "all"), ("after", "all")]
    for m in ORDER:
        b, a = by.get((m, *pops[0])), by.get((m, *pops[1]))
        if not b:
            continue
        rows.append([label(m), b["n"], ci(b, "accuracy"), f3(b.get("max_coverage_at_0.90")),
                     f3(a.get("max_coverage_at_0.90")) if a else "—", f3(b.get("max_coverage_at_0.95")),
                     f3(a.get("max_coverage_at_0.95")) if a else "—"])
    return md(["モデル", "件数", "正解率（全件）", "正解率0.90での最大カバー率 前", "後", "正解率0.95での最大カバー率 前", "後"], rows)


def t_flip(scope: str) -> str:
    d = read("stage2_flip")
    if d is None:
        return "（flip の集計は未完了）"
    d = d[d.scope == scope]
    by = {r["model"]: r for r in records(d)}
    rows = []
    for m in ORDER:
        r = by.get(m)
        if not r or not r.get("n_cases"):
            if r is not None or m in ORDER[:-2]:
                rows.append([label(m), "未測定", "", "", "", "", "", ""])
            continue
        perm = f"{r['perm_ci_low']:.3f}–{r['perm_ci_high']:.3f}" if r.get("perm_ci_low") is not None else "—"
        rows.append([label(m), int(r["n_cases"]), int(r["n_pairs"]), r["n_pairs_excluded"], ci(r, "flip_rate"), perm,
                     ci(r, "any_flip_rate"), f3(r.get("mean_tv"))])
    return md(["モデル", "件数", "（件, 並べ替え）の組", "除外した組", "flip rate（95%区間）", "並べ替えの選び方による区間",
               "1回でも変わった件の割合", "平均 TV 距離"], rows)


def t_paraphrase() -> str:
    d = read("stage1_bench_ja")
    by = {(r["model"], r["task_id"]): r for r in records(d)}
    rows = []
    for m in ORDER[:-2]:
        a, b = by.get((m, "bench_ja-churn")), by.get((m, "bench_ja-churn-para1"))
        if not a or not b or a.get("auroc") is None or b.get("auroc") is None:
            continue
        rows.append([label(m), f3(a["auroc"]), f3(b["auroc"]), f"{b['auroc'] - a['auroc']:+.3f}",
                     f3(a.get("accuracy")), f3(b.get("accuracy"))])
    return md(["モデル", "AUROC（機械生成の説明文）", "AUROC（言い換え版）", "差（言い換え − 機械生成）", "正解率（機械生成）",
               "正解率（言い換え版）"], rows)


def t_overlap() -> str:
    d = json.loads((RES / "overlap_tasks.json").read_text())
    rows = [[label(m), v["overlap"], len(v.get("tasks") or [])] for m, v in d["per_model"].items()
            if m in ORDER or m == "h2o-lightning-4b"]
    return md(["モデル", "S1MB の元データとの重なり（カード・学習記録の記載）", "重なるタスク数"], rows, "llr")


def t_overlap_tasks() -> str:
    d = json.loads((RES / "overlap_tasks.json").read_text())
    return "、".join(f"`{t['task_id']}`" for t in d["overlap_tasks"])


def t_cost(manifest) -> str:
    rows = []
    cost = RES / "cost"
    for m in ORDER[:-2]:
        mm = manifest["models"].get(m, {})
        hub = mm.get("hub") or {}
        p = params_for(m, mm.get("hf_id", ""), hub.get("revision") or "") if hub.get("revision") else None
        c = cost / f"{m}.json"
        lat = vram = "未測定"
        if c.is_file():
            j = json.loads(c.read_text())
            if j.get("latency_p50_s") is not None:
                lat = f"{j['latency_p50_s'] * 1000:.1f} / {j['latency_p95_s'] * 1000:.1f}"
                vram = f"{j['cuda_max_memory_allocated'] / 2**30:.2f}"
        rows.append([label(m), lat, "未測定", "未測定", vram, f"{p / 1e9:.2f}" if p else "未測定",
                     f"{hub['weight_file_bytes'] / 1e9:.2f}" if hub.get("weight_file_bytes") else "—"])
    return md(["モデル", "レイテンシ p50 / p95（ms、バッチ1）", "バッチ8", "バッチ32", "VRAM ピーク（GiB）", "パラメータ数（B）",
               "リポジトリの重みファイル合計（GB）"], rows)


# ---------------------------------------------------------------- 組み立て

# ---------------------------------------------------------------- Hub カーネルの一致確認（fla のみ適用）

KP = RES / "kernel_parity"


def _kp():
    p, j = KP / "comparison.csv", KP / "judgement.json"
    if not p.is_file() or not j.is_file():
        return None, None
    return records(pd.read_csv(p)), json.loads(j.read_text(encoding="utf-8"))


def _g(x) -> str:
    """小さな値を指数表記にしない表示。"""
    return "未定義" if x is None else f"{x:.4f}"


def t_kernel_parity(scope: str) -> str:
    rows_, judge = _kp()
    if rows_ is None:
        return "（カーネルの一致確認は未実施）"
    by = {(r["model"], r["pair"], r["scope"]): r for r in rows_}
    rows = []
    for m in ("d3-nano", "d3-mini", "clef-flash", "d3-lite", "d3-flash"):
        a, c = by.get((m, "A", scope)), by.get((m, "C", scope))
        if a is None:
            rows.append([label(m), "未実行", "", "", "", "", "", ""])
            continue
        n = int(a["n_both"])
        rows.append([label(m), f"{int(a['argmax_disagree'])} / {n}（{a['s1']:.4f}）",
                     f"{int(c['argmax_disagree'])} / {int(c['n_both'])}（{c['s1']:.4f}）" if c else "未実行",
                     _g(a["s2"]), _g(c["s2"]) if c else "未実行", _g(a["s3"]), _g(c["s3"]) if c else "未実行",
                     _g(a["max_abs_diff"])])
    s3 = "Task Avg の差（17タスク）" if scope == "s1mb" else "主要指標の差の最大"
    return md(["モデル", "argmax 不一致：fla のみ適用 対 なし", "argmax 不一致：bf16 対 fp32",
               "確率差 p99：fla 対 なし", "確率差 p99：bf16 対 fp32", f"{s3}：fla 対 なし", f"{s3}：bf16 対 fp32",
               "確率差の最大：fla 対 なし"], rows)


def t_kernel_parity_judgement() -> str:
    rows_, judge = _kp()
    if rows_ is None:
        return "（カーネルの一致確認は未実施）"
    by = {(r["model"], r["pair"], r["scope"]): r for r in rows_}
    rows = []
    for m, j in judge["models"].items():
        def same(pair):
            vals = [by.get((m, pair, s)) for s in ("s1mb", "bench_ja")]
            if any(v is None for v in vals):
                return "未実行"
            return "完全一致（差 0）" if all(float(v["max_abs_diff"]) == 0 for v in vals) else (
                f"差あり（最大 {max(float(v['max_abs_diff']) for v in vals):.4f}）")
        checks = j.get("checks", {})

        def mark(scope):
            c = checks.get(scope, {})
            return " ".join(f"{k[-1]}{'○' if v else ('－' if v is None else '×')}" for k, v in c.items()) or "—"
        verdict = j["verdict"]
        if verdict.startswith("一致と書けない"):
            verdict = "「一致」と書かない（基準を満たさない）"
        rows.append([label(m), same("B"), same("D"), mark("s1mb"), mark("bench_ja"), verdict])
    return md(["モデル", "なし 1回目 対 2回目", "なし 対 既存の参考値", "基準1〜5（S1MB）", "基準1〜5（bench_ja）", "判定"],
              rows, "llllll")


def build() -> str:
    manifest, status, cfg = load_meta()
    tables = {
        "models": lambda: t_models(manifest, status, cfg),
        "excluded": lambda: t_excluded(manifest, status, cfg),
        "env": lambda: t_env(manifest),
        "envs": lambda: t_envs(manifest),
        "failed_tasks": t_failed_tasks,
        "common_tasks": t_common_tasks,
        "bench_ja": t_bench_ja,
        "bench_ja_failed": t_bench_ja_failed,
        "bench_ja_ext": lambda: t_bench_ja(ext=True),
        "churn": t_churn,
        "ext_compare": t_ext_compare,
        "temperatures": t_temperatures,
        "cost": lambda: t_cost(manifest),
        "paraphrase": t_paraphrase,
        "overlap": t_overlap,
        "overlap_tasks": t_overlap_tasks,
        "kernel_parity_s1mb": lambda: t_kernel_parity("s1mb"),
        "kernel_parity_bench_ja": lambda: t_kernel_parity("bench_ja"),
        "kernel_parity_judgement": t_kernel_parity_judgement,
    }
    for s in SCOPE_LABEL:
        tables[f"s1mb_{s}"] = (lambda s=s: t_s1mb(s))
        tables[f"s1mb_ext_{s}"] = (lambda s=s: t_s1mb(s, ext=True))
    for typ in ("all", "noul", "choice", "score"):
        tables[f"calib_s1mb_{typ}"] = (lambda typ=typ: t_calib("s1mb", "non_overlap_17", typ))
        tables[f"coverage_s1mb_{typ}"] = (lambda typ=typ: t_coverage("s1mb", "non_overlap_17", typ))
    for task, typ in (("bench_ja-department", "choice"), ("bench_ja-urgency", "score"), ("bench_ja-churn", "noul")):
        key = task.split("-")[1]
        tables[f"calib_bj_{key}"] = (lambda task=task, typ=typ: t_calib("bench_ja", task, typ))
        tables[f"coverage_bj_{key}"] = (lambda task=task, typ=typ: t_coverage("bench_ja", task, typ))
    for s in ("non_overlap_17", "all_27", "bench_ja-department"):
        tables[f"flip_{s}"] = (lambda s=s: t_flip(s))

    text = TEMPLATE.read_text(encoding="utf-8")

    def sub(mo):
        name = mo.group(1)
        if name not in tables:
            raise KeyError(f"unknown table {name}")
        return tables[name]()

    return re.sub(r"\{\{table:([a-z0-9_\-]+)\}\}", sub, text)


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
