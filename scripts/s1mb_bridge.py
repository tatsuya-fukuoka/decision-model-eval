"""S1MB 公式評価器の `s1mb run` をそのまま実行し、同じアダプタのインスタンスで bench_ja も推論する。

評価器の隔離環境で、third_party/S1MB/evaluator を作業ディレクトリにして実行する:

    HF_HUB_OFFLINE=1 <env>/bin/python -I scripts/s1mb_bridge.py \
        --bench-ja data/bridge/bench_ja_requests.jsonl --bench-ja-out OUT.jsonl \
        -- run --adapter ... --category japanese-v1 --offline-dataset ...

- S1MB 側：公式 CLI の引数解釈・アダプタ構築・runner.evaluate・結果検証を一切変えない。
- bench_ja 側：公式 runner.evaluate が最初に呼ばれたときのアダプタを捕まえ、adapter.close() の直前に
  同じアダプタで bench_ja の InferenceCase を推論する（モデルを2回読み込まない）。
  予測の検査は公式の check_prediction を使い、失敗は件ごとに例外名だけを記録する（runner と同じ方針）。
- --custom-adapter FILE.py:Class（公式アダプタが無いモデル）：公式 CLI と同じ順序で
  load_category → load_benchmark → runner.evaluate を呼ぶ（採点・検証・結果形式は公式のまま）。
- --no-hub-kernels：公式 UpstreamAdapter.enable_kernels を「何もしない」版に差し替える（Hub の
  kernels-community/fla〔Triton ソース〕と kernels-community/activation〔コンパイル済み .so〕を取得・実行しない）。transformers 本体の実装で推論し、
  アダプタの settings に hub_kernels=False と差し替えた事実を残す。結果は「参考値（公式カーネル未使用）」。
- --fp32-backbone（--no-hub-kernels と併用）：公式アダプタが bf16 で読み込んだ後、enable_kernels の時点で
  バックボーン（Clef は engine 全体）を fp32 に変換する。カーネル一致確認で、数値精度によるばらつきの目安に使う。
- 公式 enable_kernels を有効にしたとき（--no-hub-kernels なし）は、kernelize 後に置き換わったモジュールと
  読み込まれたカーネルをアダプタの settings（kernel_diagnostics）に記録する。
- --requests-only：S1MB 本体は推論せず、--bench-ja に渡した入力（例：選択肢を並べ替えた data/bridge/flip_requests.jsonl）
  だけを同じアダプタで推論して終わる（公式アダプタは公式 CLI の構築経路のまま。最初の evaluate 呼び出しで捕まえる）。
- 本スクリプトは本リポジトリの src を import しない（adapters/ の自作アダプタはファイルから読み込む）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _sync() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_ja(adapter, requests: Path, out: Path, limit: int | None, timing: bool = False) -> None:
    """timing=True：件ごとに predict の所要時間（GPU は同期してから計測）を latency_s に記録し、終了時に
    torch のピーク確保メモリ（max_memory_allocated / max_memory_reserved）を <out>.stats.json に書く。"""
    from s1mb.data import InferenceCase, check_prediction

    torch = sys.modules.get("torch")
    if timing and torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rows = [json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        seen: dict[str, int] = {}
        kept = []
        for r in rows:
            seen[r["task_id"]] = seen.get(r["task_id"], 0) + 1
            if seen[r["task_id"]] <= limit:
                kept.append(r)
        rows = kept
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    start = time.perf_counter()
    with tmp.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows):
            case = InferenceCase.model_validate(r["case"])
            questions = {q.id: q for q in case.questions}
            try:
                t0 = time.perf_counter()
                preds = adapter.predict(case)
                if timing:
                    _sync()
                latency = time.perf_counter() - t0
                if len(preds) != len(questions) or {p.question_id for p in preds} != set(questions):
                    raise ValueError("Adapter returned missing or duplicate questions")
                for p in preds:
                    if p.case_id != case.case_id:
                        raise ValueError("Adapter returned an invalid case ID")
                    check_prediction(p, questions[p.question_id])
                recs = [{"probabilities": p.probabilities, "abstention": p.abstention, "error": p.error,
                         "question_id": p.question_id, **({"latency_s": latency} if timing else {})} for p in preds]
            except Exception as exc:  # noqa: BLE001 - 件ごとに失敗を残して続行
                recs = [{"probabilities": None, "abstention": None, "error": type(exc).__name__,
                         "question_id": q} for q in questions]
                print(f"  bench_ja failed {r['task_id']} {case.case_id}: {type(exc).__name__}", flush=True)
            for rec in recs:
                meta = {"meta": r["meta"]} if "meta" in r else {}
                fh.write(json.dumps({"task_id": r["task_id"], "case_id": case.case_id, **rec, **meta},
                                    ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"  bench_ja {i + 1}/{len(rows)}", flush=True)
    tmp.replace(out)
    if timing:
        stats = {"cases": len(rows), "seconds": time.perf_counter() - start}
        if torch is not None and torch.cuda.is_available():
            stats.update({"cuda_max_memory_allocated": torch.cuda.max_memory_allocated(),
                          "cuda_max_memory_reserved": torch.cuda.max_memory_reserved(),
                          "cuda_device": torch.cuda.get_device_name(0)})
        out.with_suffix(".stats.json").write_text(json.dumps(stats), encoding="utf-8")
    print(f"bench_ja: {len(rows)} cases in {time.perf_counter() - start:.1f}s -> {out}", flush=True)


def _disable_hub_kernels() -> None:
    from s1mb.adapters import upstream

    def enable_kernels(self):  # 公式版は kernels.kernelize で Hub のカーネルに差し替える
        self.settings["hub_kernels"] = False
        self.settings["hub_kernels_note"] = "disabled by decision-eval bridge (--no-hub-kernels); transformers built-in ops"

    upstream.UpstreamAdapter.enable_kernels = enable_kernels


def _kernel_diagnostics(adapter) -> dict:
    """kernelize の後に、どのモジュールが Hub カーネルに置き換わったかを数える（数値は変えない）。"""
    import collections
    import importlib

    model = adapter.attention_model
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    module_types, forwards = collections.Counter(), collections.Counter()
    for m in model.modules():
        mod = type(m).__module__
        if not mod.startswith(("torch.", "transformers.")):
            module_types[f"{mod}.{type(m).__name__}"] += 1
        fwd = getattr(m, "forward", None)
        fmod = getattr(getattr(fwd, "__func__", fwd), "__module__", "") or ""
        if fmod and not fmod.startswith(("torch.", "transformers.")):
            forwards[fmod] += 1
    kernels = importlib.import_module("kernels")
    loaded = [repr(k) for k in kernels.get_loaded_kernels()] if hasattr(kernels, "get_loaded_kernels") else []
    return {"non_core_module_types": dict(module_types), "non_core_forward_modules": dict(forwards),
            "loaded_kernels": loaded}


def _repo_ids(value) -> set[str]:
    reps = value.values() if isinstance(value, dict) else [value]
    out = set()
    for r in reps:
        if isinstance(r, dict):
            out |= _repo_ids(r)
        else:
            out.add(getattr(r, "_repo_id", None) or getattr(r, "repo_id", None))
    return out


def _call_without_repos(adapter, inner, excluded: set[str]) -> None:
    """公式 enable_kernels を、transformers の対応表から指定リポジトリの項目だけを外した状態で呼ぶ。

    外した層は kernelize の対象にならず、transformers 本体の実装のまま動く（ユーザー判断 2026-11-09：
    kernels-community/activation はコンパイル済みバイナリで内容を確認できないため適用しない）。
    """
    import importlib

    hub = importlib.import_module("transformers.integrations.hub_kernels")
    original = hub.get_kernel_mapping_transformers
    removed = []

    def filtered():
        mapping = original()
        out = {}
        for name, value in mapping.items():
            if value and _repo_ids(value) <= excluded:
                removed.append(name)
                continue
            out[name] = value
        return out

    hub.get_kernel_mapping_transformers = filtered
    try:
        inner(adapter)
    finally:
        hub.get_kernel_mapping_transformers = original
    adapter.settings["excluded_kernel_repos"] = sorted(excluded)
    adapter.settings["excluded_kernel_layers"] = sorted(set(removed))
    adapter.settings["kernel_scope"] = "fla のみ適用（activation は未適用）" if excluded == {
        "kernels-community/activation"} else f"一部のみ適用（除外：{sorted(excluded)}）"


def _patch_kernels(args) -> None:
    """--fp32-backbone（カーネルなしでのみ）と、カーネルありのときの適用状況の記録。"""
    from s1mb.adapters import upstream

    inner = upstream.UpstreamAdapter.enable_kernels

    def enable_kernels(self):
        if args.fp32_backbone:
            torch = self.torch
            engine = getattr(self, "engine", None)
            target = engine if isinstance(engine, torch.nn.Module) else self.attention_model
            target.to(torch.float32)
            self.settings["dtype_override"] = (
                "float32: bf16 で読み込んだ後に変換（decision-eval bridge --fp32-backbone）。"
                "バックボーン重みは BF16 で保存されているので、値は fp32 で読み込んだ場合と同じ"
            )
            self.settings["fp32_cast_target"] = type(target).__name__
        if args.exclude_kernel_repo and not args.no_hub_kernels:
            _call_without_repos(self, inner, set(args.exclude_kernel_repo))
        else:
            inner(self)
        if not args.no_hub_kernels:
            self.settings["kernel_diagnostics"] = _kernel_diagnostics(self)

    upstream.UpstreamAdapter.enable_kernels = enable_kernels


def _custom(args) -> None:
    import importlib.util

    from s1mb.data import DATA_DIR, load_benchmark, load_category
    from s1mb.runner import evaluate

    file, _, cls = args.custom_adapter.rpartition(":")
    spec = importlib.util.spec_from_file_location("custom_adapter", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = getattr(module, cls)(**json.loads(args.custom_kwargs))
    if args.requests_only:
        try:
            _bench_ja(adapter, args.bench_ja, args.bench_ja_out, args.bench_ja_limit, args.timing)
        finally:
            adapter.close()
        return
    failures = 0
    try:
        names = load_category(DATA_DIR, args.category).benchmarks
        for i, name in enumerate(names):
            bench = load_benchmark(DATA_DIR, name)
            print(f"[{i + 1}/{len(names)}] {bench.id}", flush=True)
            result = evaluate(adapter, DATA_DIR, bench, args.output, args.run_id, args.limit)
            failures += result.counts.failed
            print(f"  {result.status}: {result.metrics} ({result.elapsed_seconds:.1f}s)", flush=True)
        if args.bench_ja:
            _bench_ja(adapter, args.bench_ja, args.bench_ja_out, args.bench_ja_limit)
    finally:
        adapter.close()
    print(f"Results: {args.output / args.run_id}", flush=True)
    if failures:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-ja", type=Path)
    ap.add_argument("--bench-ja-out", type=Path)
    ap.add_argument("--bench-ja-limit", type=int)
    ap.add_argument("--custom-adapter", help="path/to/file.py:ClassName")
    ap.add_argument("--timing", action="store_true", help="--requests-only と併用：件ごとの所要時間とピークVRAMを記録")
    ap.add_argument("--requests-only", action="store_true", help="S1MB を推論せず --bench-ja の入力だけを推論する")
    ap.add_argument("--no-hub-kernels", action="store_true", help="公式 enable_kernels を無効化する（参考値）")
    ap.add_argument("--fp32-backbone", action="store_true",
                    help="--no-hub-kernels と併用：読み込み後にバックボーンを fp32 に変換する（カーネル一致確認の基準4）")
    ap.add_argument("--exclude-kernel-repo", action="append",
                    help="公式 enable_kernels の対応表から、このリポジトリの項目を外す（複数指定可）")
    ap.add_argument("--custom-kwargs", default="{}", help="JSON の constructor 引数")
    ap.add_argument("--category", default="japanese-v1")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--run-id")
    ap.add_argument("s1mb_args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.fp32_backbone and not args.no_hub_kernels:
        raise SystemExit("--fp32-backbone は --no-hub-kernels と併用する")
    if args.no_hub_kernels:
        _disable_hub_kernels()
    if not args.custom_adapter:
        _patch_kernels(args)
    if args.custom_adapter:
        _custom(args)
        return
    s1mb_args = args.s1mb_args[1:] if args.s1mb_args[:1] == ["--"] else args.s1mb_args

    import s1mb.runner as runner

    real_evaluate = runner.evaluate
    state = {"wrapped": False}

    def evaluate(adapter, *a, **kw):
        if args.requests_only:
            try:
                _bench_ja(adapter, args.bench_ja, args.bench_ja_out, args.bench_ja_limit, args.timing)
            finally:
                adapter.close()
            raise SystemExit(0)
        if args.bench_ja and not state["wrapped"]:
            state["wrapped"] = True
            original_close = adapter.close

            def close():
                try:
                    _bench_ja(adapter, args.bench_ja, args.bench_ja_out, args.bench_ja_limit)
                finally:
                    original_close()

            adapter.close = close
        return real_evaluate(adapter, *a, **kw)

    runner.evaluate = evaluate
    from s1mb.cli import main as cli_main

    sys.argv = ["s1mb", *s1mb_args]
    cli_main()


if __name__ == "__main__":
    main()
