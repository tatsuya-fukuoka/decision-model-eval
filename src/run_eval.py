"""ステージ1の実行（指示書 §7 手順3・5）。モデルごとに隔離環境で S1MB 公式評価器を動かし、bench_ja も推論する。

    uv run python -m src.run_eval --models saihai-v1-310m --smoke     # 2件/タスクのスモークテスト
    uv run python -m src.run_eval --models saihai-v1-310m             # 全件

手順（モデルごと。失敗しても例外で止めず、status と原因を results/model_status.json に記録して次へ）：
1. コード確認（manifest.code_review）が approved でなければ実行しない（status=not_reviewed）。
2. envs/<env>/.venv を公式 uv.lock から作る（configs/envs.yaml）。freeze を manifest に記録。
3. 固定リビジョンのスナップショットをトークンなしでダウンロード（hf_cache/）。
4. HF_HUB_OFFLINE=1 で scripts/s1mb_bridge.py を実行（公式 `s1mb run` + 同じアダプタで bench_ja）。
5. 結果を results/raw/ に取り込み、要約を results/runs/ に保存。
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import yaml

from . import import_runs
from .manifest import load_json, write_json

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "third_party" / "S1MB" / "evaluator"
HF_HOME = ROOT / "hf_cache"
LOGS = ROOT / "results" / "logs"
STATUS = ROOT / "results" / "model_status.json"
MANIFEST = ROOT / "results" / "manifest.json"
REQUESTS = ROOT / "data" / "bridge" / "bench_ja_requests.jsonl"
FLIP_REQUESTS = ROOT / "data" / "bridge" / "flip_requests.jsonl"
COST_REQUESTS = ROOT / "data" / "bridge" / "cost_requests.jsonl"
COST = ROOT / "results" / "cost"
BRIDGE_OUT = ROOT / "data" / "bridge" / "out"


@contextmanager
def locked():
    """model_status.json / manifest.json の読み書きを複数の run_eval プロセスで直列化する。"""
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / ".records.lock").open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def models_config() -> dict[str, dict]:
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    out = {}
    for tier in ("tier1", "tier2", "tier3"):
        for m in cfg.get(tier) or []:
            out[m["name"]] = {**m, "tier": tier}
    for m in cfg.get("extended") or []:
        out[m["name"]] = {**out[m["base"]], **m, "tier": out[m["base"]]["tier"]}
    return out


def clean_env(**extra: str) -> dict[str, str]:
    """トークン類を子プロセスに渡さない。"""
    env = {k: v for k, v in os.environ.items() if not re.search(r"TOKEN|API_KEY|SECRET", k)}
    env.update({"HF_HOME": str(HF_HOME), "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1", **extra})
    return env


def ensure_env(name: str, log) -> Path:
    spec = yaml.safe_load((ROOT / "configs" / "envs.yaml").read_text(encoding="utf-8"))[name]
    venv = ROOT / "envs" / name / ".venv"
    python = spec.get("python", "3.12")
    # 公式 uv.lock をハッシュ付き requirements に書き出し、--require-hashes で入れる。ファイルはハッシュで
    # 固定されるので、取得元（PyPI か PyTorch のミラー）が違っても lock と同じものが入る。
    # （uv sync --frozen は lock の URL＝PyPI からしか取れず、この環境では約 100KB/s だったため）
    req = ROOT / "envs" / name / "requirements.lock.txt"
    cmd = ["uv", "export", "--project", str(EVAL), "--frozen", "--no-dev", "--format", "requirements-txt",
           "--no-emit-project", "--python", python, "-o", str(req)]
    for e in spec.get("extras") or []:
        cmd += ["--extra", e]
    req.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, env=clean_env(), stdout=log, stderr=log)
    py = venv / "bin" / "python"
    if not py.exists():
        subprocess.run(["uv", "venv", "--python", python, str(venv)], check=True, env=clean_env(), stdout=log,
                       stderr=log)
    # PyPI が遅い環境向け：lock と同じ sha256 の NVIDIA wheel を PyTorch のミラーから取得して先に入れる
    # （ハッシュが一致したファイルだけ。続く -r のインストールでは同じ版として扱われる）
    subprocess.run(["bash", str(ROOT / "scripts" / "fetch_cuda_wheels.sh"), str(req)], env=clean_env(),
                   stdout=log, stderr=log)
    wheels = sorted((ROOT / "data" / "wheels").glob("*.whl"))
    lock_text = req.read_text(encoding="utf-8")
    wheels = [w for w in wheels if f"{w.name.split('-')[0].replace('_', '-')}=={w.name.split('-')[1]}" in lock_text]
    if wheels:
        subprocess.run(["uv", "pip", "install", "--python", str(py), "--no-deps", *map(str, wheels)], check=True,
                       env=clean_env(), stdout=log, stderr=log)
    subprocess.run(["uv", "pip", "install", "--python", str(py), "--require-hashes", "-r", str(req),
                    "--index-url", "https://download.pytorch.org/whl/cu130",
                    "--extra-index-url", "https://pypi.org/simple", "--index-strategy", "unsafe-best-match"],
                   check=True, env=clean_env(), stdout=log, stderr=log)
    # 評価器本体（uv sync と同じく editable。DATA_DIR が evaluator/data を指す）
    subprocess.run(["uv", "pip", "install", "--python", str(py), "--no-deps", "-e", str(EVAL)], check=True,
                   env=clean_env(), stdout=log, stderr=log)
    for pkg in spec.get("local_no_deps") or []:
        # 公式 uv.lock の環境に、固定コミットのローカルパッケージを依存解決なしで追加する
        subprocess.run(["uv", "pip", "install", "--python", str(py), "--no-deps", str(ROOT / pkg)], check=True,
                       env=clean_env(), stdout=log, stderr=log)
    if spec.get("overrides"):
        subprocess.run(["uv", "pip", "install", "--python", str(py), *spec["overrides"]], check=True,
                       env=clean_env(), stdout=log, stderr=log)
    if spec.get("uninstall"):
        subprocess.run(["uv", "pip", "uninstall", "--python", str(py), *spec["uninstall"]], check=True,
                       env=clean_env(), stdout=log, stderr=log)
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(py)], capture_output=True, text=True,
                            env=clean_env()).stdout.splitlines()
    with locked():
        _record_env(name, spec, python, freeze)
    return py


def _record_env(name, spec, python, freeze) -> None:
    manifest = load_json(MANIFEST)
    manifest.setdefault("environment", {}).setdefault("envs", {})[name] = {
        "python": python, "install": "uv export --frozen → uv pip install --require-hashes", "extras": spec.get("extras") or [],
        "local_no_deps": spec.get("local_no_deps") or [], "overrides": spec.get("overrides") or [],
        "uninstall": spec.get("uninstall") or [],
        "packages": freeze, "recorded_at": now(),
    }
    write_json(MANIFEST, manifest)


def ensure_snapshot(hf_id: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = str(HF_HOME)
    return Path(snapshot_download(hf_id, revision=revision, token=False, cache_dir=str(HF_HOME / "hub")))


def ensure_dependencies(m: dict) -> list[dict]:
    """モデルのコードが revision 指定なしで読み込む別リポジトリ（例：sokudan の基盤 ModernBERT）。

    オフライン時の from_pretrained(repo_id) はキャッシュの refs/main を使うので、main を一度だけ取得して
    そのコミット SHA を manifest に固定し、以後は同じ SHA であることを確認する。
    """
    from huggingface_hub import HfApi, snapshot_download

    with locked():
        return _ensure_dependencies(m, HfApi, snapshot_download)


def _ensure_dependencies(m: dict, HfApi, snapshot_download) -> list[dict]:
    out = []
    manifest = load_json(MANIFEST)
    entry = manifest["models"][m.get("base", m["name"])]
    pinned = {d["hf_id"]: d for d in entry.get("dependencies", [])}
    for dep in m.get("dependencies") or []:
        hf_id = dep["hf_id"]
        explicit = dep.get("revision")  # コードが revision を指定して読む場合（例：Strands の provenance.json）
        if hf_id not in pinned:
            info = HfApi(token=False).model_info(hf_id, revision=explicit or "main")
            pinned[hf_id] = {"hf_id": hf_id, "revision": info.sha, "license": (info.card_data or {}).get("license"),
                             "reason": dep.get("reason"), "loaded_with_explicit_revision": bool(explicit),
                             "pinned_at": now()}
        path = Path(snapshot_download(hf_id, revision=explicit or "main", token=False,
                                      cache_dir=str(HF_HOME / "hub"), allow_patterns=dep.get("allow_patterns")))
        if path.name != pinned[hf_id]["revision"]:
            raise RuntimeError(f"{hf_id} moved: {path.name} != pinned {pinned[hf_id]['revision']}")
        out.append(pinned[hf_id])
    if out:
        entry["dependencies"] = list(pinned.values())
        write_json(MANIFEST, manifest)
    return out


def classify_failure(log_text: str) -> tuple[str, str]:
    tail = log_text.strip().splitlines()[-40:]
    reason = next((l for l in reversed(tail) if re.match(r"^\w+(Error|Exception)\b", l)), tail[-1] if tail else "")
    if "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text:
        return "oom", reason
    if re.search(r"401 Client Error|GatedRepoError|RepositoryNotFoundError|is restricted\. You must have access|"
                 r"Cannot access gated repo", log_text):
        return "auth_required", reason
    return "load_failed", reason


def set_status(model: str, record: dict) -> None:
    with locked():
        status = load_json(STATUS) if STATUS.is_file() else {}
        status[model] = record
        write_json(STATUS, status)


def gpu_other_processes() -> list[str]:
    """GPU を使っている計算プロセス（コスト計測の前後に確認。並走していたら計測しない）。"""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [l.strip() for l in out.splitlines() if l.strip()]


def run_model(m: dict, smoke: bool, flip: bool = False, cost: bool = False) -> dict:
    name = m["name"]
    base = m.get("base", name)  # 拡張設定はベースのリビジョン・コード確認を使う
    manifest = load_json(MANIFEST)
    rev = manifest["models"][base]["hub"]["revision"]
    review = manifest.get("code_review", {}).get("models", {}).get(base, {})
    if review.get("verdict") != "approved":
        return {"status": "not_reviewed", "reason": f"code_review verdict={review.get('verdict')}", "revision": rev}
    if m.get("runner") == "custom" and not m.get("custom_adapter"):
        return {"status": "not_run", "reason": "custom adapter not implemented yet", "revision": rev}
    kind = ("flip-" if flip else "cost-" if cost else "") + ("smoke" if smoke else "full")
    run_id = f"{name}-{kind}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{run_id}.log"
    record = {"revision": rev, "run_id": run_id, "smoke": smoke, "started_at": now(),
              "log": str(log_path.relative_to(ROOT))}
    if m.get("reference"):
        record["reference"] = m["reference"]
    if m.get("extended"):
        record["extended"] = m["extended"]
    with log_path.open("w", encoding="utf-8") as log:
        try:
            py = ensure_env(m["env"], log)
            snapshot = ensure_snapshot(m["hf_id"], rev)
            os.environ["HF_HOME"] = str(HF_HOME)
            ensure_dependencies(m)
        except Exception as exc:  # noqa: BLE001
            log.write(f"\nsetup failed: {type(exc).__name__}: {exc}\n")
            return {**record, "status": "setup_failed", "reason": f"{type(exc).__name__}: {exc}"[:500]}
        args = [a.replace("{snapshot}", str(snapshot)) for a in m.get("s1mb_args", [])]
        out = BRIDGE_OUT / f"{run_id}.{'flip' if flip else 'cost' if cost else 'bench_ja'}.jsonl"
        requests = FLIP_REQUESTS if flip else COST_REQUESTS if cost else REQUESTS
        cmd = [str(py), "-I", str(ROOT / "scripts" / "s1mb_bridge.py"), "--bench-ja",
               str(requests), "--bench-ja-out", str(out), *m.get("bridge_args", [])]
        if flip or cost:
            cmd += ["--requests-only"]
        if cost:
            cmd += ["--timing"]
            busy = gpu_other_processes()
            if busy:
                return {**record, "status": "cost_not_measured", "reason": f"GPU shared with other processes: {busy}"}
        if smoke:
            cmd += ["--bench-ja-limit", "10" if flip else "2"]
        common = ["--category", "japanese-v1", "--output", str(import_runs.RUNS), "--run-id", run_id]
        if smoke or flip or cost:
            common += ["--limit", "2"]
        if m.get("custom_adapter"):
            kwargs = json.dumps(m.get("custom_kwargs") or {}).replace("{snapshot}", str(snapshot)).replace(
                "{revision}", rev)
            cmd += ["--custom-adapter", str(ROOT / m["custom_adapter"]), "--custom-kwargs", kwargs, *common]
        else:
            cmd += ["--", "run", "--adapter", m["s1mb_adapter"], "--model", m["hf_id"], "--revision", rev, *args,
                    "--offline-dataset", *common]
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=EVAL, env=clean_env(HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES="0"),
                              stdout=log, stderr=subprocess.STDOUT)
        record["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
        record["returncode"] = proc.returncode
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if cost:
        busy_after = gpu_other_processes()
        if not out.is_file():
            status, reason = classify_failure(text)
            return {**record, "status": status, "reason": reason[:500]}
        summary = summarize_cost(m, out, rev)
        summary["gpu_processes_after"] = busy_after
        if not smoke:
            COST.mkdir(parents=True, exist_ok=True)
            write_json(COST / f"{name}.json", {**summary, "run_id": run_id, "measured_at": now()})
        return {**record, "status": "smoke_ok" if smoke else "ok", "cost": {k: summary[k] for k in
                ("latency_p50_s", "latency_p95_s", "cuda_max_memory_allocated") if k in summary}}
    if flip:
        if not out.is_file():
            status, reason = classify_failure(text)
            return {**record, "status": status, "reason": reason[:500]}
        fl = import_runs.import_flip(name, out, write=not smoke)
        failed = sum(v["failed"] for v in fl.values())
        n = sum(v["n"] for v in fl.values())
        status = ("smoke_ok" if n > failed else "smoke_failed") if smoke else ("ok" if not failed else "partial")
        return {**record, "status": status, "flip_rows": n, "flip_failed": failed, "flip_tasks": fl,
                "reason": None if status in ("ok", "smoke_ok") else _first_failure(text)}
    run_dir = import_runs.RUNS / run_id
    if not run_dir.is_dir() or not any(run_dir.glob("*.json")):
        status, reason = classify_failure(text)
        return {**record, "status": status, "reason": reason[:500]}
    # スモークは raw に書かない（本実行の raw を上書きしない）
    s1 = import_runs.import_s1mb(name, run_id, write=not smoke)
    bj = import_runs.import_bench_ja(name, out, write=not smoke) if out.is_file() else {}
    s1["bench_ja"] = bj
    import_runs.write_summary(name, run_id, s1)
    statuses = {v["status"] for v in s1["benchmarks"].values()}
    failed = sum(v.get("counts", {}).get("failed", 0) for v in s1["benchmarks"].values())
    bj_failed = sum(v["failed"] for v in bj.values())
    complete = statuses == {"complete"} and not failed and not bj_failed and len(bj) == 4
    if smoke:
        # スモークは「読み込みと推論の経路が動くか」の確認。モデルの入力上限などによる件ごとの失敗は
        # 本実行でも同じく記録されるだけなので止めない。全タスクの結果があり、失敗が半数未満なら通す。
        n_bench = sum(1 for v in s1["benchmarks"].values() if v["status"] != "missing")
        n_ok = sum(v.get("counts", {}).get("succeeded", 0) for v in s1["benchmarks"].values())
        status = ("smoke_ok" if n_bench == 27 and bj and len(bj) == 4 and n_ok > (failed + bj_failed)
                  else "smoke_failed")
    else:
        status = "ok" if complete else "partial"
    return {**record, "status": status, "s1mb_failed_decisions": failed, "bench_ja_failed": bj_failed,
            "benchmark_status": sorted(statuses),
            "reason": None if status in ("ok", "smoke_ok") else _first_failure(text)}


def summarize_cost(m: dict, out: Path, rev: str) -> dict:
    """件ごとの所要時間（ウォームアップを除く）から p50 / p95、型ごとの p50 / p95、ピークVRAM、重みのサイズ。"""
    import numpy as np

    lat, by_type, failed = {}, {}, 0
    for line in out.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("latency_s") is None:
            failed += 1
            continue
        if r["meta"]["warmup"]:
            continue
        lat[r["case_id"]] = r["latency_s"]
        by_type.setdefault(r["meta"]["type"], {})[r["case_id"]] = r["latency_s"]
    v = np.asarray(list(lat.values()))
    stats_path = out.with_suffix(".stats.json")
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    hub = load_json(MANIFEST)["models"][m.get("base", m["name"])]["hub"]
    res = {"revision": rev, "batch_size": 1, "n_timed": int(len(v)), "n_failed": failed,
           "latency_p50_s": float(np.percentile(v, 50)) if len(v) else None,
           "latency_p95_s": float(np.percentile(v, 95)) if len(v) else None,
           "by_type": {t: {"n": len(d), "p50_s": float(np.percentile(list(d.values()), 50)),
                           "p95_s": float(np.percentile(list(d.values()), 95))} for t, d in by_type.items()},
           "weight_file_bytes": hub.get("weight_file_bytes"), **stats,
           "note": "1件=1回の adapter.predict（その件の全質問）。バッチ 8/32 は公式アダプタの predict 経路に無いため未測定"}
    return res


def _first_failure(text: str) -> str | None:
    for line in text.splitlines():
        if "failed" in line or "Error" in line:
            return line.strip()[:300]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cost", action="store_true", help="GPU が空いている専用環境で、data/bridge/cost_requests.jsonl のレイテンシとピークVRAMを測る")
    ap.add_argument("--flip", action="store_true", help="選択肢を並べ替えた入力（data/bridge/flip_requests.jsonl）だけを推論する")
    ap.add_argument("--with-smoke", action="store_true", help="モデルごとにスモークテストを先に行い、smoke_ok のときだけ全件を流す")
    args = ap.parse_args()
    if not REQUESTS.is_file():
        raise SystemExit("run `uv run python -m src.bridge_export` first")
    cfg = models_config()
    for name in args.models:
        modes = [True, False] if args.with_smoke else [args.smoke]
        for smoke in modes:
            print(f"== {name} ({'smoke' if smoke else 'full'})", flush=True)
            try:
                rec = run_model(cfg[name], smoke, flip=args.flip, cost=args.cost)
            except Exception as exc:  # noqa: BLE001 - 次のモデルへ進む
                rec = {"status": "runner_error", "reason": f"{type(exc).__name__}: {exc}"[:500]}
            rec["finished_at"] = now()
            key = name + ("#flip" if args.flip else "#cost" if args.cost else "")
            set_status(f"{key}#smoke" if smoke else key, rec)
            print(f"   -> {rec['status']} {rec.get('reason') or ''}", flush=True)
            if smoke and rec["status"] != "smoke_ok":
                if not args.smoke:
                    set_status(key, {**rec, "status": rec["status"], "note": "smoke test failed; full run skipped"})
                break


if __name__ == "__main__":
    main()
