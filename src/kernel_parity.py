"""Hub カーネル（kernels-community/fla v1）の有無で結果が一致するかの確認（ユーザー承認 2026-11-09）。

判定基準は docs/kernel_parity_criteria.md（実行前に固定）。root で実行し、推論は専用の非 root ユーザーで行う。

    uv run python -m src.kernel_parity setup            # ユーザー・評価用ディレクトリの作成と、権限・GPU の確認
    uv run python -m src.kernel_parity record-review    # fla v1 の確認結果を results/manifest.json に記録
    uv run python -m src.kernel_parity run [--models ...] [--conds nk1 nk2 k f32] [--smoke]
    uv run python -m src.kernel_parity compare          # 組 A〜D の統計量と判定

出力：
- results/kernel_parity/raw/<cond>/<model>/{s1mb,bench_ja}/<task>.jsonl（ID と確率だけ）
- results/kernel_parity/runs/<model>__<cond>.json（実行記録：状態・設定・ガードの記録・Triton キャッシュ）
- results/kernel_parity/comparison.csv、judgement.json
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from . import import_runs
from .data import bench_ja, s1mb
from .metrics import bench_ja_scores
from .metrics.s1mb_scores import aggregate, task_scores
from .results import RAW, path_for, predictions_index, read_rows

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "third_party" / "S1MB" / "evaluator"
OUT = ROOT / "results" / "kernel_parity"
USER = "kparity"
WORK = Path("/srv/kparity")
KERNELS_RO = Path("/srv/kparity-ro/kernels")
# 評価器のデータの読み取り専用コピー（root 所有）。公式評価器はデータディレクトリの2つ上の tmp/ にロックファイルを作るので、
# ここに置くとロックは評価用ディレクトリの tmp/ にできる（third_party/ には書き込まない）
S1MB_DATA = WORK / "s1mb_ro" / "data"
FLA_REPO, FLA_SHA = "kernels-community/fla", "24d8640db48c804a224b5da987b81093ff7136d1"
FLA_VARIANT = (KERNELS_RO / "kernels--kernels-community--fla" / "snapshots" / FLA_SHA / "build" / "torch-cuda")
MODELS = ["d3-nano", "d3-mini", "clef-flash", "d3-lite", "d3-flash"]
ENVS = {"clef-flash": "kparity-clef"}  # それ以外は kparity-open
CONDS = {
    "nk1": ["--no-hub-kernels"],
    "nk2": ["--no-hub-kernels"],
    # ユーザー判断 2026-11-09：activation（コンパイル済み .so）は実行しない。fla のみ適用
    "k": ["--exclude-kernel-repo", "kernels-community/activation"],
    "f32": ["--no-hub-kernels", "--fp32-backbone"],
}
# docs/kernel_parity_criteria.md と同じ値（実行前に固定）
THRESHOLDS = {"s1": 0.005, "s2": 0.01, "s3": 0.005}
C4_FACTOR = 2.0
BENCH_JA_PRIMARY = [("bench_ja-department", "accuracy"), ("bench_ja-urgency", "accuracy"), ("bench_ja-urgency", "rps"),
                    ("bench_ja-churn", "auroc"), ("bench_ja-churn-para1", "auroc")]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def non_overlap() -> list[str]:
    return json.loads((ROOT / "results" / "overlap_tasks.json").read_text(encoding="utf-8"))["non_overlap_tasks"]


def model_spec(name: str) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    for section in ("models", "tier2", "tier3", "extended"):
        for m in cfg.get(section) or []:
            if m.get("name") == name:
                return m
    for v in cfg.values():
        if isinstance(v, list):
            for m in v:
                if isinstance(m, dict) and m.get("name") == name:
                    return m
    raise KeyError(name)


def revision(name: str) -> str:
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    return manifest["models"][name]["hub"]["revision"]


def child_env(run_dir: Path | None = None) -> dict[str, str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(WORK / "home"),
        "TMPDIR": str(WORK / "tmp"),
        "XDG_CACHE_HOME": str(WORK / "home" / ".cache"),
        "LANG": "C.UTF-8",
        "HF_HOME": str(WORK / "hf_home"),
        "HF_HUB_CACHE": str(ROOT / "hf_cache" / "hub"),
        "HF_MODULES_CACHE": str(WORK / "hf_modules"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "KERNELS_CACHE": str(KERNELS_RO),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "TORCHINDUCTOR_CACHE_DIR": str(WORK / "tmp" / "inductor"),
    }
    if run_dir is not None:
        env["TRITON_CACHE_DIR"] = str(run_dir / "triton_cache")
    return env


def as_user(cmd: list[str], env: dict[str, str]) -> list[str]:
    """専用ユーザーに切り替え、権限を落とし、環境変数を空にしてから実行するコマンド列。"""
    return ["setpriv", f"--reuid={USER}", f"--regid={USER}", "--init-groups", "--inh-caps=-all",
            "--bounding-set=-all", "--no-new-privs", "env", "-i", *[f"{k}={v}" for k, v in env.items()], *cmd]


def python_for(model: str) -> Path:
    return ROOT / "envs" / ENVS.get(model, "kparity-open") / ".venv" / "bin" / "python"


# --- setup ---------------------------------------------------------------------------------------------


def setup() -> dict:
    try:
        pwd.getpwnam(USER)
    except KeyError:
        subprocess.run(["useradd", "--system", "--no-create-home", "--home-dir", str(WORK / "home"),
                        "--shell", "/usr/sbin/nologin", USER], check=True)
    pw = pwd.getpwnam(USER)
    for d in ("", "home", "tmp", "hf_home", "hf_modules", "runs"):
        p = WORK / d
        p.mkdir(parents=True, exist_ok=True)
        os.chown(p, pw.pw_uid, pw.pw_gid)
        p.chmod(0o700)
    src_data = EVAL / "data"
    if S1MB_DATA.exists():
        shutil.rmtree(S1MB_DATA)
    S1MB_DATA.parent.mkdir(exist_ok=True)
    os.chown(S1MB_DATA.parent, 0, 0)
    S1MB_DATA.parent.chmod(0o755)
    shutil.copytree(src_data, S1MB_DATA, symlinks=False)
    subprocess.run(["chmod", "-R", "a+rX,go-w", str(S1MB_DATA)], check=True)

    def tree_digest(d: Path) -> str:
        h = hashlib.sha256()
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            h.update(str(f.relative_to(d)).encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
        return h.hexdigest()

    data_copy = {"source": str(src_data.relative_to(ROOT)), "copy": str(S1MB_DATA),
                 "sha256_tree_source": tree_digest(src_data), "sha256_tree_copy": tree_digest(S1MB_DATA)}
    data_copy["identical"] = data_copy["sha256_tree_source"] == data_copy["sha256_tree_copy"]
    checks = {}
    env = child_env()
    py = str(python_for("d3-nano"))

    def probe(name: str, cmd: list[str], expect_ok: bool) -> None:
        r = subprocess.run(as_user(cmd, env), capture_output=True, text=True)
        checks[name] = {"returncode": r.returncode, "expected": "成功" if expect_ok else "失敗",
                        "as_expected": (r.returncode == 0) == expect_ok, "stdout": r.stdout.strip()[-400:]}

    probe("gpu", [py, "-I", "-c", "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0), "
                                  "torch.cuda.get_device_capability(0))"], True)
    probe("id", ["id"], True)
    probe("caps", ["grep", "-E", "^Cap(Eff|Prm|Bnd)|NoNewPrivs", "/proc/self/status"], True)
    probe("env_names", ["sh", "-c", "env | cut -d= -f1 | sort | tr '\\n' ' '"], True)
    probe("read_root_home", ["ls", "/root"], False)
    # 隔離の確認：実行ユーザーから認証情報ファイルが読めないことを確かめる（読めたら隔離失敗）
    probe("read_gh_hosts", ["test", "-r", "/root/.config/gh/hosts.yml"], False)
    probe("write_repo", ["touch", str(ROOT / ".kparity_write_test")], False)
    probe("write_hf_cache", ["touch", str(ROOT / "hf_cache" / "hub" / ".kparity_write_test")], False)
    probe("write_kernels_ro", ["touch", str(KERNELS_RO / ".kparity_write_test")], False)
    probe("write_s1mb_data_copy", ["touch", str(S1MB_DATA / ".kparity_write_test")], False)
    probe("write_work", ["touch", str(WORK / "tmp" / ".kparity_write_test")], True)
    rec = {"recorded_at": now(), "user": USER, "uid": pw.pw_uid, "work_dir": str(WORK),
           "kernels_dir": str(KERNELS_RO), "s1mb_data": data_copy, "checks": checks,
           "all_as_expected": all(c["as_expected"] for c in checks.values()) and data_copy["identical"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "setup.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return rec


# --- record-review -------------------------------------------------------------------------------------


def record_review() -> dict:
    meta_raw = (FLA_VARIANT / "metadata.json").read_bytes()
    meta = json.loads(meta_raw)
    digests = meta["digest"]["files"]
    files = sorted(str(p.relative_to(FLA_VARIANT)) for p in FLA_VARIANT.rglob("*") if p.is_file())
    present = set(files) - {"metadata.json", "metadata.json.sigstore"}
    mismatch = [f for f, d in digests.items()
                if base64.b64encode(hashlib.sha256((FLA_VARIANT / f).read_bytes()).digest()).decode() != d]
    py = str(python_for("d3-nano"))
    sig = subprocess.run([py, "-I", "-c", "import sys;from pathlib import Path;from kernels.verify import verify_variant;"
                          "print(type(verify_variant(Path(sys.argv[1]))).__name__)", str(FLA_VARIANT)],
                         capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "HOME": "/root"})
    scan = subprocess.run(["python3", "-I", str(ROOT / "scripts" / "static_scan.py"), str(FLA_VARIANT)],
                          capture_output=True, text=True, check=True).stdout.splitlines()
    summary = json.loads(scan[0])
    hits = scan[1:]
    rec = {
        "reviewed_at": now()[:10],
        "repo": FLA_REPO, "repo_type": "kernel", "branch": "v1", "revision": FLA_SHA,
        "variant": "build/torch-cuda",
        "contents": ("Python / Triton ソースのみ（.py 290、metadata.json、metadata.json.sigstore、symbols.json）。"
                     "コンパイル済みバイナリ（.so など）は含まない。GPU 用のコードは実行時に Triton がコンパイルする"),
        "files_total": len(files), "py_files": summary["files"], "py_lines": summary["lines"],
        "metadata_sha256": hashlib.sha256(meta_raw).hexdigest(),
        "metadata": {k: v for k, v in meta.items() if k != "digest"},
        "digest_check": {"algorithm": meta["digest"]["algorithm"], "entries": len(digests),
                         "missing": sorted(set(digests) - present), "extra": sorted(present - set(digests)),
                         "mismatch": mismatch},
        "signature_check": {
            "tool": "kernels 0.16.0 kernels.verify.verify_variant（sigstore 4.5.0）",
            "policy": "OIDC 発行者 token.actions.githubusercontent.com、リポジトリ github.com/huggingface/kernels-community、"
                      "署名ワークフロー build.yaml / build-mac.yaml / build-windows.yaml / sign-old-builds.yaml（refs/heads/main）",
            "result": sig.stdout.strip() or f"error: {sig.stderr.strip()[-300:]}",
        },
        "kernels_runtime_note": ("kernels 0.16.0 は読み込み時に署名を検証しない（オフラインでは発行元の確認もスキップ）。"
                                 "そのため取得後に上記の照合を別途行った"),
        "static_scan": {"tool": "scripts/static_scan.py（AST による全ファイルの走査、実行しない）",
                        "top_level_imports": summary["top_level_imports"], "n_hits": len(hits)},
        "review_findings": [
            "ネットワーク送信：なし（socket・urllib・requests・http などの import なし）。URL は docstring・コメントの文献リンクだけ",
            "シェル・プロセス実行：なし（subprocess・os.system・os.exec* なし）",
            "eval / exec / compile：なし。動的 import は ops/backends/__init__.py の __import__ 1か所で、"
            "対象は内部の演算バックエンドのモジュール名（fla.ops.<op>.backends）。ImportError は無視される",
            "ファイル書き込み：なし。読み込みは ops/utils/cache.py の autotune 設定 JSON（FLA_CACHE_MODE で有効化。既定は無効）だけ",
            "環境変数：FLA_* などの設定を読む。書き込みは utils/_device.py の TRITON_F32_DEFAULT='ieee' だけで、"
            "compute capability 8 未満（Ampere より前）の NVIDIA GPU のときに限る（本 GPU は 12.0 なので書き込まない）",
            "インラインアセンブリ：ops/utils/softplus.py（ex2/lg2/add/setp/mov の算術のみ）と ops/utils/op.py"
            "（SM100/SM120 向けの mov.f32 1命令。tl.dot の結果をそのまま渡す回避策）",
            "プロセス全体への影響：modules/fused_cross_entropy.py が torch.distributed の別名を1つ設定する"
            "（そのモジュールを import したときだけ。今回使う経路では import しない）",
            "今回使う経路：layers.py の FusedRMSNormGated、chunk_gated_delta_rule、recurrent_gated_delta_rule"
            "（transformers 5.17.0 の Qwen3.5 用対応表。causal_conv1d と rotary は公式アダプタが無効化）",
        ],
        "review_scope_note": ("290 ファイル・82,675 行のすべてを AST で走査し、該当箇所（.load の Triton 以外、environ、getattr、"
                              "__import__、open、インライン asm、URL 文字列など）をすべて目視で確認した。"
                              "全行を目視で通読したわけではない"),
        "cache_note": ("kernels のオフライン読み込みはスナップショット全体を要求するため、同じコミットの README.md・"
                       ".gitattributes・build/torch-xpu（Intel GPU 用）もキャッシュに置いた。CUDA 環境では torch-cuda が選ばれ、"
                       "torch-xpu は読み込まれない（実行記録の loaded_kernels で確認）"),
        "sm_120": "事前には判定できない（Triton 3.6.0 が実行時にコンパイルする）。実行結果で判断する",
    }
    manifest_path = ROOT / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["hub_kernel_review"] = {FLA_REPO: rec}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


# --- run -----------------------------------------------------------------------------------------------


def classify(text: str) -> tuple[str, str]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    tail = "\n".join(lines[-15:])
    low = text.lower()
    if "net_guard" in text and "blocked" in low:
        return "network_blocked", tail
    if "out of memory" in low or "outofmemoryerror" in low:
        return "oom", tail
    if "triton" in low and any(s in low for s in ("compil", "ptxas", "sm_120", "unsupported", "no kernel image")):
        return "triton_failed", tail
    return "failed", tail


def guard_summary(path: Path) -> dict:
    blocked, procs, unix = [], {}, 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["kind"] == "blocked":
                blocked.append(r)
            elif r["kind"] == "process":
                exe = os.path.basename(r["detail"].strip("'\" b"))
                procs[exe] = procs.get(exe, 0) + 1
            elif r["kind"] == "allowed_unix":
                unix += 1
    return {"blocked": len(blocked), "blocked_events": blocked[:5], "processes": procs, "unix_socket_connects": unix}


def run_one(model: str, cond: str, smoke: bool) -> dict:
    spec = model_spec(model)
    rev = revision(model)
    run_id = f"kparity-{model}-{cond}{'-smoke' if smoke else ''}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}"
    pw = pwd.getpwnam(USER)
    rd = WORK / "runs" / run_id
    rd.mkdir(parents=True)
    os.chown(rd, pw.pw_uid, pw.pw_gid)
    env = child_env(rd)
    snapshot = next((ROOT / "hf_cache" / "hub" / f"models--{spec['hf_id'].replace('/', '--')}" / "snapshots").glob(
        f"{rev}*"))
    args = [a.replace("{snapshot}", str(snapshot)) for a in spec.get("s1mb_args", [])]
    tasks = non_overlap()
    cmd = [str(python_for(model)), "-I", str(ROOT / "scripts" / "net_guard.py"), "--log", str(rd / "guard.jsonl"),
           str(ROOT / "scripts" / "s1mb_bridge.py"), "--bench-ja", str(ROOT / "data" / "bridge" / "bench_ja_requests.jsonl"),
           "--bench-ja-out", str(rd / "bench_ja.jsonl"), *CONDS[cond]]
    if smoke:
        cmd += ["--bench-ja-limit", "2"]
    cmd += ["--", "--data-dir", str(S1MB_DATA), "run", "--adapter", spec["s1mb_adapter"], "--model", spec["hf_id"], "--revision", rev, *args,
            "--offline-dataset", "--category", "japanese-v1", "--output", str(rd / "s1mb"), "--run-id", run_id]
    for t in (tasks[:2] if smoke else tasks):
        cmd += ["--benchmark", t]
    if smoke:
        cmd += ["--limit", "2"]
    rec = {"model": model, "cond": cond, "smoke": smoke, "run_id": run_id, "revision": rev, "started_at": now(),
           "flags": CONDS[cond], "env": ENVS.get(model, "kparity-open"), "user": USER,
           "child_env_names": sorted(env)}
    log_path = rd / "log.txt"
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(as_user(cmd, env), cwd=EVAL, stdout=log, stderr=subprocess.STDOUT)
    rec.update(elapsed_seconds=round(time.perf_counter() - t0, 1), returncode=proc.returncode, finished_at=now())
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rec["guard"] = guard_summary(rd / "guard.jsonl")
    tc = rd / "triton_cache"
    rec["triton_cache"] = {"exists": tc.is_dir(),
                           "cubin_files": len(list(tc.rglob("*.cubin"))) if tc.is_dir() else 0,
                           "kernel_names": sorted({p.stem for p in tc.rglob("*.cubin")})[:60] if tc.is_dir() else []}
    s1_dir = rd / "s1mb" / run_id
    results = sorted(s1_dir.glob("*.json")) if s1_dir.is_dir() else []
    if results:
        first = json.loads(results[0].read_text(encoding="utf-8"))
        rec["adapter_settings"] = first.get("model", {}).get("settings") or first.get("model")
    expected = 2 if smoke else len(tasks)
    if proc.returncode != 0 or len(results) < expected:
        rec["status"], rec["reason"] = classify(text)
    else:
        rec["status"] = "smoke_ok" if smoke else "ok"
    rec["log_tail"] = "\n".join(text.splitlines()[-12:])
    if not smoke and results:
        runs_dir = OUT / "s1mb_runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        dst = runs_dir / run_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(s1_dir, dst)
        root = OUT / "raw" / cond
        s1 = import_runs.import_s1mb(model, run_id, runs_dir=runs_dir, root=root)
        rec["s1mb"] = {t: v for t, v in s1["benchmarks"].items() if v.get("status") != "missing"}
        if (rd / "bench_ja.jsonl").is_file():
            rec["bench_ja"] = import_runs.import_bench_ja(model, rd / "bench_ja.jsonl", root=root)
    rec["work_dir"] = str(rd)
    out = OUT / "runs" / f"{model}__{cond}{'__smoke' if smoke else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return rec


# --- compare -------------------------------------------------------------------------------------------


def _raw_root(cond: str) -> Path:
    return RAW if cond == "published" else OUT / "raw" / cond


def _preds(cond: str, model: str, dataset: str, task: str) -> dict | None:
    p = path_for(model, dataset, task, _raw_root(cond))
    return predictions_index(read_rows(p)) if p.is_file() else None


def _pair_stats(a: dict, b: dict) -> dict:
    """a, b：(case_id, question_id) -> 予測。両方で予測が出た判断だけを使う。"""
    keys = sorted(set(a) | set(b))
    fa = {k for k in keys if (a.get(k) or {}).get("probs") is None}
    fb = {k for k in keys if (b.get(k) or {}).get("probs") is None}
    both = [k for k in keys if k not in fa and k not in fb]
    dis, maxdiff = 0, []
    for k in both:
        pa, pb = a[k]["probs"], b[k]["probs"]
        ids = list(pa)
        va, vb = np.array([pa[i] for i in ids]), np.array([pb[i] for i in ids])
        dis += int(np.argmax(va) != np.argmax(vb))
        maxdiff.append(float(np.abs(va - vb).max()))
    md = np.array(maxdiff) if maxdiff else np.zeros(1)
    return {"n_both": len(both), "argmax_disagree": dis, "s1": dis / len(both) if both else None,
            "s2": float(np.percentile(md, 99)), "max_abs_diff": float(md.max()), "mean_abs_diff": float(md.mean()),
            "failed_a": len(fa), "failed_b": len(fb), "same_failures": fa == fb}


def _s1mb_side(cond: str, model: str, data: dict) -> tuple[dict | None, float | None]:
    merged, per_task = {}, {}
    for t, (bench, targets) in data.items():
        p = _preds(cond, model, "s1mb", t)
        if p is None:
            return None, None
        merged.update({(t, *k): v for k, v in p.items()})
        per_task[t] = task_scores(bench.task, targets, p)
    agg = aggregate(per_task, {t: b.task for t, (b, _) in data.items()}, list(data))
    return merged, agg["task_avg"]


def _bench_ja_side(cond: str, model: str) -> tuple[dict | None, dict]:
    merged, metrics = {}, {}
    for task, qtype in {**bench_ja.TASKS, **bench_ja.VARIANT_TASKS}.items():
        p = _preds(cond, model, "bench_ja", task)
        if p is None:
            return None, {}
        merged.update({(task, *k): v for k, v in p.items()})
        _, targets = bench_ja.load_task(task)
        if any((p.get((t.case_id, t.question_id)) or {}).get("probs") is None for t in targets):
            continue
        P = np.array([[p[(t.case_id, t.question_id)]["probs"][o] for o in t.option_ids] for t in targets])
        y = np.array([int(np.argmax(t.probs)) for t in targets])
        metrics[task] = bench_ja_scores.score_task(qtype, P, y)
    return merged, metrics


def compare() -> dict:
    tasks = non_overlap()
    data = {t: (b, tg) for t in tasks for b, _, tg in [s1mb.load_benchmark(t)]}
    pairs = {"A": ("k", "nk1"), "B": ("nk1", "nk2"), "C": ("nk1", "f32"), "D": ("nk1", "published")}
    rows, judgement = [], {}
    for model in MODELS:
        status = {}
        for c in CONDS:
            f = OUT / "runs" / f"{model}__{c}.json"
            status[c] = json.loads(f.read_text(encoding="utf-8"))["status"] if f.is_file() else "未実行"
        sides = {}
        for c in [*CONDS, "published"]:
            s_preds, tavg = _s1mb_side(c, model, data)
            b_preds, bmet = _bench_ja_side(c, model)
            sides[c] = {"s1mb": s_preds, "task_avg": tavg, "bench_ja": b_preds, "bj_metrics": bmet}
        stats = {}
        for pname, (x, y) in pairs.items():
            for scope in ("s1mb", "bench_ja"):
                if sides[x][scope] is None or sides[y][scope] is None:
                    stats[(pname, scope)] = None
                    continue
                st = _pair_stats(sides[x][scope], sides[y][scope])
                if scope == "s1mb":
                    ta, tb = sides[x]["task_avg"], sides[y]["task_avg"]
                    st["s3"] = None if ta is None or tb is None else abs(ta - tb)
                    st["task_avg_a"], st["task_avg_b"] = ta, tb
                else:
                    diffs = []
                    for task, met in BENCH_JA_PRIMARY:
                        va = (sides[x]["bj_metrics"].get(task) or {}).get(met)
                        vb = (sides[y]["bj_metrics"].get(task) or {}).get(met)
                        if va is not None and vb is not None:
                            diffs.append(abs(va - vb))
                    st["s3"] = max(diffs) if len(diffs) == len(BENCH_JA_PRIMARY) else None
                stats[(pname, scope)] = st
                rows.append({"model": model, "pair": pname, "a": x, "b": y, "scope": scope,
                             **{k: v for k, v in st.items()}})
        # 判定
        if status["k"] == "triton_failed":
            judgement[model] = {"verdict": "fla のみ適用の構成では動作せず（Triton がこの GPU で動かなかった）",
                                "status": status}
            continue
        verdict, failed_criteria, details = "一致（fla のみ適用。activation は未適用）", [], {}
        for scope in ("s1mb", "bench_ja"):
            a, c = stats.get(("A", scope)), stats.get(("C", scope))
            if a is None:
                verdict = "未判定（カーネルあり、またはカーネルなしの結果がない）"
                failed_criteria.append(f"{scope}: 組 A の結果なし")
                continue
            checks = {}
            for i, s in enumerate(("s1", "s2", "s3"), start=1):
                checks[f"基準{i}"] = a[s] is not None and a[s] <= THRESHOLDS[s]
            if c is None:
                checks["基準4"] = None
            else:
                ok4 = True
                for s in ("s1", "s2", "s3"):
                    if a[s] is None or c[s] is None:
                        ok4 = None if ok4 is not False else ok4
                        continue
                    if (c[s] == 0 and a[s] != 0) or (c[s] > 0 and a[s] > C4_FACTOR * c[s]):
                        ok4 = False
                checks["基準4"] = ok4
            checks["基準5"] = a["same_failures"]
            details[scope] = checks
            for name, ok in checks.items():
                if ok is None:
                    failed_criteria.append(f"{scope}: {name} 未判定")
                    if verdict.startswith("一致"):
                        verdict = "未判定"
                elif not ok:
                    failed_criteria.append(f"{scope}: {name}")
                    verdict = "一致と書けない（基準を満たさない）"
        judgement[model] = {"verdict": verdict, "failed_criteria": failed_criteria, "checks": details,
                            "status": status}
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["model", "pair", "a", "b", "scope", "n_both", "argmax_disagree", "s1", "s2", "s3", "max_abs_diff",
            "mean_abs_diff", "failed_a", "failed_b", "same_failures", "task_avg_a", "task_avg_b"]
    with open(OUT / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    out = {"computed_at": now(), "kernel_scope": "fla のみ適用（activation は未適用）。公式構成（activation を含む）との一致は未確認",
           "thresholds": THRESHOLDS, "c4_factor": C4_FACTOR,
           "criteria_doc": "docs/kernel_parity_criteria.md", "models": judgement}
    (OUT / "judgement.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("record-review")
    r = sub.add_parser("run")
    r.add_argument("--models", nargs="+", default=MODELS)
    r.add_argument("--conds", nargs="+", default=list(CONDS))
    r.add_argument("--smoke", action="store_true")
    sub.add_parser("compare")
    args = ap.parse_args()
    if args.cmd == "setup":
        print(json.dumps(setup(), ensure_ascii=False, indent=1))
    elif args.cmd == "record-review":
        rec = record_review()
        print(json.dumps({k: rec[k] for k in ("revision", "files_total", "digest_check", "signature_check")},
                         ensure_ascii=False, indent=1))
    elif args.cmd == "run":
        for model in args.models:
            for cond in args.conds:
                rec = run_one(model, cond, args.smoke)
                print(f"{model} {cond}: {rec['status']} ({rec['elapsed_seconds']}s) "
                      f"guard_blocked={rec['guard']['blocked']} cubin={rec['triton_cache']['cubin_files']}", flush=True)
    elif args.cmd == "compare":
        out = compare()
        for m, j in out["models"].items():
            print(m, j["verdict"], j.get("failed_criteria", ""))


if __name__ == "__main__":
    main()
