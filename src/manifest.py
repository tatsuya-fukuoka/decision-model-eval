"""results/manifest.json を作成・更新する（指示書 §2）。

- モデル／データセットのリビジョン（コミットSHA）、ライセンス、gated の有無、取得日時
- 外部リポジトリ（third_party）の固定コミット
- ハードウェア、ドライバ、Python、パッケージ（pip freeze 相当）、乱数 seed

HF トークンは読み書きしない（匿名アクセスのみ）。既存の manifest にある
code_review / model_status などの節は上書きせずに保持する。

    uv run python -m src.manifest
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "manifest.json"
SEED = 20261107

DATASETS = [
    {"hf_id": "hotchpotch/S1MB-japanese-v1", "role": "ステージ1 評価データ（S1MB japanese-v1、27タスク）"},
    {"hf_id": "hotchpotch/saihai-v1-dataset", "role": "saihai 学習データ（重なり特定の根拠としてカードとmanifestのみ参照。中身は使わない）"},
]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hub_info(repo_id: str, repo_type: str) -> dict[str, Any]:
    """匿名で Hub のメタデータを取得する。失敗は例外にせず error として記録する。"""
    from huggingface_hub import HfApi

    api = HfApi(token=False)
    try:
        info = (api.dataset_info if repo_type == "dataset" else api.model_info)(
            repo_id, files_metadata=True
        )
    except Exception as exc:  # noqa: BLE001 - 失敗も記録対象
        return {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:200]}", "retrieved_at": now()}
    card = getattr(info, "card_data", None) or {}
    card = card.to_dict() if hasattr(card, "to_dict") else dict(card)
    siblings = info.siblings or []
    size = sum((s.size or 0) for s in siblings)
    weights = sum((s.size or 0) for s in siblings if s.rfilename.endswith((".safetensors", ".bin", ".pt", ".gguf")))
    return {
        "status": "ok",
        "revision": info.sha,
        "last_modified": info.last_modified.isoformat() if info.last_modified else None,
        "license": card.get("license"),
        "license_name": card.get("license_name"),
        "gated": getattr(info, "gated", None),
        "private": getattr(info, "private", None),
        "base_model": card.get("base_model"),
        "languages": card.get("language"),
        "total_file_bytes": size,
        "weight_file_bytes": weights,
        "python_files": sorted(s.rfilename for s in siblings if s.rfilename.endswith(".py")),
        "retrieved_at": now(),
    }


def environment() -> dict[str, Any]:
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    gpu_usage = run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"])
    cuda = run(["nvidia-smi"])
    cuda_version = None
    if cuda:
        for token in cuda.split("|"):
            if "CUDA Version" in token:
                cuda_version = token.split("CUDA Version:")[-1].strip()
    freeze = run(["uv", "pip", "freeze"])
    return {
        "recorded_at": now(),
        "os": run(["sh", "-c", ". /etc/os-release && echo $PRETTY_NAME"]),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "cpu_model": run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]),
        "memory_total": run(["sh", "-c", "grep MemTotal /proc/meminfo | awk '{print $2\" kB\"}'"]),
        "gpu": gpu,
        "cuda_driver_version": cuda_version,
        "gpu_usage_at_record_time": gpu_usage,
        "python": platform.python_version(),
        "uv": run(["uv", "--version"]),
        "packages_main_env": freeze.splitlines() if freeze else None,
        "seed": SEED,
        "hf_token_used": False,
    }


def third_party() -> dict[str, Any]:
    config = yaml.safe_load((ROOT / "configs" / "third_party.yaml").read_text(encoding="utf-8"))
    out = {}
    for name, entry in config.items():
        path = ROOT / "third_party" / ("S1MB" if name == "s1mb" else name)
        actual = run(["git", "-C", str(path), "rev-parse", "HEAD"])
        out[name] = {**entry, "checked_out": actual, "matches_pin": actual == entry["commit"]}
    return out


def all_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tier in ("tier1", "tier2", "tier3"):
        for m in config.get(tier, []):
            rows.append({**m, "tier": tier})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hub", action="store_true", help="Hub への問い合わせをしない（環境情報のみ更新）")
    args = parser.parse_args()

    manifest = load_json(MANIFEST)
    config = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    manifest["generated_by"] = "src/manifest.py"
    manifest["environment"] = environment()
    manifest["third_party"] = third_party()
    manifest["excluded_models"] = config.get("excluded", [])

    if not args.no_hub:
        models = manifest.get("models", {})
        for m in all_models(config):
            entry = {k: v for k, v in m.items() if k not in {"hf_id"}}
            if "hf_id" in m:
                previous = models.get(m["name"], {}).get("hub", {})
                if previous.get("status") == "ok" and previous.get("pinned"):
                    entry["hub"] = previous  # 固定済みのリビジョンは変えない
                else:
                    entry["hub"] = {**hub_info(m["hf_id"], "model"), "pinned": True}
                entry["hf_id"] = m["hf_id"]
            models[m["name"]] = {**models.get(m["name"], {}), **entry}
        manifest["models"] = models
        datasets = manifest.get("datasets", {})
        for d in DATASETS:
            previous = datasets.get(d["hf_id"], {})
            if previous.get("status") == "ok" and previous.get("pinned"):
                continue
            datasets[d["hf_id"]] = {"role": d["role"], **hub_info(d["hf_id"], "dataset"), "pinned": True}
        manifest["datasets"] = datasets
    write_json(MANIFEST, manifest)
    print(f"wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
