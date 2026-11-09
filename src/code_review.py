"""カスタムコードのレビュー結果を results/manifest.json の code_review 節に記録する。

    uv run python -m src.code_review

data/remote_code/<model>/ に固定リビジョンの .py を取得し（無ければ取得）、
SHA-256 と危険パターンの該当行を機械的に記録する。判定と所見は configs/code_review.yaml（人が読んだ結果）。
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "network": r"^\s*(import|from)\s+(socket|requests|urllib|http|httpx|aiohttp|websockets?)\b|urlopen",
    "shell": r"subprocess\.|os\.system|os\.popen|os\.exec|os\.spawn|^\s*(import|from)\s+(subprocess|pty)\b",
    "dynamic_exec": r"(^|[^.\w])eval\(|(^|[^.\w])exec\(|__import__|^\s*(import|from)\s+(marshal|pickle|dill|ctypes)\b|torch\.load",
    "hub_download": r"hf_hub_download|snapshot_download",
    "server": r"serve_forever|HTTPServer",
}


def fetch(hf_id: str, revision: str, files: list[str], dest: Path) -> None:
    for f in files:
        path = dest / f
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{hf_id}/resolve/{revision}/{f}"
        subprocess.run(["curl", "-sfL", url, "-o", str(path)], check=True)


def scan(dest: Path, files: list[str]) -> tuple[dict, dict]:
    hashes, hits = {}, {k: [] for k in PATTERNS}
    for f in files:
        data = (dest / f).read_bytes()
        hashes[f] = hashlib.sha256(data).hexdigest()
        for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if re.search(pattern, line):
                    hits[kind].append(f"{f}:{n}")
    return hashes, {k: v for k, v in hits.items() if v}


def main() -> None:
    manifest_path = ROOT / "results" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = yaml.safe_load((ROOT / "configs" / "code_review.yaml").read_text(encoding="utf-8"))
    out = {"reviewed_at": review["reviewed_at"], "method": "configs/code_review.yaml の冒頭コメント参照", "models": {}}
    for name, entry in review["models"].items():
        hub = manifest["models"][name]["hub"]
        files = hub.get("python_files") or []
        dest = ROOT / "data" / "remote_code" / name
        fetch(manifest["models"][name]["hf_id"], hub["revision"], files, dest)
        hashes, hits = scan(dest, files)
        out["models"][name] = {
            "revision": hub["revision"], **entry, "python_files_sha256": hashes, "pattern_hits": hits,
        }
    manifest["code_review"] = out
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, e in out["models"].items():
        print(f"{name:20} {e['verdict']:9} files={len(e['python_files_sha256'])} hits={ {k: len(v) for k, v in e['pattern_hits'].items()} }")


if __name__ == "__main__":
    main()
