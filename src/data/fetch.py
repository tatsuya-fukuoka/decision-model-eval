"""評価データを固定リビジョンで取得する（HF トークンは使わない）。

    uv run python -m src.data.fetch

- S1MB-japanese-v1：results/manifest.json に固定したリビジョンを data/hf/ に取得
- bench_ja：scripts/setup_third_party.sh で固定コミットの sokudan を取得済みであること
取得したデータは .gitignore 済みで、コミット・再配布しない。
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download

from . import bench_ja

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    manifest = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    repo = "hotchpotch/S1MB-japanese-v1"
    revision = manifest["datasets"][repo]["revision"]
    path = snapshot_download(repo, repo_type="dataset", revision=revision,
                             local_dir=ROOT / "data" / "hf" / "S1MB-japanese-v1", token=False)
    print(f"{repo}@{revision} -> {path}")
    if not bench_ja.DEFAULT_PATH.is_file():
        raise SystemExit("bench_ja がありません。scripts/setup_third_party.sh を先に実行してください")
    digest = bench_ja.sha256()
    manifest["datasets"]["sokudan/bench_ja"] = {
        "role": "ステージ1 評価データ（bench_ja、300件×3設問）",
        "source": "https://github.com/hiroki-abe-58/sokudan/blob/eccded37532d8a9afe5d6715ab62231a24366e65/data/bench_ja.jsonl",
        "revision": "eccded37532d8a9afe5d6715ab62231a24366e65",
        "license": "CC BY 4.0（評価専用）",
        "sha256": digest,
        "status": "ok",
        "pinned": True,
    }
    (ROOT / "results" / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"bench_ja sha256={digest}")


if __name__ == "__main__":
    main()
