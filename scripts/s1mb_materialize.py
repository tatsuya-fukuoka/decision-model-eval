"""S1MB 公式評価器のデータディレクトリに、固定リビジョンの japanese-v1 を展開する。

公式の ensure_dataset は既定で main を解決するため、manifest.json のリビジョンを
書いた source config を渡して固定する。評価器の環境で実行する:

    HF_HOME=hf_cache envs/s1mb-base/.venv/bin/python -I scripts/s1mb_materialize.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "third_party" / "S1MB" / "evaluator"


def main() -> None:
    from s1mb.data import DATA_DIR
    from s1mb.dataset_source import ensure_dataset

    manifest = json.loads((ROOT / "results" / "manifest.json").read_text(encoding="utf-8"))
    rev = manifest["datasets"]["hotchpotch/S1MB-japanese-v1"]["revision"]
    config = json.loads((EVAL / "dataset-source.json").read_text())
    config["category_sources"]["japanese-v1"]["revision"] = rev
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "dataset-source.json"
        src.write_text(json.dumps(config))
        receipt = ensure_dataset(DATA_DIR, src, category="japanese-v1")
    if receipt["revision"] != rev:
        sys.exit(f"revision mismatch {receipt['revision']} != {rev}")
    print("materialized", rev, "into", DATA_DIR)


if __name__ == "__main__":
    main()
