"""固定リビジョンのモデルと依存リポジトリを hf_cache/ に先に取得する（トークンなし、実行はしない）。

    uv run python -m src.prefetch MODEL...
"""

import sys

from .manifest import load_json
from .run_eval import MANIFEST, ensure_dependencies, ensure_snapshot, models_config


def main(names: list[str]) -> None:
    cfg, manifest = models_config(), load_json(MANIFEST)
    for n in names:
        try:
            p = ensure_snapshot(cfg[n]["hf_id"], manifest["models"][n]["hub"]["revision"])
            deps = ensure_dependencies(cfg[n])
            print(f"ok {n} {p.name} deps={[d['hf_id'] + '@' + d['revision'][:12] for d in deps]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"failed {n}: {type(exc).__name__}: {exc}"[:400], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
