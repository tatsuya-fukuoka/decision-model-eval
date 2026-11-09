# 実行コスト計測の手順（EC2 の専用環境、指示書 §5-4）

レイテンシと VRAM は、GPU を他のジョブと共有していない専用環境だけで測ります。共有環境で測った値は使いません。
この作業環境（GPU を別ジョブと共有、AWS の認証情報なし）では測っておらず、結果は「未測定」です。

## 前提

- NVIDIA GPU 1枚、CUDA 13 対応ドライバ、Linux x86_64。インスタンス種別、GPU 名、ドライバは結果に記録されます（`cuda_device`）。
- 計測中は、他の GPU プロセスを動かさないでください。`run_eval --cost` は計測の前後に
  `nvidia-smi --query-compute-apps` を確認します。計測前に他のプロセスがあれば `cost_not_measured` で止まり、計測後のプロセスは `gpu_processes_after` に残ります。
- HF トークンは不要です。モデルは固定リビジョンをトークンなしで取得します。

## 手順

```sh
git clone <this repo> && cd decision-eval && git checkout <commit>
uv sync
bash scripts/setup_third_party.sh
bash scripts/fetch_packages.sh
uv run python -m src.data.fetch
# S1MB 評価器の環境とデータ（README「環境構築」）
UV_PROJECT_ENVIRONMENT=$PWD/envs/s1mb-base/.venv uv sync --project third_party/S1MB/evaluator --frozen --no-dev --python 3.12
(cd third_party/S1MB/evaluator && HF_HOME=$PWD/../../../hf_cache ../../../envs/s1mb-base/.venv/bin/python -I ../../../scripts/s1mb_materialize.py)
# 計測用の入力（全モデル共通の120件＋ウォームアップ12件）
uv run python -m src.bridge_export
(cd third_party/S1MB/evaluator && HF_HUB_OFFLINE=1 HF_HOME=$PWD/../../../hf_cache \
  ../../../envs/s1mb-base/.venv/bin/python -I ../../../scripts/cost_export.py --out ../../../data/bridge/cost_requests.jsonl)
uv run python -m src.prefetch <models...>
# 計測（モデルを1つずつ順に実行する。並列にしない）
uv run python -m src.run_eval --cost --with-smoke --models saihai-v1-310m saihai-v1-8b sokudan-ja-310m \
  strands-decider-2b julia-1 laya-multilingual d3-nano d3-mini clef-flash
```

結果は `results/cost/<model>.json` に保存されます。

## 記録される値

- `latency_p50_s` / `latency_p95_s`：ウォームアップ12件を除いた120件の、1件あたりの `adapter.predict` の所要時間です。GPU は計測の終わりに同期します。全体の値と、型ごとの値（`by_type`）があります。
- `cuda_max_memory_allocated` / `cuda_max_memory_reserved`：PyTorch がプロセス内で確保したメモリのピークです。CUDA コンテキストの分は含みません。
- `weight_file_bytes`：固定したリビジョンの重みファイルの合計サイズです。

## 測れない項目（「未測定」と書きます）

- **バッチ 8 / 32**：公式アダプタの `predict` は1件ずつ呼ぶ経路なので測れません。
- **CPU だけでのレイテンシ**：公式アダプタの多くが CPU での実行を拒否する（"no CPU fallback"）ため測れません。encoder 系（saihai-310m、sokudan、Laya、Julia）で測る場合は、アダプタごとに別途確認します。
- **d3 / Clef**：Hub のカーネルを無効にしているため、値は公式の実行構成より遅い可能性があります。「参考値（公式カーネル未使用）」と書きます。
