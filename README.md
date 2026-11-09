# decision-model-eval

公開日本語データ（S1MB-japanese-v1、sokudan `bench_ja`）を使った、約9BまでのDecision Modelのzero-shot比較評価です。
仕様は [`docs/decision-model-eval-instructions.md`](docs/decision-model-eval-instructions.md) です。結果は一次スクリーニング用の材料にとどまり、特定の業務に最適なモデルを決めるものではありません。

- 採点：S1MB は公式評価器（`third_party/S1MB`、固定コミット）の式をそのまま使います。メイン環境の移植版（`src/metrics/s1mb_scores.py`）は、公式コードの出力と一致することを `tests/test_s1mb_official_parity.py` で確認しています。
- モデルの実行：モデルごとに隔離した仮想環境（`envs/<name>/.venv`）を使います。モデルとデータは、固定したリビジョンでトークンなしで取得します。実行は `HF_HUB_OFFLINE=1` で行います。
- 記録：
  - `results/manifest.json`：リビジョン、ライセンス、環境の freeze、コード確認の結果
  - `results/model_status.json`：モデルごとの成否と原因
  - `results/raw/`：件ごとの確率（ID・確率のみ。入力テキストと正解は含みません）
- 評価データ（`data/`、`third_party/`、`hf_cache/`）は git に含めず、再配布もしません。

## ライセンスとデータの扱い

- **このリポジトリのコード：** MIT License（`LICENSE`）。
- **評価データは含みません。** S1MB-japanese-v1 はタスクごとに元データのライセンスに従い、generalization-* の8タスクは CC BY-NC 4.0 です。bench_ja は CC BY 4.0（評価専用）です。データは上記の手順で各自取得してください。
- **`results/` には、これらのデータに由来する評価結果（件ID・予測確率・集計値）が含まれます。** 入力テキストと正解ラベルは含みません。非商用条件のデータを使って得た結果を含むため、利用にあたっては各データのライセンスを確認してください。
- 評価したモデルのライセンスは `report.md` §6 を参照してください。
- 結果は公開データでの一次スクリーニングです。レイテンシと VRAM は未測定です。

## 必要なもの

- Linux と NVIDIA GPU（CUDA 13 対応ドライバ）、`git`、`curl`、[`uv`](https://docs.astral.sh/uv/)（0.13 以上）
- HF トークンは不要です（認証が必要なモデルは `auth_required` と記録してスキップします）。

## 環境構築

```sh
uv sync                                   # メイン環境（集計・テスト。Python 3.12）
bash scripts/setup_third_party.sh         # S1MB 評価器・sokudan を固定コミットで third_party/ に取得
bash scripts/fetch_packages.sh            # 自作アダプタ用の PyPI パッケージを sha256 確認付きで取得
uv run python -m src.manifest             # モデル・データのリビジョン/ライセンスを解決して manifest に固定
uv run python -m src.data.fetch           # S1MB-japanese-v1 を固定リビジョンで data/hf/ に取得
uv run python -m src.code_review          # trust_remote_code 等のコード確認結果を manifest に記録
```

S1MB 公式評価器の環境と、評価器用のデータを用意します。

```sh
UV_PROJECT_ENVIRONMENT=$PWD/envs/s1mb-base/.venv \
  uv sync --project third_party/S1MB/evaluator --frozen --no-dev --python 3.12
(cd third_party/S1MB/evaluator && HF_HOME=$PWD/../../../hf_cache \
  ../../../envs/s1mb-base/.venv/bin/python -I ../../../scripts/s1mb_materialize.py)
```

## テストと採点の一致確認

```sh
uv run python -m src.run_baselines        # 多数派・ランダムのベースライン
uv run python -m src.parity_predictions   # 一致確認用の人工予測（正解入りのものは gitignore 対象）
(cd third_party/S1MB/evaluator && HF_HUB_OFFLINE=1 ../../../envs/s1mb-base/.venv/bin/python -I \
  ../../../scripts/s1mb_official_score.py parity-dirichlet parity-oracle-noisy baseline-majority baseline-random)
uv run pytest
```

## ステージ1の実行（指示書 §7 手順3・5）

```sh
uv run python -m src.bridge_export                          # bench_ja の入力を data/bridge/ に書き出す
uv run python -m src.prefetch saihai-v1-310m ...           # （任意）固定リビジョンのモデルを先に取得
uv run python -m src.run_eval --models saihai-v1-310m --smoke   # 各タスク2件のスモークテスト
uv run python -m src.run_eval --models saihai-v1-310m saihai-v1-8b julia-1 d3-nano d3-mini clef-flash \
  sokudan-ja-310m laya-multilingual strands-decider-2b     # 全件
uv run python -m src.stage1_tables --n-boot 1000           # results/tables/ に集計表と 95% 信頼区間
```

`run_eval` は、モデルごとに次の順で処理します。

1. コード確認が `approved` かを確かめます。
2. `configs/envs.yaml` に従って隔離環境を作ります（公式の `uv.lock` から `--frozen`）。
3. スナップショットを取得します。
4. `scripts/s1mb_bridge.py` を実行します。中身は公式の `s1mb run` と、同じアダプタのインスタンスでの bench_ja 推論です。
5. 結果を `results/raw/` に取り込みます。

失敗しても例外では止まりません。`results/model_status.json` に status と原因を記録して、次のモデルへ進みます。

## ステージ2と報告書

```sh
uv run python -m src.stage2_calibration --n-boot 1000      # 温度合わせ（17タスクの calibration 側のみ）、reliability diagram
uv run python -m src.bench_ja_churn                        # bench_ja 解約示唆の P(true) の分布
uv run python -m src.run_eval --flip --with-smoke --models <models...>   # 選択肢の並べ替え（入力は scripts/flip_export.py）
uv run python -m src.stage2_flip --n-boot 1000             # flip rate
uv run python -m src.make_report                           # report.md を docs/report_template.md と results/ から生成
```

レイテンシと VRAM は EC2 の専用環境で測ります（`docs/ec2_cost_runbook.md`）。

## 構成

| パス | 内容 |
| --- | --- |
| `configs/models.yaml` | 対象モデル（Tier）、アダプタ、`s1mb run` の引数、依存リポジトリ |
| `configs/envs.yaml` | 隔離環境の extra と追加パッケージ |
| `configs/code_review.yaml` | 実行前に読んだカスタムコードの判定と所見 |
| `adapters/` | 公式アダプタが無いモデル用の S1MB 形式アダプタ（sokudan、Strands） |
| `scripts/s1mb_bridge.py` | 公式評価器の環境で動く橋渡し（本リポジトリの `src` は import しない） |
| `src/data/` | S1MB・bench_ja のローダ（正解はモデル入力に含めない） |
| `src/metrics/` | 指標、S1MB の採点（移植版）、高速ブートストラップ |
| `src/calibration/` | 温度スケーリングと group_id 単位の分割 |
| `results/overlap_tasks.json` | saihai-v1-8b の学習データと重なる10タスクと、その根拠 |

## 注意

- レイテンシと VRAM は、他のジョブと GPU を共有した状態の値は使いません。GPU が空いた専用環境で別途測ります（未測定）。
- 有料 API（Microsoft-Decision-1）は承認がない限り使いません。
