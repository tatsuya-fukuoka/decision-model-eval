# Decision Model 日本語評価：実装・実行指示書（公開データのみ・9Bクラスまで）

## 0. この指示書について

- 対象：コーディングエージェント
- 目的：日本語のDecision Modelを、**公開されている日本語評価セットだけ**で同一条件に揃えて比較評価するコードを実装し、実行し、結果を報告する。
- 範囲：自前の業務データでの評価は**行わない**。サイズは**約9Bまで**のモデルを対象にする。
- 背景：Decision Modelは、stateと型付きの質問（Choice / Noul / Score）を入力し、選択肢ごとの確率を1回のforward passで返すモデル。詳細は同梱の記事 `zenn-decision-model-survey.md` を参照する。
- 原則：
  1. **数値を創作しない。** 測れなかった項目は「未測定」と書く。
  2. **事実（実測・カード記載）と解釈を分けて書く。**
  3. 判断に迷う点、ライセンスや認証情報が絡む点は、実行を止めてユーザーに確認する。
- 重要な制約：公開セットは業務の分類そのものではない。結果は**一次スクリーニングの材料**であり、採用判断には使わない（報告書にそう明記する）。

## 1. 着手前にユーザーへ確認すること（不明なら質問してから進める）

| 項目 | 内容 | 未回答時の扱い |
| --- | --- | --- |
| 計算環境 | GPUの機種・VRAM、CPUコア数、OS | 実行環境を自動取得して記録し、収まらないモデルはスキップして理由を記録する。目安：9Bクラスのbf16推論には約20GB以上のVRAMが必要 |
| Hugging Faceトークン | 取得が必要なモデルがあるか | 認証が必要なモデルはスキップして記録する |
| 外部API利用の可否 | Microsoft-Decision-1（Foundry）。費用と認証情報が発生する | 許可があるまで使わない |
| 時間・予算の上限 | 全体の目安 | 上限が不明なら、Tier 1（§3）の完了までを最優先とする |

## 2. 安全・再現性のルール

- 対象モデルの多くは `trust_remote_code=True` を要求する。**リビジョン（コミットSHA）を固定**し、実行前にカスタムコードを読んで、ネットワーク送信やシェル実行がないことを確認する。隔離した仮想環境で実行する。
- すべてのモデル・データセットについて、取得時のリビジョン、ライセンス、取得日時を `results/manifest.json` に記録する。
- 乱数seed、ライブラリのバージョン（`pip freeze`）、CUDA／ドライバのバージョン、ハードウェア情報を記録する。
- 評価データ（S1MB、`bench_ja`）は**学習に使わない。** 再配布もしない。

## 3. 評価対象モデル

「zero-shot（追加学習なし）」で評価する。fine-tuningは範囲外（§9）。

### Tier 1（必須）

| モデル | Hugging Face ID | メモ（カード記載ベース） |
| --- | --- | --- |
| saihai-v1-310m | `hotchpotch/saihai-v1-310m` | 日本語・英語。Apache-2.0。`AutoModel.from_pretrained(..., trust_remote_code=True)` で読み込み、`model.systemone(request)` が `{"answers": ...}` を返す。Python 3.12+とtorch等の指定バージョンはカードで確認。CUDA BF16とCPU FP32で結果が異なる場合がある |
| saihai-v1-8b | `hotchpotch/saihai-v1-8b` | 日本語・英語。Apache-2.0。llama系（約8.6B）のカスタムコード。**このベンチマークの10タスクの元データのtrainを学習に含む**（§4-1） |
| sokudan-ja-310m | `GeneLab/sokudan-ja-310m` | 日本語のみ。Apache-2.0。`pip install sokudan`（Python 3.11）。使い方はPyPIのREADMEに従う |
| Laya multilingual | `convaiinnovations/laya-multilingual` | Apache-2.0。カードに「ScoreはPrimitiveの中で最も弱い」「Noulがtrueを過小報告する場合がある」「選択肢は約20個以下を推奨」と記載 |
| Julia-1 | `SupersonicLabs/Julia-1` | 多言語。Apache-2.0。選択肢は2〜20個。使い方はカード参照 |
| Strands Decider 2B | `StrandsAgents/strands-decider-2B-hobson-v19` | Qwen3.5-2B-Base + LoRA + readout head。Apache-2.0。コードはGitHubの `strands-labs/strands-decider` |
| Decision 3.0 d3-nano | `vllm-sr/d3-nano` | Qwen3.5-2Bベース。Apache-2.0。言語の記載なし |
| Decision 3.0 d3-mini | `vllm-sr/d3-mini` | Qwen3.5-4Bベース（約4.5B）。Apache-2.0。言語の記載なし |
| H2O-Lightning-4B | `h2oai/h2o-lightning-4b` | Qwen3.5-4Bベース。Apache-2.0。two-way読み出しには作者のvLLMビルドが必要。標準のvLLMはone-wayのみ。**本タスクではone-wayで評価し、two-wayは未実施と明記する** |
| Clef-Flash | `Cloudflare/clef-flash` | Qwen3.5-9Bのpost-train（約9.4B）、マルチモーダル。Apache-2.0。`AutoModelForMultimodalLM`、カスタムコード。レコードは `state` と `questions` を持ち、質問タイプは `noul` / `choice` / `score`。言語の記載なし。**テキストのみの入力で評価する** |

### Tier 2（時間・資源があれば）

| モデル | Hugging Face ID | メモ |
| --- | --- | --- |
| d3-lite | `vllm-sr/d3-lite` | Qwen3.5-0.8Bベース（約0.85B） |
| Decision 2.0 Nox 4B | `vllm-sr/Decision-2.0-Nox-4B` | Qwen3.5-4B-Baseベース。Apache-2.0。Decision 3.0との世代比較用 |
| Microsoft-Decision-1 | Foundry API | 9B級（Qwen3.5-9Bのpost-training）。**§1の承認がある場合のみ。** エンドポイントは `.../providers/microsoft/v1/systemone`、質問タイプは `noul` / `choice` / `score` |

### Tier 3（存在とライセンスを先に確認できた場合のみ）

次のモデルは、元の調査メモに名前があるが、**このプロジェクトでカードを確認できていない。** 存在、サイズ、ライセンス、言語を確認して、条件を満たせばTier 2と同じ扱いで追加する。確認できなければスキップして理由を記録する。

- `vllm-sr/d3-flash`（8B級とされる）
- `vllm-sr/Decision-2.0-Lux-9B`（9Bとされる）

### 対象外（理由を報告書に記載する）

- 英語のみ：Kev、Bekko、CLM（CLM-8Bも英語のみ）
- 非商用ライセンス（CC BY-NC 4.0）：Nimble（9B）、StartLux（9B Instant／AutoThinkを含む）、Jeff-1.1。9Bクラスだが、評価する場合はユーザーに確認する
- 約9Bを超える：pplx-decider（27B）、AutoTrust-JEV（27B）、Clef（27B）、Jeff 27B／31B
- 重みが非公開：Jev

## 4. ステージ1：公開データでの評価

### 4-1. S1MB-japanese-v1

- 取得：`hotchpotch/S1MB-japanese-v1`（Hugging Face Datasets）。27のconfig、各 `test` splitのみ。
- 構成：既存の日本語データセット19タスクと、合成8タスク（`generalization-*`）。合計8,067件。
- 読み込みと推論ルール（データセットカードに従う）：
  - `input.state_json`、`input.decisions[].instructions_json`、`criteria[].description_json` をJSONとして復号し、**それだけを**モデルに渡す。
  - ラベル、`group_id`、`provenance_json`、評価ファミリのフラグなどはプロンプトに入れない。
  - 選択肢のIDは匿名の中立IDに置き換えてよいが、**選択肢の意味と順序は変えない。** 予測は選択肢IDで元に戻して照合する。
  - Noulの `true` は「良い」とは限らない。作者が定めた意味を保つ。
  - Scoreは、選択肢のインデックスではなく、`criteria[].value`（数値）を使う。
  - 確率ベクトルは選択肢IDで揃える。ソフトターゲットを丸めない。長い入力を黙って切り詰めない（切り詰めが必要なら件数と理由を記録する）。
- 指標：
  1. データセット内の `benchmarks/` 配下の記述子と `categories/jdmb-v1.json` を読み、各タスクの**primary metric**を確認する。
  2. 作者の公開した評価コードが、データセットカードまたは紹介記事（https://secon.dev/entry/2026/10/15/100000-saihai-japanese-decision-benchmark/ ）からリンクされていれば、それを使う。
  3. 公式の評価コードが見つからない場合は、Choice／Noulは正解率、Scoreは期待値と正解値の誤差で算出し、**「基準値調整済みスコアではない」と明記する。** 基準値調整の式を推測で実装しない。
- 集計：Noul・Choice・Scoreの型ごとの平均と、それらの等重み平均（Task Avg）。「合成8タスクのみ」の平均も別に出す。
- **学習データとの重なりへの対処（必須）**：
  - saihai-v1-8bは、このベンチマークの10タスクの元データのtrainを学習に含む（JNLI、JSTS、JCommonsenseQAなど）。その10タスクを `hotchpotch/saihai-v1-8b` のカードと `hotchpotch/saihai-v1-dataset` から特定し、`results/overlap_tasks.json` に根拠付きで記録する。特定できなければユーザーに確認する。
  - 他モデルの学習データも、各カードの記載から重なりの有無を調べて記録する。記載がなければ「不明」とする。
  - 集計は「全27タスク」と「重ならない17タスク」の2通りを出し、**モデル間の比較は17タスクの結果を主とする。**
- ライセンス：サブセットごとに異なり、合成タスクはCC BY-NC 4.0。評価は社内の研究目的にとどめ、データを再配布しない。

### 4-2. sokudanの `bench_ja`

- 取得：sokudanのGitHubリポジトリ（`hiroki-abe-58/sokudan`）の `bench_ja`。日本語ビジネス文300件、3課題（4択の部署ルーティング、3段階の緊急度、解約示唆）。CC BY 4.0、評価専用。
- 指標：Choiceの正解率、Score RPS（小さいほど良い）、Boolの正解率とAUROC。リポジトリの評価スクリプトがあればそれを使う。
- ベースライン：多数派クラスとランダムを併記する。
- 注意：データはLLM生成で、sokudanの学習データと同じ生成元。**sokudanに有利に出る可能性**を報告書に書く。

### 4-3. 補足データ（任意）

- OpenDecisions（`jgeuter/OpenDecisions`、CC BY-SA 4.0、22言語）：日本語の件数を確認し、日本語サブセットが十分あれば追加する。「学習に使わない」を守る。

## 5. ステージ2：確率の質・頑健性・コスト

公開データ（S1MB、`bench_ja`）に対して、全モデルで同じコードを通す。

### 5-1. 分類性能

- Accuracy、Macro-F1（Choice／Noul）。
- 切り口：型別（Choice／Noul／Score）、既存19タスク／合成8タスク、重ならない17タスク／重なる10タスク、タスク別。
- ベースライン：多数派クラス、ランダム。

### 5-2. 確率の質（calibration）

- NLL、Brier Score（多クラス）、ECE（15ビン。ビニング方法を記載）。
- 信頼度図（reliability diagram）を画像として保存する。
- **温度合わせの前後を両方出す。** 手順：
  1. 各タスクを `group_id` 単位で、calibration 30%／test 70%に分ける（seedを固定して保存）。
  2. 型（Choice／Noul／Score）ごとに1つのスカラー温度Tを、calibration側でNLL最小化により求める（確率pに対して、p^(1/T)を再正規化する形）。
  3. **温度は、学習データと重ならない17タスクのcalibration側だけで求める**（saihai-8bのように学習と重なるタスクで求めると偏るため）。
  4. testには触れない。
  5. 温度合わせ前の指標は全データ、合わせた後の指標はtest 70%で出し、どちらの母集団かを表に明記する。
- 選択的予測：確率の閾値を変えながら「coverage」と「その中の正解率」を曲線で出す。目標正解率（例：95%）を満たす最大のcoverageを表にする。

### 5-3. 頑健性

- **選択肢順序の入れ替え**：Choiceタスクの各件で、ランダムに5通りの順序に並べ替え、予測が変わる割合（flip rate）を出す。
- 「答えを変える条件が1つだけ異なるペア」の正解率は、公開データにペアの情報がないため**未測定**とする（ペア構造を持つタスクがあれば、その旨を記録して追加を検討する）。
- 日英混在・未知の言い回しの評価は、公開データでは行わない。**未測定**と明記する。

### 5-4. 実行コスト

- p50／p95レイテンシ（バッチサイズ1と、可能なら8／32）。ウォームアップを除き、各100回以上。
- GPUのピークVRAM、CPUのみでのレイテンシ（encoder系のみで可）、モデルサイズ。
- 実行環境を固定し、他のジョブと並走させない。

### 5-5. 統計

- 主要指標にブートストラップ（1,000回）の95%信頼区間を付ける（タスクごとの件数が最大300件と少ないため、タスク単位の値は区間の幅に注意する）。
- 乱数を含む処理は3回以上実行し、ばらつきを記録する（決定的なら、そう記録する）。

## 6. 実装方針

### 6-1. ディレクトリ構成（目安）

```
decision-eval/
  README.md
  pyproject.toml / requirements.lock
  configs/            # モデル、データセット、実行設定（YAML）
  src/
    adapters/         # モデルごとのアダプタ（共通インタフェース）
    data/             # S1MB、bench_ja のローダ
    metrics/          # accuracy, F1, NLL, Brier, ECE, flip rate, coverage curve
    calibration/      # 温度合わせ
    bench/            # レイテンシ・VRAM計測
    run_eval.py
    make_report.py
  results/
    manifest.json
    overlap_tasks.json
    raw/              # 件ごとの予測確率（parquet/JSONL）
    tables/           # 集計CSV
    figures/
  report.md
  tests/
```

### 6-2. 共通インタフェース

```python
class DecisionModelAdapter:
    name: str
    def load(self, device: str, dtype: str) -> None: ...
    def predict(self, requests: list[DecisionRequest]) -> list[DecisionResult]: ...
    # DecisionRequest: state, question(type, instructions, options[id, description, value?])
    # DecisionResult: probs{option_id: float}, extra{expected_value?, latency_ms?}
```

- モデルごとの入出力の違い（`systemone(request)` 形式、JSON形式、logit読み出しなど）はアダプタ内に閉じ込める。
- **件ごとの予測確率を必ず保存する**（後から指標を再計算できるように）。
- 選択肢数が上限を超える場合（例：Laya 約20、Julia-1 20）は、そのモデルの仕様に従った処理にするか、スキップして件数を記録する。黙って切り捨てない。
- 失敗（読み込み不可、推論エラー、OOM）は例外で止めず、モデルごとの `status` と原因を記録して次へ進む。

### 6-3. テスト

- 指標関数の単体テスト：既知の小さな例（完全予測、一様分布、既知のECE）で期待値を検証する。
- アダプタのスモークテスト：各モデルで2〜3件を通し、確率が正規化されている（和が1）ことを確認する。
- 選択肢の並べ替えと元IDへの復元が正しいことのテスト。
- calibrationの分割が `group_id` をまたがないことのテスト。

## 7. 実行順序

1. 環境構築、リビジョン固定、`manifest.json` 作成。
2. 指標・温度合わせ・データローダの実装とテスト。
3. Tier 1モデルのアダプタ実装とスモークテスト。
4. 重なり10タスクの特定と、各モデルの学習データ重なりの確認。
5. ステージ1（S1MB、`bench_ja`）を全Tier 1で実行。
6. **中間報告をユーザーに送る**（モデルの読み込み可否、ステージ1の暫定結果）。
7. ステージ2（calibration、頑健性、コスト）。
8. Tier 2、Tier 3（資源がある場合）。
9. `report.md` の作成。

## 8. 報告書（`report.md`）の要件

1. 実施概要：目的、日付、対象モデルと除外モデル（理由付き）。**自前の業務データでの評価は行っていない**ことを冒頭に明記する。
2. 環境：ハードウェア、ライブラリ、各モデルのリビジョン。
3. 結果表：
   - ステージ1：S1MB（27タスク／重ならない17タスク）、`bench_ja`。型別と総合。
   - calibration：温度合わせの前後のNLL・Brier・ECE、coverage–accuracy表（母集団を明記）。
   - 頑健性：flip rate。
   - コスト：レイテンシ、VRAM、サイズ。
   すべて95%信頼区間つき。
4. 事実と解釈の分離：「実測結果」と「そこから言えること」を別の節にする。因果を断定しない。
5. 制約：未測定の項目（ペア正解率、日英混在、自前データ）、スキップしたモデルと理由、合成データの限界、学習データの重なり、`bench_ja` がLLM生成でsokudanに有利な可能性、one-wayでの評価（H2O）、タスクあたりの件数が少ないことによる区間の広さ。
6. ライセンス：各モデルと評価データのライセンス、商用利用の可否に関する注意。
7. 再現手順：コマンド一式と、結果ファイルの場所。

禁止事項：実測していない数値の記載、カードや論文の値を実測値として扱うこと、公開データの結果をもとにした「業務に最適なモデル」の断定。

## 9. 範囲外（ユーザーの指示があるまで行わない）

- 自前の業務データでの評価
- fine-tuning（Strands、Kevなど。zero-shotの結果を見たうえで次に検討する）
- 非商用ライセンスのモデルの評価
- 約9Bを超えるモデルの評価
- 評価結果に基づくモデルの採用判断（比較結果の提示までを範囲とする）

## 10. 完了条件

- [ ] Tier 1の全モデルについて、読み込み成否と原因が記録されている
- [ ] ステージ1の結果が、全27タスクと重ならない17タスクの両方で出ている
- [ ] 公式の評価コードを使ったか、使わずに代替指標を使ったかが明記されている
- [ ] 件ごとの予測確率が保存され、指標を再計算できる
- [ ] 温度合わせの前後の結果と、reliability diagramがある（温度は17タスクのcalibration側のみで求めている）
- [ ] 選択肢順序のflip rateがある
- [ ] レイテンシ・VRAMが、固定環境で測られている
- [ ] 指標関数と分割のテストが通っている
- [ ] `report.md` に、事実と解釈の区別、制約、ライセンスが記載されている
- [ ] 未測定項目がすべて「未測定」と明示されている
