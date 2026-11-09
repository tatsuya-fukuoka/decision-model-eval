# 日本語 Decision Model 比較評価 報告書

> このファイルは `uv run python -m src.make_report` で `docs/report_template.md` から生成しています。表の数値はすべて `results/` の CSV・JSON から読み込んだ実測値です。カードや論文の値は使っていません。

**自前の業務データでの評価は行っていません。** 公開データ（S1MB-japanese-v1、sokudan `bench_ja`）での zero-shot 評価の結果で、一次スクリーニングの材料です。ここから「業務に最適なモデル」は決められません。

## 1. 実施概要

- **目的：** 約 9B までの公開 Decision Model を、日本語の公開データで同じ条件のもとで比べ、次の検討（自前データでの評価、fine-tuning）に進める候補を絞るための材料を作ること（指示書 §1）。
- **実施日：** 2026-11-07〜2026-11-08（ステージ1・ステージ2）。作業ブランチは `feat/eval-pipeline` です。
- **範囲：**
  - S1MB-japanese-v1 の27タスク（8,067件）。
  - bench_ja の300件 × 3設問（部署＝Choice、緊急度＝Score、解約示唆＝Noul）と、解約示唆の言い換え版1設問。
  - すべて zero-shot、`cuda:0`、`HF_HUB_OFFLINE=1`、HF トークンなしで実行しました。
- **採点：**
  - S1MB は**公式評価器の式**を使いました（`third_party/S1MB` @ `dbfc746`、基準値で調整したスコア baseline-adjusted-v1、0〜1）。手元の移植版が公式の出力と 1e-9 以内で一致することをテストで確認しています。
  - bench_ja は sokudan と同じ指標の定義（確率の下限 5e-5）で、本リポジトリの実装で採点しました（公式の基準値調整はありません）。
- **主比較の範囲：** saihai-v1-8b の学習データと元データが重ならない17タスクです（§5「学習データの重なり」）。

### 対象モデルと結果の状態

※ が付いたモデルは、公式アダプタの Hub カーネル（`kernels-community/fla` と `kernels-community/activation`）を無効にし、transformers 本体の実装で推論しました。**参考値（公式カーネル未使用）**として扱います（ユーザー判断 2026-11-07、選択肢 b）。fla だけを適用した場合との比較は §3.7 にあります。

{{table:models}}

- 「重みファイル」は、固定リビジョンのリポジトリにある重みファイルの合計サイズです（Hub API の値）。
- 「パラメータ数」は、固定リビジョンの safetensors のヘッダから数えました。
  - saihai-v1-310m は、推論用（`0_BekkoInference`）と `checkpoint/` に同じ重みが2組あるので、推論用だけを数えています。
  - Strands は、LoRA アダプタ（0.07 GB）と基盤の Qwen3.5-2B-Base を足した数です。
- 「S1MB 失敗」は判定の単位で数えています。

### 除外したモデル・実行しなかったモデル

{{table:excluded}}

## 2. 環境

{{table:env}}

モデルごとに隔離した仮想環境を使いました。

{{table:envs}}

- リビジョン：§1 の表のとおりです。完全な SHA は `results/manifest.json` にあります。
- 環境の freeze：`results/manifest.json` の `environment.envs`。
- `trust_remote_code` が必要なモデル：固定リビジョンのカスタムコードを実行前に読み、ネットワーク送信・シェル実行・`eval`/`exec` がないことを確認しました。結果は `results/manifest.json` の `code_review` にあります。

## 3. 結果（実測）

かっこ内は95%信頼区間です。タスク内で件を復元抽出するブートストラップを1,000回行い、範囲内のタスクで層化して求めました。「未定義」は、公式の規則で値が出ないことを表します（範囲内に失敗が残ったタスクがあると、そのタスクを含む平均は未定義になります）。

### 3.1 ステージ1：S1MB

#### 主比較：学習データと重ならない17タスク

{{table:s1mb_non_overlap_17}}

#### 全27タスク

{{table:s1mb_all_27}}

#### 重なる10タスク（参考。saihai の学習データと元データが同じ）

対象タスク：{{table:overlap_tasks}}

{{table:s1mb_overlap_10}}

#### 合成8タスク（generalization-*。17タスクに含まれる）

{{table:s1mb_synthetic_8}}

#### 補助：既定設定の6モデルがすべて完了したタスクだけの平均

**公式の Task Avg ではありません。** 入力上限による失敗で欠けるタスクを除き、同じタスク集合でモデルを並べるための参考値です。基準にする6モデルは固定しているので、参考値のモデルや拡張設定を足しても、タスク集合は変わりません。

{{table:common_tasks}}

{{table:s1mb_common_complete_non_overlap}}

{{table:s1mb_common_complete_all}}

#### 失敗した件数（既定設定、タスク別）

失敗はすべて、各モデルの実装が持つ入力上限・選択肢数の上限を超えたときの例外です（saihai-310m・Julia・Laya は `ValueError`、sokudan は `QuestionTooLongError`、Strands はプロンプトが選択肢の部分まで切れたときの例外）。補完せず、失敗として数えています。

{{table:failed_tasks}}

### 3.2 拡張設定（入力上限を変えた追加実行。別表）

主表（3.1）は既定設定の結果です。下の表は、入力上限を変えて追加で実行した結果です（ユーザー判断 2026-11-07）。

{{table:ext_compare}}

- 「回復」は、既定設定で失敗し、拡張設定で予測が出た件の数です。
- 「共通完了件で argmax が変化」は、両方の設定で予測が出た件のうち、最大確率の選択肢が変わった件の数です。
- **切り詰めの扱い：**
  - saihai-310m と Julia の `--native-adaptive-budget` は作者の規則で入力を切り詰めます。切り詰めた件数は「回復」の件数と同じです（既定設定で上限を超えた件だけが切り詰めの対象になるため）。
  - Laya・sokudan・Strands は上限を上げただけで、切り詰めは行っていません。sokudan の状態（state）は作者の規則で切り詰められ、その件数は実行設定（`results/runs/sokudan-ja-310m__ext/`）に記録しています。
- Laya（拡張）は、8,192 トークンでも収まらない35件が引き続き失敗しています。

拡張設定での S1MB（重ならない17タスク）：

{{table:s1mb_ext_non_overlap_17}}

拡張設定での bench_ja（入力が短く、既定設定と同じ結果）：

{{table:bench_ja_ext}}

### 3.3 ステージ1：bench_ja

数値は正解率・macro-F1・RPS・AUROC で、かっこ内は95%区間です。ECE・Brier・NLL と区間は `results/tables/stage1_bench_ja.csv` にあります。

{{table:bench_ja}}

{{table:bench_ja_failed}}

#### 解約示唆（Noul）：P(true) の分布

正解率が多数派（常に「示唆していない」）と同じになるモデルがあるので、AUROC と確率の分布を並べて示します。温度合わせ前の値です。ヒストグラムは `results/figures/bench_ja_churn_ptrue.png` にあります。

{{table:churn}}

#### 解約示唆の説明文を言い換えたときの差

Noul の説明文（true / false の定義）は、機械で作った版に固定し、全モデル共通にしています。言い換え版は1パターンだけで、主結果には入れていません。

{{table:paraphrase}}

### 3.4 ステージ2：温度合わせ（calibration）

- **分割：** S1MB の件を `group_id` 単位で 30% / 70% に分けました（seed 20261107、`results/splits/s1mb_calibration_split.json`）。
- **温度：** 型（Noul / Choice / Score）ごとにスカラーの温度を1つ、**重ならない17タスクの calibration 側（30%）だけ**で NLL 最小化により求めました。探索範囲は 0.05〜20 です。
- **評価：** 下の S1MB の表は、重ならない17タスクの **test 側（70%）** での値です。bench_ja には、S1MB で求めた温度をそのまま当てて、全300件で評価しました。
- **除外：** 失敗した件は補完せず、「除外」の件数に入れています。
- reliability diagram は `results/figures/reliability/<model>.png` にあります。全件・test 側・温度合わせの後の3つの母集団を描いています。

#### 温度

「（上限）」は、探索範囲の上限 20 に張り付いたことを表します。確率を一様に近づけても NLL が下がり続けた、という意味です。

{{table:temperatures}}

#### S1MB 重ならない17タスク・test 70%（全型）

{{table:calib_s1mb_all}}

型別（Noul / Choice / Score）：

{{table:calib_s1mb_noul}}

{{table:calib_s1mb_choice}}

{{table:calib_s1mb_score}}

#### coverage–accuracy（S1MB 重ならない17タスク・test 70%、全型）

確信度（最大確率）の高い順に件を受け付けたとき、受け付けた件の正解率が 0.90 / 0.95 以上になる最大の割合です。母集団は test 70% の全件で、失敗した件は含みません。曲線の全点は `results/tables/stage2_coverage_curves.csv` にあります。

{{table:coverage_s1mb_all}}

#### bench_ja（S1MB で求めた温度を適用。全300件）

部署（Choice）：

{{table:calib_bj_department}}

緊急度（Score）：

{{table:calib_bj_urgency}}

解約示唆（Noul）：

{{table:calib_bj_churn}}

coverage–accuracy（bench_ja、全300件）：

{{table:coverage_bj_department}}

{{table:coverage_bj_urgency}}

{{table:coverage_bj_churn}}

### 3.5 ステージ2：選択肢順序の入れ替え（flip rate）

- **対象：** S1MB の Choice 型16タスクの全件と、bench_ja の部署（Choice）300件です。
- **並べ替え：** 1件ごとに、恒等でない並べ替えを最大5通り、重複なしで選びました（K=2 は1通り、K=3 は5通りすべて）。
- **比べ方：** 予測は元の選択肢 ID で比べ、最大確率の選択肢が変わった（件, 並べ替え）の組の割合を flip rate としています。
- **除外：** 元の順序と並べ替え後のどちらかで失敗した組は除外し、件数を示しています。
- **区間：** 95%区間は、件ごと（その件の並べ替えをまとめて）の復元抽出1,000回です。並べ替えの選び方の乱数は1回（seed 固定）なので、そのばらつきは「並べ替えの選び方による区間」に分けて示しています。こちらは件を固定し、各件の並べ替えを復元抽出したものです。

重ならない17タスクの Choice 型：

{{table:flip_non_overlap_17}}

全27タスクの Choice 型：

{{table:flip_all_27}}

bench_ja 部署：

{{table:flip_bench_ja-department}}

### 3.6 実行コスト

**レイテンシと VRAM は未測定です。** この環境の GPU は別のジョブと共有しているので、指示（専用環境の EC2 で測る）に従い、ここでは測っていません。EC2 での手順は `docs/ec2_cost_runbook.md` にあります。サイズは固定リビジョンのファイルから求めた値です。

{{table:cost}}

### 3.7 Hub カーネルの一致確認（fla のみ適用。activation は未適用）

※ の5モデルについて、Hub カーネルを使った場合に結果が変わるかを確かめました（ユーザー承認 2026-11-09）。判定基準は実行前に固定しました（`docs/kernel_parity_criteria.md`）。

- **適用したカーネル：** `kernels-community/fla` v1（コミット `24d8640d`）だけです。
  - 公式構成は `kernels-community/activation` v1（SiLU・GELU 系）も使います。しかし activation はコンパイル済みの CUDA バイナリ（`.so`）で、実行していません（§5）。
  - そのため、ここでの結果は**「fla のみ適用（activation は未適用）」**の構成についてのもので、公式構成との一致を示すものではありません。
- **fla v1 の中身：** Python / Triton のソース（290ファイル）だけで、コンパイル済みバイナリは含みません。GPU 用のコードは、実行時に Triton 3.6.0 がこの GPU（compute capability 12.0）向けにコンパイルしました（5モデルとも正常に完了）。
- **条件：** モデルごとに4回実行しました。
  - カーネルなし bf16 を2回（既存の参考値と同じ構成）
  - fla のみ適用 bf16
  - カーネルなし fp32（数値精度によるばらつきの目安）
- **範囲：** S1MB の重ならない17タスク（5,083判断）と bench_ja（1,200判断）です。
- **実行方法：**
  - 専用の非 root ユーザーで、権限（capability）をすべて外し、`env -i` で実行しました。
  - 書き込めるのは評価用ディレクトリだけです。`HF_HUB_OFFLINE=1` とし、Python の audit hook でソケット接続と名前解決を拒否しました。
  - 拒否したのは、urllib3 が IPv6 対応を調べるループバックへの bind（各実行1件）だけでした。

S1MB（重ならない17タスク）：

{{table:kernel_parity_s1mb}}

bench_ja（4設問）：

{{table:kernel_parity_bench_ja}}

判定（○＝満たす、×＝満たさない）：

{{table:kernel_parity_judgement}}

- 基準：
  1. argmax の不一致率 ≤ 0.005
  2. 確率差の99パーセンタイル ≤ 0.01
  3. Task Avg（bench_ja は主要指標）の差 ≤ 0.005
  4. 1〜3の値が bf16 対 fp32 の差の2倍以下
  5. 失敗した判断の集合が同じ
- **5モデルとも基準を満たさない項目があるので、「一致」とは書きません。** 上の表に、fla のみ適用とカーネルなしの差と、bf16 と fp32 の差を並べています。

## 4. 解釈（測定結果と分けて書いています）

ここに書くのは、上の実測値から読み取れることです。因果は断定していません。どれも公開データでの一次スクリーニングの材料で、自前の業務データでの性能を表すものではありません。

- **主比較の範囲（重ならない17タスク）で既定設定の Task Avg が出たのは、失敗が0件の8モデルです。** 値が最も高いのは d3-flash（参考値）で、d3-mini（参考値）、Lux-9B が続きます。ただし、d3・Clef の各モデルは Hub カーネルを無効にした参考値です。公式の実行構成では値が変わる可能性があります。
- **既定設定で入力上限による失敗が残る5モデル**（saihai-310m・sokudan・Strands・Julia・Laya）は、公式の規則では Task Avg が未定義です。
  - 補助の表（共通完了タスクだけ）では、この5モデルの点推定は、いずれも失敗のない8モデルより低い値でした。ただし、Strands（0.248）と d3-lite（0.252）は区間が重なっていて、差は付けられません。
  - 拡張設定で全タスクが完了した4モデル（saihai-310m・sokudan・Strands・Julia）でも、17タスクの Task Avg の点推定は8モデルより低い値でした。
- **拡張設定で回復した件は、もともと長い入力や選択肢の多い難しい件です。** 両方の設定で予測が出た件では、argmax の変化は sokudan の1件だけでした。上限を変えても、既存の件の予測はほとんど変わらず、増えた件の採点が加わったと読めます。
- **17タスクの Score 型（2タスク）で調整スコアが 0 にクリップされたモデルが5つあります。** saihai-310m・sokudan・Strands・Julia・d3-lite で、基準値（ベースライン）を上回れませんでした（Laya は未定義）。
- **saihai-v1-310m と saihai-v1-8b は、重なる10タスクで高く出ています。** カードには、この元データの train split を学習に使ったと書かれています（310m は7タスク、8b は10タスク）。この範囲の値は汎化性能の指標になりません。
- **bench_ja の部署（Choice）では、13モデルのうち11モデルの正解率が 0.880〜0.933 の範囲にあり、区間が重なっています。** 300件しかないので、この11モデルの間で順位は付けられません。Julia（0.563）と Laya（0.773）は、ほかより低い値でした。
- **bench_ja の解約示唆では、正解率が多数派と同じ 0.703 のモデルが4つあります**（saihai-310m・Strands・d3-lite と多数派自身）。これらのモデルは P(true) ≥ 0.5 の割合が 0 で、閾値 0.5 の判定では「常に示唆なし」と同じです。一方で AUROC は 0.69〜0.82 で、順位づけの情報は残っています。逆に Julia と Laya は、AUROC が 0.5 前後で、正解が true の件の P(true) が false の件より高い、という傾向が見られません（中央値：Julia は true 0.583 / false 0.749、Laya は 0.954 / 0.946）。
- **温度合わせは、S1MB の test 側（重ならない17タスク・全型）では、ベースライン以外の13モデルすべてで NLL と ECE を下げました。** ただし、bench_ja に同じ温度を当てると、悪化するモデルもあります（例：saihai-8b の解約示唆、saihai-310m・sokudan の緊急度）。S1MB で求めた温度は、データの分布が違う bench_ja にはそのまま移らない、という結果です。温度が上限 20 に張り付いた型（Julia・Laya・sokudan の Score など）は、確率がほとんど情報を持たない状態に近く、温度だけでは較正しきれません。
- **選択肢の順序を入れ替えたときの予測の変わりやすさ（flip rate）は、モデルによって大きく違います。**
  - 重ならない17タスクの Choice 型では、Laya（0.599）と Julia（0.449）が最も高い値でした（1回でも予測が変わった件の割合は、それぞれ 0.896 と 0.761）。続くのは Strands（0.302）、sokudan（0.283）、saihai-8b（0.235）です。
  - 低い値は saihai-310m（0.074）、d3-flash（0.081）、d3-mini（0.086）で、この3モデルは区間が重なっていて、差は付けられません。
  - saihai-8b は、失敗のない8モデルのなかで最も高い値でした（次は Clef-Flash の 0.158）。Task Avg が近い Nox-4B（0.150）や Lux-9B（0.139）と比べても、順序の影響を受けやすい結果です。
  - bench_ja の部署では、Julia（0.355）と Laya（0.211）以外の11モデルはすべて 0.031 以下でした。
  - 注意：失敗した組を除外しているので、Laya・Julia・sokudan・Strands・saihai-310m は、比べている件の集合がほかのモデルと同じではありません（除外した組の数は表のとおり。Laya は17タスクで 2,540 組）。
- **Hub カーネル（fla のみ適用）の有無による差は、§3.7 の基準をどのモデルも満たしませんでした。** 「一致」とは書きません。
  - S1MB では、bf16 と fp32 の差だけでも基準1・2を超えていました（argmax の不一致率 0.0049〜0.0073、確率差の99パーセンタイル 0.0174〜0.0259）。
  - fla のみ適用とカーネルなしの差は、argmax の不一致率 0.0067〜0.0087、確率差の99パーセンタイル 0.0227〜0.0364 でした。
  - 17タスクの Task Avg の差は 0.0000〜0.0022 で、この5モデルの §3.1 の95%区間の幅（0.028〜0.038）より小さい値でした。
  - fla のみ適用での Task Avg（d3-flash 0.476、d3-mini 0.432、Clef-Flash 0.383、d3-nano 0.271、d3-lite 0.199）に置き換えても、§3.1 の主比較の順位は変わりませんでした。ただし、activation を含む公式構成での値は未確認です。
- **言い換えへの感度（AUROC の差）は、モデルによって違います。** sokudan は −0.057、saihai-310m は −0.036、Laya は +0.034 で、ほかのモデルは ±0.016 以内でした。言い換えは1パターンだけなので、どちらの書き方が良いかは言えません。

## 5. 制約

- **未測定の項目：**
  - ペア正解率、日英混在の入力、自前の業務データでの評価（指示書 §9 で範囲外）。
  - レイテンシ・VRAM（専用環境の EC2 で測る指示。この環境では測っていない）。
  - バッチ 8 / 32 のレイテンシ（公式アダプタの `predict` が1件ずつ処理する経路のため、測れない）。
  - CPU だけでのレイテンシ（公式アダプタの多くが CPU での実行を拒否するため）。
  - H2O-Lightning-4B の全項目（スキップ）。
- **スキップしたモデル：**
  - H2O-Lightning-4B はユーザー判断（2026-11-08）でスキップしました。作者が推奨する two-way 読み出しには作者の vLLM ビルドが必要です。標準 vLLM での one-way 読み出しは自作のアダプタが要り、作者の読み出しとの一致確認にも工数がかかります。ロードも推論もしていないので、one-way での評価結果もありません。
  - Microsoft-Decision-1 は有料 API のため、承認がなく実行していません。
  - そのほかの除外は §1 の表のとおりです。
- **Hub カーネルを無効にしたモデル（参考値）：**
  - d3-nano・d3-mini・Clef-Flash・d3-lite・d3-flash は、公式アダプタの `enable_kernels()` を何もしない関数に差し替えて実行しました（`scripts/s1mb_bridge.py --no-hub-kernels`）。差し替えたことは、実行設定の `hub_kernels=False` に記録されています。
  - 公式構成で使う Hub カーネルは2つです（transformers 5.17.0 の対応表）。
    - **`kernels-community/fla` v1：** gated delta rule と FusedRMSNormGated を担当します。中身は Python / Triton のソースだけで、コンパイル済みバイナリは含みません。290ファイルを AST で全件走査し、該当箇所を目視で確認しました。ネットワーク送信・シェル実行・`eval`/`exec` はありませんでした。署名と digest も照合しました（結果は `results/manifest.json` の `hub_kernel_review`）。
    - **`kernels-community/activation` v1：** SiLU と GELU 系を担当します。コンパイル済みの CUDA バイナリ（`_activation_cuda_1b107cb.abi3.so`）で、内容を読んで確認できません。また、ネイティブコードが直接行う通信は、この環境（`CAP_NET_ADMIN` / `CAP_SYS_ADMIN` がなく、ネットワーク名前空間を分けられない）では防げません。署名・digest の照合と、sm_120 向けの機械語が含まれていることは確認しましたが、**実行していません**。
  - 当初（2026-11-07）は、両方を「コンパイル済みバイナリで事前に確認できない」と扱って外していました。これは fla については正確ではありませんでした（fla はソースだけ）。
  - fla だけを適用した場合との比較は §3.7 にあります。5モデルとも、実行前に固定した基準を満たさない項目があり、「一致」とは書いていません。
  - **activation を含む公式構成との一致は未確認です。** 上記のとおり、activation がコンパイル済みバイナリで内容を読めず、ネイティブコードの通信を現環境では防げないため、実行していません。
- **saihai-v1-8b のリビジョン変更：**
  - 当初は main の `907eb3b2` に固定しましたが、公式アダプタが読み込む `saihai/` パッケージがこの版にはなく、`ModuleNotFoundError` で読み込めませんでした。そこで、S1MB の OPEN_MODELS.md が指定する `c8aa3bbb` に固定し直しました。
  - 2つの版で同じもの（SHA-256 一致）：重み（`model.safetensors`）、トークナイザ、`training-provenance.json`、`decision-inference.json`、`generation_config.json`、`chat_template.jinja`、`llmjp4_*.py`。
  - 違うもの：`config.json`（旧版の `auto_map` が新版では削除）、`release-manifest.json`、`README.md`、推論コード（`inference_saihai.py` が `saihai/` パッケージと `pyproject.toml` に置き換わった）。
  - 新しい版のコードも実行前に読み、確認結果を記録しました。
  - 影響：旧版での読み出しは一度も実行できていないので、旧版との結果の差は**未測定**です。
  - 実行時に、スナップショット内に `saihai/__pycache__` ができています（Python のバイトコードのキャッシュで、重みとコードは変えていません）。
- **入力上限：**
  - 主表は各モデルの既定設定のまま（上限を超えた件は失敗）です。
  - 上限を変えた結果は §3.2 の別表にしています。拡張設定のうち `--native-adaptive-budget` は入力を切り詰めるので、その件では、モデルが入力の全体を見ていません。
- **Noul の説明文：**
  - bench_ja の解約示唆の説明文は、機械で作った版に固定し、全モデル共通にしています。
  - 言い換え版（1パターン）での AUROC の差は −0.057（sokudan）〜 +0.034（Laya）でした（§3.3）。説明文の書き方によって結果が変わりうるので、主結果はこの固定した定義のもとでの値です。
- **学習データの重なり：**
  - 「重ならない17タスク」は、saihai-v1-8b の学習データを基準にした分類です。ほかのモデルにとって未学習であることは保証できません（多くのモデルはカードに学習データの記載がなく、不明）。
  - 事前学習（ベースモデル）のデータとの重なりは、どのモデルも不明です。
  - 各モデルの記載状況：

{{table:overlap}}

- **bench_ja：** LLM が生成したデータで、sokudan の作者が作ったものです。sokudan に有利な可能性があります。1設問あたり300件と少ないので、区間は広くなります。
- **合成データ：** S1MB の generalization-* の8タスクは、OpenAI GPT-6.1-Sol で生成されたと作者が明記しています。実データの分布と違う可能性があります。
- **件数の少なさ：** S1MB は1タスクあたり最大300件です。特に Score 型（17タスク中2タスク）と Noul 型の区間は広くなっています。
- **ブートストラップ区間：** 調整スコアは 0〜1 にクリップされるので、点推定が 0 付近では、パーセンタイル区間が点推定を含まないことがあります（ベースラインの行がその例です）。
- **並べ替え：** 並べ替えの選び方の乱数は1回だけです。その影響は「並べ替えの選び方による区間」で示しています。
- **GPU の共有：** 精度の評価は、別のジョブと GPU を共有した状態で行いました。精度の結果には影響しませんが、実行時間（ログに残る所要時間）は性能の指標として使えません。
- **自作アダプタ：**
  - sokudan と Strands は公式アダプタがないので、作者のパッケージを呼ぶ自作のアダプタで実行しました。読み出し・温度は作者の既定どおりです。
  - 作者のコードは表示用に確率を小数4桁に丸めるので、丸める前の値を取り出しています。
  - 作者の公式の評価経路と完全に同じであるとは確認できていません。

## 6. ライセンス

- **モデル：** 評価した13モデルは、いずれも Hub 上のライセンスが Apache-2.0 です（§1 の表）。
  - 商用利用の可否は、基盤モデルのライセンス、学習データの条件、各カードの利用条件にも左右されます。採用を検討する段階で、個別に確認が必要です。
  - 非商用ライセンスのモデル（Nimble、StartLux、Jeff-1.1）は評価していません。
- **S1MB-japanese-v1：**
  - `license: other`（mixed-source-licenses）で、タスクごとに元データのライセンスに従います（MIT、CC BY 4.0、CC BY-SA 4.0、BSD-3-Clause など）。
  - generalization-* の8タスクは **CC BY-NC 4.0** で、生成に使った OpenAI のサービスの規約も別に関わります。
  - jmmlu-japan-choice の一部（CC BY-NC-ND 4.0 など）と jmteb-retrieval の一部の元データには、研究・評価目的の限定や、権利関係が未確定な部分があると作者が記載しています。
  - 本評価は評価だけに使い、学習には使っていません。データは再配布しておらず、git にも含めていません。
- **bench_ja：** CC BY 4.0（評価専用、sokudan @ `eccded37`）。sokudan のコードは Apache-2.0 です。
- **S1MB 評価器：** MIT（`third_party/S1MB` @ `dbfc746`）。
- `results/raw/` には、件の ID と予測確率だけを保存しています。入力テキストと正解は含んでいません。

## 7. 再現手順

README の「環境構築」を行ったあと、次のコマンドを順に実行します。

```sh
# ステージ1（既定設定・Tier 1〜3）
uv run python -m src.bridge_export
uv run python -m src.run_eval --with-smoke --models saihai-v1-8b saihai-v1-310m sokudan-ja-310m strands-decider-2b \
  julia-1 laya-multilingual d3-nano d3-mini clef-flash d3-lite decision-2.0-nox-4b d3-flash decision-2.0-lux-9b
# 拡張設定（入力上限の変更。configs/models.yaml の extended:）
uv run python -m src.run_eval --with-smoke --models saihai-v1-310m__ext sokudan-ja-310m__ext strands-decider-2b__ext \
  julia-1__ext laya-multilingual__ext
uv run python -m src.stage1_tables --n-boot 1000
# ステージ2：温度合わせ・解約示唆の分布
uv run python -m src.stage2_calibration --n-boot 1000
uv run python -m src.bench_ja_churn
# ステージ2：並べ替え（入力は評価器の環境で作る）
(cd third_party/S1MB/evaluator && HF_HUB_OFFLINE=1 ../../../envs/s1mb-base/.venv/bin/python -I \
  ../../../scripts/flip_export.py --bench-ja ../../../data/bridge/bench_ja_requests.jsonl \
  --out ../../../data/bridge/flip_requests.jsonl)
uv run python -m src.run_eval --flip --with-smoke --models <上と同じ13モデル>
uv run python -m src.stage2_flip --n-boot 1000
# コスト（EC2 の専用環境で。docs/ec2_cost_runbook.md）
uv run python -m src.run_eval --cost --with-smoke --models <モデル>
# 報告書
uv run python -m src.make_report
uv run pytest
```

結果ファイルの場所：

| パス | 内容 |
|---|---|
| `results/manifest.json` | リビジョン、ライセンス、環境の freeze、コード確認の結果 |
| `results/model_status.json` | モデル・設定ごとの成否、失敗件数、原因 |
| `results/raw/<model>/{s1mb,bench_ja,flip}/` | 件ごとの予測確率（ID と確率のみ） |
| `results/runs/<model>/` | 実行ごとの設定（アダプタの settings、引数） |
| `results/tables/stage1_*.csv` | ステージ1の表（タスク別・範囲別・bench_ja、拡張設定の別表） |
| `results/tables/stage2_*.csv`、`stage2_temperatures.json` | 温度合わせ、coverage 曲線、flip rate |
| `results/tables/bench_ja_churn_distribution.csv` | 解約示唆の P(true) の分布 |
| `results/figures/` | reliability diagram、解約示唆のヒストグラム |
| `results/splits/s1mb_calibration_split.json` | calibration / test の分割（group_id 単位） |
| `results/overlap_tasks.json` | 重なる10タスクとその根拠、モデルごとの記載状況 |
| `results/kernel_parity/` | Hub カーネルの一致確認（実行記録・件ごとの確率・比較・判定） |

## 8. 未測定の項目（一覧）

| 項目 | 理由 |
|---|---|
| レイテンシ（p50 / p95）・VRAM | 専用環境（EC2）で測る指示のため。この環境は GPU を共有していて、EC2 の認証情報もない |
| バッチ 8 / 32 のレイテンシ | 公式アダプタの `predict` が1件ずつ処理する経路のため |
| CPU だけでのレイテンシ | 公式アダプタの多くが CPU での実行を拒否するため |
| H2O-Lightning-4B の全項目 | スキップ（ユーザー判断 2026-11-08） |
| Microsoft-Decision-1 の全項目 | 有料 API。承認なし |
| ペア正解率・日英混在・自前データ | 指示書で範囲外 |
| d3・Clef の公式構成（fla と activation の両方を適用）での値 | activation がコンパイル済みバイナリで内容を読めず、ネイティブコードの通信を現環境では防げないため実行していない（fla のみ適用の比較は §3.7） |
| saihai-v1-8b の旧リビジョン（`907eb3b2`）での値 | 旧版は公式アダプタで読み込めなかったため |
