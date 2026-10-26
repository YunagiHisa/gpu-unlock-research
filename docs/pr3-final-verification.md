# PR #3 最終検証（2026-10-26、Asia/Tokyo）

ブランチは `feat/nvml-telemetry-metadata`。指定された正式仕様
`/tmp/gpu-unlock-issue-1.md` を再読し、`issue-1-conformance.md` と照合した。
SHA-256は `f75d8a0824c60464adad37388bde2a50fcb4d0eb16b105c5fadbabde62872fb9`
で前回と同一。仕様を変更していない。

## 今回の修正とCPU検証

- timed区間の0サンプルは以前からcoverageがmissingだったが、全区間の統計と
  混同される余地があった。`telemetry.timed_summary` を追加し、各フィールドの
  統計を `unavailable`、min/max/meanをnull、理由を明示した。既存のsummaryは
  warm-upと前後の読み取りを含む全区間の統計と明記した。
- 比較統計も0件ならunavailableとし、CVは有限な有効サンプルが2件未満、
  または平均が0なら算出せず、理由を付ける。非有限値を数に含めない。
- NVML共通クエリ関数はDeviceGet以外を拒否する。設定変更関数名を渡した
  モックテストでも実際の関数が呼ばれないことを確認した。
- 修正前・修正後とも `python -m unittest discover -s tests -v` の56件が成功。
  既存15件を維持した。`git diff --check` も成功。
- 0件のtimed統計に非ゼロのendpoint値が流用されないこと、CVの0/1有効サンプル、
  物理GPU対応矛盾時のテンソル確保・GEMM前の停止、終了コード4と診断保存を
  CPU上の模擬テストで検証した。

CUDA Eventsと最終同期の計測方式は変更していない。

## 両GPUの短時間・読み取り専用スモーク

PyTorchのimport、CUDAコンテキスト作成、テンソル確保、GEMMは行っていない。
ドライバーAPIのデバイス列挙とPCI ID読み取りからNVMLの対応を確認した。
したがって、この試験はCUDA実行コンテキスト経由の実機GEMM検証ではない。
GPU番号は `CUDA_DEVICE_ORDER=PCI_BUS_ID` でのドライバー列挙番号である。

| 対象 | 所要時間 | 温度 前→後 | 初期電力 | 最終ファン率 | 計算プロセス数 前→後 | 結果 |
|---|---:|---:|---:|---:|---:|---|
| GPU 0 | 約0.49秒 | 35→35℃ | 11.717 W | 30% | 0→0 | PCI照合・短時間ポーリング成功 |
| GPU 1 | 約0.03秒 | 41→41℃ | 13.431 W | 0% | 1→1 | PCI照合・単発読み取り成功。使用中のためポーリング省略 |

GPU 1のプロセスの名前・PIDは照会結果に保持せず、サービスの種類も推定していない。
既存サービスを停止せず、GPUへの計算負荷も加えていない。GPU 0のプロセス数0も
期間中の無競合を証明するものではない。
試験には最大約0.5秒、75℃以上または2回連続の温度上昇で中断する保守的な条件を
設けた。75℃は今回の試験用中断条件であり、GPUメーカーの温度限界ではない。
観測中の温度上昇はなく、GPU 1は最初から使用中として単発に限定された。

結果はローカルの `results/diagnostics/pr3-final-nvml-smoke-2026-10-26.json`。
試験用スクリプトは `/tmp/pr3-final-nvml-smoke.py`。鍵は生成していない。
この読み取り試験も正式Before baselineには数えない。

## JSONと禁止操作の検査

ローカルresultsのJSON 71件を検査した。そのうちPR #3の診断JSON 10件で、
生UUID、実際のホスト名・ユーザー名、既知の環境秘密値、秘密鍵形式、
不要な絶対パス、PIDなどの禁止フィールドの検出は0件だった。
HMAC鍵の実ファイルを検査のために読み出したり、その内容を表示したりしていない。
これは既知値・パターンによる検査であり、任意のユーザー入力の匿名性の証明ではない。
公開前のレビューは引き続き必要である。

PR #3とは別系統の既存ローカル結果
`results/smoke/llama/qwen38-llama-smoke.json` の2レコードのmodel_filenameに
絶対パスが検出された。値は表示せず、過去の原本も変更していない。
したがって「ローカルJSON全件が匿名」とは判定していない。このファイルを公開する
場合には別途サニタイズが必要であり、PR #3の出力からの漏洩ではない。

本番実装8モジュールをASTと目視で検査した。NVMLの設定変更API、sudo、
modprobe/insmod/rmmod、nvidia-smiによる変更、gpu-unlock有効化の使用は0件。
NVMLの直接呼出しは初期化・終了とgetter、動的クエリはgetterに限定される。
外部コマンドは証拠のdmesg、Gitの読み取り、比較用Python子プロセスである。
`torch.cuda.set_device` はプロセスのCUDAデバイス選択であり、電力・クロック・
ファン・永続化設定を変更するAPIではない。精度設定は既存GEMMのPyTorch設定である。
今回のスモークではこれらのPyTorch処理も呼び出していない。

検査記録は `results/diagnostics/pr3-final-audit-reviewed-2026-10-26.json`。
ローカル全件の判定とPR #3の判定は分けて記録している。

## Issue #1への適合と残るリスク

GPU識別、NVMLサンプリング、環境メタデータ、解除状態ラベルの保守的な判定は
今回の範囲で実装・CPU検証済み。追加要件も維持している。
ただしIssue全体の完了や、正式baseline採用可能とは判定していない。

- カーネルログは取得可否と成功・失敗マーカーの件数のみ。サニタイズした個別行、
  時系列、GPUごとの対応、失敗した解除試行の詳細保存は未実装で、§3は部分適合。
  件数から解除成功・失敗を自動確定しない。取得不能や詳細不足を「証拠不要」として
  扱わず、正式採用の証拠審査に残す。
- 正式JSON Schema・検証器、保存先/ファイル名の自動化、正式集計と採用判定、CIは
  後続PRの未達項目。一般baselineの構成順序/冷却手順と任意出力ハッシュも未実装。
- 全GPU/精度/サイズの実機検証、72件の正式baseline、変動評価は未完了。
  GPU 1の今回の試験は使用中の単発読み取りに限り、負荷時の冷却性能を再検証しない。
- HMAC鍵なしの場合は安定仮名IDが利用できず、PCI照合だけでは長期の個体連続性を
  証明できない。鍵を紛失すると以前の仮名IDとの自動照合ができなくなる。
- NVMLのポーリングは短いイベントを取り逃がし得る。制限カウンターAPIには
  廃止予定の互換APIを使用しており、未対応は欠損として扱う。

すべてのPR #3結果で `baseline_eligible=false` を維持する。GPU 1の冷却変更以前の
スモーク、ON/OFF比較、今回の読み取り試験を正式baselineへ流用しない。
§3の証拠不足を含め、Issue #1の元要件を独断で縮小・変更していない。

## 変更ファイル一覧（未commit）

| 区分 | ファイル |
|---|---|
| 実装 | benchmarks/core.py、benchmarks/gemm.py、benchmarks/compare_telemetry.py、benchmarks/evidence.py、benchmarks/identity.py、benchmarks/metadata.py、benchmarks/telemetry.py |
| テスト | tests/test_accuracy.py、tests/test_comparison.py、tests/test_duration.py、tests/test_evidence.py、tests/test_identity.py、tests/test_integration.py、tests/test_telemetry.py |
| 文書 | README.md、docs/telemetry-metadata.md、docs/issue-1-conformance.md、docs/sampler-diagnostic-2026-10-26.md、docs/pr3-final-verification.md |
| 依存・除外 | requirements-gpu.txt、.gitignore |

合計21ファイル。ローカル診断JSONと一時検証スクリプトはこの一覧に含めない。
commit・push・PR作成・merge・解除有効化・システム設定変更は実行していない。
