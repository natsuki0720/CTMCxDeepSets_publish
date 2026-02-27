# notebook/ のデータセット作成と `scripts/train_entrypoint.py` の整合性検証

## 結論
- 現在の `notebook/pretrained_model_eval.ipynb` には、**train と同等のスクリーニング機構を任意で有効化できる実装**を追加済みです。
- そのため、`enable_screening=True` の場合は、評価用 notebook でも `min_lambda` / `max_lambda` / 構造チェック / NaN・Inf チェックで除外できます。
- ただし notebook は評価用途のため、train 側にある `n` 件ランダム抽出や train/val 分割は行いません。

## notebook 側で追加した機構
- ユーザー設定セルで以下の閾値・フラグを変更可能。
  - `enable_screening`
  - `screening_min_lambda`
  - `screening_max_lambda`
  - `screening_check_nan_inf`
  - `screening_require_structure`
- `apply_eval_screening(...)` を新設し、`ScreeningConfig` と `screen_datasets(...)` を用いて通過データのみを返す。
- スクリーニング結果（入力件数・通過件数・除外件数）を表示。
- 通過件数が 0 の場合は、閾値見直しを促すエラーを明示。

## train と notebook の比較（最新）
- **揃っている点**
  - `screen_datasets(...)` によるラベル品質ベースの除外（有効化時）。
  - `state`, `delta_t`, `target(=extract_lambdas_from_Q(q_mle))` の変換方針。
- **異なる点**
  - train は `n` 件ランダム抽出・train/val 分割を行う。
  - notebook は評価用のため、原則としてフィルタ後データをそのまま評価に用いる。
