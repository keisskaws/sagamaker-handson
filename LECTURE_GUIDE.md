# 🎓 SageMaker講義実施ガイド

## 📋 講義前の準備

### 1. 環境セットアップ
- [ ] AWS SageMaker Studioアクセス確認
- [ ] 必要なIAMロール設定
- [ ] S3バケット作成権限確認
- [ ] このリポジトリをSageMaker Studioにアップロード

### 2. 講師用チェックリスト
- [ ] 各ノートブックの事前実行確認
- [ ] データファイルの存在確認
- [ ] 推定実行時間の把握
- [ ] エラー対処法の確認

## 🕐 講義パターン別実施方法

### パターンA: 短時間講義（90分）
**対象**: 高校生・大学1年生
**内容**: 基本概念の理解

```
1. 導入（10分）
   - 機械学習とSageMakerの概要説明

2. 01_data_management.ipynb（20分）
   - データ管理の基礎
   - S3との連携

3. 02_script_mode_training_complete.ipynb（30分）
   - Script Modeの実践
   - 迅速な実験手法

4. 04_comparison_and_summary.ipynb（20分）
   - 手法の比較
   - まとめと質疑応答（10分）
```

### パターンB: 標準講義（3時間）
**対象**: 大学2-3年生
**内容**: 実践的なスキル習得

```
1. 導入（15分）
   - 機械学習プロジェクトの全体像

2. 01_data_management.ipynb（25分）
   - データ管理とベストプラクティス

3. 02_script_mode_training_complete.ipynb（35分）
   - Script Modeの詳細実践

4. 休憩（10分）

5. 03_training_jobs.ipynb（35分）
   - Training Jobsの活用

6. 04_comparison_and_summary.ipynb（25分）
   - 手法の比較と選択指針

7. 05_builtin_algorithms.ipynb または 06_byol_script_mode.ipynb（30分）
   - 発展的内容（どちらか選択）

8. まとめと質疑応答（15分）
```

### パターンC: 集中講義（1日・6時間）
**対象**: 大学3-4年生・大学院生
**内容**: 包括的なスキル習得

```
午前の部（3時間）
1. 導入（20分）
2. 01_data_management.ipynb（30分）
3. 02_script_mode_training_complete.ipynb（40分）
4. 03_training_jobs.ipynb（40分）
5. 休憩（10分）

午後の部（3時間）
6. 04_comparison_and_summary.ipynb（30分）
7. 05_builtin_algorithms.ipynb（45分）
8. 休憩（15分）
9. 06_byol_script_mode.ipynb（60分）
10. 総合まとめと質疑応答（30分）
```

## 📝 各ノートブックの実施ポイント

### 01_data_management.ipynb
**重要ポイント**:
- S3バケットの命名規則
- データのバージョン管理
- セキュリティ設定

**よくある質問**:
- Q: なぜS3を使うのか？
- A: スケーラビリティ、耐久性、コスト効率

### 02_script_mode_training_complete.ipynb
**重要ポイント**:
- Script Modeの利点
- デバッグの容易さ
- 迅速な実験サイクル

**よくある質問**:
- Q: いつScript Modeを使うべきか？
- A: 実験段階、プロトタイピング、デバッグ時

### 03_training_jobs.ipynb
**重要ポイント**:
- スケーラビリティ
- リソース管理
- 本番環境での利用

**よくある質問**:
- Q: Training Jobsのコストは？
- A: 使用時間分のみ課金、自動停止

### 04_comparison_and_summary.ipynb
**重要ポイント**:
- 使い分けの判断基準
- 実際のプロジェクトでの選択

**よくある質問**:
- Q: どちらを選ぶべきか？
- A: プロジェクトの段階と要件による

### 05_builtin_algorithms.ipynb
**重要ポイント**:
- 組み込みアルゴリズムの利点
- アルゴリズム選択の指針

**よくある質問**:
- Q: カスタムアルゴリズムとの違いは？
- A: 最適化済み、メンテナンス不要

### 06_byol_script_mode.ipynb
**重要ポイント**:
- Dockerの活用
- カスタムライブラリの統合

**よくある質問**:
- Q: Dockerの知識は必要か？
- A: 基本的な理解があれば十分

## 🚨 トラブルシューティング

### よくあるエラーと対処法

#### 1. 権限エラー
```
ClientError: An error occurred (AccessDenied)
```
**対処法**: IAMロールの権限を確認

#### 2. S3バケット名重複エラー
```
BucketAlreadyExists
```
**対処法**: ユニークなバケット名を使用

#### 3. インスタンス起動エラー
```
ResourceLimitExceeded
```
**対処法**: 別のインスタンスタイプを選択

#### 4. Docker関連エラー
```
CannotPullContainerError
```
**対処法**: ECRの権限とリージョンを確認

## 📊 評価とフィードバック

### 学習達成度チェック

#### 基礎レベル
- [ ] SageMakerの基本概念を説明できる
- [ ] データをS3にアップロードできる
- [ ] Script Modeで訓練を実行できる

#### 中級レベル
- [ ] Training Jobsを設定・実行できる
- [ ] Script ModeとTraining Jobsの違いを説明できる
- [ ] 適切な手法を選択できる

#### 上級レベル
- [ ] 組み込みアルゴリズムを活用できる
- [ ] カスタムライブラリを統合できる
- [ ] 実際のプロジェクトを設計できる

### フィードバック収集

講義後に以下の点について確認：
1. 理解度（1-5段階）
2. 難易度（適切/難しい/簡単）
3. 実行時間（適切/長い/短い）
4. 改善提案

## 🔄 継続学習の提案

### 次のステップ
1. より大きなデータセットでの実験
2. 他のAWSサービスとの連携
3. MLOpsパイプラインの構築
4. 実際のビジネス問題への適用

### 推奨リソース
- [SageMaker公式ドキュメント](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)
- [Coursera Machine Learning Course](https://www.coursera.org/learn/machine-learning)

---

**📞 サポート**: 講義実施中に問題が発生した場合は、各ノートブックのトラブルシューティングセクションを参照してください。
