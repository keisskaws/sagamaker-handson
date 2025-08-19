# 🎓 SageMaker機械学習講義（完全版）

高校生から大学生向けの機械学習講義用に最適化されたプロジェクトです。
01から06まで段階的に学習できるように構成されています。

## 📚 講義構成（6つのノートブック）

### 01. データ管理とS3連携
**ファイル**: `01_data_management.ipynb`
**学習時間**: 20分
**内容**:
- S3バケットの作成と管理
- データのアップロード・ダウンロード
- SageMakerでのデータ管理ベストプラクティス

### 02. Script Mode Training
**ファイル**: `02_script_mode_training_complete.ipynb`
**学習時間**: 30分
**内容**:
- Script Modeの基本概念
- カスタム訓練スクリプトの作成
- 迅速な実験・デバッグ手法

### 03. Training Jobs
**ファイル**: `03_training_jobs.ipynb`
**学習時間**: 30分
**内容**:
- Training Jobsの概念と利点
- スケーラブルな訓練の実行
- 本番環境での機械学習開発

### 04. 比較とまとめ
**ファイル**: `04_comparison_and_summary.ipynb`
**学習時間**: 20分
**内容**:
- Script Mode vs Training Jobsの比較
- 使い分けの指針
- 実際のプロジェクトでの選択基準

### 05. Built-in Algorithms
**ファイル**: `05_builtin_algorithms.ipynb`
**学習時間**: 25分
**内容**:
- SageMaker組み込みアルゴリズムの活用
- XGBoost、Linear Learnerなどの実践
- アルゴリズム選択の指針

### 06. BYOL (Bring Your Own Library)
**ファイル**: `06_byol_script_mode.ipynb`
**学習時間**: 35分
**内容**:
- カスタムライブラリの持ち込み
- Dockerコンテナの活用
- 高度なカスタマイゼーション

## ⏱️ 総学習時間: 約160分（2時間40分）

## 🎯 学習目標

### 基礎レベル（01-02）
- SageMakerの基本操作
- データ管理の基礎
- Script Modeでの開発

### 中級レベル（03-04）
- Training Jobsの活用
- 開発手法の比較・選択
- スケーラブルな機械学習

### 上級レベル（05-06）
- 組み込みアルゴリズムの活用
- カスタムライブラリの統合
- 本格的な機械学習システム構築

## 🚀 使用方法

### 1. SageMaker Studioでの実行

1. SageMaker Studioにログイン
2. JupyterLabを起動
3. このディレクトリをアップロード
4. 01から順番にノートブックを実行

### 2. 推奨学習順序

```
01_data_management.ipynb
    ↓
02_script_mode_training_complete.ipynb
    ↓
03_training_jobs.ipynb
    ↓
04_comparison_and_summary.ipynb
    ↓
05_builtin_algorithms.ipynb
    ↓
06_byol_script_mode.ipynb
```

## 📁 ファイル構成

```
sagemaker-lecture-organized/
├── README.md                                    # 📖 このファイル
├── 01_data_management.ipynb                     # 📊 データ管理
├── 02_script_mode_training_complete.ipynb       # 🚀 Script Mode
├── 03_training_jobs.ipynb                       # ☁️ Training Jobs
├── 04_comparison_and_summary.ipynb              # 📊 比較・まとめ
├── 05_builtin_algorithms.ipynb                  # 🔧 組み込みアルゴリズム
├── 06_byol_script_mode.ipynb                    # 🐳 BYOL
├── scripts/                                     # 📝 訓練スクリプト
│   ├── train_lecture.py                         # 基本訓練スクリプト
│   ├── train_lecture_extended.py                # 拡張訓練スクリプト
│   ├── builtin_algorithms_example.py            # 組み込みアルゴリズム例
│   └── builtin_simple_example.py                # シンプル例
├── byol_docker/                                 # 🐳 BYOL用Docker
│   ├── Dockerfile                               # Dockerファイル
│   ├── requirements.txt                         # 依存関係
│   ├── train.py                                 # 訓練スクリプト
│   └── custom_ml_lib/                           # カスタムライブラリ
├── data/                                        # 📈 データセット
│   ├── create_lecture_dataset.py                # データ生成
│   ├── train_lecture.csv                        # 訓練データ
│   ├── validation_lecture.csv                   # 検証データ
│   └── test_lecture.csv                         # テストデータ
├── config/                                      # ⚙️ 設定ファイル
│   ├── training_patterns_config.py              # 訓練パターン設定
│   └── multiple_training_patterns.py            # 複数パターン
└── docs/                                        # 📚 ドキュメント
    ├── QUICKSTART.md                            # クイックスタート
    ├── BUILTIN_QUICKSTART.md                    # 組み込みアルゴリズム
    ├── BYOL_QUICKSTART.md                       # BYOL
    ├── BYOL_ERROR_FIX.md                        # エラー対処法
    └── MULTIPLE_PATTERNS_GUIDE.md               # 複数パターンガイド
```

## 💡 講義での活用方法

### 短時間講義（90分）の場合
- 01, 02, 04を中心に実施
- 基本概念の理解に重点

### 標準講義（3時間）の場合
- 01-04を完全実施
- 05または06のどちらかを選択

### 集中講義（1日）の場合
- 01-06を全て実施
- 各セクション間で質疑応答時間を確保

## 🔧 前提条件

### 必要な環境
- AWS SageMaker Studio
- Python 3.8以上
- 必要なライブラリ（requirements.txtに記載）

### 推奨スペック
- ml.t3.medium以上のインスタンス
- 最低10GB以上のストレージ

## 📊 データセット詳細

### 特徴
- **サンプル数**: 3,000個
- **特徴量**: 30個（feature_00 〜 feature_29）
- **クラス**: 3種類（0, 1, 2）
- **クラス分布**: 60% : 30% : 10%

### 現実的な要素
- ✅ 欠損値あり
- ✅ ノイズあり
- ✅ 相関のある特徴量
- ✅ 不均衡データ

## 🎉 期待される学習成果

この講義を完了すると、学生は以下を習得できます：

- ✅ SageMakerの基本操作と概念
- ✅ データ管理とS3連携
- ✅ Script ModeとTraining Jobsの使い分け
- ✅ 組み込みアルゴリズムの活用
- ✅ カスタムライブラリの統合
- ✅ 実際のプロジェクトでの開発フロー

## 🔍 トラブルシューティング

### よくある問題と解決方法

1. **権限エラー**
   - IAMロールの設定を確認
   - SageMaker実行ロールの権限を確認

2. **データアクセスエラー**
   - S3バケットの権限設定を確認
   - リージョンの一致を確認

3. **Docker関連エラー**
   - ECRの権限設定を確認
   - Dockerfileの構文を確認

## 📈 次のステップ

### 発展学習
1. より大きなデータセットでの実験
2. 他のアルゴリズム（Deep Learning）
3. MLOpsパイプラインの構築
4. 実際のビジネス問題への適用

### 推奨リソース
- [SageMaker公式ドキュメント](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Examples](https://github.com/aws/amazon-sagemaker-examples)
- [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/)

---

**🎓 講師の方へ**: このプロジェクトは段階的学習に最適化されています。学習者のレベルに応じて、適切なノートブックを選択してご活用ください。
