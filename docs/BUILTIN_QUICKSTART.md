# 🚀 SageMakerビルドインアルゴリズム クイックスタート

## ❌ エラー解決：FileNotFoundError

`FileNotFoundError: [Errno 2] No such file or directory: './data/train_lecture.csv'` が発生した場合の解決方法です。

## 🔧 解決方法

### 1. 正しいディレクトリで実行

```bash
# 正しいディレクトリに移動
cd ~/sagemaker-lecture-version

# ファイルが存在することを確認
ls -la data/

# 期待される出力:
# train_lecture.csv
# test_lecture.csv
# validation_lecture.csv
```

### 2. Jupyterノートブック実行

```bash
# sagemaker-lecture-version ディレクトリで実行
cd ~/sagemaker-lecture-version
jupyter notebook 05_builtin_algorithms.ipynb
```

### 3. Pythonスクリプト実行

```bash
# sagemaker-lecture-version ディレクトリで実行
cd ~/sagemaker-lecture-version
python3 builtin_simple_example.py
```

## 📁 ディレクトリ構造確認

正しいディレクトリ構造：

```
sagemaker-lecture-version/
├── data/
│   ├── train_lecture.csv          ← これが必要
│   ├── test_lecture.csv           ← これが必要
│   ├── validation_lecture.csv     ← これが必要
│   └── create_lecture_dataset.py
├── 05_builtin_algorithms.ipynb    ← 新しいノートブック
├── builtin_simple_example.py      ← 新しいスクリプト
└── ... (その他のファイル)
```

## 🛠️ データファイルが見つからない場合

### データファイル生成

```bash
cd ~/sagemaker-lecture-version/data
python3 create_lecture_dataset.py
```

### データファイル確認

```bash
cd ~/sagemaker-lecture-version
find . -name "*lecture*.csv" -type f
```

期待される出力：
```
./data/train_lecture.csv
./data/test_lecture.csv
./data/validation_lecture.csv
```

## 🎯 実行手順（完全版）

### Step 1: 環境確認
```bash
# 現在のディレクトリ確認
pwd
# 期待される出力: /Users/keissk/sagemaker-lecture-version

# データファイル確認
ls -la data/*.csv
```

### Step 2: Jupyter実行
```bash
# Jupyterノートブック起動
jupyter notebook

# ブラウザで 05_builtin_algorithms.ipynb を開く
# セルを順番に実行
```

### Step 3: Python実行（代替方法）
```bash
# シンプルなスクリプト実行
python3 builtin_simple_example.py
```

## 🔍 トラブルシューティング

### エラー1: データファイルが見つからない
```bash
# 解決方法
cd ~/sagemaker-lecture-version
ls -la data/
```

### エラー2: SageMaker権限エラー
```python
# SageMaker Studio または SageMaker Notebook Instance で実行
# IAMロールに以下の権限が必要:
# - AmazonSageMakerFullAccess
# - S3アクセス権限
```

### エラー3: インスタンスタイプエラー
```python
# ml.m5.large が利用できない場合
# ノートブックで instance_type を変更:
instance_type='ml.t3.medium'  # より軽量なインスタンス
```

## 📊 期待される実行結果

### 成功時の出力例：
```
=== SageMakerビルドインアルゴリズム シンプル例 ===
Region: us-west-2
S3 bucket: sagemaker-us-west-2-123456789012

1. データファイル検索中...
✅ データファイル発見: /Users/keissk/sagemaker-lecture-version/data/

2. データ準備中...
データ読み込み成功: (2100, 31)
クラス分布: {0: 1260, 1: 630, 2: 210}
ビルドインアルゴリズム用フォーマット変換完了

3. S3アップロード中...
✅ アップロード完了: s3://sagemaker-us-west-2-123456789012/lecture-builtin-simple/train/train_builtin_simple.csv

4. XGBoostビルドインアルゴリズム実行中...
XGBoostトレーニング開始...
✅ XGBoostトレーニング完了: 180.45秒

🎉 ビルドインアルゴリズム実行成功！
```

## 🎓 学習のポイント

### Script Mode vs ビルドインアルゴリズム

| 項目 | Script Mode | ビルドインアルゴリズム |
|------|-------------|----------------------|
| **データ形式** | 任意 | 特定形式（target列が最初） |
| **カスタマイズ** | 高い | 限定的 |
| **パフォーマンス** | 標準 | AWS最適化済み |
| **実行時間** | 2-5分 | 3-8分 |
| **学習効果** | アルゴリズム理解 | クラウドML理解 |

### 次のステップ

1. **モデルデプロイ**: エンドポイント作成と予測
2. **性能比較**: Script Modeとの精度・速度比較
3. **他のアルゴリズム**: Linear Learner、Image Classification
4. **ハイパーパラメータ最適化**: SageMaker Automatic Model Tuning

## 💡 よくある質問

**Q: なぜtarget列を最初に移動するのですか？**
A: SageMakerビルドインアルゴリズムは、最初の列をターゲット変数として認識するためです。

**Q: Script Modeとどちらが良いですか？**
A: 学習目的ならScript Mode、本番運用ならビルドインアルゴリズムが適しています。

**Q: コストはどのくらいかかりますか？**
A: 講義用の設定（10 rounds、ml.m5.large）で約$1-2程度です。

---

**🎯 重要**: 必ず `sagemaker-lecture-version` ディレクトリで実行してください！
