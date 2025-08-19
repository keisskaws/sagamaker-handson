# 🚀 BYOL (Bring Your Own Library) Script Mode クイックスタート

## 🎯 BYOLとは？

**BYOL (Bring Your Own Library)** は、独自のライブラリやフレームワークをSageMakerで使用する方法です。

### 主な利点
- 🔧 **独自アルゴリズム**: 企業や研究機関の独自手法を使用
- 📦 **既存コード活用**: レガシーシステムとの統合
- ⚡ **最適化**: 特定問題に特化したソリューション
- 🔄 **再利用性**: 既存のコードベースを活用

## 📁 作成されたファイル構成

```
sagemaker-lecture-version/
├── byol_docker/                          # BYOLコンテナ関連
│   ├── Dockerfile                        # カスタムコンテナ定義
│   ├── requirements.txt                  # 依存関係
│   ├── train.py                         # BYOLトレーニングスクリプト
│   └── custom_ml_lib/                   # カスタムライブラリ
│       ├── setup.py                     # ライブラリ設定
│       └── custom_ml_lib/
│           ├── __init__.py
│           ├── custom_classifier.py     # カスタム分類器
│           └── custom_preprocessor.py   # カスタム前処理
├── 06_byol_script_mode.ipynb           # BYOLノートブック
└── BYOL_QUICKSTART.md                  # このファイル
```

## 🚀 実行方法

### 1. Jupyterノートブック実行

```bash
cd ~/sagemaker-lecture-version
jupyter notebook 06_byol_script_mode.ipynb
```

### 2. カスタムライブラリのテスト

```python
# ローカルでカスタムライブラリをテスト
import sys
sys.path.append('./byol_docker/custom_ml_lib')

from custom_ml_lib.custom_classifier import CustomEnsembleClassifier

# テストデータで動作確認
classifier = CustomEnsembleClassifier()
# ... (詳細はノートブック参照)
```

### 3. SageMaker Script Mode実行

```python
from sagemaker.sklearn.estimator import SKLearn

sklearn_estimator = SKLearn(
    entry_point='train.py',
    source_dir='./byol_docker',  # カスタムライブラリを含む
    role=role,
    instance_type='ml.m5.large',
    hyperparameters={
        'use-custom-ensemble': 'true',
        'ensemble-rf': 'true',
        'ensemble-gb': 'true',
        'ensemble-lr': 'true'
    }
)

sklearn_estimator.fit({
    'train': train_s3_path,
    'test': test_s3_path,
    'validation': validation_s3_path
})
```

## 🔧 カスタムライブラリの特徴

### CustomEnsembleClassifier

独自のアンサンブル分類器：

```python
class CustomEnsembleClassifier:
    """
    複数のベースモデルを組み合わせた
    重み付き投票による予測を行う分類器
    """
    
    def __init__(self, use_rf=True, use_gb=True, use_lr=True):
        # RandomForest, GradientBoosting, LogisticRegression
        # を組み合わせ
        
    def fit(self, X, y):
        # クロスバリデーションスコアに基づく重み計算
        
    def predict(self, X):
        # 重み付き投票による予測
```

### 主な機能

1. **自動重み計算**: CVスコアに基づく最適重み
2. **柔軟な構成**: 使用するモデルを選択可能
3. **特徴量重要度**: 各モデルの重要度を取得
4. **scikit-learn互換**: 標準的なAPIを提供

## 🎓 学習の流れ

### Phase 1: ローカルテスト
1. カスタムライブラリの作成
2. ローカル環境での動作確認
3. 基本機能のテスト

### Phase 2: SageMaker統合
1. Script Modeでの実行
2. S3データとの連携
3. ハイパーパラメータ調整

### Phase 3: 本格運用（発展）
1. Dockerコンテナ化
2. ECRへのプッシュ
3. 本番環境でのデプロイ

## 🔍 トラブルシューティング

### エラー1: カスタムライブラリのインポートエラー
```python
# 解決方法
import sys
sys.path.append('./byol_docker/custom_ml_lib')

# または
pip install -e ./byol_docker/custom_ml_lib
```

### エラー2: 依存関係の問題
```bash
# requirements.txtの確認
cat ./byol_docker/requirements.txt

# 必要に応じて追加
echo "your-library==1.0.0" >> ./byol_docker/requirements.txt
```

### エラー3: SageMakerでのパスエラー
```python
# source_dirを正しく設定
sklearn_estimator = SKLearn(
    entry_point='train.py',
    source_dir='./byol_docker',  # 重要: 正しいパス
    # ...
)
```

## 📊 期待される結果

### 成功時の出力例：
```
=== BYOL (Bring Your Own Library) トレーニング開始 ===
✅ カスタムライブラリのインポート成功

モデル重み:
  RandomForest: 0.340
  GradientBoosting: 0.380
  LogisticRegression: 0.280

訓練精度: 0.8952
検証精度: 0.8711
テスト精度: 0.8644

=== BYOL トレーニング完了 ===
総実行時間: 245.67秒
モデルタイプ: custom_ensemble
```

## 🎯 実際の使用例

### 1. 金融業界
```python
# 独自のリスク評価アルゴリズム
class CustomRiskClassifier:
    def __init__(self):
        # 業界特有のルールを組み込み
        
    def predict_risk(self, financial_data):
        # 規制要件に準拠した予測
```

### 2. 医療業界
```python
# 医療画像解析用カスタムモデル
class MedicalImageClassifier:
    def __init__(self):
        # FDA承認済みアルゴリズムを使用
        
    def diagnose(self, medical_image):
        # 医療基準に準拠した診断
```

### 3. 製造業
```python
# 品質管理用カスタムアルゴリズム
class QualityControlClassifier:
    def __init__(self):
        # 製造プロセス特有の特徴を考慮
        
    def predict_defect(self, sensor_data):
        # 製造ラインに最適化された予測
```

## 🚀 次のステップ

### 初級レベル
- [x] カスタムライブラリの基本作成
- [x] SageMaker Script Modeでの実行
- [ ] 他のアルゴリズムの実装

### 中級レベル
- [ ] Dockerコンテナの完全活用
- [ ] ECRを使った本格的なBYOL
- [ ] カスタムメトリクスの実装

### 上級レベル
- [ ] マルチモデルエンドポイント
- [ ] A/Bテスト機能の実装
- [ ] MLOpsパイプラインとの統合

## 💡 重要なポイント

### BYOLの選択基準
- **独自性**: 既存ソリューションでは対応できない要件
- **規制対応**: 業界特有の規制要件への対応
- **レガシー統合**: 既存システムとの互換性
- **競争優位**: 独自のアルゴリズムによる差別化

### 注意事項
- **ライセンス**: 使用するライブラリのライセンス確認
- **セキュリティ**: カスタムコードのセキュリティ監査
- **保守性**: 長期的なメンテナンス計画
- **テスト**: 十分なテストカバレッジの確保

---

**🎓 重要**: BYOLは強力な機能ですが、適切な設計と実装が重要です。まずは小さなカスタムライブラリから始めて、段階的に複雑な機能を追加していくことをお勧めします。
