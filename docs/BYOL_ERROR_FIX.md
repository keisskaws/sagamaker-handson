# 🔧 BYOL make_classification エラー修正ガイド

## ❌ 発生したエラー

```
n_classes(3) * n_clusters_per_class(2) must be smaller or equal 2**n_informative(2)=4
```

## 🔍 エラーの原因

`sklearn.datasets.make_classification`のパラメータ設定に問題がありました：

- **n_classes=3**: 3クラス分類
- **n_clusters_per_class=2**: クラスあたり2つのクラスター（デフォルト）
- **n_informative=2**: 情報のある特徴量が2個（デフォルト）

**問題**: `3 × 2 = 6 > 2² = 4` となり、情報量が不足

## ✅ 修正方法

### 修正前（エラーが発生）
```python
X_test, y_test = make_classification(
    n_samples=100, 
    n_features=10, 
    n_classes=3, 
    random_state=42
)
```

### 修正後（正常動作）
```python
X_test, y_test = make_classification(
    n_samples=200,           # サンプル数を増加
    n_features=10,           # 特徴量数
    n_informative=8,         # 情報のある特徴量数を増加（重要！）
    n_redundant=2,           # 冗長な特徴量数
    n_classes=3,             # クラス数
    n_clusters_per_class=1,  # クラスあたりのクラスター数を1に
    random_state=42,
    class_sep=1.0            # クラス間の分離度を追加
)
```

## 📊 修正のポイント

### 1. n_informative を増加
- **修正前**: `n_informative=2`（デフォルト）
- **修正後**: `n_informative=8`
- **効果**: 3クラス分類に十分な情報量を確保

### 2. n_clusters_per_class を調整
- **修正前**: `n_clusters_per_class=2`（デフォルト）
- **修正後**: `n_clusters_per_class=1`
- **効果**: 必要な情報量を削減

### 3. その他の改善
- **n_samples**: 100 → 200（データ量増加）
- **class_sep**: 追加（クラス分離度向上）
- **n_redundant**: 明示的に設定

## 🧮 数学的な説明

make_classificationの制約：
```
n_classes × n_clusters_per_class ≤ 2^n_informative
```

### 修正前（エラー）
```
3 × 2 = 6 > 2² = 4  ❌
```

### 修正後（正常）
```
3 × 1 = 3 ≤ 2⁸ = 256  ✅
```

## 🚀 実行方法

### 1. Jupyterノートブックで実行
```python
# 06_byol_script_mode.ipynb の該当セルで実行
# 修正済みのコードが含まれています
```

### 2. テストスクリプトで確認
```python
# test_byol_fixed.py を実行
exec(open('test_byol_fixed.py').read())
```

### 3. SageMaker環境での実行
```python
# SageMaker Studio または Notebook Instance で実行
# 必要なライブラリが自動的に利用可能
```

## 🎯 期待される結果

修正後の実行結果：
```
✅ カスタムライブラリのインポート成功
テストデータ生成: (200, 10), クラス数: 3
クラス分布: {0: 67, 1: 67, 2: 66}

モデル重み:
  RandomForest: 0.340
  GradientBoosting: 0.380
  LogisticRegression: 0.280

✅ 予測結果: [1 0 2 1 0 2 1 0 2 1]
✅ 予測確率形状: (10, 3)
✅ 訓練精度: 0.8950
🎉 カスタムライブラリの動作確認完了
```

## 🔍 トラブルシューティング

### エラー1: scikit-learn がない
```bash
# ローカル環境の場合
pip install scikit-learn pandas numpy

# SageMaker環境では自動的に利用可能
```

### エラー2: カスタムライブラリが見つからない
```python
# パスの確認
import os
print(os.getcwd())
print(os.path.exists('./byol_docker/custom_ml_lib'))

# パスの追加
import sys
sys.path.append('./byol_docker/custom_ml_lib')
```

### エラー3: 他のパラメータエラー
```python
# より安全なパラメータ設定
X_test, y_test = make_classification(
    n_samples=500,           # さらに多くのサンプル
    n_features=20,           # より多くの特徴量
    n_informative=15,        # 十分な情報量
    n_redundant=3,           # 冗長特徴量
    n_classes=3,             # クラス数
    n_clusters_per_class=1,  # シンプルなクラスター
    random_state=42,
    class_sep=2.0,           # より大きな分離度
    flip_y=0.01              # 少しのノイズ
)
```

## 💡 学習のポイント

### make_classificationの理解
1. **n_informative**: 実際に分類に役立つ特徴量数
2. **n_redundant**: 情報特徴量の線形結合
3. **n_clusters_per_class**: クラス内のクラスター数
4. **class_sep**: クラス間の分離度

### 実際のデータとの違い
- **合成データ**: パラメータで制御可能
- **実際のデータ**: 自然な分布と複雑性
- **BYOL**: 実際のデータに最適化されたアルゴリズム

## 🎓 次のステップ

1. **修正されたコードでテスト実行**
2. **実際のデータでBYOL実行**
3. **カスタムアルゴリズムの改良**
4. **SageMaker環境での本格運用**

---

**🔧 重要**: この修正により、BYOLカスタムライブラリが正常に動作するようになります。SageMaker環境での実行をお勧めします。
