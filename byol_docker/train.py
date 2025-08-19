#!/usr/bin/env python3
"""
BYOL (Bring Your Own Library) トレーニングスクリプト
カスタムライブラリを使用したSageMakerトレーニング
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# カスタムライブラリをインポート
try:
    from custom_ml_lib import CustomEnsembleClassifier, CustomPreprocessor
    from custom_ml_lib.utils import load_custom_data, save_custom_model
    print("✅ カスタムライブラリのインポート成功")
except ImportError as e:
    print(f"⚠️ カスタムライブラリのインポートに失敗: {e}")
    print("標準ライブラリを使用します")
    CustomEnsembleClassifier = None

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser()
    
    # SageMakerの環境変数
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # カスタムハイパーパラメータ
    parser.add_argument('--use-custom-ensemble', type=str, default='true')
    parser.add_argument('--ensemble-rf', type=str, default='true')
    parser.add_argument('--ensemble-gb', type=str, default='true')
    parser.add_argument('--ensemble-lr', type=str, default='true')
    parser.add_argument('--feature-selection-k', type=int, default=20)
    
    return parser.parse_args()

def load_data(train_path, test_path=None, validation_path=None):
    """データの読み込み"""
    print("データを読み込み中...")
    
    datasets = {}
    
    # 訓練データ
    train_file = os.path.join(train_path, 'train_lecture.csv')
    if os.path.exists(train_file):
        datasets['train'] = pd.read_csv(train_file)
        print(f"  Train: {datasets['train'].shape}")
    
    # テストデータ
    if test_path:
        test_file = os.path.join(test_path, 'test_lecture.csv')
        if os.path.exists(test_file):
            datasets['test'] = pd.read_csv(test_file)
            print(f"  Test: {datasets['test'].shape}")
    
    # 検証データ
    if validation_path:
        val_file = os.path.join(validation_path, 'validation_lecture.csv')
        if os.path.exists(val_file):
            datasets['validation'] = pd.read_csv(val_file)
            print(f"  Validation: {datasets['validation'].shape}")
    
    return datasets

def create_preprocessing_pipeline(feature_selection_k=20):
    """前処理パイプラインの作成"""
    print(f"前処理パイプラインを作成中... (特徴選択: {feature_selection_k})")
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=feature_selection_k))
    ])
    
    return pipeline

def train_custom_model(X_train, y_train, X_val=None, y_val=None, args=None):
    """カスタムモデルでトレーニング"""
    print("\n=== BYOL カスタムモデルトレーニング開始 ===")
    
    if CustomEnsembleClassifier is None:
        print("❌ カスタムライブラリが利用できません。標準モデルを使用します。")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, {"model_type": "fallback_random_forest"}
    
    # カスタムアンサンブル分類器の設定
    use_rf = args.ensemble_rf.lower() == 'true' if args else True
    use_gb = args.ensemble_gb.lower() == 'true' if args else True
    use_lr = args.ensemble_lr.lower() == 'true' if args else True
    
    print(f"アンサンブル設定:")
    print(f"  RandomForest: {use_rf}")
    print(f"  GradientBoosting: {use_gb}")
    print(f"  LogisticRegression: {use_lr}")
    
    # カスタムモデル作成
    model = CustomEnsembleClassifier(
        use_rf=use_rf,
        use_gb=use_gb,
        use_lr=use_lr,
        rf_params={'n_estimators': 50, 'max_depth': 10},
        gb_params={'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.1},
        lr_params={'C': 1.0, 'penalty': 'l2'}
    )
    
    # トレーニング実行
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 訓練精度
    train_score = model.score(X_train, y_train)
    print(f"訓練精度: {train_score:.4f}")
    
    # 検証精度
    val_score = None
    if X_val is not None and y_val is not None:
        val_score = model.score(X_val, y_val)
        print(f"検証精度: {val_score:.4f}")
    
    print(f"トレーニング時間: {training_time:.2f}秒")
    
    # 特徴量重要度
    try:
        importances = model.get_feature_importance()
        if importances:
            print("\n特徴量重要度（上位5個）:")
            for model_name, importance in importances.items():
                if hasattr(importance, '__len__'):
                    top_indices = np.argsort(importance)[::-1][:5]
                    print(f"  {model_name}:")
                    for i, idx in enumerate(top_indices):
                        print(f"    {i+1}. 特徴量 {idx}: {importance[idx]:.4f}")
    except Exception as e:
        print(f"特徴量重要度の取得に失敗: {e}")
    
    results = {
        "model_type": "custom_ensemble",
        "training_time": training_time,
        "train_score": train_score,
        "val_score": val_score,
        "ensemble_config": {
            "use_rf": use_rf,
            "use_gb": use_gb,
            "use_lr": use_lr
        }
    }
    
    return model, results

def evaluate_model(model, X_test, y_test):
    """モデルの評価"""
    print("\nモデルを評価中...")
    
    predictions = model.predict(X_test)
    
    # 予測確率（可能な場合）
    probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(X_test)
        except Exception as e:
            print(f"予測確率の取得に失敗: {e}")
    
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"テスト精度: {accuracy:.4f}")
    print("\n分類レポート:")
    print(classification_report(y_test, predictions))
    
    print("\n混同行列:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    results = {
        'accuracy': accuracy,
        'predictions': predictions.tolist(),
    }
    
    if probabilities is not None:
        results['probabilities'] = probabilities.tolist()
    
    return results

def model_fn(model_dir):
    """SageMaker推論用のモデル読み込み関数"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    return {'model': model, 'preprocessor': preprocessor}

def main():
    """メイン実行関数"""
    args = parse_args()
    
    print("=== BYOL (Bring Your Own Library) トレーニング開始 ===")
    print("引数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    total_start_time = time.time()
    
    # データの読み込み
    datasets = load_data(args.train, args.test, args.validation)
    
    if 'train' not in datasets:
        print("❌ 訓練データが見つかりません")
        return
    
    # 前処理パイプラインの作成
    preprocessor = create_preprocessing_pipeline(args.feature_selection_k)
    
    # 特徴量とターゲットの分離
    X_train = datasets['train'].drop('target', axis=1)
    y_train = datasets['train']['target']
    
    X_test = None
    y_test = None
    if 'test' in datasets:
        X_test = datasets['test'].drop('target', axis=1)
        y_test = datasets['test']['target']
    
    X_val = None
    y_val = None
    if 'validation' in datasets:
        X_val = datasets['validation'].drop('target', axis=1)
        y_val = datasets['validation']['target']
    
    print(f"\nデータ分割:")
    print(f"  訓練: {X_train.shape}")
    if X_val is not None:
        print(f"  検証: {X_val.shape}")
    if X_test is not None:
        print(f"  テスト: {X_test.shape}")
    
    # 前処理の実行
    print("\n前処理を実行中...")
    preprocessing_start = time.time()
    
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    if X_val is not None:
        X_val_processed = preprocessor.transform(X_val)
    else:
        X_val_processed = None
    if X_test is not None:
        X_test_processed = preprocessor.transform(X_test)
    else:
        X_test_processed = None
    
    preprocessing_time = time.time() - preprocessing_start
    print(f"前処理完了: {preprocessing_time:.2f}秒")
    print(f"処理後の特徴量数: {X_train_processed.shape[1]}")
    
    # カスタムモデルトレーニング
    if args.use_custom_ensemble.lower() == 'true':
        model, training_results = train_custom_model(
            X_train_processed, y_train, 
            X_val_processed, y_val,
            args
        )
    else:
        print("標準のRandomForestを使用します...")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_processed, y_train)
        training_results = {"model_type": "standard_random_forest"}
    
    # モデル評価
    evaluation_results = {}
    if X_test_processed is not None and y_test is not None:
        evaluation_results = evaluate_model(model, X_test_processed, y_test)
    
    total_time = time.time() - total_start_time
    
    # 結果の保存
    print(f"\nモデルとメタデータを保存中...")
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(args.model_dir, "preprocessor.joblib"))
    
    # メトリクスの保存
    metrics = {
        'total_training_time': total_time,
        'preprocessing_time': preprocessing_time,
        'data_shape': {
            'train': list(X_train.shape),
            'features_after_preprocessing': X_train_processed.shape[1]
        },
        'training_results': training_results
    }
    
    if evaluation_results:
        metrics['evaluation_results'] = evaluation_results
    
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n=== BYOL トレーニング完了 ===")
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"モデルタイプ: {training_results.get('model_type', 'unknown')}")
    if evaluation_results:
        print(f"テスト精度: {evaluation_results.get('accuracy', 'N/A'):.4f}")

if __name__ == '__main__':
    main()
