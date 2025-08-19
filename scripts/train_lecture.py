#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(train_path, test_path=None, val_path=None):
    """データの読み込みと前処理"""
    print("データを読み込み中...")
    
    # データ読み込み
    train_data = pd.read_csv(os.path.join(train_path, 'train_lecture.csv'))
    
    datasets = {'train': train_data}
    
    if test_path:
        test_data = pd.read_csv(os.path.join(test_path, 'test_lecture.csv'))
        datasets['test'] = test_data
    
    if val_path:
        val_data = pd.read_csv(os.path.join(val_path, 'validation_lecture.csv'))
        datasets['validation'] = val_data
    
    print(f"データ読み込み完了:")
    for name, data in datasets.items():
        print(f"  {name}: {data.shape}")
        print(f"  欠損値: {data.isnull().sum().sum()}")
    
    return datasets

def create_preprocessing_pipeline():
    """前処理パイプラインの作成"""
    print("前処理パイプラインを作成中...")
    
    # 欠損値補完 → 標準化 → 特徴選択（軽量化）
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=20))  # 30→20に削減
    ])
    
    return preprocessing_pipeline

def train_multiple_models_lecture(X_train, y_train, X_val=None, y_val=None):
    """講義用：軽量化された複数モデル訓練"""
    print("講義用：軽量モデル比較開始...")
    
    # パラメータ数を大幅削減
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],      # [100,200,300] → [50,100]
                'max_depth': [10, 20],          # [10,20,None] → [10,20]
                'min_samples_split': [2, 5]     # [2,5,10] → [2,5]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100],      # [100,200] → [50,100]
                'learning_rate': [0.1, 0.2],    # [0.05,0.1,0.2] → [0.1,0.2]
                'max_depth': [3, 5]             # [3,5,7] → [3,5]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=500),  # 1000→500
            'params': {
                'C': [0.1, 1.0],                # [0.1,1.0,10.0] → [0.1,1.0]
                'penalty': ['l2'],              # ['l1','l2'] → ['l2']のみ
                'solver': ['lbfgs']             # ['liblinear'] → ['lbfgs']（高速）
            }
        }
    }
    
    best_models = {}
    results = {}
    
    for name, config in models.items():
        print(f"\n{name} を軽量グリッドサーチで最適化中...")
        start_time = time.time()
        
        # 軽量グリッドサーチ
        grid_search = GridSearchCV(
            config['model'],
            config['params'],
            cv=3,  # 3-fold cross validation
            scoring='accuracy',
            n_jobs=-1,
            verbose=0  # ログを簡略化
        )
        
        grid_search.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        best_models[name] = grid_search.best_estimator_
        
        # 訓練精度
        train_score = grid_search.best_estimator_.score(X_train, y_train)
        
        # 検証精度（もしあれば）
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = grid_search.best_estimator_.score(X_val, y_val)
        
        # 軽量クロスバリデーション
        cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=3)  # 5→3
        
        results[name] = {
            'best_params': grid_search.best_params_,
            'train_score': train_score,
            'val_score': val_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }
        
        print(f"  最適パラメータ: {grid_search.best_params_}")
        print(f"  訓練精度: {train_score:.4f}")
        if val_score:
            print(f"  検証精度: {val_score:.4f}")
        print(f"  CV精度: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  訓練時間: {training_time:.2f}秒")
    
    return best_models, results

def select_best_model(models, results):
    """最適なモデルを選択"""
    print("\n最適なモデルを選択中...")
    
    # 検証精度またはCV精度で選択
    best_score = -1
    best_model_name = None
    
    for name, result in results.items():
        score = result['val_score'] if result['val_score'] else result['cv_mean']
        if score > best_score:
            best_score = score
            best_model_name = name
    
    print(f"最適モデル: {best_model_name} (スコア: {best_score:.4f})")
    return models[best_model_name], best_model_name, results[best_model_name]

def evaluate_model(model, X_test, y_test):
    """モデルの詳細評価"""
    print("\nモデルを評価中...")
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"テスト精度: {accuracy:.4f}")
    print("\n分類レポート:")
    print(classification_report(y_test, predictions))
    
    print("\n混同行列:")
    cm = confusion_matrix(y_test, predictions)
    print(cm)
    
    # 特徴量重要度（可能な場合）
    if hasattr(model, 'feature_importances_'):
        print("\n上位5個の重要な特徴量:")  # 10→5に削減
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(indices):
            print(f"  {i+1}. 特徴量 {idx}: {importances[idx]:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    }

def model_fn(model_dir):
    """SageMaker推論用のモデル読み込み関数"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    return {'model': model, 'preprocessor': preprocessor}

def train():
    parser = argparse.ArgumentParser()
    
    # SageMakerの環境変数
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # カスタムハイパーパラメータ
    parser.add_argument('--enable-grid-search', type=str, default='true')
    parser.add_argument('--n-jobs', type=int, default=-1)
    
    args = parser.parse_args()
    
    print("=== 講義用：軽量機械学習トレーニング開始 ===")
    print("引数:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    total_start_time = time.time()
    
    # データの読み込み
    datasets = load_and_preprocess_data(
        args.train, 
        args.test, 
        args.validation
    )
    
    # 前処理パイプラインの作成
    preprocessor = create_preprocessing_pipeline()
    
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
    
    # モデル訓練
    if args.enable_grid_search.lower() == 'true':
        models, results = train_multiple_models_lecture(
            X_train_processed, y_train, 
            X_val_processed, y_val
        )
        best_model, best_model_name, best_result = select_best_model(models, results)
    else:
        print("シンプルなRandomForestで訓練中...")
        best_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            n_jobs=args.n_jobs
        )
        best_model.fit(X_train_processed, y_train)
        best_model_name = "RandomForest"
    
    # モデル評価
    if X_test_processed is not None and y_test is not None:
        evaluation_results = evaluate_model(best_model, X_test_processed, y_test)
    else:
        evaluation_results = {}
    
    total_time = time.time() - total_start_time
    
    # 結果の保存
    print(f"\nモデルとメタデータを保存中...")
    joblib.dump(best_model, os.path.join(args.model_dir, "model.joblib"))
    joblib.dump(preprocessor, os.path.join(args.model_dir, "preprocessor.joblib"))
    
    # メトリクスの保存
    metrics = {
        'best_model': best_model_name,
        'total_training_time': total_time,
        'preprocessing_time': preprocessing_time,
        'data_shape': {
            'train': list(X_train.shape),
            'features_after_preprocessing': X_train_processed.shape[1]
        }
    }
    
    if args.enable_grid_search.lower() == 'true':
        metrics['model_comparison'] = results
    
    if evaluation_results:
        metrics['test_accuracy'] = evaluation_results['accuracy']
    
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n=== 講義用訓練完了 ===")
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"最適モデル: {best_model_name}")
    if evaluation_results:
        print(f"テスト精度: {evaluation_results['accuracy']:.4f}")

if __name__ == '__main__':
    train()
