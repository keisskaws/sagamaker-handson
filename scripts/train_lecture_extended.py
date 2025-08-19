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
    
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(f_classif, k=20))
    ])
    
    return preprocessing_pipeline

def get_model_configs(primary_algorithm=None, quick_mode=False):
    """モデル設定を取得（パラメータ化対応）"""
    
    if quick_mode:
        # クイックモード: パラメータ数を最小限に
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {'n_estimators': [50]}
            }
        }
    elif primary_algorithm == 'RandomForest':
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None]
                }
            }
        }
    elif primary_algorithm == 'GradientBoosting':
        models = {
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.1, 0.2],
                    'max_depth': [3, 5]
                }
            }
        }
    elif primary_algorithm == 'LogisticRegression':
        models = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=500),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            }
        }
    else:
        # デフォルト: 全モデル（軽量版）
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'max_depth': [10, 20]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.1, 0.2]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=500),
                'params': {
                    'C': [0.1, 1.0]
                }
            }
        }
    
    return models

def train_models_with_config(X_train, y_train, X_val, y_val, 
                           enable_grid_search=True, n_jobs=-1, 
                           primary_algorithm=None, quick_mode=False):
    """設定可能なモデル訓練"""
    print(f"=== モデル訓練開始 ===")
    print(f"GridSearch: {enable_grid_search}")
    print(f"並列ジョブ数: {n_jobs}")
    print(f"主要アルゴリズム: {primary_algorithm or '全て'}")
    print(f"クイックモード: {quick_mode}")
    
    models = get_model_configs(primary_algorithm, quick_mode)
    results = {}
    best_models = {}
    
    for name, config in models.items():
        print(f"\n{name} を訓練中...")
        start_time = time.time()
        
        if enable_grid_search and not quick_mode:
            # グリッドサーチ実行
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=3,
                scoring='accuracy',
                n_jobs=n_jobs,
                verbose=0
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # シンプル訓練
            model = config['model']
            model.fit(X_train, y_train)
            best_model = model
            best_params = {}
        
        training_time = time.time() - start_time
        
        # 評価
        train_score = best_model.score(X_train, y_train)
        val_score = best_model.score(X_val, y_val) if X_val is not None else None
        
        # クロスバリデーション（軽量）
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=3)
        
        best_models[name] = best_model
        results[name] = {
            'best_params': best_params,
            'train_score': train_score,
            'val_score': val_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time
        }
        
        print(f"  最適パラメータ: {best_params}")
        print(f"  訓練精度: {train_score:.4f}")
        if val_score:
            print(f"  検証精度: {val_score:.4f}")
        print(f"  CV精度: {cv_scores.mean():.4f}")
        print(f"  訓練時間: {training_time:.2f}秒")
    
    return best_models, results

def select_best_model(models, results):
    """最適なモデルを選択"""
    print("\n最適なモデルを選択中...")
    
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
        print("\n上位5個の重要な特徴量:")
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
    
    # 拡張ハイパーパラメータ
    parser.add_argument('--enable-grid-search', type=str, default='true')
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--primary-algorithm', type=str, default=None)
    parser.add_argument('--quick-mode', type=str, default='false')
    parser.add_argument('--performance-test', type=str, default='false')
    
    args = parser.parse_args()
    
    print("=== 拡張機械学習トレーニング開始 ===")
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
    
    # パラメータの解析
    enable_grid_search = args.enable_grid_search.lower() == 'true'
    quick_mode = args.quick_mode.lower() == 'true'
    
    # モデル訓練
    models, results = train_models_with_config(
        X_train_processed, y_train, 
        X_val_processed, y_val,
        enable_grid_search=enable_grid_search,
        n_jobs=args.n_jobs,
        primary_algorithm=args.primary_algorithm,
        quick_mode=quick_mode
    )
    
    best_model, best_model_name, best_result = select_best_model(models, results)
    
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
    
    # 拡張メトリクスの保存
    metrics = {
        'best_model': best_model_name,
        'total_training_time': total_time,
        'preprocessing_time': preprocessing_time,
        'configuration': {
            'enable_grid_search': enable_grid_search,
            'n_jobs': args.n_jobs,
            'primary_algorithm': args.primary_algorithm,
            'quick_mode': quick_mode
        },
        'data_shape': {
            'train': list(X_train.shape),
            'features_after_preprocessing': X_train_processed.shape[1]
        },
        'model_comparison': results
    }
    
    if evaluation_results:
        metrics['test_accuracy'] = evaluation_results['accuracy']
    
    with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n=== 拡張訓練完了 ===")
    print(f"総実行時間: {total_time:.2f}秒")
    print(f"最適モデル: {best_model_name}")
    if evaluation_results:
        print(f"テスト精度: {evaluation_results['accuracy']:.4f}")

if __name__ == '__main__':
    train()
