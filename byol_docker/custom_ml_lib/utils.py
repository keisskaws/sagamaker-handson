"""
カスタムMLライブラリのユーティリティ関数
"""

import pandas as pd
import numpy as np
import joblib
import os

def load_custom_data(file_path, target_column='target'):
    """
    カスタムデータ読み込み関数
    
    Parameters:
    -----------
    file_path : str
        データファイルのパス
    target_column : str, default='target'
        ターゲット列の名前
        
    Returns:
    --------
    X : DataFrame
        特徴量データ
    y : Series
        ターゲットデータ
    """
    try:
        data = pd.read_csv(file_path)
        
        if target_column in data.columns:
            X = data.drop(target_column, axis=1)
            y = data[target_column]
        else:
            X = data
            y = None
            
        return X, y
        
    except Exception as e:
        raise ValueError(f"データの読み込みに失敗しました: {e}")

def save_custom_model(model, file_path, metadata=None):
    """
    カスタムモデル保存関数
    
    Parameters:
    -----------
    model : object
        保存するモデル
    file_path : str
        保存先のパス
    metadata : dict, optional
        モデルのメタデータ
    """
    try:
        # モデル本体を保存
        joblib.dump(model, file_path)
        
        # メタデータがある場合は別ファイルで保存
        if metadata:
            metadata_path = file_path.replace('.joblib', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        print(f"モデルを保存しました: {file_path}")
        
    except Exception as e:
        raise ValueError(f"モデルの保存に失敗しました: {e}")

def load_custom_model(file_path):
    """
    カスタムモデル読み込み関数
    
    Parameters:
    -----------
    file_path : str
        モデルファイルのパス
        
    Returns:
    --------
    model : object
        読み込まれたモデル
    metadata : dict or None
        メタデータ（存在する場合）
    """
    try:
        # モデル本体を読み込み
        model = joblib.load(file_path)
        
        # メタデータの読み込みを試行
        metadata = None
        metadata_path = file_path.replace('.joblib', '_metadata.json')
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
        return model, metadata
        
    except Exception as e:
        raise ValueError(f"モデルの読み込みに失敗しました: {e}")

def validate_data(X, y=None):
    """
    データの妥当性チェック
    
    Parameters:
    -----------
    X : array-like
        特徴量データ
    y : array-like, optional
        ターゲットデータ
        
    Returns:
    --------
    is_valid : bool
        データが妥当かどうか
    issues : list
        発見された問題のリスト
    """
    issues = []
    
    # 基本的なチェック
    if X is None:
        issues.append("特徴量データがNoneです")
        return False, issues
    
    # 形状チェック
    if hasattr(X, 'shape'):
        if len(X.shape) != 2:
            issues.append(f"特徴量データの次元が不正です: {X.shape}")
        
        if X.shape[0] == 0:
            issues.append("特徴量データが空です")
            
        if X.shape[1] == 0:
            issues.append("特徴量が存在しません")
    
    # ターゲットデータのチェック
    if y is not None:
        if hasattr(y, 'shape'):
            if len(y.shape) > 1 and y.shape[1] > 1:
                issues.append(f"ターゲットデータの次元が不正です: {y.shape}")
        
        if hasattr(X, 'shape') and hasattr(y, 'shape'):
            if X.shape[0] != y.shape[0]:
                issues.append(f"特徴量とターゲットのサンプル数が一致しません: {X.shape[0]} vs {y.shape[0]}")
    
    # 欠損値チェック
    if hasattr(X, 'isnull'):
        if X.isnull().any().any():
            issues.append("特徴量データに欠損値が含まれています")
    elif hasattr(X, 'isnan'):
        if np.isnan(X).any():
            issues.append("特徴量データに欠損値が含まれています")
    
    return len(issues) == 0, issues

def preprocess_data(X, y=None, strategy='standard'):
    """
    データの前処理
    
    Parameters:
    -----------
    X : array-like
        特徴量データ
    y : array-like, optional
        ターゲットデータ
    strategy : str, default='standard'
        前処理の戦略 ('standard', 'minmax', 'robust')
        
    Returns:
    --------
    X_processed : array-like
        前処理済み特徴量データ
    preprocessor : object
        前処理オブジェクト
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    # 前処理パイプラインの構築
    steps = [('imputer', SimpleImputer(strategy='median'))]
    
    if strategy == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif strategy == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif strategy == 'robust':
        steps.append(('scaler', RobustScaler()))
    else:
        raise ValueError(f"未知の前処理戦略: {strategy}")
    
    preprocessor = Pipeline(steps)
    
    # 前処理の実行
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, preprocessor

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    評価メトリクスの計算
    
    Parameters:
    -----------
    y_true : array-like
        正解ラベル
    y_pred : array-like
        予測ラベル
    y_pred_proba : array-like, optional
        予測確率
        
    Returns:
    --------
    metrics : dict
        計算されたメトリクス
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'classification_report': classification_report(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    # 確率が提供されている場合の追加メトリクス
    if y_pred_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score, log_loss
            
            # マルチクラスの場合
            if len(np.unique(y_true)) > 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            
        except Exception as e:
            print(f"追加メトリクスの計算に失敗: {e}")
    
    return metrics
