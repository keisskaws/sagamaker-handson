"""
カスタム前処理クラス
独自の前処理ロジックを実装
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    """
    カスタム前処理クラス
    
    独自の前処理ロジックを組み合わせた前処理器
    """
    
    def __init__(self, 
                 impute_strategy='median',
                 scale_features=True,
                 select_features=True,
                 n_features=20,
                 remove_outliers=True,
                 outlier_threshold=3.0):
        """
        Parameters:
        -----------
        impute_strategy : str, default='median'
            欠損値補完の戦略
        scale_features : bool, default=True
            特徴量をスケーリングするかどうか
        select_features : bool, default=True
            特徴量選択を行うかどうか
        n_features : int, default=20
            選択する特徴量数
        remove_outliers : bool, default=True
            外れ値を除去するかどうか
        outlier_threshold : float, default=3.0
            外れ値判定の閾値（標準偏差の倍数）
        """
        self.impute_strategy = impute_strategy
        self.scale_features = scale_features
        self.select_features = select_features
        self.n_features = n_features
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        # 内部で使用するトランスフォーマー
        self.imputer = None
        self.scaler = None
        self.feature_selector = None
        self.outlier_mask = None
        
    def fit(self, X, y=None):
        """
        前処理器を訓練データに適合
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            訓練データの特徴量
        y : array-like, shape (n_samples,), optional
            訓練データのターゲット
        """
        X = self._validate_input(X)
        
        # 1. 欠損値補完器の適合
        self.imputer = SimpleImputer(strategy=self.impute_strategy)
        X_imputed = self.imputer.fit_transform(X)
        
        # 2. 外れ値検出（訓練時のみ）
        if self.remove_outliers:
            self.outlier_mask = self._detect_outliers(X_imputed)
            X_clean = X_imputed[~self.outlier_mask]
            if y is not None:
                y_clean = np.array(y)[~self.outlier_mask]
            else:
                y_clean = None
        else:
            X_clean = X_imputed
            y_clean = y
        
        # 3. スケーラーの適合
        if self.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean
        
        # 4. 特徴量選択器の適合
        if self.select_features and y_clean is not None:
            n_features = min(self.n_features, X_scaled.shape[1])
            self.feature_selector = SelectKBest(f_classif, k=n_features)
            self.feature_selector.fit(X_scaled, y_clean)
        
        return self
    
    def transform(self, X):
        """
        データを変換
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            変換対象の特徴量
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_selected_features)
            変換後の特徴量
        """
        X = self._validate_input(X)
        
        # 1. 欠損値補完
        if self.imputer is None:
            raise ValueError("前処理器が適合されていません。先にfit()を実行してください。")
        
        X_imputed = self.imputer.transform(X)
        
        # 2. スケーリング
        if self.scale_features and self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        # 3. 特徴量選択
        if self.select_features and self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_scaled)
        else:
            X_selected = X_scaled
        
        return X_selected
    
    def fit_transform(self, X, y=None):
        """
        適合と変換を同時に実行
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            訓練データの特徴量
        y : array-like, shape (n_samples,), optional
            訓練データのターゲット
            
        Returns:
        --------
        X_transformed : array, shape (n_samples, n_selected_features)
            変換後の特徴量
        """
        return self.fit(X, y).transform(X)
    
    def _validate_input(self, X):
        """
        入力データの検証
        """
        if X is None:
            raise ValueError("入力データがNoneです")
        
        # DataFrameをnumpy配列に変換
        if hasattr(X, 'values'):
            X = X.values
        
        # 2次元配列に変換
        X = np.atleast_2d(X)
        
        return X
    
    def _detect_outliers(self, X):
        """
        外れ値を検出
        
        Parameters:
        -----------
        X : array-like
            特徴量データ
            
        Returns:
        --------
        outlier_mask : array, shape (n_samples,)
            外れ値のマスク（Trueが外れ値）
        """
        # Z-scoreベースの外れ値検出
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        outlier_mask = np.any(z_scores > self.outlier_threshold, axis=1)
        
        outlier_count = np.sum(outlier_mask)
        total_count = len(outlier_mask)
        
        print(f"外れ値検出: {outlier_count}/{total_count} ({outlier_count/total_count*100:.1f}%)")
        
        return outlier_mask
    
    def get_feature_names_out(self, input_features=None):
        """
        出力特徴量名を取得
        """
        if self.feature_selector is not None:
            if input_features is not None:
                selected_indices = self.feature_selector.get_support(indices=True)
                return [input_features[i] for i in selected_indices]
            else:
                return [f"feature_{i}" for i in range(self.feature_selector.k_)]
        else:
            if input_features is not None:
                return input_features
            else:
                return None
    
    def get_params(self, deep=True):
        """
        パラメータを取得
        """
        return {
            'impute_strategy': self.impute_strategy,
            'scale_features': self.scale_features,
            'select_features': self.select_features,
            'n_features': self.n_features,
            'remove_outliers': self.remove_outliers,
            'outlier_threshold': self.outlier_threshold
        }
    
    def set_params(self, **params):
        """
        パラメータを設定
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"無効なパラメータ: {key}")
        return self
    
    def get_preprocessing_info(self):
        """
        前処理の情報を取得
        """
        info = {
            'impute_strategy': self.impute_strategy,
            'scale_features': self.scale_features,
            'select_features': self.select_features,
            'n_features': self.n_features,
            'remove_outliers': self.remove_outliers,
            'outlier_threshold': self.outlier_threshold
        }
        
        if self.imputer is not None:
            info['imputer_statistics'] = getattr(self.imputer, 'statistics_', None)
        
        if self.scaler is not None:
            info['scaler_mean'] = getattr(self.scaler, 'mean_', None)
            info['scaler_scale'] = getattr(self.scaler, 'scale_', None)
        
        if self.feature_selector is not None:
            info['selected_features'] = self.feature_selector.get_support(indices=True).tolist()
            info['feature_scores'] = getattr(self.feature_selector, 'scores_', None)
        
        if self.outlier_mask is not None:
            info['outliers_removed'] = int(np.sum(self.outlier_mask))
            info['outlier_ratio'] = float(np.mean(self.outlier_mask))
        
        return info
