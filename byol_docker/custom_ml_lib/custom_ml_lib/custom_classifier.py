"""
カスタムアンサンブル分類器
複数のアルゴリズムを組み合わせた独自の分類器
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score

class CustomEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    カスタムアンサンブル分類器
    
    複数のベースモデルを組み合わせて、
    重み付き投票による予測を行う独自の分類器
    """
    
    def __init__(self, 
                 use_rf=True, 
                 use_gb=True, 
                 use_lr=True,
                 rf_params=None,
                 gb_params=None,
                 lr_params=None,
                 voting_weights=None):
        """
        Parameters:
        -----------
        use_rf : bool, default=True
            RandomForestを使用するかどうか
        use_gb : bool, default=True  
            GradientBoostingを使用するかどうか
        use_lr : bool, default=True
            LogisticRegressionを使用するかどうか
        rf_params : dict, optional
            RandomForestのパラメータ
        gb_params : dict, optional
            GradientBoostingのパラメータ
        lr_params : dict, optional
            LogisticRegressionのパラメータ
        voting_weights : list, optional
            各モデルの重み
        """
        self.use_rf = use_rf
        self.use_gb = use_gb
        self.use_lr = use_lr
        self.rf_params = rf_params or {}
        self.gb_params = gb_params or {}
        self.lr_params = lr_params or {}
        self.voting_weights = voting_weights
        
        self.models = []
        self.model_weights = []
        
    def fit(self, X, y):
        """
        モデルを訓練
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            訓練データの特徴量
        y : array-like, shape (n_samples,)
            訓練データのターゲット
        """
        self.classes_ = np.unique(y)
        self.models = []
        self.model_weights = []
        
        # RandomForest
        if self.use_rf:
            rf = RandomForestClassifier(
                random_state=42,
                **self.rf_params
            )
            rf.fit(X, y)
            self.models.append(('RandomForest', rf))
            
        # GradientBoosting
        if self.use_gb:
            gb = GradientBoostingClassifier(
                random_state=42,
                **self.gb_params
            )
            gb.fit(X, y)
            self.models.append(('GradientBoosting', gb))
            
        # LogisticRegression
        if self.use_lr:
            lr = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **self.lr_params
            )
            lr.fit(X, y)
            self.models.append(('LogisticRegression', lr))
        
        # 重みの設定
        if self.voting_weights is None:
            # クロスバリデーションスコアに基づく重み計算
            self._calculate_weights(X, y)
        else:
            self.model_weights = self.voting_weights[:len(self.models)]
            
        return self
    
    def _calculate_weights(self, X, y):
        """
        クロスバリデーションスコアに基づいて重みを計算
        """
        scores = []
        for name, model in self.models:
            cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            scores.append(cv_scores.mean())
        
        # スコアを正規化して重みとする
        total_score = sum(scores)
        self.model_weights = [score / total_score for score in scores]
        
        print("モデル重み:")
        for (name, _), weight in zip(self.models, self.model_weights):
            print(f"  {name}: {weight:.3f}")
    
    def predict(self, X):
        """
        予測を実行
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            予測対象の特徴量
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            予測結果
        """
        if not self.models:
            raise ValueError("モデルが訓練されていません。先にfit()を実行してください。")
        
        # 各モデルの予測確率を取得
        all_probas = []
        for (name, model), weight in zip(self.models, self.model_weights):
            probas = model.predict_proba(X)
            weighted_probas = probas * weight
            all_probas.append(weighted_probas)
        
        # 重み付き平均
        ensemble_probas = np.sum(all_probas, axis=0)
        
        # 最も確率の高いクラスを予測
        predictions = self.classes_[np.argmax(ensemble_probas, axis=1)]
        
        return predictions
    
    def predict_proba(self, X):
        """
        予測確率を返す
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            予測対象の特徴量
            
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            各クラスの予測確率
        """
        if not self.models:
            raise ValueError("モデルが訓練されていません。先にfit()を実行してください。")
        
        # 各モデルの予測確率を取得
        all_probas = []
        for (name, model), weight in zip(self.models, self.model_weights):
            probas = model.predict_proba(X)
            weighted_probas = probas * weight
            all_probas.append(weighted_probas)
        
        # 重み付き平均
        ensemble_probas = np.sum(all_probas, axis=0)
        
        return ensemble_probas
    
    def get_feature_importance(self):
        """
        特徴量重要度を取得（可能なモデルのみ）
        """
        importances = {}
        
        for name, model in self.models:
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
        
        return importances
