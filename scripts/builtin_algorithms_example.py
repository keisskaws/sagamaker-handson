#!/usr/bin/env python3
"""
SageMakerビルドインアルゴリズムの使用例
講義用に最適化されたビルドインアルゴリズムの比較
"""

import sagemaker
import boto3
import pandas as pd
import numpy as np
import time
import json
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

class SageMakerBuiltinDemo:
    def __init__(self):
        self.sagemaker_session = sagemaker.Session()
        self.role = get_execution_role()
        self.region = boto3.Session().region_name
        self.bucket = self.sagemaker_session.default_bucket()
        
        print(f"SageMaker role: {self.role}")
        print(f"Region: {self.region}")
        print(f"S3 bucket: {self.bucket}")
    
    def prepare_data_for_builtin(self, data_path='./data'):
        """ビルドインアルゴリズム用にデータを準備"""
        print("ビルドインアルゴリズム用データ準備中...")
        
        # データ読み込み
        train_data = pd.read_csv(f'{data_path}/train_lecture.csv')
        test_data = pd.read_csv(f'{data_path}/test_lecture.csv')
        validation_data = pd.read_csv(f'{data_path}/validation_lecture.csv')
        
        # ビルドインアルゴリズム用フォーマット（target列を最初に移動）
        def reformat_for_builtin(df):
            target = df['target']
            features = df.drop('target', axis=1)
            return pd.concat([target, features], axis=1)
        
        train_builtin = reformat_for_builtin(train_data)
        test_builtin = reformat_for_builtin(test_data)
        validation_builtin = reformat_for_builtin(validation_data)
        
        # S3にアップロード
        train_s3_path = f's3://{self.bucket}/lecture-builtin/train/train.csv'
        test_s3_path = f's3://{self.bucket}/lecture-builtin/test/test.csv'
        validation_s3_path = f's3://{self.bucket}/lecture-builtin/validation/validation.csv'
        
        train_builtin.to_csv('train_builtin.csv', index=False, header=False)
        test_builtin.to_csv('test_builtin.csv', index=False, header=False)
        validation_builtin.to_csv('validation_builtin.csv', index=False, header=False)
        
        # S3アップロード
        self.sagemaker_session.upload_data('train_builtin.csv', 
                                          bucket=self.bucket, 
                                          key_prefix='lecture-builtin/train')
        self.sagemaker_session.upload_data('test_builtin.csv', 
                                          bucket=self.bucket, 
                                          key_prefix='lecture-builtin/test')
        self.sagemaker_session.upload_data('validation_builtin.csv', 
                                          bucket=self.bucket, 
                                          key_prefix='lecture-builtin/validation')
        
        print(f"データをS3にアップロード完了:")
        print(f"  Train: {train_s3_path}")
        print(f"  Test: {test_s3_path}")
        print(f"  Validation: {validation_s3_path}")
        
        return {
            'train': train_s3_path,
            'test': test_s3_path,
            'validation': validation_s3_path
        }
    
    def train_xgboost_builtin(self, data_paths):
        """XGBoostビルドインアルゴリズムでトレーニング"""
        print("\n=== XGBoostビルドインアルゴリズム ===")
        
        # XGBoostコンテナイメージ取得
        from sagemaker.image_uris import retrieve
        xgboost_image = retrieve('xgboost', self.region, version='1.5-1')
        
        # XGBoostエスティメーター作成
        xgboost_estimator = sagemaker.estimator.Estimator(
            image_uri=xgboost_image,
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.large',
            hyperparameters={
                'max_depth': 5,
                'eta': 0.2,
                'gamma': 4,
                'min_child_weight': 6,
                'subsample': 0.8,
                'objective': 'multi:softprob',
                'num_class': 3,
                'num_round': 50,  # 講義用に削減
                'eval_metric': 'mlogloss'
            },
            output_path=f's3://{self.bucket}/lecture-builtin/xgboost-output'
        )
        
        # トレーニング入力設定
        train_input = TrainingInput(data_paths['train'], content_type='text/csv')
        validation_input = TrainingInput(data_paths['validation'], content_type='text/csv')
        
        # トレーニング実行
        start_time = time.time()
        print("XGBoostトレーニング開始...")
        
        xgboost_estimator.fit({
            'train': train_input,
            'validation': validation_input
        }, wait=True)
        
        training_time = time.time() - start_time
        print(f"XGBoostトレーニング完了: {training_time:.2f}秒")
        
        return xgboost_estimator, training_time
    
    def train_linear_learner_builtin(self, data_paths):
        """Linear Learnerビルドインアルゴリズムでトレーニング"""
        print("\n=== Linear Learnerビルドインアルゴリズム ===")
        
        # Linear Learnerコンテナイメージ取得
        from sagemaker.image_uris import retrieve
        linear_image = retrieve('linear-learner', self.region)
        
        # Linear Learnerエスティメーター作成
        linear_estimator = sagemaker.estimator.Estimator(
            image_uri=linear_image,
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.large',
            hyperparameters={
                'feature_dim': 30,  # 特徴量数
                'mini_batch_size': 100,
                'predictor_type': 'multiclass_classifier',
                'num_classes': 3,
                'epochs': 10,  # 講義用に削減
                'learning_rate': 0.1,
                'l1': 0.0,
                'l2': 0.0
            },
            output_path=f's3://{self.bucket}/lecture-builtin/linear-output'
        )
        
        # トレーニング入力設定
        train_input = TrainingInput(data_paths['train'], content_type='text/csv')
        validation_input = TrainingInput(data_paths['validation'], content_type='text/csv')
        
        # トレーニング実行
        start_time = time.time()
        print("Linear Learnerトレーニング開始...")
        
        linear_estimator.fit({
            'train': train_input,
            'validation': validation_input
        }, wait=True)
        
        training_time = time.time() - start_time
        print(f"Linear Learnerトレーニング完了: {training_time:.2f}秒")
        
        return linear_estimator, training_time
    
    def deploy_and_test_model(self, estimator, model_name, test_data_path):
        """モデルをデプロイしてテスト"""
        print(f"\n{model_name}モデルをデプロイ中...")
        
        # エンドポイントデプロイ
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',  # 講義用に軽量インスタンス
            serializer=CSVSerializer(),
            deserializer=JSONDeserializer()
        )
        
        # テストデータで予測
        test_data = pd.read_csv(test_data_path)
        test_features = test_data.drop('target', axis=1)
        test_labels = test_data['target']
        
        # 予測実行（小さなバッチで）
        predictions = []
        batch_size = 100
        
        for i in range(0, len(test_features), batch_size):
            batch = test_features.iloc[i:i+batch_size]
            batch_pred = predictor.predict(batch.values)
            predictions.extend(batch_pred['predictions'])
        
        # 精度計算
        predicted_classes = [np.argmax(pred) for pred in predictions]
        accuracy = np.mean(predicted_classes == test_labels)
        
        print(f"{model_name}テスト精度: {accuracy:.4f}")
        
        # エンドポイント削除（コスト削減）
        predictor.delete_endpoint()
        print(f"{model_name}エンドポイントを削除しました")
        
        return accuracy, predictions
    
    def run_builtin_comparison(self):
        """ビルドインアルゴリズム比較実行"""
        print("=== SageMakerビルドインアルゴリズム比較開始 ===")
        
        # データ準備
        data_paths = self.prepare_data_for_builtin()
        
        results = {}
        
        try:
            # XGBoost
            xgb_estimator, xgb_time = self.train_xgboost_builtin(data_paths)
            xgb_accuracy, _ = self.deploy_and_test_model(
                xgb_estimator, "XGBoost", './data/test_lecture.csv'
            )
            results['XGBoost'] = {
                'training_time': xgb_time,
                'test_accuracy': xgb_accuracy
            }
            
        except Exception as e:
            print(f"XGBoostエラー: {e}")
            results['XGBoost'] = {'error': str(e)}
        
        try:
            # Linear Learner
            linear_estimator, linear_time = self.train_linear_learner_builtin(data_paths)
            linear_accuracy, _ = self.deploy_and_test_model(
                linear_estimator, "Linear Learner", './data/test_lecture.csv'
            )
            results['Linear Learner'] = {
                'training_time': linear_time,
                'test_accuracy': linear_accuracy
            }
            
        except Exception as e:
            print(f"Linear Learnerエラー: {e}")
            results['Linear Learner'] = {'error': str(e)}
        
        # 結果表示
        print("\n=== ビルドインアルゴリズム比較結果 ===")
        for model_name, result in results.items():
            if 'error' not in result:
                print(f"{model_name}:")
                print(f"  トレーニング時間: {result['training_time']:.2f}秒")
                print(f"  テスト精度: {result['test_accuracy']:.4f}")
            else:
                print(f"{model_name}: エラー - {result['error']}")
        
        # 結果保存
        with open('builtin_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    """メイン実行関数"""
    demo = SageMakerBuiltinDemo()
    results = demo.run_builtin_comparison()
    
    print("\n🎓 講義用ビルドインアルゴリズム比較完了！")
    print("結果はbuiltin_results.jsonに保存されました。")

if __name__ == '__main__':
    main()
