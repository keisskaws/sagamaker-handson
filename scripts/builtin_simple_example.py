#!/usr/bin/env python3
"""
SageMakerビルドインアルゴリズム - シンプル実行例
講義用に最適化された最小限のコード（パス問題修正版）
"""

import sagemaker
import boto3
import pandas as pd
import numpy as np
import time
import os
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput

def find_data_path():
    """データファイルのパスを自動検出"""
    possible_paths = [
        './data/',
        '../data/',
        '~/sagemaker-lecture-version/data/',
        '/Users/keissk/sagemaker-lecture-version/data/',
        os.path.expanduser('~/sagemaker-lecture-version/data/')
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        train_file = os.path.join(expanded_path, 'train_lecture.csv')
        if os.path.exists(train_file):
            return expanded_path
    
    return None

def main():
    print("=== SageMakerビルドインアルゴリズム シンプル例 ===")
    
    # 基本設定
    try:
        sagemaker_session = sagemaker.Session()
        role = get_execution_role()
        region = boto3.Session().region_name
        bucket = sagemaker_session.default_bucket()
        
        print(f"Region: {region}")
        print(f"S3 bucket: {bucket}")
        print(f"Current directory: {os.getcwd()}")
        
    except Exception as e:
        print(f"❌ SageMaker設定エラー: {e}")
        print("💡 このスクリプトはSageMaker環境で実行してください")
        return
    
    # データファイル検索
    print("\n1. データファイル検索中...")
    data_path = find_data_path()
    
    if data_path is None:
        print("❌ データファイルが見つかりません")
        print("現在のディレクトリ:", os.getcwd())
        print("\n利用可能なCSVファイル:")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.csv') and 'lecture' in file:
                    print(f"  {os.path.join(root, file)}")
        
        print("\n💡 解決方法:")
        print("1. sagemaker-lecture-version ディレクトリで実行してください")
        print("2. または、データファイルのパスを確認してください")
        return
    
    print(f"✅ データファイル発見: {data_path}")
    
    # データ準備
    print("\n2. データ準備中...")
    try:
        train_file = os.path.join(data_path, 'train_lecture.csv')
        train_data = pd.read_csv(train_file)
        
        print(f"データ読み込み成功: {train_data.shape}")
        print(f"クラス分布: {train_data['target'].value_counts().to_dict()}")
        
        # ビルドインアルゴリズム用フォーマット（target列を最初に移動）
        target = train_data['target']
        features = train_data.drop('target', axis=1)
        train_builtin = pd.concat([target, features], axis=1)
        
        # CSVファイルとして保存（ヘッダーなし）
        train_builtin.to_csv('train_builtin_simple.csv', index=False, header=False)
        print("ビルドインアルゴリズム用フォーマット変換完了")
        
    except Exception as e:
        print(f"❌ データ準備エラー: {e}")
        return
    
    # S3アップロード
    print("\n3. S3アップロード中...")
    try:
        train_s3_path = sagemaker_session.upload_data(
            'train_builtin_simple.csv', 
            bucket=bucket, 
            key_prefix='lecture-builtin-simple/train'
        )
        print(f"✅ アップロード完了: {train_s3_path}")
        
    except Exception as e:
        print(f"❌ S3アップロードエラー: {e}")
        return
    
    # XGBoostビルドインアルゴリズム
    print("\n4. XGBoostビルドインアルゴリズム実行中...")
    
    try:
        # XGBoostコンテナイメージ取得
        from sagemaker.image_uris import retrieve
        xgboost_image = retrieve('xgboost', region, version='1.5-1')
        
        # XGBoostエスティメーター作成（最小限の設定）
        xgboost_estimator = sagemaker.estimator.Estimator(
            image_uri=xgboost_image,
            role=role,
            instance_count=1,
            instance_type='ml.m5.large',
            hyperparameters={
                'max_depth': 3,
                'eta': 0.3,
                'objective': 'multi:softprob',
                'num_class': 3,
                'num_round': 10,  # 講義用に大幅削減
                'eval_metric': 'mlogloss'
            },
            output_path=f's3://{bucket}/lecture-builtin-simple/output'
        )
        
        # トレーニング実行
        start_time = time.time()
        train_input = TrainingInput(train_s3_path, content_type='text/csv')
        
        print("XGBoostトレーニング開始...")
        xgboost_estimator.fit({'train': train_input}, wait=True)
        
        training_time = time.time() - start_time
        print(f"✅ XGBoostトレーニング完了: {training_time:.2f}秒")
        
        # 成功メッセージ
        print("\n🎉 ビルドインアルゴリズム実行成功！")
        print(f"トレーニング時間: {training_time:.2f}秒")
        print("モデルはS3に保存されました。")
        
        # 結果サマリー
        print("\n📊 実行サマリー:")
        print(f"  データサイズ: {train_data.shape}")
        print(f"  アルゴリズム: XGBoost (ビルドイン)")
        print(f"  トレーニング時間: {training_time:.2f}秒")
        print(f"  出力場所: s3://{bucket}/lecture-builtin-simple/output")
        
    except Exception as e:
        print(f"❌ XGBoostトレーニングエラー: {e}")
        print("\n💡 トラブルシューティング:")
        print("1. SageMaker実行ロールが正しく設定されているか確認")
        print("2. 必要な権限（S3、SageMaker）があるか確認")
        print("3. インスタンスタイプ ml.m5.large が利用可能か確認")
        print("4. リージョンでXGBoostが利用可能か確認")
        return
        
    print("\n=== 実行完了 ===")
    print("🎓 SageMakerビルドインアルゴリズムの基本的な使い方を学習しました！")

if __name__ == '__main__':
    main()
