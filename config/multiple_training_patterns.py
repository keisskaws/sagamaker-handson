#!/usr/bin/env python3
"""
SageMaker Training Jobs 複数パターンテスト用スクリプト

このスクリプトでは以下のパターンをテストできます：
1. 異なるインスタンスタイプでの性能比較
2. 異なるハイパーパラメータセットでの比較
3. 異なるアルゴリズムでの比較
4. 並列実行による効率化
"""

import sagemaker
import boto3
import time
import json
import pandas as pd
from datetime import datetime
from sagemaker.sklearn.estimator import SKLearn
from sagemaker import get_execution_role
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultipleTrainingPatterns:
    def __init__(self):
        self.sagemaker_session = sagemaker.Session()
        self.role = get_execution_role()
        self.region = boto3.Session().region_name
        self.bucket = self.sagemaker_session.default_bucket()
        
        # S3データパスを読み込み
        try:
            with open('s3_data_paths.json', 'r') as f:
                self.s3_paths = json.load(f)
            logger.info("S3データパス情報を読み込み完了")
        except FileNotFoundError:
            logger.error("S3データパス情報が見つかりません")
            raise
    
    def create_estimator(self, config):
        """設定に基づいてEstimatorを作成"""
        return SKLearn(
            entry_point=config['entry_point'],
            framework_version='1.0-1',
            py_version='py3',
            instance_type=config['instance_type'],
            instance_count=config.get('instance_count', 1),
            role=self.role,
            hyperparameters=config['hyperparameters'],
            base_job_name=config['job_name_prefix'],
            max_run=config.get('max_run', 3600)
        )
    
    def run_single_training(self, config):
        """単一のTraining Jobを実行"""
        logger.info(f"Training Job開始: {config['name']}")
        start_time = time.time()
        
        try:
            estimator = self.create_estimator(config)
            
            # Training Job実行
            estimator.fit(self.s3_paths)
            
            # 結果の取得
            job_name = estimator.latest_training_job.job_name
            job_description = self.sagemaker_session.describe_training_job(job_name)
            
            # 実行時間とコストの計算
            total_time = time.time() - start_time
            billable_seconds = job_description.get('BillableTimeInSeconds', 0)
            
            # インスタンス別の時間単価（USD/時間）
            instance_costs = {
                'ml.m5.large': 0.115,
                'ml.m5.xlarge': 0.230,
                'ml.m5.2xlarge': 0.460,
                'ml.c5.large': 0.108,
                'ml.c5.xlarge': 0.216
            }
            
            hourly_cost = instance_costs.get(config['instance_type'], 0.115)
            actual_cost = (billable_seconds / 3600) * hourly_cost
            
            result = {
                'name': config['name'],
                'job_name': job_name,
                'instance_type': config['instance_type'],
                'hyperparameters': config['hyperparameters'],
                'total_time': total_time,
                'training_time': billable_seconds,
                'cost': actual_cost,
                'status': job_description['TrainingJobStatus'],
                'model_artifacts': estimator.model_data
            }
            
            logger.info(f"Training Job完了: {config['name']} ({total_time:.1f}秒, ${actual_cost:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Training Job失敗: {config['name']} - {str(e)}")
            return {
                'name': config['name'],
                'status': 'Failed',
                'error': str(e),
                'total_time': time.time() - start_time
            }
    
    def run_parallel_training(self, configs, max_workers=3):
        """複数のTraining Jobを並列実行"""
        logger.info(f"{len(configs)}個のTraining Jobを並列実行開始 (最大{max_workers}並列)")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 全てのジョブを投入
            future_to_config = {
                executor.submit(self.run_single_training, config): config 
                for config in configs
            }
            
            # 完了したものから結果を収集
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"並列実行エラー: {config['name']} - {str(e)}")
                    results.append({
                        'name': config['name'],
                        'status': 'Failed',
                        'error': str(e)
                    })
        
        return results
    
    def analyze_results(self, results):
        """結果の分析とレポート生成"""
        logger.info("結果分析中...")
        
        # 成功したジョブのみを分析
        successful_results = [r for r in results if r.get('status') == 'Completed']
        
        if not successful_results:
            logger.warning("成功したTraining Jobがありません")
            return
        
        # 結果をDataFrameに変換
        df = pd.DataFrame(successful_results)
        
        print("\n" + "="*80)
        print("🎯 Training Jobs 複数パターン実行結果")
        print("="*80)
        
        # 基本統計
        print(f"\n📊 実行統計:")
        print(f"  総実行数: {len(results)}")
        print(f"  成功数: {len(successful_results)}")
        print(f"  失敗数: {len(results) - len(successful_results)}")
        
        if len(successful_results) > 0:
            print(f"\n⏱️ 実行時間統計:")
            print(f"  平均実行時間: {df['total_time'].mean():.1f}秒")
            print(f"  最短実行時間: {df['total_time'].min():.1f}秒")
            print(f"  最長実行時間: {df['total_time'].max():.1f}秒")
            
            print(f"\n💰 コスト統計:")
            print(f"  総コスト: ${df['cost'].sum():.4f}")
            print(f"  平均コスト: ${df['cost'].mean():.4f}")
            print(f"  最小コスト: ${df['cost'].min():.4f}")
            print(f"  最大コスト: ${df['cost'].max():.4f}")
            
            # 詳細結果テーブル
            print(f"\n📋 詳細結果:")
            display_df = df[['name', 'instance_type', 'total_time', 'cost']].copy()
            display_df['total_time'] = display_df['total_time'].round(1)
            display_df['cost'] = display_df['cost'].round(4)
            print(display_df.to_string(index=False))
            
            # 最適な設定の推奨
            best_time = df.loc[df['total_time'].idxmin()]
            best_cost = df.loc[df['cost'].idxmin()]
            
            print(f"\n🏆 推奨設定:")
            print(f"  最速実行: {best_time['name']} ({best_time['total_time']:.1f}秒)")
            print(f"  最安実行: {best_cost['name']} (${best_cost['cost']:.4f})")
        
        # 結果をJSONファイルに保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'training_patterns_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 詳細結果を保存: {results_file}")
        
        return results

def main():
    """メイン実行関数"""
    trainer = MultipleTrainingPatterns()
    
    print("🚀 SageMaker Training Jobs 複数パターンテスト")
    print("="*60)
    
    # テストパターンの選択
    print("\n実行するテストパターンを選択してください:")
    print("1. インスタンスタイプ比較")
    print("2. ハイパーパラメータ比較") 
    print("3. アルゴリズム比較")
    print("4. カスタム設定")
    
    choice = input("\n選択 (1-4): ").strip()
    
    if choice == '1':
        configs = get_instance_comparison_configs()
    elif choice == '2':
        configs = get_hyperparameter_comparison_configs()
    elif choice == '3':
        configs = get_algorithm_comparison_configs()
    elif choice == '4':
        configs = get_custom_configs()
    else:
        print("無効な選択です")
        return
    
    print(f"\n{len(configs)}個のパターンを実行します...")
    
    # 並列実行の確認
    parallel = input("並列実行しますか？ (y/n): ").strip().lower() == 'y'
    
    if parallel:
        max_workers = min(3, len(configs))  # 最大3並列
        results = trainer.run_parallel_training(configs, max_workers)
    else:
        results = []
        for config in configs:
            result = trainer.run_single_training(config)
            results.append(result)
    
    # 結果分析
    trainer.analyze_results(results)

if __name__ == '__main__':
    main()
