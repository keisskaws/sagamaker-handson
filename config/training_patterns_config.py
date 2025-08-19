#!/usr/bin/env python3
"""
Training Jobs テストパターン設定

様々なテストパターンの設定を提供します
"""

def get_instance_comparison_configs():
    """インスタンスタイプ比較用の設定"""
    base_hyperparameters = {
        'enable-grid-search': 'true',
        'n-jobs': -1
    }
    
    configs = [
        {
            'name': 'ml.m5.large',
            'entry_point': 'train_lecture.py',
            'instance_type': 'ml.m5.large',
            'hyperparameters': base_hyperparameters,
            'job_name_prefix': 'instance-comparison-m5-large',
            'max_run': 1800
        },
        {
            'name': 'ml.m5.xlarge',
            'entry_point': 'train_lecture.py',
            'instance_type': 'ml.m5.xlarge',
            'hyperparameters': base_hyperparameters,
            'job_name_prefix': 'instance-comparison-m5-xlarge',
            'max_run': 1800
        },
        {
            'name': 'ml.c5.large',
            'entry_point': 'train_lecture.py',
            'instance_type': 'ml.c5.large',
            'hyperparameters': base_hyperparameters,
            'job_name_prefix': 'instance-comparison-c5-large',
            'max_run': 1800
        },
        {
            'name': 'ml.c5.xlarge',
            'entry_point': 'train_lecture.py',
            'instance_type': 'ml.c5.xlarge',
            'hyperparameters': base_hyperparameters,
            'job_name_prefix': 'instance-comparison-c5-xlarge',
            'max_run': 1800
        }
    ]
    
    return configs

def get_hyperparameter_comparison_configs():
    """ハイパーパラメータ比較用の設定"""
    base_config = {
        'entry_point': 'train_lecture.py',
        'instance_type': 'ml.m5.large',
        'job_name_prefix': 'hyperparameter-comparison',
        'max_run': 1800
    }
    
    configs = [
        {
            **base_config,
            'name': 'GridSearch-Enabled',
            'hyperparameters': {
                'enable-grid-search': 'true',
                'n-jobs': -1
            }
        },
        {
            **base_config,
            'name': 'GridSearch-Disabled',
            'hyperparameters': {
                'enable-grid-search': 'false',
                'n-jobs': -1
            }
        },
        {
            **base_config,
            'name': 'Single-Thread',
            'hyperparameters': {
                'enable-grid-search': 'true',
                'n-jobs': 1
            }
        },
        {
            **base_config,
            'name': 'Dual-Thread',
            'hyperparameters': {
                'enable-grid-search': 'true',
                'n-jobs': 2
            }
        }
    ]
    
    return configs

def get_algorithm_comparison_configs():
    """アルゴリズム比較用の設定（異なるトレーニングスクリプト）"""
    # 注意: この例では同じスクリプトを使用していますが、
    # 実際のプロジェクトでは異なるアルゴリズム用のスクリプトを作成します
    
    base_config = {
        'entry_point': 'train_lecture.py',
        'instance_type': 'ml.m5.large',
        'job_name_prefix': 'algorithm-comparison',
        'max_run': 1800
    }
    
    configs = [
        {
            **base_config,
            'name': 'RandomForest-Focus',
            'hyperparameters': {
                'enable-grid-search': 'true',
                'primary-algorithm': 'RandomForest',
                'n-jobs': -1
            }
        },
        {
            **base_config,
            'name': 'GradientBoosting-Focus',
            'hyperparameters': {
                'enable-grid-search': 'true',
                'primary-algorithm': 'GradientBoosting',
                'n-jobs': -1
            }
        },
        {
            **base_config,
            'name': 'LogisticRegression-Focus',
            'hyperparameters': {
                'enable-grid-search': 'true',
                'primary-algorithm': 'LogisticRegression',
                'n-jobs': -1
            }
        }
    ]
    
    return configs

def get_custom_configs():
    """カスタム設定（ユーザー入力ベース）"""
    print("\nカスタム設定を作成します...")
    
    configs = []
    
    while True:
        print(f"\n設定 {len(configs) + 1} を作成中...")
        
        name = input("設定名: ").strip()
        if not name:
            break
            
        print("\nインスタンスタイプを選択:")
        print("1. ml.m5.large")
        print("2. ml.m5.xlarge")
        print("3. ml.c5.large")
        print("4. ml.c5.xlarge")
        
        instance_choice = input("選択 (1-4): ").strip()
        instance_map = {
            '1': 'ml.m5.large',
            '2': 'ml.m5.xlarge',
            '3': 'ml.c5.large',
            '4': 'ml.c5.xlarge'
        }
        
        instance_type = instance_map.get(instance_choice, 'ml.m5.large')
        
        # ハイパーパラメータ設定
        enable_grid_search = input("GridSearchを有効にしますか？ (y/n): ").strip().lower() == 'y'
        n_jobs = input("並列ジョブ数 (-1 for all cores): ").strip()
        
        try:
            n_jobs = int(n_jobs) if n_jobs else -1
        except ValueError:
            n_jobs = -1
        
        config = {
            'name': name,
            'entry_point': 'train_lecture.py',
            'instance_type': instance_type,
            'hyperparameters': {
                'enable-grid-search': 'true' if enable_grid_search else 'false',
                'n-jobs': n_jobs
            },
            'job_name_prefix': f'custom-{name.lower().replace(" ", "-")}',
            'max_run': 1800
        }
        
        configs.append(config)
        
        if input("\n他の設定を追加しますか？ (y/n): ").strip().lower() != 'y':
            break
    
    return configs

def get_performance_test_configs():
    """性能テスト用の設定（大規模データ対応）"""
    configs = [
        {
            'name': 'Performance-Small-Instance',
            'entry_point': 'train_lecture.py',
            'instance_type': 'ml.m5.large',
            'hyperparameters': {
                'enable-grid-search': 'true',
                'n-jobs': -1,
                'performance-test': 'true'
            },
            'job_name_prefix': 'performance-small',
            'max_run': 3600
        },
        {
            'name': 'Performance-Large-Instance',
            'entry_point': 'train_lecture.py',
            'instance_type': 'ml.m5.2xlarge',
            'hyperparameters': {
                'enable-grid-search': 'true',
                'n-jobs': -1,
                'performance-test': 'true'
            },
            'job_name_prefix': 'performance-large',
            'max_run': 3600
        }
    ]
    
    return configs

def get_cost_optimization_configs():
    """コスト最適化テスト用の設定"""
    configs = [
        {
            'name': 'Cost-Optimized-Basic',
            'entry_point': 'train_lecture.py',
            'instance_type': 'ml.m5.large',
            'hyperparameters': {
                'enable-grid-search': 'false',  # GridSearchを無効化
                'n-jobs': 2,  # 並列数を制限
                'quick-mode': 'true'
            },
            'job_name_prefix': 'cost-optimized-basic',
            'max_run': 1200
        },
        {
            'name': 'Cost-Optimized-Minimal',
            'entry_point': 'train_lecture.py',
            'instance_type': 'ml.t3.medium',  # より安価なインスタンス
            'hyperparameters': {
                'enable-grid-search': 'false',
                'n-jobs': 1,
                'quick-mode': 'true'
            },
            'job_name_prefix': 'cost-optimized-minimal',
            'max_run': 1200
        }
    ]
    
    return configs
