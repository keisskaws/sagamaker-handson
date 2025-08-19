import csv
import random
import math
import numpy as np

def generate_lecture_dataset():
    """
    講義用の軽量データセットを生成
    - 3,000サンプル（元の30%）
    - 30個の特徴量（元の60%）
    - 処理時間を大幅短縮
    """
    print("講義用データセットを生成中...")
    
    random.seed(42)
    np.random.seed(42)
    
    n_samples = 3000  # 10,000 → 3,000
    n_features = 30   # 50 → 30
    n_classes = 3
    
    data = []
    
    # クラス分布を不均衡にする（現実的）
    class_probabilities = [0.6, 0.3, 0.1]  # クラス0が多数、クラス2が少数
    
    for i in range(n_samples):
        # クラスを不均衡に選択
        target = np.random.choice(n_classes, p=class_probabilities)
        
        # 特徴量を生成（クラスに依存）
        features = []
        
        for j in range(n_features):
            if j < 8:  # 最初の8個は重要な特徴量（元10個→8個）
                if target == 0:
                    value = np.random.normal(0, 1)
                elif target == 1:
                    value = np.random.normal(2, 1.5)
                else:  # target == 2
                    value = np.random.normal(-1, 0.8)
            elif j < 16:  # 次の8個は中程度に重要（元10個→8個）
                if target == 0:
                    value = np.random.normal(1, 2)
                elif target == 1:
                    value = np.random.normal(-0.5, 1.2)
                else:
                    value = np.random.normal(1.5, 1.8)
            elif j < 22:  # 相関のある特徴量（元10個→6個）
                if len(features) > 0:
                    value = features[0] * 0.7 + np.random.normal(0, 0.5)
                else:
                    value = np.random.normal(0, 1)
            else:  # 残りはノイズ特徴量
                value = np.random.normal(0, 1)
            
            # 外れ値を時々追加（現実的）
            if random.random() < 0.02:  # 2%の確率で外れ値
                value *= random.uniform(3, 8)
            
            features.append(round(value, 4))
        
        # 欠損値をシミュレート（一部の特徴量で）
        for j in range(25, 28):  # 特定の特徴量で欠損値
            if random.random() < 0.05:  # 5%の確率で欠損
                features[j] = None
        
        data.append(features + [target])
    
    return data, n_features

def save_lecture_dataset():
    data, n_features = generate_lecture_dataset()
    
    # 特徴量名を生成
    feature_names = [f'feature_{i:02d}' for i in range(n_features)]
    header = feature_names + ['target']
    
    # 訓練・検証・テストに分割 (70:15:15)
    random.shuffle(data)
    
    n_total = len(data)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    
    # CSVファイルとして保存
    datasets = [
        ('train_lecture.csv', train_data),
        ('validation_lecture.csv', val_data),
        ('test_lecture.csv', test_data)
    ]
    
    for filename, dataset in datasets:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for row in dataset:
                # 欠損値をNaNとして保存
                processed_row = []
                for value in row:
                    if value is None:
                        processed_row.append('')  # CSVでは空文字で欠損値を表現
                    else:
                        processed_row.append(value)
                writer.writerow(processed_row)
    
    print(f"講義用データセット生成完了:")
    print(f"  訓練データ: {len(train_data)}サンプル")
    print(f"  検証データ: {len(val_data)}サンプル") 
    print(f"  テストデータ: {len(test_data)}サンプル")
    print(f"  特徴量数: {n_features}")
    print(f"  予想実行時間: Script Mode 2-5分, Training Jobs 3-8分")
    
    # クラス分布を確認
    for name, dataset in datasets:
        class_counts = {}
        for row in dataset:
            target = row[-1]
            class_counts[target] = class_counts.get(target, 0) + 1
        print(f"  {name} クラス分布: {class_counts}")

if __name__ == '__main__':
    save_lecture_dataset()
