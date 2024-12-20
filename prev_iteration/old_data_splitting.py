# data_splitting.py

import pandas as pd
import numpy as np
from datetime import datetime

def split_by_time(data, time_column='Datetime', split_ratio=0.8):
    data = data.sort_values(by=[time_column])
    split_index = int(len(data) * split_ratio)
    return data.iloc[:split_index], data.iloc[split_index:]

def generate_soft_bucket_labels(data, price_column='Close', buckets=[-3, -1, 1, 3]):
    data = data.copy()
    data['Next_Return'] = ((data[price_column].shift(-1) - data[price_column]) / data[price_column]) * 100
    data = data.dropna(subset=['Next_Return'])
    bucket_labels = pd.cut(data['Next_Return'], bins=[-float('inf')] + buckets + [float('inf')], labels=False)
    data = data.drop(['Next_Return'], axis=1)
    return bucket_labels.values, data

def save_split_data(train_data, test_data, y_train, y_test, prefix):
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_file_name = f"{prefix}_train_data_{current_datetime}.parquet"
    test_file_name = f"{prefix}_test_data_{current_datetime}.parquet"
    y_train_file_name = f"{prefix}_y_train_{current_datetime}.npy"
    y_test_file_name = f"{prefix}_y_test_{current_datetime}.npy"
    train_data.to_parquet(train_file_name)
    test_data.to_parquet(test_file_name)
    np.save(y_train_file_name, y_train)
    np.save(y_test_file_name, y_test)
    print(f"Train data saved to {train_file_name}")
    print(f"Test data saved to {test_file_name}")
    print(f"y_train saved to {y_train_file_name}")
    print(f"y_test saved to {y_test_file_name}")

def main_data_splitting():
    ohlcv_1h = pd.read_parquet('/content/preprocessed_1h_ohlcv_data_20240926_041859.parquet')
    ohlcv_1d = pd.read_parquet('/content/preprocessed_1d_ohlcv_data_20240926_041859.parquet')

    # Split data
    train_ohlcv_1h, test_ohlcv_1h = split_by_time(ohlcv_1h)
    train_ohlcv_1d, test_ohlcv_1d = split_by_time(ohlcv_1d)

    # Generate labels
    y_train_1h, train_ohlcv_1h = generate_soft_bucket_labels(train_ohlcv_1h)
    y_test_1h, test_ohlcv_1h = generate_soft_bucket_labels(test_ohlcv_1h)
    y_train_1d, train_ohlcv_1d = generate_soft_bucket_labels(train_ohlcv_1d)
    y_test_1d, test_ohlcv_1d = generate_soft_bucket_labels(test_ohlcv_1d)

    # Save data
    save_split_data(train_ohlcv_1h, test_ohlcv_1h, y_train_1h, y_test_1h, prefix='1h')
    save_split_data(train_ohlcv_1d, test_ohlcv_1d, y_train_1d, y_test_1d, prefix='1d')

# Run data splitting
main_data_splitting()
