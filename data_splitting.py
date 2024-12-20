# data_splitting.py

import pandas as pd
import numpy as np
from datetime import datetime

def split_by_time(data, time_column='Datetime', split_ratio=0.8):
    data = data.sort_values(by=[time_column])
    split_index = int(len(data) * split_ratio)
    return data.iloc[:split_index], data.iloc[split_index:]

def generate_soft_bucket_labels(data, price_column='Close', buckets=[-1.5, -.5, .5, 1.5]): #buckets=[-1.5, -.5, .5, 1.5] buckets=[-3, -1, 1, 3]
    data = data.copy()
    data['Next_Return'] = ((data[price_column].shift(-1) - data[price_column]) / data[price_column]) * 100
    data = data.dropna(subset=['Next_Return'])
    bucket_labels = pd.cut(data['Next_Return'], bins=[-float('inf')] + buckets + [float('inf')], labels=False)
    data = data.drop(['Next_Return'], axis=1)
    return bucket_labels.values, data

def save_split_data(train_data, test_data, y_train, y_test, prefix, date=False):
    if date == True:
        stamp = '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    else: stamp = ''
    train_file_name = f"{prefix}_train_data{stamp}.parquet"
    test_file_name = f"{prefix}_test_data{stamp}.parquet"
    y_train_file_name = f"{prefix}_y_train{stamp}.npy"
    y_test_file_name = f"{prefix}_y_test{stamp}.npy"
    train_data.to_parquet(train_file_name)
    test_data.to_parquet(test_file_name)
    np.save(y_train_file_name, y_train)
    np.save(y_test_file_name, y_test)
    print(f"Train data saved to {train_file_name}")
    print(f"Test data saved to {test_file_name}")
    print(f"y_train saved to {y_train_file_name}")
    print(f"y_test saved to {y_test_file_name}")

def main_data_splitting():
    # Read preprocessed 1h and 1d data
    ohlcv_1h = pd.read_parquet('/content/preprocessed_1h_ohlcv_data.parquet')
    ohlcv_1d = pd.read_parquet('/content/preprocessed_1d_ohlcv_data.parquet')

    # Get unique tickers
    tickers = ohlcv_1h['Ticker'].unique()

    # Drop Interval flag
    ohlcv_1h = ohlcv_1h.drop('Interval', axis=1)
    ohlcv_1d = ohlcv_1d.drop('Interval', axis=1)

    # Initialize lists to accumulate split data
    train_1h_list, test_1h_list = [], []
    y_train_1h_list, y_test_1h_list = [], []
    train_1d_list, test_1d_list = [], []
    y_train_1d_list, y_test_1d_list = [], []

    for ticker in tickers:
        # Filter data for the current ticker
        ticker_1h = ohlcv_1h[ohlcv_1h['Ticker'] == ticker]
        ticker_1d = ohlcv_1d[ohlcv_1d['Ticker'] == ticker]

        # Split 1h data
        train_ohlcv_1h, test_ohlcv_1h = split_by_time(ticker_1h)

        # Generate labels for 1h data
        y_train_1h, train_ohlcv_1h = generate_soft_bucket_labels(train_ohlcv_1h)
        y_test_1h, test_ohlcv_1h = generate_soft_bucket_labels(test_ohlcv_1h)

        # Append to the lists
        train_1h_list.append(train_ohlcv_1h)
        test_1h_list.append(test_ohlcv_1h)
        y_train_1h_list.append(y_train_1h)
        y_test_1h_list.append(y_test_1h)

        # Split 1d data
        train_ohlcv_1d, test_ohlcv_1d = split_by_time(ticker_1d)

        # Generate labels for 1d data
        y_train_1d, train_ohlcv_1d = generate_soft_bucket_labels(train_ohlcv_1d)
        y_test_1d, test_ohlcv_1d = generate_soft_bucket_labels(test_ohlcv_1d)

        # Append to the lists
        train_1d_list.append(train_ohlcv_1d)
        test_1d_list.append(test_ohlcv_1d)
        y_train_1d_list.append(y_train_1d)
        y_test_1d_list.append(y_test_1d)

    # Concatenate all tickers' train and test data
    combined_train_1h = pd.concat(train_1h_list, ignore_index=True)
    combined_test_1h = pd.concat(test_1h_list, ignore_index=True)
    combined_y_train_1h = np.concatenate(y_train_1h_list)
    combined_y_test_1h = np.concatenate(y_test_1h_list)

    combined_train_1d = pd.concat(train_1d_list, ignore_index=True)
    combined_test_1d = pd.concat(test_1d_list, ignore_index=True)
    combined_y_train_1d = np.concatenate(y_train_1d_list)
    combined_y_test_1d = np.concatenate(y_test_1d_list)

    # Save combined split data
    save_split_data(combined_train_1h, combined_test_1h, combined_y_train_1h, combined_y_test_1h, prefix='1h')
    save_split_data(combined_train_1d, combined_test_1d, combined_y_train_1d, combined_y_test_1d, prefix='1d')

# Run data splitting
main_data_splitting()
