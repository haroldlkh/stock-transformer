# data_transformation.py

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import h5py

def encode_categorical_columns(df, categorical_cols, label_encoders=None, category_mappings=None):
    """
    Encode categorical columns in the DataFrame using LabelEncoder.
    If label_encoders are provided, use them; otherwise, fit new encoders.
    Returns the encoded DataFrame, updated label_encoders, and category_mappings.
    """
    if label_encoders is None:
        label_encoders = {}
    if category_mappings is None:
        category_mappings = {}

    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels by assigning a new code
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
            le.classes_ = np.append(le.classes_, 'Unknown') if 'Unknown' not in le.classes_ else le.classes_
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        # Create mapping
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        category_mappings[col] = mapping
    return df, label_encoders, category_mappings

def create_sequences_generator(data, target, sequence_length, price_columns):
    """
    Generator that yields sequences per ticker for efficiency, including encoded tickers.
    """
    tickers = data['Ticker'].unique()
    for ticker in tickers:
        ticker_data = data[data['Ticker'] == ticker].reset_index(drop=True)
        ticker_target = target[data['Ticker'] == ticker]
        ticker_indices = ticker_data['Ticker'].values  # Encoded tickers

        num_sequences = len(ticker_data) - sequence_length
        if num_sequences <= 0:
            continue  # Skip tickers with insufficient data

        # Pre-extract data to minimize slicing in the loop
        ticker_prices = ticker_data[price_columns].values.astype(np.float32)
        for i in range(num_sequences):
            X_seq = ticker_prices[i:i+sequence_length]
            y_seq = ticker_target[i + sequence_length]
            ticker_idx = ticker_indices[i + sequence_length - 1]  # Encoded ticker
            yield X_seq, y_seq, ticker_idx

def save_sequences_to_hdf5(generator, num_sequences, sequence_shape, y_shape, prefix):
    """
    Save sequences to HDF5 file incrementally.
    """
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"{prefix}_data_{current_datetime}.h5"
    with h5py.File(file_name, 'w') as h5f:
        X_dataset = h5f.create_dataset('X_sequences', shape=(num_sequences, *sequence_shape), dtype='float32')
        y_dataset = h5f.create_dataset('y_sequences', shape=(num_sequences, y_shape), dtype='float32')
        ticker_indices_dataset = h5f.create_dataset('ticker_indices', shape=(num_sequences,), dtype='int32')

        idx = 0  # Global index across all tickers
        for X_seq, y_seq, ticker_idx in generator:
            X_dataset[idx] = X_seq
            y_dataset[idx] = y_seq
            ticker_indices_dataset[idx] = ticker_idx
            idx += 1

    print(f"Sequences saved to {file_name}")

def process_ohlcv_data(ohlcv_data, y_data, ticker_info_data, ticker_categorical_cols, ohlcv_categorical_cols, prefix, label_encoders=None, category_mappings=None):
    """
    Process OHLCV data: merge with ticker info, encode categorical columns, create sequences, and save.
    """
    # Merge ticker_info_data with OHLCV data on 'Ticker'
    ohlcv_data = ohlcv_data.merge(ticker_info_data, on='Ticker', how='left')

    # Encode categorical columns in ohlcv_data
    ohlcv_data, label_encoders, category_mappings = encode_categorical_columns(
        ohlcv_data,
        ohlcv_categorical_cols,
        label_encoders=label_encoders,
        category_mappings=category_mappings
    )

    # Price columns (numerical features)
    exclude_cols = ['Ticker', 'Datetime'] + ticker_categorical_cols + ohlcv_categorical_cols
    price_columns = [col for col in ohlcv_data.columns if col not in exclude_cols]

    # Convert numerical data to float32
    ohlcv_data[price_columns] = ohlcv_data[price_columns].astype(np.float32)

    # One-hot encode labels with predefined categories [0,1,2,3,4]
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', categories=[np.arange(5)])
    y_data = one_hot_encoder.fit_transform(y_data.reshape(-1, 1))

    # Save OneHotEncoder and possible labels
    with open(f'one_hot_encoder_{prefix}.pkl', 'wb') as f:
        pickle.dump(one_hot_encoder, f)

    possible_labels = np.arange(5)
    with open(f'possible_labels_{prefix}.pkl', 'wb') as f:
        pickle.dump(possible_labels, f)

    # Define sequence length
    sequence_length = 10

    # Calculate total number of sequences
    total_sequences = 0
    tickers = ohlcv_data['Ticker'].unique()
    for ticker in tickers:
        ticker_data_length = len(ohlcv_data[ohlcv_data['Ticker'] == ticker])
        num_sequences = max(0, ticker_data_length - sequence_length)
        total_sequences += num_sequences

    # If no sequences can be formed, skip saving
    if total_sequences == 0:
        print(f"No sequences can be formed for prefix '{prefix}'. Skipping.")
        return label_encoders, category_mappings

    # Get shape of sequences
    sample_sequence = ohlcv_data[ohlcv_data['Ticker'] == tickers[0]][price_columns].iloc[:sequence_length].values
    sequence_shape = sample_sequence.shape
    y_shape = y_data.shape[1]

    # Create generator
    generator = create_sequences_generator(
        ohlcv_data, y_data, sequence_length, price_columns
    )

    # Save sequences using generator
    save_sequences_to_hdf5(generator, total_sequences, sequence_shape, y_shape, prefix=f'{prefix}_train')

    return label_encoders, category_mappings

def main_data_transformation():
    """
    Main function to encode categorical variables, create sequences, and save data.
    """
    # Load split data for both 1h and 1d intervals
    train_ohlcv_1h = pd.read_parquet('/content/1h_train_data_20240926_041931.parquet')
    y_train_1h = np.load('/content/1h_y_train_20240926_041931.npy')
    train_ohlcv_1d = pd.read_parquet('/content/1d_train_data_20240926_041936.parquet')
    y_train_1d = np.load('/content/1d_y_train_20240926_041936.npy')

    # Load ticker_info_data
    ticker_info_data = pd.read_parquet('/content/raw_ticker_info_data_20240926_041538.parquet')  # Replace with actual filename

    # Define categorical columns
    ticker_categorical_cols = ['Industry', 'Sector', 'QuoteType']
    ohlcv_categorical_cols = ['Ticker', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'Interval', 'ExtendedHours']

    # Initialize label encoders and category mappings
    label_encoders = {}
    category_mappings = {}

    # Encode categorical columns in ticker_info_data
    ticker_info_data, label_encoders, category_mappings = encode_categorical_columns(
        ticker_info_data,
        ticker_categorical_cols,
        label_encoders=label_encoders,
        category_mappings=category_mappings
    )

    # Save ticker label encoders and category mappings
    with open('label_encoders_ticker.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open('category_mappings_ticker.pkl', 'wb') as f:
        pickle.dump(category_mappings, f)

    # Reset label encoders and category mappings for OHLCV data
    label_encoders = {}
    category_mappings = {}

    # Process 1h data
    label_encoders, category_mappings = process_ohlcv_data(
        ohlcv_data=train_ohlcv_1h,
        y_data=y_train_1h,
        ticker_info_data=ticker_info_data,
        ticker_categorical_cols=ticker_categorical_cols,
        ohlcv_categorical_cols=ohlcv_categorical_cols,
        prefix='1h',
        label_encoders=label_encoders,
        category_mappings=category_mappings
    )

    # Process 1d data
    label_encoders, category_mappings = process_ohlcv_data(
        ohlcv_data=train_ohlcv_1d,
        y_data=y_train_1d,
        ticker_info_data=ticker_info_data,
        ticker_categorical_cols=ticker_categorical_cols,
        ohlcv_categorical_cols=ohlcv_categorical_cols,
        prefix='1d',
        label_encoders=label_encoders,
        category_mappings=category_mappings
    )

    # Combine category mappings for Interval to ensure consistent encoding
    # if 'Interval' in category_mappings:
    #     interval_mapping_1h = category_mappings['Interval']
    # else:
    #     interval_mapping_1h = {}
    # if 'Interval' in category_mappings:
    #     interval_mapping_1d = category_mappings['Interval']
    # else:
    #     interval_mapping_1d = {}

    # all_intervals = set(interval_mapping_1h.keys()).union(set(interval_mapping_1d.keys()))
    # combined_interval_mapping = {interval: idx for idx, interval in enumerate(sorted(all_intervals))}
    # category_mappings['Interval'] = combined_interval_mapping

    # # Save combined category mappings
    # with open('full_ohlcv_category_mappings.pkl', 'wb') as f:
    #     pickle.dump(category_mappings, f)

# Run data transformation
main_data_transformation()
