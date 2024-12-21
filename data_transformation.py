# data_transformation.py

import polars as pl
import numpy as np
import pickle
from datetime import timedelta
from sklearn.preprocessing import LabelEncoder
import h5py

# ----------------------------
# Encoding Utilities
# ----------------------------
class CategoricalEncoder:
    def __init__(self, categorical_cols):
        self.categorical_cols = categorical_cols
        self.encoders = {col: LabelEncoder() for col in categorical_cols}
        self.mapping_dicts = {col: {} for col in categorical_cols}

    def fit(self, df):
        for col in self.categorical_cols:
            self.encoders[col].fit(df[col].to_list())
            self.mapping_dicts[col] = {label: idx for idx, label in enumerate(self.encoders[col].classes_)}

    def transform(self, df):
        for col in self.categorical_cols:
            # Handle unseen labels by assigning a default value, e.g., -1
            transformed = []
            for item in df[col].to_list():
                if item in self.encoders[col].classes_:
                    transformed.append(self.encoders[col].transform([item])[0])
                else:
                    transformed.append(-1)  # Unknown category
            df = df.with_columns([
                pl.Series(col, transformed).cast(pl.Int64)
            ])
        return df

    def save(self, encoders_path='categorical_encoders.pkl', mappings_path='categorical_mappings.pkl'):
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.encoders, f)
        with open(mappings_path, 'wb') as f:
            pickle.dump(self.mapping_dicts, f)

    def load(encoders_path='categorical_encoders.pkl', mappings_path='categorical_mappings.pkl'):
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
        with open(mappings_path, 'rb') as f:
            mapping_dicts = pickle.load(f)
        return encoders, mapping_dicts

# ----------------------------
# Data Processing Utilities
# ----------------------------
def merge_1h_1d(ticker_1h, ticker_1d):
    """
    Merge 1h and 1d data for a single ticker by shifting 1d Datetime to align with the last 1h interval.
    """
    # Shift 1d Datetime from 00:00 to 19:00 of the same day and ensure datetime[ns] precision
    ticker_1d = ticker_1d.with_columns([
        (pl.col('Datetime') + timedelta(hours=19)).cast(pl.Datetime('ns')).alias('Datetime')
    ])

    # Rename 1d features to avoid collision, excluding 'Ticker', 'Datetime', 'Interval', 'Label_d'
    features_to_rename = [col for col in ticker_1d.columns if col not in ['Ticker', 'Datetime', 'Interval', 'Label_d']]
    renamed_1d = {col: f"{col}_1d" for col in features_to_rename}
    ticker_1d = ticker_1d.rename(renamed_1d)

    # Drop 'Interval' column from 1h data if present
    if 'Interval' in ticker_1h.columns:
        ticker_1h = ticker_1h.drop('Interval')

    # Perform asof join to merge 1d features into 1h data
    merged = ticker_1h.join_asof(
        ticker_1d,
        on='Datetime',
        by='Ticker',
        strategy='forward'
    )

    return merged.drop_nulls()

def create_separated_sequences(data, sequence_length, feature_cols_1h, feature_cols_1d, label_col_1h, label_col_1d):
    """
    Create separated sequences for 1h and 1d features along with their labels.
    Returns:
        X_1h: np.ndarray of shape (num_sequences, sequence_length, num_features_1h)
        X_1d: np.ndarray of shape (num_sequences, sequence_length, num_features_1d)
        y_1h: np.ndarray of shape (num_sequences, 1)
        y_1d: np.ndarray of shape (num_sequences, 1)
    """
    X_1h, X_1d, y_1h, y_1d = [], [], [], []
    for i in range(len(data) - sequence_length):
        X_seq_1h = data[i:i+sequence_length][feature_cols_1h].to_numpy()
        X_seq_1d = data[i:i+sequence_length][feature_cols_1d].to_numpy()
        y_seq_1h = data[i+sequence_length][label_col_1h]
        y_seq_1d = data[i+sequence_length][label_col_1d]
        X_1h.append(X_seq_1h)
        X_1d.append(X_seq_1d)
        y_1h.append(y_seq_1h)
        y_1d.append(y_seq_1d)
    X_1h = np.array(X_1h, dtype=np.float32)
    X_1d = np.array(X_1d, dtype=np.float32)
    y_1h = np.array(y_1h, dtype=np.int64).reshape(-1, 1)
    y_1d = np.array(y_1d, dtype=np.int64).reshape(-1, 1)
    return X_1h, X_1d, y_1h, y_1d

# ----------------------------
# Main Transformation Workflow
# ----------------------------
def process_dataset(split='train', sequence_length=30):
    """
    Process either 'train' or 'test' dataset.
    Args:
        split (str): 'train' or 'test'
        sequence_length (int): Length of the sequences
    """
    # ----------------------------
    # 1. Define File Paths
    # ----------------------------
    data_paths = {
        'train': {
            '1h_data': '/content/1h_train_data.parquet',
            'y_train_1h': '/content/1h_y_train.npy',
            '1d_data': '/content/1d_train_data.parquet',
            'y_train_1d': '/content/1d_y_train.npy'
        },
        'test': {
            '1h_data': '/content/1h_test_data.parquet',
            'y_test_1h': '/content/1h_y_test.npy',
            '1d_data': '/content/1d_test_data.parquet',
            'y_test_1d': '/content/1d_y_test.npy'
        }
    }
    ticker_info_path = '/content/raw_ticker_info_data_20240929_221653.parquet'

    # ----------------------------
    # 2. Load Data
    # ----------------------------
    if split == 'train':
        data = {
            '1h': pl.read_parquet(data_paths['train']['1h_data']),
            'y_1h': np.load(data_paths['train']['y_train_1h']),
            '1d': pl.read_parquet(data_paths['train']['1d_data']),
            'y_1d': np.load(data_paths['train']['y_train_1d'])
        }
    elif split == 'test':
        data = {
            '1h': pl.read_parquet(data_paths['test']['1h_data']),
            'y_1h': np.load(data_paths['test']['y_test_1h']),
            '1d': pl.read_parquet(data_paths['test']['1d_data']),
            'y_1d': np.load(data_paths['test']['y_test_1d'])
        }
    else:
        raise ValueError("split must be 'train' or 'test'")

    # ----------------------------
    # 3. Standardize Datetime Precision
    # ----------------------------
    data['1h'] = data['1h'].with_columns([
        pl.col('Datetime').cast(pl.Datetime('ns'))
    ])
    data['1d'] = data['1d'].with_columns([
        pl.col('Datetime').cast(pl.Datetime('ns'))
    ])

    # ----------------------------
    # 4. Merge Labels
    # ----------------------------
    # Ensure that the length of labels matches the DataFrame
    if split == 'train':
        if len(data['y_1h']) != len(data['1h']):
            raise ValueError("Length of y_train_1h does not match train 1h DataFrame.")
        if len(data['y_1d']) != len(data['1d']):
            raise ValueError("Length of y_train_1d does not match train 1d DataFrame.")

        # Add 'Label' and 'Label_d' columns
        data['1h'] = data['1h'].with_columns([
            pl.Series('Label', data['y_1h']).cast(pl.Int64)
        ])
        data['1d'] = data['1d'].with_columns([
            pl.Series('Label_d', data['y_1d']).cast(pl.Int64)
        ])
    elif split == 'test':
        if len(data['y_1h']) != len(data['1h']):
            raise ValueError("Length of y_test_1h does not match test 1h DataFrame.")
        if len(data['y_1d']) != len(data['1d']):
            raise ValueError("Length of y_test_1d does not match test 1d DataFrame.")

        # Add 'Label' and 'Label_d' columns
        data['1h'] = data['1h'].with_columns([
            pl.Series('Label', data['y_1h']).cast(pl.Int64)
        ])
        data['1d'] = data['1d'].with_columns([
            pl.Series('Label_d', data['y_1d']).cast(pl.Int64)
        ])

    # ----------------------------
    # 5. Merge Ticker Information
    # ----------------------------
    # Define categorical columns
    categorical_cols = ['Industry', 'Sector', 'QuoteType']
    ohlcv_categorical = ['DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'ExtendedHours']
    all_categorical = categorical_cols + ohlcv_categorical

    # Load ticker info
    ticker_info = pl.read_parquet(ticker_info_path)

    # Merge ticker info with data
    data['1h'] = data['1h'].join(ticker_info, on='Ticker', how='left')
    data['1d'] = data['1d'].join(ticker_info, on='Ticker', how='left')

    # ----------------------------
    # 6. Encode Categorical Columns
    # ----------------------------
    if split == 'train':
        # Exclude label columns before concatenation
        combined_for_encoding = pl.concat([
            data['1h'].drop(['Datetime', 'Label']),
            data['1d'].drop(['Datetime', 'Label_d'])
        ])

        # Initialize and fit the encoder globally on training data
        encoder = CategoricalEncoder(categorical_cols=all_categorical)
        encoder.fit(combined_for_encoding)

        # Save Encoders for later use (training and inference)
        encoder.save()
    else:
        # Load previously saved encoders
        _, _ = CategoricalEncoder.load(encoders_path='categorical_encoders.pkl', mappings_path='categorical_mappings.pkl')
        encoder = CategoricalEncoder(categorical_cols=all_categorical)
        encoder.encoders, encoder.mapping_dicts = CategoricalEncoder.load()

    # Transform data
    data['1h'] = encoder.transform(data['1h'])
    data['1d'] = encoder.transform(data['1d'])

    # ----------------------------
    # 7. Dynamic Feature Selection
    # ----------------------------
    def get_feature_columns(df, exclude_cols):
        """
        Automatically select feature columns by excluding specified columns.
        """
        return [col for col in df.columns if col not in exclude_cols]

    # Exclude columns used for joining and labels
    exclude_columns = ['Ticker', 'Datetime', 'Label', 'Label_d']

    # Get feature columns from 1h data
    feature_cols_1h = get_feature_columns(data['1h'], exclude_columns)

    # Get feature columns from 1d data (technical indicators)
    feature_cols_1d = get_feature_columns(data['1d'], exclude_columns)

    # ----------------------------
    # 8. Initialize HDF5 File
    # ----------------------------
    if split == 'train':
        h5_file_name = 'train_sequences.h5'
    else:
        h5_file_name = 'test_sequences.h5'

    h5f = h5py.File(h5_file_name, 'w')
    num_ticker_info_features = len(categorical_cols)  # 3

    # Determine the number of features
    num_features_1h = len(feature_cols_1h)
    num_features_1d = len(feature_cols_1d)

    # Create datasets with initial size 0 and enable resizing
    X_1h_dataset = h5f.create_dataset(
        'X_1h',
        shape=(0, sequence_length, num_features_1h),
        maxshape=(None, sequence_length, num_features_1h),
        dtype='float32',
        chunks=True,
        compression="gzip"
    )
    X_1d_dataset = h5f.create_dataset(
        'X_1d',
        shape=(0, sequence_length, num_features_1d),
        maxshape=(None, sequence_length, num_features_1d),
        dtype='float32',
        chunks=True,
        compression="gzip"
    )
    y_1h_dataset = h5f.create_dataset(
        'y_1h',
        shape=(0, 1),
        maxshape=(None, 1),
        dtype='int64',
        chunks=True,
        compression="gzip"
    )
    y_1d_dataset = h5f.create_dataset(
        'y_1d',
        shape=(0, 1),
        maxshape=(None, 1),
        dtype='int64',
        chunks=True,
        compression="gzip"
    )
    ticker_info_dataset = h5f.create_dataset(
        'ticker_info_encoded',
        shape=(0, num_ticker_info_features),  # 3 features
        maxshape=(None, num_ticker_info_features),
        dtype='int32',
        chunks=True,
        compression="gzip"
    )
    ticker_indices_dataset = h5f.create_dataset(
        'ticker_indices',
        shape=(0,),
        maxshape=(None,),
        dtype='int32',
        chunks=True,
        compression="gzip"
    )

    # Initialize LabelEncoder for Ticker (only once during training)
    if split == 'train':
        ticker_le = LabelEncoder()
        ticker_le.fit(data['1h']['Ticker'].unique().to_list())
        ticker_vocab_size = len(ticker_le.classes_)

        # Save the Ticker LabelEncoder
        with open('ticker_label_encoder.pkl', 'wb') as f:
            pickle.dump(ticker_le, f)
    else:
        # Load the Ticker LabelEncoder
        with open('ticker_label_encoder.pkl', 'rb') as f:
            ticker_le = pickle.load(f)

    # Get unique tickers
    tickers = data['1h']['Ticker'].unique().to_list()

    for ticker in tickers:
        # Filter data for the current ticker and sort by Datetime
        ticker_data_1h = data['1h'].filter(pl.col('Ticker') == ticker).sort('Datetime')
        ticker_data_1d = data['1d'].filter(pl.col('Ticker') == ticker).sort('Datetime')

        # Merge 1h and 1d data
        merged_df = merge_1h_1d(ticker_data_1h, ticker_data_1d)

        # Encode the ticker
        ticker_encoded = ticker_le.transform([ticker])[0]

        # Define ticker info columns (excluding 'Ticker')
        # Already defined above as categorical_cols
        ticker_info_cols = categorical_cols  # ['Industry', 'Sector', 'QuoteType']

        # Create separated sequences using 'Label' and 'Label_d' as labels
        X_1h, X_1d, y_1h, y_1d = create_separated_sequences(
            merged_df,
            sequence_length=sequence_length,
            feature_cols_1h=feature_cols_1h,
            feature_cols_1d=feature_cols_1d,
            label_col_1h='Label',
            label_col_1d='Label_d'
        )

        num_sequences = X_1h.shape[0]

        if num_sequences > 0:
            # Extract ticker info once (static per ticker)
            ticker_info_array = merged_df[0][ticker_info_cols].to_numpy().flatten()  # 1D array with 3 elements

            # Create a (num_sequences, num_ticker_info_features) array by tiling
            ticker_info_encoded_seq = np.tile(ticker_info_array, (num_sequences, 1))

            # Resize datasets to accommodate new data
            X_1h_dataset.resize((X_1h_dataset.shape[0] + num_sequences, sequence_length, num_features_1h))
            X_1d_dataset.resize((X_1d_dataset.shape[0] + num_sequences, sequence_length, num_features_1d))
            y_1h_dataset.resize((y_1h_dataset.shape[0] + num_sequences, 1))
            y_1d_dataset.resize((y_1d_dataset.shape[0] + num_sequences, 1))
            ticker_info_dataset.resize((ticker_info_dataset.shape[0] + num_sequences, num_ticker_info_features))
            ticker_indices_dataset.resize((ticker_indices_dataset.shape[0] + num_sequences,))

            # Append new data
            X_1h_dataset[-num_sequences:, :, :] = X_1h
            X_1d_dataset[-num_sequences:, :, :] = X_1d
            y_1h_dataset[-num_sequences:, :] = y_1h
            y_1d_dataset[-num_sequences:, :] = y_1d
            ticker_info_dataset[-num_sequences:, :] = ticker_info_encoded_seq
            ticker_indices_dataset[-num_sequences:] = np.full(num_sequences, ticker_encoded, dtype=np.int32)

    # ----------------------------
    # 9. Close and Save Sequences
    # ----------------------------
    h5f.close()
    print(f"Sequences saved to '{h5_file_name}'.")

def main_data_transformation():
    """
    Main function to process both train and test datasets.
    """
    # ----------------------------
    # 1. Process Training Data
    # ----------------------------
    print("Processing training data...")
    process_dataset(split='train', sequence_length=30)

    # ----------------------------
    # 2. Process Test Data
    # ----------------------------
    print("Processing test data...")
    process_dataset(split='test', sequence_length=30)

    print("Data transformation completed for both train and test datasets.")

# Execute Transformation
main_data_transformation()
