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
            self.encoders[col].fit(df[col].unique().to_list())
            self.mapping_dicts[col] = {label: idx for idx, label in enumerate(self.encoders[col].classes_)}

    def transform(self, df):
        for col in self.categorical_cols:
            df = df.with_columns([
                pl.Series(col, self.encoders[col].transform(df[col].to_list())).cast(pl.Int64)
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

def create_sequences(data, sequence_length, feature_cols, label_col):
    """
    Create sequences and corresponding labels from the DataFrame.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X_seq = data[i:i+sequence_length][feature_cols].to_numpy()
        y_seq = data[i+sequence_length][label_col]
        X.append(X_seq)
        y.append(y_seq)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)  # Ensure labels are integers
    return X, y

def save_sequences_incrementally(h5f, X, y):
    """
    Append sequences and labels to existing HDF5 datasets.
    """
    current_X_size = h5f['X'].shape[0]
    current_y_size = h5f['y'].shape[0]

    new_X_size = X.shape[0]
    new_y_size = y.shape[0]

    # Resize datasets to accommodate new data
    h5f['X'].resize((current_X_size + new_X_size, X.shape[1], X.shape[2]))
    h5f['y'].resize((current_y_size + new_y_size, y.shape[1]))

    # Append new data
    h5f['X'][current_X_size:current_X_size + new_X_size, :, :] = X
    h5f['y'][current_y_size:current_y_size + new_y_size, :] = y

def save_encoders(encoder, encoder_path='categorical_encoders.pkl', mappings_path='categorical_mappings.pkl'):
    """
    Save the categorical encoders and their mappings.
    """
    encoder.save(encoders_path=encoder_path, mappings_path=mappings_path)

# ----------------------------
# Main Transformation Workflow
# ----------------------------
def main_data_transformation():
    # ----------------------------
    # 1. Define File Paths
    # ----------------------------
    train_1h_path = '/content/1h_train_data_20240929_234402.parquet'
    y_train_1h_path = '/content/1h_y_train_20240929_234402.npy'
    train_1d_path = '/content/1d_train_data_20240929_234406.parquet'
    y_train_1d_path = '/content/1d_y_train_20240929_234406.npy'
    ticker_info_path = '/content/raw_ticker_info_data_20240929_221653.parquet'

    # ----------------------------
    # 2. Load Data
    # ----------------------------
    train_1h = pl.read_parquet(train_1h_path)
    y_train_1h = np.load(y_train_1h_path)
    train_1d = pl.read_parquet(train_1d_path)
    y_train_1d = np.load(y_train_1d_path)
    ticker_info = pl.read_parquet(ticker_info_path)

    # ----------------------------
    # 3. Standardize Datetime Precision
    # ----------------------------
    # Convert 'Datetime' to datetime[ns] in both DataFrames
    train_1h = train_1h.with_columns([
        pl.col('Datetime').cast(pl.Datetime('ns'))
    ])
    train_1d = train_1d.with_columns([
        pl.col('Datetime').cast(pl.Datetime('ns'))
    ])

    # ----------------------------
    # 4. Merge Labels
    # ----------------------------
    # Ensure that the length of y_train matches the DataFrame
    if len(y_train_1h) != len(train_1h):
        raise ValueError("Length of y_train_1h does not match train_1h DataFrame.")
    if len(y_train_1d) != len(train_1d):
        raise ValueError("Length of y_train_1d does not match train_1d DataFrame.")

    # Add 'Label' column to 1h DataFrame and 'Label_d' to 1d DataFrame
    train_1h = train_1h.with_columns([
        pl.Series('Label', y_train_1h).cast(pl.Int64)
    ])
    train_1d = train_1d.with_columns([
        pl.Series('Label_d', y_train_1d).cast(pl.Int64)
    ])

    # ----------------------------
    # 5. Merge Ticker Information
    # ----------------------------
    # Define categorical columns
    categorical_cols = ['Industry', 'Sector', 'QuoteType']
    ohlcv_categorical = ['Ticker', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'ExtendedHours']
    all_categorical = categorical_cols + ohlcv_categorical

    # Merge ticker info with training data
    train_1h = train_1h.join(ticker_info, on='Ticker', how='left')
    train_1d = train_1d.join(ticker_info, on='Ticker', how='left')

    # ----------------------------
    # 6. Encode Categorical Columns
    # ----------------------------
    # Exclude label columns before concatenation
    combined_for_encoding = pl.concat([train_1h.drop(['Label']), train_1d.drop(['Label_d'])])

    # Initialize and fit the encoder
    encoder = CategoricalEncoder(categorical_cols=all_categorical)
    encoder.fit(combined_for_encoding)

    # Transform DataFrames
    train_1h = encoder.transform(train_1h)
    train_1d = encoder.transform(train_1d)

    # Save Encoders
    save_encoders(encoder, encoder_path='categorical_encoders.pkl', mappings_path='categorical_mappings.pkl')

    # ----------------------------
    # 7. Dynamic Feature Selection
    # ----------------------------
    def get_feature_columns(df, exclude_cols):
        """
        Automatically select feature columns by excluding specified columns.
        """
        return [col for col in df.columns if col not in exclude_cols]

    # Exclude columns used for joining and labels
    exclude_columns = ['Ticker', 'Datetime', 'Interval', 'Label', 'Label_d']

    # Get feature columns from 1h data
    feature_cols_1h = get_feature_columns(train_1h, exclude_columns)

    # Get feature columns from 1d data (technical indicators)
    feature_cols_1d = get_feature_columns(train_1d, exclude_columns)

    # Combine feature columns, renaming 1d features with '_1d' suffix
    combined_features = feature_cols_1h + [f"{col}_1d" for col in feature_cols_1d]

    # ----------------------------
    # 8. Merge 1h and 1d Data, Create Sequences
    # ----------------------------
    # Initialize HDF5 file with extendable datasets
    h5f = h5py.File('train_sequences.h5', 'w')
    max_shape_X = (None, 30, len(combined_features))  # (num_sequences, sequence_length, num_features)
    max_shape_y = (None, 1)  # (num_sequences, 1)

    # Create datasets with initial size 0 and enable resizing
    X_dataset = h5f.create_dataset(
        'X',
        shape=(0, 30, len(combined_features)),
        maxshape=max_shape_X,
        dtype='float32',
        chunks=True
    )
    y_dataset = h5f.create_dataset(
        'y',
        shape=(0, 1),
        maxshape=max_shape_y,
        dtype='int64',
        chunks=True
    )

    # Get unique tickers
    tickers = train_1h['Ticker'].unique().to_list()

    for ticker in tickers:
        # Filter data for the current ticker and sort by Datetime
        ticker_train_1h = train_1h.filter(pl.col('Ticker') == ticker).sort('Datetime')
        ticker_train_1d = train_1d.filter(pl.col('Ticker') == ticker).sort('Datetime')

        # Merge 1h and 1d data
        merged_df = merge_1h_1d(ticker_train_1h, ticker_train_1d)

        # Create sequences using 'Label_d' as the label
        X, y = create_sequences(
            merged_df,
            sequence_length=30,  # Example sequence length
            feature_cols=combined_features,
            label_col='Label_d'    # Use 'Label_d' as the target label
        )

        if X.size > 0:
            # Reshape y to (n_sequences, 1)
            y = y.reshape(-1, 1)

            # Resize datasets to accommodate new data
            X_dataset.resize((X_dataset.shape[0] + X.shape[0], 30, len(combined_features)))
            y_dataset.resize((y_dataset.shape[0] + y.shape[0], 1))

            # Append new data
            X_dataset[-X.shape[0]:, :, :] = X
            y_dataset[-y.shape[0]:, :] = y

    # ----------------------------
    # 9. Close and Save Sequences
    # ----------------------------
    h5f.close()
    print("Sequences saved to 'train_sequences.h5'.")

# ----------------------------
# Execute Transformation
# ----------------------------
if __name__ == "__main__":
    main_data_transformation()
