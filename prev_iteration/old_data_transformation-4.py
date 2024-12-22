# # data_transformation.py

# import polars as pl
# import numpy as np
# import pickle
# from datetime import timedelta
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import h5py

# # ----------------------------
# # Encoding Utilities
# # ----------------------------
# class CategoricalEncoder:
#     def __init__(self, categorical_cols):
#         self.categorical_cols = categorical_cols
#         self.encoders = {col: LabelEncoder() for col in categorical_cols}
#         self.mapping_dicts = {col: {} for col in categorical_cols}

#     def fit(self, df):
#         for col in self.categorical_cols:
#             self.encoders[col].fit(df[col].unique().to_list())
#             self.mapping_dicts[col] = {label: idx for idx, label in enumerate(self.encoders[col].classes_)}

#     def transform(self, df):
#         for col in self.categorical_cols:
#             df = df.with_columns([
#                 pl.Series(col, self.encoders[col].transform(df[col].to_list())).cast(pl.Int64)
#             ])
#         return df

#     def save(self, encoders_path='categorical_encoders.pkl', mappings_path='categorical_mappings.pkl'):
#         with open(encoders_path, 'wb') as f:
#             pickle.dump(self.encoders, f)
#         with open(mappings_path, 'wb') as f:
#             pickle.dump(self.mapping_dicts, f)

#     @staticmethod
#     def load(encoders_path='categorical_encoders.pkl', mappings_path='categorical_mappings.pkl'):
#         with open(encoders_path, 'rb') as f:
#             encoders = pickle.load(f)
#         with open(mappings_path, 'rb') as f:
#             mapping_dicts = pickle.load(f)
#         return encoders, mapping_dicts

# # ----------------------------
# # Data Processing Utilities
# # ----------------------------
# def merge_1h_1d(ticker_1h, ticker_1d):
#     """
#     Merge 1h and 1d data for a single ticker by shifting 1d Datetime to align with the last 1h interval.
#     """
#     # Shift 1d Datetime from 00:00 to 19:00 of the same day
#     ticker_1d = ticker_1d.with_columns([
#         (pl.col('Datetime') + timedelta(hours=19)).alias('Datetime')
#     ])

#     # Rename 1d features to avoid collision, excluding 'Ticker', 'Datetime', 'Interval'
#     features_to_rename = [col for col in ticker_1d.columns if col not in ['Ticker', 'Datetime', 'Interval']]
#     renamed_1d = {col: f"{col}_1d" for col in features_to_rename}
#     ticker_1d = ticker_1d.rename(renamed_1d)

#     # Drop 'Interval' column from 1h data if present
#     if 'Interval' in ticker_1h.columns:
#         ticker_1h = ticker_1h.drop('Interval')

#     # Perform asof join to merge 1d features into 1h data
#     merged = ticker_1h.join_asof(
#         ticker_1d,
#         on='Datetime',
#         by='Ticker',
#         strategy='forward'
#     )

#     return merged.drop_nulls()

# def create_sequences(data, sequence_length, feature_cols, label_col):
#     """
#     Create sequences and corresponding labels from the DataFrame.
#     """
#     X, y = [], []
#     for i in range(len(data) - sequence_length):
#         X_seq = data[i:i+sequence_length][feature_cols].to_numpy()
#         y_seq = data[i+sequence_length][label_col]
#         X.append(X_seq)
#         y.append(y_seq)
#     X = np.array(X, dtype=np.float32)
#     y = np.array(y, dtype=np.int64)

#     # One-hot encode labels if there are multiple classes
#     if len(np.unique(y)) > 1:
#         one_hot = OneHotEncoder(sparse_output=False, categories='auto')
#         y = one_hot.fit_transform(y.reshape(-1, 1))
#     else:
#         y = y.reshape(-1, 1)  # Handle single class

#     return X, y

# def save_sequences(X, y, prefix='train'):
#     """
#     Save sequences and labels to HDF5 files.
#     """
#     timestamp = pl.datetime.now().strftime('%Y%m%d_%H%M%S')
#     file_name = f"{prefix}_sequences_{timestamp}.h5"
#     with h5py.File(file_name, 'w') as h5f:
#         h5f.create_dataset('X', data=X)
#         h5f.create_dataset('y', data=y)
#     print(f"Sequences saved to {file_name}")

# # ----------------------------
# # Main Transformation Workflow
# # ----------------------------
# def main_data_transformation():
#     # ----------------------------
#     # 1. Define File Paths
#     # ----------------------------
#     train_1h_path = '/content/1h_train_data_20240929_234402.parquet'
#     y_train_1h_path = '/content/1h_y_train_20240929_234402.npy'
#     train_1d_path = '/content/1d_train_data_20240929_234406.parquet'
#     y_train_1d_path = '/content/1d_y_train_20240929_234406.npy'
#     ticker_info_path = '/content/raw_ticker_info_data_20240929_221653.parquet'

#     # ----------------------------
#     # 2. Load Data
#     # ----------------------------
#     train_1h = pl.read_parquet(train_1h_path)
#     y_train_1h = np.load(y_train_1h_path)
#     train_1d = pl.read_parquet(train_1d_path)
#     y_train_1d = np.load(y_train_1d_path)
#     ticker_info = pl.read_parquet(ticker_info_path)

#     # ----------------------------
#     # 3. Merge Labels
#     # ----------------------------
#     # Ensure that the length of y_train matches the DataFrame
#     if len(y_train_1h) != len(train_1h):
#         raise ValueError("Length of y_train_1h does not match train_1h DataFrame.")
#     if len(y_train_1d) != len(train_1d):
#         raise ValueError("Length of y_train_1d does not match train_1d DataFrame.")

#     # Add 'Label' column to both DataFrames
#     train_1h = train_1h.with_columns([
#         pl.Series('Label', y_train_1h).cast(pl.Int64)
#     ])
#     train_1d = train_1d.with_columns([
#         pl.Series('Label_d', y_train_1d).cast(pl.Int64)  # If labels are different for 1d
#     ])

#     # ----------------------------
#     # 4. Merge Ticker Information
#     # ----------------------------
#     # Define categorical columns
#     categorical_cols = ['Industry', 'Sector', 'QuoteType']
#     ohlcv_categorical = ['Ticker', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'ExtendedHours']
#     all_categorical = categorical_cols + ohlcv_categorical

#     # Merge ticker info with training data
#     train_1h = train_1h.join(ticker_info, on='Ticker', how='left')
#     train_1d = train_1d.join(ticker_info, on='Ticker', how='left')

#     # ----------------------------
#     # 5. Encode Categorical Columns
#     # ----------------------------
#     encoder = CategoricalEncoder(categorical_cols=all_categorical)
#     encoder.fit(pl.concat([train_1h, train_1d]))

#     # Transform DataFrames
#     train_1h = encoder.transform(train_1h)
#     train_1d = encoder.transform(train_1d)

#     # Save Encoders
#     encoder.save(encoders_path='categorical_encoders.pkl', mappings_path='categorical_mappings.pkl')

#     # ----------------------------
#     # 6. Dynamic Feature Selection
#     # ----------------------------
#     def get_feature_columns(df, exclude_cols):
#         """
#         Automatically select feature columns by excluding specified columns.
#         """
#         return [col for col in df.columns if col not in exclude_cols]

#     # Exclude columns used for joining and labels
#     exclude_columns = ['Ticker', 'Datetime', 'Interval']
#     shared_columns = ['DayOfWeek', 'DayOfMonth', 'WeekOfYear']

#     # Get feature columns from 1h data
#     feature_cols_1h = get_feature_columns(train_1h, exclude_columns)

#     # Get feature columns from 1d data (technical indicators)
#     feature_cols_1d = get_feature_columns(train_1d, exclude_columns+shared_columns)

#     # Combine feature columns
#     combined_features = feature_cols_1h + [f"{col}_1d" for col in feature_cols_1d]

#     # ----------------------------
#     # 7. Merge 1h and 1d Data, Create Sequences
#     # ----------------------------
#     X_train_list = []
#     y_train_list = []

#     # Get unique tickers
#     tickers = train_1h['Ticker'].unique().to_list()

#     for ticker in tickers:
#         # Filter data for the current ticker and sort by Datetime
#         ticker_train_1h = train_1h.filter(pl.col('Ticker') == ticker).sort('Datetime')
#         ticker_train_1d = train_1d.filter(pl.col('Ticker') == ticker).sort('Datetime')

#         # Merge 1h and 1d data
#         merged_df = merge_1h_1d(ticker_train_1h, ticker_train_1d)

#         # Create sequences
#         X, y = create_sequences(
#             merged_df,
#             sequence_length=30,  # Example sequence length
#             feature_cols=combined_features,
#             label_col='Label'
#         )

#         if X.size > 0:
#             X_train_list.append(X)
#             y_train_list.append(y)

#     # ----------------------------
#     # 8. Concatenate and Save Sequences
#     # ----------------------------
#     if X_train_list and y_train_list:
#         X_train = np.concatenate(X_train_list, axis=0)
#         y_train = np.concatenate(y_train_list, axis=0)
#         save_sequences(X_train, y_train, prefix='train')
#     else:
#         print("No sequences created. Check data for sufficient length.")

# # ----------------------------
# # Execute Transformation
# # ----------------------------
# if __name__ == "__main__":
#     main_data_transformation()
