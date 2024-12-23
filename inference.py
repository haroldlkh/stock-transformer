# inference.py

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
import pickle
import h5py
from datetime import datetime, timedelta
import tensorflow as tf

# Assuming necessary functions from previous scripts are imported:
# - ohlcv_pipeline
# - normalize_datetime
# - get_valid_day_range
# - ticker_info_pipeline
# - add_features_pipeline
# - CategoricalEncoder
# - merge_1h_1d

def main_inference():
    """
    Main function for inference.
    """
    # ----------------------------
    # 1. Define Tickers and Date Range
    # ----------------------------
    tickers = ['TSLA', 'NVDA','MSFT','AMZN', 'COST']

    # Define date range: From (Sept 1 - 200 days) to Oct 4
    end_date = datetime(2024, 10, 3)
    start_date = datetime(2024, 9, 1) - timedelta(days=300)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # ----------------------------
    # 2. Data Collection
    # ----------------------------
    # Collect OHLCV data and Ticker info
    ohlcv_data_list = []
    ticker_info_list = []
    intervals = ['1d', '1h']
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        # Fetch OHLCV data
        ohlcv_data = ohlcv_pipeline(ticker, intervals, start_date_str, end_date_str)
        ohlcv_data_list.append(ohlcv_data)
        # Fetch ticker info
        ticker_info_data = ticker_info_pipeline(ticker)
        ticker_info_list.append(ticker_info_data)
    # Concatenate data
    combined_ohlcv_data = pl.concat(ohlcv_data_list)
    combined_ticker_info = pl.concat(ticker_info_list)

    # ----------------------------
    # 3. Feature Engineering
    # ----------------------------
    # Apply feature engineering to each ticker's data
    # (Assuming add_features_pipeline function is defined elsewhere)
    preprocessed_1d_list = []
    preprocessed_1h_list = []
    for ticker in tickers:
        print(f"Processing features for {ticker}...")
        df_ticker = combined_ohlcv_data.filter(pl.col('Ticker') == ticker).sort('Datetime')
        df_1d = df_ticker.filter(pl.col('Interval') == '1d').sort('Datetime')
        df_1h = df_ticker.filter(pl.col('Interval') == '1h').sort('Datetime')
        preprocessed_1d = add_features_pipeline(df_1d)
        preprocessed_1h = add_features_pipeline(df_1h)
        preprocessed_1d_list.append(preprocessed_1d)
        preprocessed_1h_list.append(preprocessed_1h)
    combined_preprocessed_1d = pl.concat(preprocessed_1d_list)
    combined_preprocessed_1h = pl.concat(preprocessed_1h_list)

    # ----------------------------
    # 4. Data Transformation
    # ----------------------------
    # Load the encoders and mappings generated during training
    with open('categorical_encoders.pkl', 'rb') as f:
        categorical_encoders = pickle.load(f)
    with open('categorical_mappings.pkl', 'rb') as f:
        categorical_mappings = pickle.load(f)
    # Create an instance of CategoricalEncoder and assign loaded encoders and mappings
    all_categorical = ['Industry', 'Sector', 'QuoteType', 'DayOfWeek', 'DayOfMonth', 'WeekOfYear', 'ExtendedHours']
    encoder = CategoricalEncoder(categorical_cols=all_categorical)
    encoder.encoders = categorical_encoders
    encoder.mapping_dicts = categorical_mappings
    # Load Ticker LabelEncoder
    with open('ticker_label_encoder.pkl', 'rb') as f:
        ticker_le = pickle.load(f)
    # Standardize Datetime Precision
    combined_preprocessed_1h = combined_preprocessed_1h.with_columns([
        pl.col('Datetime').cast(pl.Datetime('ns'))
    ])
    combined_preprocessed_1d = combined_preprocessed_1d.with_columns([
        pl.col('Datetime').cast(pl.Datetime('ns'))
    ])
    # Merge Ticker Information
    combined_preprocessed_1h = combined_preprocessed_1h.join(combined_ticker_info, on='Ticker', how='left')
    combined_preprocessed_1d = combined_preprocessed_1d.join(combined_ticker_info, on='Ticker', how='left')
    # Encode Categorical Columns
    combined_preprocessed_1h = encoder.transform(combined_preprocessed_1h)
    combined_preprocessed_1d = encoder.transform(combined_preprocessed_1d)

    # ----------------------------
    # 5. Feature Selection
    # ----------------------------
    exclude_columns = ['Ticker', 'Datetime', 'Interval']
    feature_cols_1h = [col for col in combined_preprocessed_1h.columns if col not in exclude_columns]
    feature_cols_1d = [col for col in combined_preprocessed_1d.columns if col not in exclude_columns]

    # ----------------------------
    # 6. Prepare Sequences for Each Ticker
    # ----------------------------
    # Initialize lists to store data
    X_1h_list = []
    X_1d_list = []
    ticker_info_list = []
    ticker_indices_list = []
    sequence_length = 30  # Same as training
    for ticker in tickers:
        print(f"Preparing sequences for {ticker}...")
        ticker_data_1h = combined_preprocessed_1h.filter(pl.col('Ticker') == ticker).sort('Datetime')
        ticker_data_1d = combined_preprocessed_1d.filter(pl.col('Ticker') == ticker).sort('Datetime')
        merged_df = merge_1h_1d(ticker_data_1h, ticker_data_1d)
        ticker_encoded = ticker_le.transform([ticker])[0]
        ticker_info_cols = ['Industry', 'Sector', 'QuoteType']
        if len(merged_df) >= sequence_length:
            data_seq = merged_df.tail(sequence_length)
            X_seq_1h = data_seq[feature_cols_1h].to_numpy().astype(np.float32)
            X_seq_1d = data_seq[feature_cols_1d].to_numpy().astype(np.float32)
            X_seq_1h = X_seq_1h.reshape(1, sequence_length, -1)
            X_seq_1d = X_seq_1d.reshape(1, sequence_length, -1)
            ticker_info_array = data_seq[0][ticker_info_cols].to_numpy().flatten().astype(np.int32)
            ticker_info_array = ticker_info_array.reshape(1, -1)
            X_1h_list.append(X_seq_1h)
            X_1d_list.append(X_seq_1d)
            ticker_info_list.append(ticker_info_array)
            ticker_indices_list.append(np.array([[ticker_encoded]], dtype=np.int32))
        else:
            print(f"Not enough data for {ticker} to form a sequence of length {sequence_length}")
    if X_1h_list:
        X_1h = np.vstack(X_1h_list)
        X_1d = np.vstack(X_1d_list)
        ticker_info_encoded = np.vstack(ticker_info_list)
        ticker_indices = np.vstack(ticker_indices_list)
    else:
        print("No data available for inference.")
        return

    # ----------------------------
    # 7. Load the Trained Model
    # ----------------------------
    print("Loading the trained model...")
    # Loading the model with safe_mode=False
    model = tf.keras.models.load_model('transformer_cross_attention_model3 rebucket.keras', safe_mode=False) # need to fix use of lambda functions.

    # ----------------------------
    # 8. Make Predictions
    # ----------------------------
    print("Making predictions...")
    input_data = {
        'ohlcv_input_1h': X_1h,
        'ohlcv_input_1d': X_1d,
        'ticker_info_input': ticker_info_encoded,
        'ticker_input': ticker_indices
    }
    predictions = model.predict(input_data)

    # ----------------------------
    # 9. Interpret Predictions
    # ----------------------------
    buckets = [-1.5, -0.5, 0.5, 1.5]
    bucket_labels = [0, 1, 2, 3, 4]
    bucket_ranges = [(-np.inf, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, np.inf)]
    predicted_classes = np.argmax(predictions, axis=1)
    results = []
    for i, ticker in enumerate(tickers):
        predicted_class = predicted_classes[i]
        predicted_range = bucket_ranges[predicted_class]
        probability = predictions[i, predicted_class]
        results.append({
            'Ticker': ticker,
            'Predicted_Class': predicted_class,
            'Predicted_Range': predicted_range,
            'Probability': probability
        })

    # ----------------------------
    # 10. Display Results
    # ----------------------------
    for res in results:
        print(f"Ticker: {res['Ticker']}")
        print(f"Predicted Price Change Range for Next Day: {res['Predicted_Range']}")
        print(f"Probability: {res['Probability']:.4f}")
        print('---')

main_inference()
