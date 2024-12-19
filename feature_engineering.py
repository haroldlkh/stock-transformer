# feature_engineering.py

import polars as pl
import numpy as np
from datetime import datetime
from tqdm import tqdm

def add_time_and_day_features(df):
    """
    Add time-based features to the dataframe.
    """
    df = df.with_columns([
        pl.col('Datetime').dt.weekday().alias('DayOfWeek'),  # Day of the week (0 = Monday)
        pl.col('Datetime').dt.day().alias('DayOfMonth'),     # Day of the month
        pl.col('Datetime').dt.week().alias('WeekOfYear')     # Week of the year
    ])
    return df

def calculate_historical_volatility(df, column='Close', period=20):
    """
    Calculate historical volatility.
    """
    df = df.with_columns([
        (pl.col(column).pct_change() * 100).alias('Daily_Returns')  # Percentage change (returns)
    ])
    df = df.with_columns([
        pl.col('Daily_Returns').rolling_std(window_size=period).alias(f'HV_{period}')
    ])
    df = df.with_columns([
        (pl.col(f'HV_{period}') * np.sqrt(252)).alias(f'Annualized_HV_{period}')
    ])
    return df

def calculate_rolling_sd(df, column='Close', period=20):
    """
    Calculate rolling standard deviation.
    """
    # Calculate daily returns
    df = df.with_columns([
        (pl.col(column).pct_change() * 100).alias('Daily_Returns')
    ])
    # Calculate rolling standard deviation of the returns
    df = df.with_columns([
        pl.col('Daily_Returns').rolling_std(window_size=period).alias(f'Rolling_SD_{period}')
    ])
    return df

def calculate_macd(df, column='Close', fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD and MACD histogram.
    """
    # Step 1: Calculate the fast and slow EMAs for MACD
    df = df.with_columns([
        pl.col(column).ewm_mean(span=fast_period).alias(f'EMA_{fast_period}'),
        pl.col(column).ewm_mean(span=slow_period).alias(f'EMA_{slow_period}')
    ])
    # Step 2: Calculate the MACD line
    df = df.with_columns([
        (pl.col(f'EMA_{fast_period}') - pl.col(f'EMA_{slow_period}')).alias(f'MACD_{fast_period}_{slow_period}')
    ])
    # Step 3: Calculate the MACD signal line (EMA of MACD line)
    df = df.with_columns([
        pl.col(f'MACD_{fast_period}_{slow_period}').ewm_mean(span=signal_period).alias(f'MACD_Signal_{signal_period}')
    ])
    # Step 4: Calculate the MACD histogram
    df = df.with_columns([
        (pl.col(f'MACD_{fast_period}_{slow_period}') - pl.col(f'MACD_Signal_{signal_period}')).alias(f'MACD_Histogram_{fast_period}_{slow_period}_{signal_period}')
    ])
    # Drop intermediary EMAs
    df = df.drop([f'EMA_{fast_period}', f'EMA_{slow_period}'])
    return df

def calculate_bollinger_bands(df, column='Close', period=20, num_std_dev=2):
    """
    Calculate Bollinger Bands.
    """
    df = df.with_columns([
        pl.col(column).rolling_mean(window_size=period).alias(f'BB_Middle_{period}'),
        pl.col(column).rolling_std(window_size=period).alias(f'BB_Std_Dev_{period}')
    ])
    df = df.with_columns([
        (pl.col(f'BB_Middle_{period}') + num_std_dev * pl.col(f'BB_Std_Dev_{period}')).alias(f'BB_Upper_{period}_{num_std_dev}'),
        (pl.col(f'BB_Middle_{period}') - num_std_dev * pl.col(f'BB_Std_Dev_{period}')).alias(f'BB_Lower_{period}_{num_std_dev}')
    ])
    # Calculate Bollinger Band Width (BBW)
    df = df.with_columns([
        ((pl.col(f'BB_Upper_{period}_{num_std_dev}') - pl.col(f'BB_Lower_{period}_{num_std_dev}')) / pl.col(f'BB_Middle_{period}')).alias(f'BBW_{period}_{num_std_dev}')
    ])
    return df

def calculate_stochastic(df, high_column='High', low_column='Low', close_column='Close', period=14, smooth_k=3, smooth_d=3):
    """
    Calculate Stochastic Oscillator.
    """
    # Step 1: Calculate the Highest High and Lowest Low over the period
    df = df.with_columns([
        pl.col(high_column).rolling_max(window_size=period).alias(f'Highest_High_{period}'),
        pl.col(low_column).rolling_min(window_size=period).alias(f'Lowest_Low_{period}')
    ])
    # Step 2: Calculate the raw Stochastic %K line
    df = df.with_columns([
        (100 * ((pl.col(close_column) - pl.col(f'Lowest_Low_{period}')) / (pl.col(f'Highest_High_{period}') - pl.col(f'Lowest_Low_{period}')))).alias(f'Stoch_%K_{period}')
    ])
    # Step 3: Apply smoothing to %K to get smoothed %K
    df = df.with_columns([
        pl.col(f'Stoch_%K_{period}').rolling_mean(window_size=smooth_k).alias(f'Stoch_%K_{period}_{smooth_k}')
    ])
    # Step 4: Apply smoothing to smoothed %K to get %D
    df = df.with_columns([
        pl.col(f'Stoch_%K_{period}_{smooth_k}').rolling_mean(window_size=smooth_d).alias(f'Stoch_%D_{period}_{smooth_d}')
    ])
    # Drop intermediary columns
    df = df.drop([f'Highest_High_{period}', f'Lowest_Low_{period}', f'Stoch_%K_{period}'])
    return df

def calculate_atr(df, high_column='High', low_column='Low', close_column='Close', period=14):
    """
    Calculate ATR.
    """
    # Calculate True Range components
    df = df.with_columns([
        (pl.col(high_column) - pl.col(low_column)).alias('TR1'),
        (pl.col(high_column) - pl.col(close_column).shift(1)).abs().alias('TR2'),
        (pl.col(low_column) - pl.col(close_column).shift(1)).abs().alias('TR3')
    ])
    # True Range is the max of the three components
    df = df.with_columns([
        pl.max_horizontal(['TR1', 'TR2', 'TR3']).alias('True_Range')
    ])
    # Calculate ATR
    df = df.with_columns([
        pl.col('True_Range').rolling_mean(window_size=period).alias(f'ATR_{period}')
    ])
    # Drop intermediary columns
    df = df.drop(['TR1', 'TR2', 'TR3', 'True_Range'])
    return df

def calculate_sma(df, column='Close', period=20):
    """
    Calculate Simple Moving Average.
    """
    df = df.with_columns([
        pl.col(column).rolling_mean(window_size=period).alias(f'SMA_{period}')
    ])
    return df

def calculate_rsi(df, column='Close', period=14):
    """
    Calculate Relative Strength Index.
    """
    df = df.with_columns([
        (pl.col(column) - pl.col(column).shift(1)).alias('Close_Change')
    ])
    df = df.with_columns([
        pl.when(pl.col('Close_Change') > 0)
          .then(pl.col('Close_Change'))
          .otherwise(0)
          .rolling_mean(window_size=period)
          .alias(f'Avg_Gain_{period}'),
        pl.when(pl.col('Close_Change') < 0)
          .then(-pl.col('Close_Change'))
          .otherwise(0)
          .rolling_mean(window_size=period)
          .alias(f'Avg_Loss_{period}')
    ])
    df = df.with_columns([
        (100 - (100 / (1 + (pl.col(f'Avg_Gain_{period}') / pl.col(f'Avg_Loss_{period}'))))).alias(f'RSI_{period}')
    ])
    # Drop intermediary columns
    df = df.drop(['Close_Change', f'Avg_Gain_{period}', f'Avg_Loss_{period}'])
    return df

def calculate_ema(df, column='Close', period=20):
    """
    Calculate Exponential Moving Average.
    """
    df = df.with_columns([
        pl.col(column).ewm_mean(span=period).alias(f'EMA_{period}')
    ])
    return df

def calculate_vwap(df, price_column='Close', volume_column='Volume'):
    """
    Calculate Volume Weighted Average Price.
    """
    df = df.with_columns([
        (pl.col(price_column) * pl.col(volume_column)).alias('PriceVolume')
    ])
    df = df.with_columns([
        (pl.col('PriceVolume').cum_sum() / pl.col(volume_column).cum_sum()).alias('VWAP')
    ])
    # Drop intermediary column
    df = df.drop(['PriceVolume'])
    return df

def calculate_roc(df, column='Close', period=14):
    """
    Calculate Rate of Change.
    """
    df = df.with_columns([
        (((pl.col(column) - pl.col(column).shift(period)) / pl.col(column).shift(period)) * 100).alias(f'ROC_{period}')
    ])
    return df

def calculate_pvt(df, price_column='Close', volume_column='Volume'):
    """
    Calculate Price-Volume Trend.
    """
    df = df.with_columns([
        ((pl.col(price_column).pct_change() * pl.col(volume_column)).alias('PVT_Change'))
    ])
    df = df.with_columns([
        pl.col('PVT_Change').cum_sum().alias('PVT')
    ])
    df = df.drop(['PVT_Change'])
    return df

def calculate_cumulative_returns(df, column='Close', period=20):
    """
    Calculate cumulative returns.
    """
    df = df.with_columns([
        ((pl.col(column) / pl.col(column).shift(period)) - 1).alias(f'Cumulative_Returns_{period}')
    ])
    return df

def calculate_relative_volume(df, volume_column='Volume', period=14):
    """
    Calculate relative volume.
    """
    df = df.with_columns([
        (pl.col(volume_column) / pl.col(volume_column).rolling_mean(window_size=period)).alias(f'Relative_Volume_{period}')
    ])
    return df

def calculate_close_open_ratio(df):
    """
    Calculate the ratio of close price to open price.
    """
    df = df.with_columns([
        (pl.col('Close').shift(1) / pl.col('Open')).alias('Close_Open_Ratio')
    ])
    return df

def add_features_pipeline(df):
    """
    Apply all feature engineering steps to the dataframe.
    """
    # Add time and day features
    df = add_time_and_day_features(df)

    # Calculate technical indicators
    df = calculate_sma(df, period=20)
    df = calculate_rsi(df, period=14)
    df = calculate_bollinger_bands(df, period=20, num_std_dev=2)
    df = calculate_macd(df, fast_period=12, slow_period=26, signal_period=9)
    df = calculate_ema(df, period=20)
    df = calculate_vwap(df)

    # Add custom indicators
    df = calculate_bollinger_bands(df, period=30, num_std_dev=2.2)
    df = calculate_bollinger_bands(df, period=200, num_std_dev=2)
    df = calculate_ema(df, period=200)
    df = calculate_stochastic(df, period=14, smooth_k=3, smooth_d=3)
    df = calculate_stochastic(df, period=70, smooth_k=10, smooth_d=3)
    df = calculate_rsi(df, period=13)

    # Add volatility indicators
    df = calculate_atr(df, period=14)
    df = calculate_historical_volatility(df, period=20)
    df = calculate_rolling_sd(df, period=20)

    # Add additional features
    df = calculate_close_open_ratio(df)
    df = calculate_roc(df, period=14)
    df = calculate_pvt(df)
    df = calculate_cumulative_returns(df, period=20)
    df = calculate_relative_volume(df, period=14)

    # Drop any rows with null values after feature engineering
    df = df.drop_nulls()

    return df

def save_preprocessed_data(ohlcv_data_1d, ohlcv_data_1h, file_name=None, date=False):
    if date == True:
        stamp = '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    else: stamp = ''
    if file_name is None:
        ohlcv_file_name_1d = f"preprocessed_1d_ohlcv_data{stamp}.parquet"
        ohlcv_file_name_1h = f"preprocessed_1h_ohlcv_data{stamp}.parquet"
    else:
        ohlcv_file_name_1d = f"{file_name}_1d_ohlcv{stamp}.parquet"
        ohlcv_file_name_1h = f"{file_name}_1h_ohlcv{stamp}.parquet"
    ohlcv_data_1d.write_parquet(ohlcv_file_name_1d)
    ohlcv_data_1h.write_parquet(ohlcv_file_name_1h)
    print(f"Preprocessed 1d OHLCV data saved to {ohlcv_file_name_1d}")
    print(f"Preprocessed 1h OHLCV data saved to {ohlcv_file_name_1h}")

def main_feature_engineering(parquet_path='/content/raw_ohlcv_data_20240929_221653.parquet'):
    ohlcv_data = pl.read_parquet(parquet_path)

    # Step 2: Identify Unique Tickers
    unique_tickers = ohlcv_data['Ticker'].unique().to_list()

    # Initialize lists to collect preprocessed data
    preprocessed_1d_list = []
    preprocessed_1h_list = []

    # Iterate over each ticker and process
    for ticker in tqdm(unique_tickers, desc="Processing Tickers"):
        # Filter data for the current ticker
        df_ticker = ohlcv_data.filter(pl.col('Ticker') == ticker).sort('Datetime')

        # Split into 1d and 1h data
        df_1d = df_ticker.filter(pl.col('Interval') == '1d').sort('Datetime')
        df_1h = df_ticker.filter(pl.col('Interval') == '1h').sort('Datetime')

        # Apply feature engineering to 1d and 1h data
        preprocessed_1d = add_features_pipeline(df_1d)
        preprocessed_1h = add_features_pipeline(df_1h)

        # Find the earliest datetime in 1d data
        earliest_1d_datetime = preprocessed_1d.select(pl.col('Datetime').min()).to_numpy()[0][0]

        # Filter 1h data to include only records >= earliest 1d datetime
        preprocessed_1h_filtered = preprocessed_1h.filter(pl.col('Datetime') >= earliest_1d_datetime)

        # Append to the respective lists
        preprocessed_1d_list.append(preprocessed_1d)
        preprocessed_1h_list.append(preprocessed_1h_filtered)

    # Step 4: Concatenate All Processed Data
    if preprocessed_1d_list:
        combined_preprocessed_1d = pl.concat(preprocessed_1d_list)
    else:
        combined_preprocessed_1d = pl.DataFrame()

    if preprocessed_1h_list:
        combined_preprocessed_1h = pl.concat(preprocessed_1h_list)
    else:
        combined_preprocessed_1h = pl.DataFrame()

    # Step 5: Save the Preprocessed Data
    save_preprocessed_data(combined_preprocessed_1d, combined_preprocessed_1h, file_name=None)

# Run feature engineering
main_feature_engineering()
