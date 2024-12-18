# data_collection.py

import requests
from bs4 import BeautifulSoup
import yfinance as yf
import polars as pl
from datetime import datetime, timedelta

# Function to scrape S&P 500 tickers
def get_sp500_tickers(exclude_list=[]):
    """
    Scrape the list of S&P 500 companies from Wikipedia and return their tickers.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Parse the table and extract ticker symbols
    table = soup.find('table', {'id': 'constituents'})
    tickers = []

    # Loop through the table rows to get the tickers
    for row in table.find_all('tr')[1:]:  # Skip the header row
        ticker = row.find_all('td')[0].text.strip()
        if ticker != 'GOOGL' and ticker not in exclude_list:
            tickers.append(ticker)

    return tickers

# Function to retrieve the market cap for a single ticker
def get_market_cap(ticker):
    """
    Retrieve the market capitalization for a given ticker using yfinance.
    """
    ticker_obj = yf.Ticker(ticker)
    try:
        market_cap = ticker_obj.info.get('marketCap', 0)  # Get market cap (0 if not available)
    except Exception as e:
        print(f"Error retrieving market cap for {ticker}: {e}")
        market_cap = 0
    return market_cap

# Function to get the top companies by market cap
def get_top_n_by_market_cap(tickers, n):
    """
    Get the top N companies from a list of tickers based on market capitalization.
    """
    ticker_data = []

    for ticker in tickers:
        market_cap = get_market_cap(ticker)
        ticker_data.append((ticker, market_cap))

    # Sort the companies by market cap in descending order
    sorted_companies = sorted(ticker_data, key=lambda x: x[1], reverse=True)

    # Get the top N companies
    top_n = sorted_companies[:n]

    return [ticker for ticker, market_cap in top_n]

# Function to normalize datetime columns
def normalize_datetime(df):
    """
    Standardize datetime columns and ensure they are timezone-free.
    """
    if 'Date' in df.columns:
        df = df.rename({'Date': 'Datetime'})
    if 'Datetime' in df.columns:
        df = df.with_columns([pl.col('Datetime').dt.replace_time_zone(None)])
    return df

# Function to get the valid date range for data collection
def get_valid_day_range(start_date=None, end_date=None):
    """
    Get the start and end dates within 700 days (adjusted to the first of the month).
    """
    if end_date is None:
        end_date = datetime.today().replace(day=1)  # Set to first day of the current month
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date is None:
        start_date = end_date - timedelta(days=700)
        start_date = start_date.replace(day=1)  # Adjust to the first day of the month
    else:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# Function to fetch static ticker information
def ticker_info_pipeline(ticker):
    """
    Retrieve static ticker information such as Industry, Sector, and QuoteType.
    """
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info

    # Extract industry, sector, and quoteType
    industry = info.get('industry', 'Unknown')
    sector = info.get('sector', 'Unknown')
    quote_type = info.get('quoteType', 'Unknown')

    # Adjust industry and sector based on the quote type
    if quote_type == 'ETF':
        industry = 'ETF'
        sector = 'ETF'
    elif quote_type == 'EQUITY' and industry == 'Unknown' and sector == 'Unknown':
        industry = 'Unclassified Equity'
        sector = 'Unclassified Equity'

    # Create a DataFrame for embedding use later
    info_df = pl.DataFrame({
        'Ticker': [ticker],
        'Industry': [industry],
        'Sector': [sector],
        'QuoteType': [quote_type]
    })

    return info_df

# Function to fetch OHLCV data
def ohlcv_pipeline(ticker, intervals=['1d', '1h'], startdate=None, enddate=None):
    """
    Fetch OHLCV data for a given ticker over specified intervals.
    """
    ticker_obj = yf.Ticker(ticker)
    start_date, end_date = get_valid_day_range(startdate, enddate)
    data_frames = []

    for interval in intervals:
        ohlcv_data = ticker_obj.history(interval=interval, start=start_date, end=end_date, prepost=True)
        df = pl.from_pandas(ohlcv_data.reset_index())

        # Normalize datetime column
        df = normalize_datetime(df)

        # Remove unnecessary columns
        if 'Capital Gains' in df.columns:
            df = df.drop('Capital Gains')
        if 'Dividends' in df.columns:
            df = df.drop('Dividends')
        if 'Stock Splits' in df.columns:
            df = df.drop('Stock Splits')

        # Add additional columns
        df = df.with_columns([
            pl.lit(ticker).alias('Ticker'),
            pl.lit(interval).alias('Interval'),
            pl.when(pl.col('Datetime').dt.hour() == 0)
              .then(pl.lit('na'))
              .when(
                  (pl.col('Datetime').dt.hour() < 9) |
                  (pl.col('Datetime').dt.hour() >= 16) |
                  ((pl.col('Datetime').dt.hour() == 9) & (pl.col('Datetime').dt.minute() < 30))
              )
              .then(pl.lit('extended'))
              .otherwise(pl.lit('regular'))
              .alias('ExtendedHours')
        ])

        data_frames.append(df)

    # Concatenate data for different intervals
    return pl.concat(data_frames)

# Function to download stock data for multiple tickers
def download_stock_data(tickers, intervals=['1d', '1h'], startdate=None, enddate=None):
    """
    Download OHLCV data and ticker info for a list of tickers.
    """
    ohlcv_data_list = []
    ticker_info_list = []

    for ticker in tickers:
        print(f"Downloading data for {ticker}...")

        # Fetch OHLCV data
        ohlcv_data = ohlcv_pipeline(ticker, intervals, startdate, enddate)
        ohlcv_data_list.append(ohlcv_data)

        # Fetch ticker info
        ticker_info_data = ticker_info_pipeline(ticker)
        ticker_info_list.append(ticker_info_data)

    # Concatenate all data
    combined_ohlcv_data = pl.concat(ohlcv_data_list)
    combined_ticker_info = pl.concat(ticker_info_list)

    return combined_ohlcv_data, combined_ticker_info

# Function to save raw data to disk
def save_raw_data(ohlcv_data, ticker_info_data, file_name=None):
    """
    Save raw OHLCV and ticker info data to disk.
    """
    current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

    if file_name is None:
        ohlcv_file_name = f"raw_ohlcv_data_{current_datetime}.parquet"
        ticker_info_file_name = f"raw_ticker_info_data_{current_datetime}.parquet"
    else:
        ohlcv_file_name = f"{file_name}_ohlcv_{current_datetime}.parquet"
        ticker_info_file_name = f"{file_name}_ticker_info_{current_datetime}.parquet"

    ohlcv_data.write_parquet(ohlcv_file_name)
    ticker_info_data.write_parquet(ticker_info_file_name)

    print(f"Raw OHLCV data saved to {ohlcv_file_name}")
    print(f"Ticker info data saved to {ticker_info_file_name}")

# Main function to collect data
def main_data_collection():
    """
    Main function to collect and save raw data.
    """
    # Define custom tickers and get top tickers by market cap
    custom_tickers = ['SPY', 'RSP', 'QQQ', 'PLTR', 'LUNR', 'RKLB', 'ASTS', 'AMD', 'INTC', 'COST', 'GS', 'JPM', 'SBUX', 'MCD', 'PG', 'DELL']
    sp500_tickers = get_sp500_tickers(exclude_list=custom_tickers)
    top_tickers = get_top_n_by_market_cap(sp500_tickers, 100)

    # Combine tickers
    tickers = custom_tickers + top_tickers

    # Download data
    ohlcv_data, ticker_info_data = download_stock_data(tickers)

    # Save raw data
    save_raw_data(ohlcv_data, ticker_info_data)

# Run data collection
main_data_collection()
