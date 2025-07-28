import yfinance as yf
import pandas as pd
import numpy as np  # Import numpy
from datetime import datetime, timedelta
import os


def get_daily_prices(ticker_symbol, start_date, end_date, output_filename=None):
    """
    Downloads daily historical prices for a given ticker symbol using Yahoo Finance.

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        output_filename (str, optional): Name of the CSV file to save the data.
                                         If None, data is not saved to a file.

    Returns:
        pandas.DataFrame: DataFrame containing daily historical prices.
    """
    # print(f"Fetching daily prices for {ticker_symbol} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker_symbol, start=start_date,
                           end=end_date, auto_adjust=True, progress=False)
        if not data.empty:
            # print(f"Successfully fetched {len(data)} daily records for {ticker_symbol}.")
            if output_filename:
                # Ensure the index is named 'datetime'
                data.index.name = 'datetime'
                # Reset index to make 'datetime' a regular column for cleaning
                data_to_save = data.reset_index()

                # Filter out rows where 'datetime' column contains 'Ticker' or 'datetime' strings
                data_to_save = data_to_save[~data_to_save['datetime'].astype(
                    str).isin(['Ticker', 'datetime'])]

                # Convert 'datetime' column to datetime objects and set as index
                data_to_save['datetime'] = pd.to_datetime(
                    data_to_save['datetime'], errors='coerce', utc=True)
                data_to_save = data_to_save.set_index('datetime')
                # Save the cleaned DataFrame to CSV with the datetime index
                data_to_save.to_csv(
                    output_filename, index=True, index_label='datetime')
                # print(f"Daily prices saved to {output_filename}")
            return data_to_save  # Return the cleaned data
        else:
            # print(f"No daily data found for {ticker_symbol} in the specified range.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching daily prices for {ticker_symbol}: {e}")
        return pd.DataFrame()


def get_hourly_prices(ticker_symbol, period="7d", interval="1h", output_filename=None):
    """
    Downloads hourly historical prices for a given ticker symbol using Yahoo Finance.
    Note: yfinance has limitations on the period for hourly data (max 60 days for 1h interval).

    Args:
        ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').
        period (str): Data period (e.g., '7d', '60d', '1y'). Max '60d' for '1h' interval.
        interval (str): Data interval (e.g., '1h', '2h', '15m').
        output_filename (str, optional): Name of the CSV file to save the data.
                                         If None, data is not saved to a file.

    Returns:
        pandas.DataFrame: DataFrame containing hourly historical prices.
    """
    # print(f"Fetching hourly prices for {ticker_symbol} for period {period} with interval {interval}...")
    try:
        data = yf.download(ticker_symbol, period=period,
                           interval=interval, auto_adjust=True, progress=False)
        if not data.empty:
            # print(f"Successfully fetched {len(data)} hourly records for {ticker_symbol}.")
            if output_filename:
                # Ensure the index is named 'datetime'
                data.index.name = 'datetime'
                # Reset index to make 'datetime' a regular column for cleaning
                data_to_save = data.reset_index()

                # Filter out rows where 'datetime' column contains 'Ticker' or 'datetime' strings
                data_to_save = data_to_save[~data_to_save['datetime'].astype(
                    str).isin(['Ticker', 'datetime'])]

                # Convert 'datetime' column to datetime objects and set as index
                data_to_save['datetime'] = pd.to_datetime(
                    data_to_save['datetime'], errors='coerce', utc=True)
                data_to_save = data_to_save.set_index('datetime')
                # Save the cleaned DataFrame to CSV with the datetime index
                data_to_save.to_csv(
                    output_filename, index=True, index_label='datetime')
                # print(f"Hourly prices saved to {output_filename}")
            return data_to_save  # Return the cleaned data
        else:
            print(
                f"No hourly data found for {ticker_symbol} in the specified range/period.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching hourly prices for {ticker_symbol}: {e}")
        return pd.DataFrame()
