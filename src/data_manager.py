"""
Data Manager - Handles efficient data fetching and caching.
Only fetches data for tickers that haven't been fetched yet.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from price_downloader import get_hourly_prices, get_daily_prices
from clean_data import clean_data
from returns import calculate_log_returns, normalize_series_rolling_zscore


class DataManager:
    def __init__(self, data_directory="price_downloader"):
        self.data_directory = data_directory
        self.processed_data_cache = {}

        # Ensure data directory exists
        os.makedirs(data_directory, exist_ok=True)

    def get_processed_data(self, ticker, period="60d", interval="1h", force_refresh=False):
        """
        Get processed data for a ticker. Only fetch if not already processed.

        Args:
            ticker (str): Ticker symbol
            period (str): Data period (e.g., "60d")
            interval (str): Data interval (e.g., "1h")
            force_refresh (bool): Force re-fetch even if data exists

        Returns:
            pd.DataFrame: Processed data with log returns and normalization
        """
        processed_filename = os.path.join(
            self.data_directory, f"{ticker}_processed_hourly_prices.csv")

        # Check if we already have processed data and don't need to refresh
        if not force_refresh and os.path.exists(processed_filename):
            try:
                # Try to load from cache first
                if ticker in self.processed_data_cache:
                    return self.processed_data_cache[ticker]

                # Load from file
                processed_data = pd.read_csv(
                    processed_filename, index_col='datetime', parse_dates=True)
                self.processed_data_cache[ticker] = processed_data
                print(f"✓ Loaded cached data for {ticker}")
                return processed_data
            except Exception as e:
                print(f"Warning: Error loading cached data for {ticker}: {e}")
                # Fall through to fetching new data

        # If we get here, we need to fetch and process the data
        print(f"Fetching and processing data for {ticker}...")

        # Get raw hourly data
        hourly_data = get_hourly_prices(
            ticker_symbol=ticker,
            period=period,
            interval=interval,
            output_filename=os.path.join(
                self.data_directory, f"{ticker}_hourly_prices.csv")
        )

        if hourly_data.empty:
            print(f"Warning: No hourly data fetched for {ticker}")
            return pd.DataFrame()

        # Clean the data
        cleaned_data = clean_data(hourly_data.copy())

        # Calculate log returns
        if 'Close' in cleaned_data.columns:
            log_returns = calculate_log_returns(
                cleaned_data.copy(), price_column='Close')
            cleaned_data['Log_Returns'] = log_returns
        else:
            print(f"Warning: No Close column for {ticker}")
            cleaned_data['Log_Returns'] = pd.NA

        # Normalize log returns
        if 'Log_Returns' in cleaned_data.columns and not cleaned_data['Log_Returns'].dropna().empty:
            normalized_log_returns = normalize_series_rolling_zscore(
                cleaned_data['Log_Returns'].dropna(), window=20)
            cleaned_data['Normalized_Log_Returns'] = normalized_log_returns
        else:
            cleaned_data['Normalized_Log_Returns'] = pd.NA

        # Make index timezone-naive
        if cleaned_data.index.tz is not None:
            cleaned_data.index = cleaned_data.index.tz_localize(None)

        # Save processed data
        cleaned_data.index.name = 'datetime'
        cleaned_data.to_csv(processed_filename, index_label='datetime')

        # Cache the data
        self.processed_data_cache[ticker] = cleaned_data

        print(f"✓ Processed and saved data for {ticker}")
        return cleaned_data

    def get_all_processed_data(self, tickers, period="60d", interval="1h", force_refresh=False):
        """
        Get processed data for multiple tickers efficiently.

        Args:
            tickers (list): List of ticker symbols
            period (str): Data period
            interval (str): Data interval
            force_refresh (bool): Force re-fetch for all tickers

        Returns:
            dict: Dictionary of processed dataframes keyed by ticker
        """
        all_data = {}

        for ticker in tickers:
            data = self.get_processed_data(
                ticker, period, interval, force_refresh)
            if not data.empty:
                all_data[ticker] = data

        return all_data

    def check_data_exists(self, ticker):
        """
        Check if processed data exists for a ticker.

        Args:
            ticker (str): Ticker symbol

        Returns:
            bool: True if processed data exists
        """
        processed_filename = os.path.join(
            self.data_directory, f"{ticker}_processed_hourly_prices.csv")
        return os.path.exists(processed_filename)

    def get_missing_tickers(self, tickers):
        """
        Get list of tickers that don't have processed data yet.

        Args:
            tickers (list): List of ticker symbols

        Returns:
            list: Tickers that need to be fetched
        """
        return [ticker for ticker in tickers if not self.check_data_exists(ticker)]

    def clear_cache(self):
        """Clear the in-memory data cache."""
        self.processed_data_cache = {}
        print("Data cache cleared")


# Global instance for easy access
data_manager = DataManager()

# Helper functions for backward compatibility


def get_processed_data(ticker, period="60d", interval="1h", force_refresh=False):
    """Helper function to get processed data for a single ticker."""
    return data_manager.get_processed_data(ticker, period, interval, force_refresh)


def get_all_processed_data(tickers, period="60d", interval="1h", force_refresh=False):
    """Helper function to get processed data for multiple tickers."""
    return data_manager.get_all_processed_data(tickers, period, interval, force_refresh)


def check_data_exists(ticker):
    """Helper function to check if data exists for a ticker."""
    return data_manager.check_data_exists(ticker)


def get_missing_tickers(tickers):
    """Helper function to get missing tickers."""
    return data_manager.get_missing_tickers(tickers)
