import pandas as pd
import numpy as np  # Import numpy


def calculate_log_returns(df, price_column='Close'):
    """
    Calculates hourly log-returns for a given DataFrame.
    r_t = log(P_t) - log(P_{t-1})
    """
    if df.empty or price_column not in df.columns:
        print(
            f"Warning: DataFrame is empty or '{price_column}' column not found for log-return calculation.")
        return pd.Series(dtype='float64')
    # print(f"Calculating log-returns for column '{price_column}'...")
    # Calculate ratio, then apply log, and ensure float dtype
    log_returns = np.log(df[price_column] /
                         df[price_column].shift(1)).astype(float)
    return log_returns


def normalize_series_rolling_zscore(series, window=10):
    """
    Normalizes a series using a rolling z-score.
    Z-score = (X - rolling_mean) / rolling_std
    """
    if series.empty:
        print("Warning: Series is empty for normalization.")
        return pd.Series(dtype='float64')
    if len(series) < window:
        print(
            f"Warning: Series length ({len(series)}) is less than rolling window ({window}). Cannot normalize.")
        return pd.Series(dtype='float64')

    # print(
    #     f"Normalizing series using rolling z-score with window {window}...")
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    normalized_series = (series - rolling_mean) / rolling_std
    return normalized_series
