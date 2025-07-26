import pandas as pd


def clean_data(df):
    """
    Cleans the DataFrame by handling missing values, removing duplicates,
    and potentially filtering out quiet hours (if applicable based on index frequency).
    """
    if df.empty:
        print("Warning: DataFrame is empty for cleaning.")
        return df

    original_rows = len(df)
    # print(f"Cleaning data. Original rows: {original_rows}")

    # Handle missing values: forward fill then backward fill
    # print("Handling missing values (ffill then bfill)...")
    df_cleaned = df.ffill().bfill()
    if df_cleaned.isnull().sum().sum() > 0:
        print(
            "Warning: Some NaN values remain after ffill/bfill. Dropping remaining NaNs.")
        df_cleaned = df_cleaned.dropna()

    # Deduplication: remove duplicate index entries (e.g., duplicate timestamps)
    # print("Removing duplicate index entries...")
    df_cleaned = df_cleaned[~df_cleaned.index.duplicated(keep='first')]

    # Optional: Filter out quiet hours/weekends if the index is datetime and has a frequency
    # This part is more complex and depends on the exact nature of "quiet hours".
    # For now, we assume the data is already filtered or that 'quiet hours' means
    # simply missing entries, which are handled by ffill/bfill.
    # If specific non-trading hours need to be removed, more logic would be needed here.

    # print(
    #     f"Cleaned data rows: {len(df_cleaned)}. Removed {original_rows - len(df_cleaned)} rows.")
    return df_cleaned


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
