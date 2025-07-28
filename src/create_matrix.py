import pandas as pd
from clean_data import clean_data
from returns import calculate_log_returns


def create_returns_matrix(tickers, period="30d", interval="1h", processed_data=None):
    """
    Creates a matrix X_t of hourly log-returns for a basket of assets.
    X_t is of shape n x T, where n is the number of assets and T is the number of observations.

    Args:
        tickers (list): List of ticker symbols.
        period (str): Data period (e.g., "30d", "1h").
        interval (str): Data interval (e.g., "1h", "1m").
        processed_data (dict, optional): A dictionary where keys are ticker symbols
                                         and values are pre-processed DataFrames.
                                         If provided, data is taken from here instead of CSVs.

    returns : pandas.DataFrame
        DataFrame containing the log-returns matrix with assets as columns and timestamps as index.
    """
    print(
        f"\n--- Creating Returns Matrix for {tickers} over last {period} ({interval} interval) ---")
    all_hourly_returns = {}

    for ticker in tickers:
        if processed_data and ticker in processed_data:
            # Use pre-processed data if available
            cleaned_hourly_data = processed_data[ticker]
            # Ensure 'Close' column is numeric and handle potential NaNs, as this might not be fully guaranteed by previous steps
            if 'Close' in cleaned_hourly_data.columns:
                cleaned_hourly_data['Close'] = pd.to_numeric(
                    cleaned_hourly_data['Close'], errors='coerce')
                cleaned_hourly_data.dropna(subset=['Close'], inplace=True)
            else:
                print(
                    f"Warning: 'Close' column not found in processed data for {ticker}. Skipping for matrix creation.")
                continue

            # Calculate Log-Returns
            if 'Close' in cleaned_hourly_data.columns:
                log_returns = calculate_log_returns(
                    cleaned_hourly_data.copy(), price_column='Close')
                all_hourly_returns[ticker] = log_returns
            else:
                print(
                    f"Warning: 'Close' column not found in cleaned data for {ticker}. Skipping log-return calculation for matrix.")

        else:
            # Fallback to loading from CSV if pre-processed data is not provided or missing
            hourly_file_path = f"price_downloader/{ticker}_hourly_prices.csv"
            try:
                # Load hourly data (assuming price_downloader now saves standard CSVs)
                hourly_data = pd.read_csv(
                    hourly_file_path, index_col='datetime', parse_dates=True, date_format='%Y-%m-%d %H:%M:%S%z')

                # Ensure 'Close' column is numeric
                if 'Close' in hourly_data.columns:
                    hourly_data['Close'] = pd.to_numeric(
                        hourly_data['Close'], errors='coerce')
                    # Drop NaNs after conversion
                    hourly_data.dropna(subset=['Close'], inplace=True)

                # Clean data (this might be redundant if price_downloader already cleans thoroughly)
                cleaned_hourly_data = clean_data(hourly_data.copy())

                # Calculate Log-Returns
                if 'Close' in cleaned_hourly_data.columns:
                    log_returns = calculate_log_returns(
                        cleaned_hourly_data.copy(), price_column='Close')
                    all_hourly_returns[ticker] = log_returns
                else:
                    print(
                        f"Warning: 'Close' column not found for {ticker}. Skipping log-return calculation for matrix.")

            except FileNotFoundError:
                print(
                    f"Error: Hourly data file not found for {ticker} at {hourly_file_path}. Skipping for matrix creation.")
            except Exception as e:
                print(
                    f"Error loading or processing hourly data for {ticker} for matrix creation: {e}")

    if not all_hourly_returns:
        print("No hourly returns data available to create the matrix.")
        return pd.DataFrame()

    # Combine all log returns into a single DataFrame
    returns_df = pd.DataFrame(all_hourly_returns)

    # Drop rows with all NaN values (e.g., non-overlapping timestamps)
    returns_df = returns_df.dropna(how='all')

    # Drop rows that still contain any NaN values after alignment.
    returns_df = returns_df.dropna(how='any')

    print(returns_df.head())

    # Transpose to get n x T as requested (n assets, T observations)
    X_t_matrix = returns_df.T
    print(
        f"\nTransposed matrix X_t with shape: {X_t_matrix.shape} (n rows, T columns)")

    return X_t_matrix
