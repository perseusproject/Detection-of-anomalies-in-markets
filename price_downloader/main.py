import yfinance as yf
import pandas as pd
import numpy as np  # Import numpy
from datetime import datetime, timedelta


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
    print(
        f"Fetching daily prices for {ticker_symbol} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker_symbol, start=start_date, end=end_date)
        if not data.empty:
            print(
                f"Successfully fetched {len(data)} daily records for {ticker_symbol}.")
            if output_filename:
                data.to_csv(output_filename)
                print(f"Daily prices saved to {output_filename}")
            return data
        else:
            print(
                f"No daily data found for {ticker_symbol} in the specified range.")
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
    print(
        f"Fetching hourly prices for {ticker_symbol} for period {period} with interval {interval}...")
    try:
        data = yf.download(ticker_symbol, period=period, interval=interval)
        if not data.empty:
            print(
                f"Successfully fetched {len(data)} hourly records for {ticker_symbol}.")
            if output_filename:
                data.to_csv(output_filename)
                print(f"Hourly prices saved to {output_filename}")
            return data
        else:
            print(
                f"No hourly data found for {ticker_symbol} in the specified range/period.")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching hourly prices for {ticker_symbol}: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Example Usage:
    # Define a list of ticker symbols
    tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA"]

    # Define start and end dates for daily data
    end_date_daily = datetime.now().strftime('%Y-%m-%d')
    start_date_daily = (datetime.now() - timedelta(days=365)
                        ).strftime('%Y-%m-%d')  # Last 1 year

    for ticker in tickers:
        print(f"\n--- Processing Data for {ticker} ---")

        # --- Get Daily Prices ---
        print(f"\n--- Daily Price Data for {ticker} ---")
        daily_data = get_daily_prices(
            ticker_symbol=ticker,
            start_date=start_date_daily,
            end_date=end_date_daily,
            output_filename=f"price_downloader/{ticker}_daily_prices.csv"
        )
        if not daily_data.empty:
            print(f"\nFirst 5 daily records for {ticker}:")
            print(daily_data.head())

        # --- Get Hourly Prices ---
    # Note: For hourly data, yfinance has limitations. Max period is typically 60 days for 1h interval.
    # Changed period to 30 days as requested for the matrix X_t
    print(f"\n--- Hourly Price Data (last 30 days) for {ticker} ---")
    hourly_data = get_hourly_prices(
        ticker_symbol=ticker,
        period="30d",  # Max 60 days for 1h interval
        interval="1h",
        output_filename=f"price_downloader/{ticker}_hourly_prices.csv"
    )
    if not hourly_data.empty:
        print(f"\nFirst 5 hourly records for {ticker}:")
        print(hourly_data.head())

    print("\nAll data fetching complete. Check the 'price_downloader' directory for CSV files.")

    # --- Data Processing ---
    print("\n--- Starting Data Processing ---")

    def calculate_log_returns(df, price_column='Close'):
        """
        Calculates hourly log-returns for a given DataFrame.
        r_t = log(P_t) - log(P_{t-1})
        """
        if df.empty or price_column not in df.columns:
            print(
                f"Warning: DataFrame is empty or '{price_column}' column not found for log-return calculation.")
            return pd.Series(dtype='float64')
        print(f"Calculating log-returns for column '{price_column}'...")
        df['Log_Returns'] = (df[price_column] / df[price_column].shift(1)
                             # Use np.log
                             ).apply(lambda x: pd.NA if pd.isna(x) else np.log(x))
        return df['Log_Returns']

    def clean_data(df):
        """
        Cleans the DataFrame by handling missing values, removing duplicates,
        and potentially filtering out quiet hours (if applicable based on index frequency).
        """
        if df.empty:
            print("Warning: DataFrame is empty for cleaning.")
            return df

        original_rows = len(df)
        print(f"Cleaning data. Original rows: {original_rows}")

        # Handle missing values: forward fill then backward fill
        print("Handling missing values (ffill then bfill)...")
        df_cleaned = df.ffill().bfill()
        if df_cleaned.isnull().sum().sum() > 0:
            print(
                "Warning: Some NaN values remain after ffill/bfill. Dropping remaining NaNs.")
            df_cleaned = df_cleaned.dropna()

        # Deduplication: remove duplicate index entries (e.g., duplicate timestamps)
        print("Removing duplicate index entries...")
        df_cleaned = df_cleaned[~df_cleaned.index.duplicated(keep='first')]

        # Optional: Filter out quiet hours/weekends if the index is datetime and has a frequency
        # This part is more complex and depends on the exact nature of "quiet hours".
        # For now, we assume the data is already filtered or that 'quiet hours' means
        # simply missing entries, which are handled by ffill/bfill.
        # If specific non-trading hours need to be removed, more logic would be needed here.

        print(
            f"Cleaned data rows: {len(df_cleaned)}. Removed {original_rows - len(df_cleaned)} rows.")
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

        print(
            f"Normalizing series using rolling z-score with window {window}...")
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        normalized_series = (series - rolling_mean) / rolling_std
        return normalized_series

    for ticker in tickers:
        print(f"\n--- Processing Hourly Data for {ticker} ---")
        hourly_file_path = f"price_downloader/{ticker}_hourly_prices.csv"
        try:
            # Load hourly data
            hourly_data = pd.read_csv(
                hourly_file_path, index_col=0, parse_dates=True)
            print(f"Loaded {len(hourly_data)} hourly records for {ticker}.")

            # Ensure 'Close' column is numeric
            if 'Close' in hourly_data.columns:
                hourly_data['Close'] = pd.to_numeric(
                    hourly_data['Close'], errors='coerce')
                print(f"Converted 'Close' column to numeric for {ticker}.")
            else:
                print(
                    f"Warning: 'Close' column not found in raw data for {ticker}.")

            # 1. Cleaning
            cleaned_hourly_data = clean_data(hourly_data.copy())

            # 2. Calculate Log-Returns
            # Ensure 'Close' column exists after cleaning
            if 'Close' in cleaned_hourly_data.columns:
                log_returns = calculate_log_returns(
                    cleaned_hourly_data.copy(), price_column='Close')
                cleaned_hourly_data['Log_Returns'] = log_returns
                print(f"Log-returns calculated for {ticker}.")
            else:
                print(
                    f"Warning: 'Close' column not found in cleaned data for {ticker}. Skipping log-return calculation.")
                # Add column with NA if not found
                cleaned_hourly_data['Log_Returns'] = pd.NA

            # 3. Normalization of Log-Returns
            if 'Log_Returns' in cleaned_hourly_data.columns and not cleaned_hourly_data['Log_Returns'].dropna().empty:
                normalized_log_returns = normalize_series_rolling_zscore(
                    cleaned_hourly_data['Log_Returns'].dropna(), window=20)
                cleaned_hourly_data['Normalized_Log_Returns'] = normalized_log_returns
                print(f"Normalized log-returns calculated for {ticker}.")
            else:
                print(
                    f"Warning: No valid Log_Returns to normalize for {ticker}.")
                # Add column with NA if not found
                cleaned_hourly_data['Normalized_Log_Returns'] = pd.NA

            # Save processed data
            processed_output_filename = f"price_downloader/{ticker}_processed_hourly_prices.csv"
            cleaned_hourly_data.to_csv(processed_output_filename)
            print(
                f"Processed hourly data saved to {processed_output_filename}")

            print(f"\nFirst 5 processed hourly records for {ticker}:")
            print(cleaned_hourly_data.head())

        except FileNotFoundError:
            print(
                f"Error: Hourly data file not found for {ticker} at {hourly_file_path}. Skipping processing.")
        except Exception as e:
            print(f"Error processing hourly data for {ticker}: {e}")

    print("\n--- All Data Processing Complete ---")

    def create_returns_matrix(tickers, period="30d", interval="1h"):
        """
        Creates a matrix X_t of hourly log-returns for a basket of assets.
        X_t is of shape n x T, where n is the number of assets and T is the number of observations.
        """
        print(
            f"\n--- Creating Returns Matrix for {tickers} over last {period} ({interval} interval) ---")
        all_hourly_returns = {}

        for ticker in tickers:
            hourly_file_path = f"price_downloader/{ticker}_hourly_prices.csv"
            try:
                # Load hourly data
                hourly_data = pd.read_csv(
                    hourly_file_path, index_col=0, parse_dates=True)

                # Ensure 'Close' column is numeric
                if 'Close' in hourly_data.columns:
                    hourly_data['Close'] = pd.to_numeric(
                        hourly_data['Close'], errors='coerce')

                # Clean data
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

        # Instead of filling with 0, which can skew analysis,
        # we will drop rows that still contain any NaN values after alignment.
        # This ensures that all returns in the matrix correspond to actual observations
        # across all assets for that specific timestamp.
        print("Dropping rows with any remaining NaN values to ensure strict alignment...")
        returns_df = returns_df.dropna(how='any')

        print(
            f"\nCreated returns matrix X_t with shape: {returns_df.shape} (T rows, n columns)")
        print("First 5 rows of the returns matrix:")
        print(returns_df.head())

        # Transpose to get n x T as requested (n assets, T observations)
        X_t_matrix = returns_df.T
        print(
            f"\nTransposed matrix X_t with shape: {X_t_matrix.shape} (n rows, T columns)")
        print("First 5 columns of the transposed matrix:")
        print(X_t_matrix.iloc[:, :5])  # Display first 5 columns

        return X_t_matrix

    # Call the function to create the returns matrix
    returns_matrix_Xt = create_returns_matrix(
        tickers, period="30d", interval="1h")

    if not returns_matrix_Xt.empty:
        # You can save this matrix to a file if needed
        returns_matrix_Xt.to_csv("price_downloader/returns_matrix_Xt.csv")
        print("\nReturns matrix X_t saved to price_downloader/returns_matrix_Xt.csv")
    else:
        print("\nCould not create returns matrix X_t.")
