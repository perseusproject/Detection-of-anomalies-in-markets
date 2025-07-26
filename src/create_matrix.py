import pandas as pd
from clean_data import clean_data
from returns import calculate_log_returns


def create_returns_matrix(tickers, period="30d", interval="1h"):
    """
    Creates a matrix X_t of hourly log-returns for a basket of assets.
    X_t is of shape n x T, where n is the number of assets and T is the number of observations.
    """
    # print(f"\n--- Creating Returns Matrix for {tickers} over last {period} ({interval} interval) ---")
    all_hourly_returns = {}

    for ticker in tickers:
        hourly_file_path = f"price_downloader/{ticker}_hourly_prices.csv"
        try:
            # Load hourly data
            hourly_data = pd.read_csv(
                hourly_file_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')

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
    # print("Dropping rows with any remaining NaN values to ensure strict alignment...")
    returns_df = returns_df.dropna(how='any')

    # print(f"\nCreated returns matrix X_t with shape: {returns_df.shape} (T rows, n columns)")
    # print("First 5 rows of the returns matrix:")
    print(returns_df.head())

    # Transpose to get n x T as requested (n assets, T observations)
    X_t_matrix = returns_df.T
    print(
        f"\nTransposed matrix X_t with shape: {X_t_matrix.shape} (n rows, T columns)")
    # print("First 5 columns of the transposed matrix:")
    # print(X_t_matrix.iloc[:, :5])  # Display first 5 columns

    return X_t_matrix
