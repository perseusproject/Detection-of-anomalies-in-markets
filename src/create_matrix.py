import pandas as pd
from clean_data import clean_data
from returns import calculate_log_returns
from price_downloader import get_hourly_prices  # Import the downloader


def create_returns_matrix(tickers, period="30d", interval="1h"):
    """
    Creates a matrix X_t of hourly log-returns for a basket of assets.
    X_t is of shape n x T, where n is the number of assets and T is the number of observations.

    returns : pandas.DataFrame
        DataFrame containing the log-returns matrix with assets as columns and timestamps as index.
    """
    # print(f"\n--- Creating Returns Matrix for {tickers} over last {period} ({interval} interval) ---")
    all_hourly_returns = {}

    for ticker in tickers:
        hourly_file_path = f"price_downloader/{ticker}_hourly_prices.csv"
        hourly_data = pd.DataFrame()  # Initialize empty DataFrame

        # Check if file exists and has recent data, otherwise download
        try:
            # Attempt to load hourly data, skipping the first 3 rows
            temp_hourly_data = pd.read_csv(
                # Read without parsing dates initially
                hourly_file_path, skiprows=3, header=None)

            # Assign meaningful column names
            temp_hourly_data.columns = [
                'Datetime', 'Close', 'High', 'Low', 'Open', 'Volume']

            # Explicitly convert 'Datetime' column to datetime objects and set as index
            temp_hourly_data['Datetime'] = pd.to_datetime(
                temp_hourly_data['Datetime'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
            temp_hourly_data = temp_hourly_data.set_index('Datetime').dropna(
                # Drop rows where Datetime conversion failed or Close is NaN
                subset=['Close'])

            # Check if the latest data point is recent enough (e.g., within the last 'period' days)
            period_timedelta = pd.to_timedelta(period)

            if not temp_hourly_data.empty and (pd.Timestamp.now(tz='UTC') - temp_hourly_data.index.max()) < period_timedelta:
                hourly_data = temp_hourly_data
                print(f"Using existing recent hourly data for {ticker}.")
            else:
                print(
                    f"Hourly data for {ticker} is old or empty. Attempting to download new data.")
                hourly_data = get_hourly_prices(
                    ticker, period=period, interval=interval, output_filename=hourly_file_path)

        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(
                f"Hourly data file not found or empty for {ticker}. Attempting to download new data.")
            hourly_data = get_hourly_prices(
                ticker, period=period, interval=interval, output_filename=hourly_file_path)
        except Exception as e:
            print(
                f"Error loading existing hourly data for {ticker}: {e}. Attempting to download new data.")
            hourly_data = get_hourly_prices(
                ticker, period=period, interval=interval, output_filename=hourly_file_path)

            # Debug prints after download attempt
            print(
                f"Columns of hourly_data for {ticker} after download: {hourly_data.columns}")
            print(
                f"Head of hourly_data for {ticker} after download:\n{hourly_data.head()}")

        if hourly_data.empty:
            print(
                f"Could not get hourly data for {ticker}. Skipping for matrix creation.")
            continue

        # Ensure 'Close' column is numeric after loading or downloading
        # This check is still necessary as downloaded data might also have issues
        if 'Close' in hourly_data.columns:
            hourly_data['Close'] = pd.to_numeric(
                hourly_data['Close'], errors='coerce')
        else:
            print(
                f"Error: 'Close' column not found in {ticker} data after loading/downloading. Skipping.")
            continue

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
