import pandas as pd
import numpy as np  # Import numpy
from datetime import datetime, timedelta
import os
from price_downloader import get_daily_prices, get_hourly_prices
from clean_data import clean_data
from returns import calculate_log_returns, normalize_series_rolling_zscore
from create_matrix import create_returns_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PCA import perform_pca, robust_pca
from backtester import strategy, calculate_pnl


# Define a list of ticker symbols
tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA"]

# Define start and end dates for daily data
end_date_daily = datetime.now().strftime('%Y-%m-%d')
start_date_daily = (datetime.now() - timedelta(days=365)
                    ).strftime('%Y-%m-%d')  # Last 1 year

for ticker in tickers:
    # print(f"\n--- Processing Data for {ticker} ---")

    # --- Get Daily Prices ---
    # print(f"\n--- Daily Price Data for {ticker} ---")
    daily_data = get_daily_prices(
        ticker_symbol=ticker,
        start_date=start_date_daily,
        end_date=end_date_daily,
        output_filename=os.path.join(
            os.getcwd(), "price_downloader", f"{ticker}_daily_prices.csv")
    )
    # --- Get Hourly Prices ---
    # print(f"\n--- Hourly Price Data (last 30 days) for {ticker} ---")
    hourly_data = get_hourly_prices(
        ticker_symbol=ticker,
        period="7d",  # Last week as per objective
        interval="1h",
        output_filename=os.path.join(
            os.getcwd(), "price_downloader", f"{ticker}_hourly_prices.csv")
    )
# print("\nAll data fetching complete. Check the 'price_downloader' directory for CSV files.")

# --- Data Processing ---
# print("\n--- Starting Data Processing ---")

for ticker in tickers:
    # print(f"\n--- Processing Hourly Data for {ticker} ---")
    hourly_file_path = f"price_downloader/{ticker}_hourly_prices.csv"
    try:

        # Load hourly data
        hourly_data = pd.read_csv(
            hourly_file_path, index_col=0, parse_dates=True, date_format='%Y-%m-%d %H:%M:%S')
        # print(f"Loaded {len(hourly_data)} hourly records for {ticker}.")

        # Ensure 'Close' column is numeric
        if 'Close' in hourly_data.columns:
            hourly_data['Close'] = pd.to_numeric(
                hourly_data['Close'], errors='coerce')
            # print(f"Converted 'Close' column to numeric for {ticker}.")
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
            # print(f"Log-returns calculated for {ticker}.")
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
            # print(f"Normalized log-returns calculated for {ticker}.")
        else:
            print(
                f"Warning: No valid Log_Returns to normalize for {ticker}.")
            # Add column with NA if not found
            cleaned_hourly_data['Normalized_Log_Returns'] = pd.NA

        # Save processed data
        processed_output_filename = f"price_downloader/{ticker}_processed_hourly_prices.csv"
        cleaned_hourly_data.to_csv(processed_output_filename)
        # print(f"Processed hourly data saved to {processed_output_filename}")

        # print(f"\nFirst 5 processed hourly records for {ticker}:")
        # print(cleaned_hourly_data.head())

    except FileNotFoundError:
        print(
            f"Error: Hourly data file not found for {ticker} at {hourly_file_path}. Skipping processing.")
    except Exception as e:
        print(f"Error processing hourly data for {ticker}: {e}")

# print("\n--- All Data Processing Complete ---")

# Call the function to create the returns matrix
returns_matrix_Xt = create_returns_matrix(
    tickers, period="7d", interval="1h")

if not returns_matrix_Xt.empty:
    # You can save this matrix to a file if needed
    returns_matrix_Xt.to_csv("price_downloader/returns_matrix_Xt.csv")
    print("\nReturns matrix X_t saved to price_downloader/returns_matrix_Xt.csv")
else:
    print("\nCould not create returns matrix X_t.")

# Perform robust PCA on the returns matrix


if not returns_matrix_Xt.empty:
    # Perform robust PCA on the returns matrix, specifying n_components for a low-rank approximation
    # Setting n_components to a value less than the full rank will ensure a non-zero sparse component.
    # For a matrix of shape (n_assets, n_observations), n_components should be less than n_assets.
    low_rank_component, sparse_component, singular_values = robust_pca(
        returns_matrix_Xt, n_components=2)

    # Plot anomalous elements (sparse component) as a heatmap
    print("\n--- Plotting Anomalous Elements (Sparse Component) Heatmap ---")
    plt.figure(figsize=(12, 8))

    # Fill any NaN values in the sparse_component with 0 before plotting
    # This is necessary because heatmap cannot plot 'object' dtype (which pd.NA can cause)
    sparse_component_filled = sparse_component.fillna(0)

    sns.heatmap(sparse_component_filled, cmap='viridis',
                cbar_kws={'label': 'Anomaly Magnitude'})
    plt.title('Heatmap of Anomalous Elements (Sparse Component)')
    plt.xlabel('Assets')
    plt.ylabel('Time')
    plt.tight_layout()
    plt.show()

    # --- Apply Strategy and Plot PnL for each index ---
    print("\n--- Applying Strategy and Plotting PnL for each index ---")
    plt.figure(figsize=(14, 10))

    for j, ticker in enumerate(tickers):
        # Get the returns series for the current ticker
        returns_series = returns_matrix_Xt.iloc[j]

        # Generate strategy signals
        strategy_signals = strategy(returns_matrix_Xt, sparse_component, j)

        # Calculate PnL
        cumulative_pnl = calculate_pnl(returns_series, strategy_signals)

        # Plot PnL
        plt.plot(cumulative_pnl.index, cumulative_pnl, label=f'{ticker} PnL')

    plt.title('Cumulative PnL for Each Index using Anomaly Detection Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
