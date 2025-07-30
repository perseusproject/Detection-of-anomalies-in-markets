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
from backtester import strategy, run_backtest  # Import run_backtest

# Define a list of ticker symbols
tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

# Define period and interval for hourly data
period = "300d"
interval = "4h"

# Define start and end dates for daily data
end_date_daily = datetime.now().strftime('%Y-%m-%d')
start_date_daily = (datetime.now() - timedelta(days=365)
                    ).strftime('%Y-%m-%d')  # Last 1 year

# --- Data Fetching and Initial Processing ---
print("\n--- Starting Data Fetching and Initial Processing ---")

# Dictionary to store processed hourly data for each ticker
all_processed_hourly_data = {}

for ticker in tickers:
    print(f"\n--- Processing Data for {ticker} ---")

    # --- Get Daily Prices ---
    # Daily prices are fetched but not used in subsequent steps for this anomaly detection.
    # They are saved to CSV by get_daily_prices.
    daily_data = get_daily_prices(
        ticker_symbol=ticker,
        start_date=start_date_daily,
        end_date=end_date_daily,
        output_filename=os.path.join(
            os.getcwd(), "price_downloader", f"{ticker}_daily_prices.csv")
    )
    if daily_data.empty:
        print(
            f"Warning: No daily data fetched for {ticker}. Skipping daily data processing.")

    # --- Get Hourly Prices ---
    # get_hourly_prices now returns a cleaned DataFrame directly.
    hourly_data = get_hourly_prices(
        ticker_symbol=ticker,
        period=period,
        interval=interval,
        output_filename=os.path.join(
            os.getcwd(), "price_downloader", f"{ticker}_hourly_prices.csv")
    )

    if hourly_data.empty:
        print(
            f"Warning: No hourly data fetched for {ticker}. Skipping hourly data processing.")
        continue  # Skip to next ticker if no hourly data

    # print(f"\n--- Processing Hourly Data for {ticker} ---")

    # Flatten MultiIndex columns if present (e.g., from yfinance output)
    if isinstance(hourly_data.columns, pd.MultiIndex):
        hourly_data.columns = hourly_data.columns.droplevel(1)
        hourly_data.columns.name = None  # Remove the MultiIndex name

    # Ensure 'Close' column is numeric
    if 'Close' in hourly_data.columns:
        hourly_data['Close'] = pd.to_numeric(
            hourly_data['Close'], errors='coerce')
        # Drop rows where 'Close' price is NaN after conversion
        hourly_data.dropna(subset=['Close'], inplace=True)
    else:
        print(
            f"Warning: 'Close' column not found in raw data for {ticker}. Skipping further processing for this ticker.")
        continue  # Skip to next ticker if no 'Close' column

    # 1. Cleaning (additional cleaning if needed, get_hourly_prices already does some)
    # clean_data function can handle further general cleaning like ffill/bfill
    cleaned_hourly_data = clean_data(hourly_data.copy())

    # 2. Calculate Log-Returns
    if 'Close' in cleaned_hourly_data.columns:
        log_returns = calculate_log_returns(
            cleaned_hourly_data.copy(), price_column='Close')
        cleaned_hourly_data['Log_Returns'] = log_returns
    else:
        print(
            f"Warning: 'Close' column not found in cleaned data for {ticker}. Skipping log-return calculation.")
        cleaned_hourly_data['Log_Returns'] = pd.NA

    # 3. Normalization of Log-Returns
    if 'Log_Returns' in cleaned_hourly_data.columns and not cleaned_hourly_data['Log_Returns'].dropna().empty:
        normalized_log_returns = normalize_series_rolling_zscore(
            cleaned_hourly_data['Log_Returns'].dropna(), window=20)
        cleaned_hourly_data['Normalized_Log_Returns'] = normalized_log_returns
    else:
        print(
            f"Warning: No valid Log_Returns to normalize for {ticker}.")
        cleaned_hourly_data['Normalized_Log_Returns'] = pd.NA

    # Make the index timezone-naive before saving and passing to other functions
    if cleaned_hourly_data.index.tz is not None:
        cleaned_hourly_data.index = cleaned_hourly_data.index.tz_localize(None)

    # Save processed data
    processed_output_filename = f"price_downloader/{ticker}_processed_hourly_prices.csv"
    # Ensure the index is named 'datetime' and save it with that label
    cleaned_hourly_data.index.name = 'datetime'
    cleaned_hourly_data.to_csv(
        processed_output_filename, index_label='datetime')
    # print(f"Processed hourly data saved to {processed_output_filename}")

    # Store processed data for matrix creation
    all_processed_hourly_data[ticker] = cleaned_hourly_data

print("\n--- All Data Fetching and Initial Processing Complete ---")

# Call the function to create the returns matrix
# Pass the dictionary of processed dataframes instead of re-reading from CSVs
returns_matrix_Xt = create_returns_matrix(
    tickers, period=period, interval=interval, processed_data=all_processed_hourly_data)

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
        returns_matrix_Xt, n_components=1)

    # Plot anomalous elements (sparse component) as a heatmap
    print("\n--- Plotting Anomalous Elements (Sparse Component) Heatmap ---")
    plt.figure(figsize=(12, 8))

    # Fill any NaN values in the sparse_component with 0 before plotting
    # This is necessary because heatmap cannot plot 'object' dtype (which pd.NA can cause)
    sparse_component_filled = sparse_component.fillna(0)

    # sns.heatmap(sparse_component_filled, cmap='viridis',
    #             cbar_kws={'label': 'Anomaly Magnitude'})
    # plt.title('Heatmap of Anomalous Elements (Sparse Component)')
    # plt.xlabel('Assets')
    # plt.ylabel('Time')
    # plt.tight_layout()

    # # Define filename for heatmap
    # heatmap_filename = os.path.join("Figures", f"{period}{interval}HM.png")

    # # Check if the heatmap file already exists before saving
    # if not os.path.exists(heatmap_filename):
    #     plt.savefig(heatmap_filename)
    #     print(f"Heatmap saved to {heatmap_filename}")
    # else:
    #     print(
    #         f"Heatmap file already exists at {heatmap_filename}, skipping save.")

    # plt.show()

    # --- Run Backtest ---
