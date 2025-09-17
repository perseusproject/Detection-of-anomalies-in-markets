import pandas as pd
import numpy as np  # Import numpy
from datetime import datetime, timedelta
import os
from price_downloader import get_daily_prices, get_hourly_prices
from clean_data import clean_data
from returns import calculate_log_returns, normalize_series_rolling_zscore
from create_matrix import create_returns_matrix
from data_manager import get_all_processed_data, get_missing_tickers
import matplotlib.pyplot as plt
import seaborn as sns
from PCA import perform_pca, robust_pca, apply_robust_pca_model
from backtester import strategy, backtest_strategy, plot_pnl_chart
from parameter_optimizer import grid_search_optimization, analyze_optimization_results
from optimized_parameters import get_optimized_parameters, run_parameter_optimization

# Define a list of ticker symbols
tickers = [
    "^IXIC",  # Nasdaq Composite
    "^NDX",   # Nasdaq-100
    "XSW",    # SPDR S&P Software & Services ETF
    "XLK",    # Technology Select Sector SPDR
    "VGT",    # Vanguard Information Technology ETF
    "IXN",    # iShares Global Tech ETF
    "FTEC",   # Fidelity MSCI Info Tech ETF
    "IYW",    # iShares U.S. Technology ETF
    "QQQ",    # Invesco QQQ Trust
    "IGM",    # iShares Expanded Tech Sector ETF
    "SMH",    # VanEck Semiconductor ETF
    "SOXX",   # iShares Semiconductor ETF
    "XSD",    # SPDR S&P Semiconductor ETF
    "TECL",   # Direxion Daily Technology Bull 3X Shares
    "FANG",   # AdvisorShares New Tech & Media ETF
    "SKYY",   # First Trust Cloud Computing ETF
    "BOTZ",   # Global X Robotics & AI ETF
    "ARKK",   # ARK Innovation ETF (tech-heavy)
    "PNQI",   # Invesco Nasdaq Internet ETF
    "FDN",    # First Trust Dow Jones Internet ETF
    "HACK",   # ETFMG Prime Cyber Security ETF
    "CIBR",   # First Trust NASDAQ Cybersecurity ETF
    "XLC",    # Communication Services Select Sector SPDR (tech-linked)
    "XNTK",   # SPDR NYSE Technology ETF
    "QTEC"    # First Trust NASDAQ-100 Technology Sector ETF
]

# Define period and interval for hourly data
period = "60d"  # Get more data for proper train-test split
interval = "1h"

# Define start and end dates for daily data
end_date_daily = datetime.now().strftime('%Y-%m-%d')
start_date_daily = (datetime.now() - timedelta(days=365)
                    ).strftime('%Y-%m-%d')  # Last 1 year

# --- Data Fetching and Initial Processing ---
print("\n--- Starting Data Fetching and Initial Processing ---")

# Check which tickers need to be fetched
missing_tickers = get_missing_tickers(tickers)
if missing_tickers:
    print(f"Fetching data for missing tickers: {missing_tickers}")
else:
    print("All tickers already have processed data. Using cached data.")

# Get all processed data efficiently (only fetches missing tickers)
all_processed_hourly_data = get_all_processed_data(tickers, period, interval)

print("\n--- All Data Fetching and Initial Processing Complete ---")

# Call the function to create the returns matrix
# Pass the dictionary of processed dataframes instead of re-reading from CSVs
returns_matrix_Xt = create_returns_matrix(
    tickers, period=period, interval=interval, processed_data=all_processed_hourly_data)

if not returns_matrix_Xt.empty:
    # You can save this matrix to a file if needed
    returns_matrix_Xt.to_csv("price_downloader/returns_matrix_Xt.csv")
    print("\nReturns matrix X_t saved to price_downloader/returns_matrix_Xt.csv")

    # --- PCA Analysis and Anomaly Detection ---
    print("\n--- Performing PCA Analysis and Anomaly Detection ---")

    # Data validation and preparation
    print("Preparing data for PCA...")
    # Ensure we have a clean numeric matrix
    X_t_for_pca = returns_matrix_Xt.T.copy()

    # Check for NaN values
    nan_count = X_t_for_pca.isna().sum().sum()
    if nan_count > 0:
        print(
            f"Warning: Found {nan_count} NaN values in the matrix. Dropping rows with NaN values.")
        X_t_for_pca = X_t_for_pca.dropna()
        print(f"Matrix shape after dropping NaN values: {X_t_for_pca.shape}")

    # Ensure all values are numeric
    X_t_for_pca = X_t_for_pca.astype(float)

    print(f"Final matrix shape for PCA: {X_t_for_pca.shape}")
    print(f"Matrix data type: {X_t_for_pca.dtypes.iloc[0]}")

    # 1. Perform standard PCA
    print("Performing standard PCA...")
    principal_components, scores, singular_values = perform_pca(X_t_for_pca)

    # Calculate explained variance
    explained_variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    cumulative_variance = np.cumsum(explained_variance)

    print(f"\nPCA Results:")
    print(f"Number of components: {len(singular_values)}")
    print(
        f"Explained variance by first 5 components: {explained_variance[:5]}")
    print(
        f"Cumulative variance by first 5 components: {cumulative_variance[:5]}")

    # Plot explained variance
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance) + 1),
            explained_variance, alpha=0.7)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.tight_layout()
    plt.savefig('pca_explained_variance.png')
    plt.close()
    print("PCA explained variance plot saved to pca_explained_variance.png")

    # 2. Perform Robust PCA with proper train-test split to avoid overfitting
    print("\n--- Performing Robust PCA with Train-Test Split ---")

    # Split data into training (first 70%) and testing (last 30%)
    train_size = int(len(X_t_for_pca) * 0.7)
    train_data = X_t_for_pca.iloc[:train_size]
    test_data = X_t_for_pca.iloc[train_size:]

    print(f"Training data shape: {train_data.shape}")
    print(f"Testing data shape: {test_data.shape}")

    # Train robust PCA on training data and get the model components
    print("Training Robust PCA on training data...")
    train_low_rank, train_sparse, train_singular_values, model_components = robust_pca(
        train_data, n_components=5, return_model=True)

    # Apply the trained model to testing data for proper out-of-sample testing
    print("Applying trained model to testing data...")
    test_low_rank, test_sparse = apply_robust_pca_model(
        test_data, model_components)

    # Combine results for full analysis
    low_rank_component = pd.concat([train_low_rank, test_low_rank])
    sparse_component = pd.concat([train_sparse, test_sparse])
    robust_singular_values = train_singular_values

    # Perform Robust PCA on full data for comparison (this is the old approach)
    print("\nPerforming Robust PCA on full data for comparison...")
    full_low_rank, full_sparse, full_singular_values = robust_pca(
        X_t_for_pca, n_components=5)

    # Calculate anomaly scores (absolute values of sparse component)
    anomaly_scores = np.abs(sparse_component.values)

    # Create anomaly heatmap
    plt.figure(figsize=(15, 10))

    # Create heatmap of anomaly scores
    sns.heatmap(anomaly_scores.T,
                xticklabels=50,  # Show fewer x-axis labels for readability
                yticklabels=returns_matrix_Xt.index,  # Asset names on y-axis
                cmap='Reds',
                cbar_kws={'label': 'Anomaly Score (Absolute Sparse Component)'})

    plt.title(
        'Anomaly Heatmap - Absolute Values of Sparse Component from Robust PCA')
    plt.xlabel('Time Observation Index')
    plt.ylabel('Assets')
    plt.tight_layout()
    plt.savefig('anomaly_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Anomaly heatmap saved to anomaly_heatmap.png")

    # 3. Identify top anomalies
    print("\nTop anomalies detected:")
    # Flatten the anomaly scores and get indices of top anomalies
    flat_anomalies = anomaly_scores.flatten()
    top_indices = np.argsort(flat_anomalies)[-10:][::-1]  # Top 10 anomalies

    # Convert flat indices to 2D indices
    time_indices, asset_indices = np.unravel_index(
        top_indices, anomaly_scores.shape)

    for i, (time_idx, asset_idx) in enumerate(zip(time_indices, asset_indices)):
        asset_name = returns_matrix_Xt.index[asset_idx]
        anomaly_value = flat_anomalies[top_indices[i]]
        print(
            f"Anomaly {i+1}: Asset '{asset_name}' at time index {time_idx}, score: {anomaly_value:.6f}")

    # 4. Save results
    sparse_component_df = pd.DataFrame(sparse_component,
                                       index=returns_matrix_Xt.columns,
                                       columns=returns_matrix_Xt.index)
    sparse_component_df.to_csv("robust_pca_sparse_component.csv")
    print("Sparse component (anomalies) saved to robust_pca_sparse_component.csv")

    low_rank_component_df = pd.DataFrame(low_rank_component,
                                         index=returns_matrix_Xt.columns,
                                         columns=returns_matrix_Xt.index)
    low_rank_component_df.to_csv("robust_pca_low_rank_component.csv")
    print("Low-rank component saved to robust_pca_low_rank_component.csv")

    # 5. Additional analysis: Compare training vs testing performance
    print(f"\n--- Additional Analysis ---")
    print(
        f"Training period: {X_t_for_pca.index[0]} to {X_t_for_pca.index[train_size-1]}")
    print(
        f"Testing period: {X_t_for_pca.index[train_size]} to {X_t_for_pca.index[-1]}")
    print("Successfully implemented proper out-of-sample testing!")
    print("Model trained on training data and applied to testing data to avoid overfitting.")

    # 6. Run Parameter Optimization
    print(f"\n--- Running Parameter Optimization ---")
    print("Finding optimal parameter combination with trade constraints...")

    # Run optimization to find the best parameters
    best_params = run_parameter_optimization(
        sparse_component,
        returns_matrix_Xt.T,
        interval
    )

    print(
        f"Optimal parameters found: alpha={best_params['alpha']}, quantile={best_params['quantile_threshold']}, ewma={best_params['ewma_span']}")
    print(f"Total trades generated: {best_params.get('total_trades', 'N/A')}")

    trading_positions = strategy(
        sparse_component, interval,
        alpha=best_params['alpha'],
        ewma_span=best_params['ewma_span'],
        quantile_threshold=best_params['quantile_threshold']
    )

    print("Running final backtest with 0.1% transaction costs...")
    portfolio_values, cumulative_returns, trades_df = backtest_strategy(
        trading_positions,
        returns_matrix_Xt.T,  # Transpose to match positions format
        transaction_cost=0.001,  # 0.1% transaction cost
        initial_capital=10000
    )

    print("Generating PnL chart...")
    plot_pnl_chart(portfolio_values, cumulative_returns, trades_df,
                   title=f"Optimized Robust PCA Strategy (alpha={best_params['alpha']}, quantile={best_params['quantile_threshold']}, ewma={best_params['ewma_span']})")

    print("Optimized backtest completed! PnL chart saved to strategy_pnl_chart.png")

else:
    print("\nCould not create returns matrix X_t.")
