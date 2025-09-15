import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from price_downloader import get_hourly_prices
from returns import calculate_log_returns
from create_matrix import create_returns_matrix
from PCA import robust_pca, get_last_n_days_submatrix
plt.style.use("default")  # ggplot is also fine
plt.rcParams["figure.figsize"] = (15, 12)


# Improved strategy function with adaptive volatility and dynamic thresholds
def strategy(sparse_component, interval, alpha=0.2, ewma_span=20, quantile_threshold=0.99):
    """
    Generates trading signals for all assets based on their sparse components with improved anomaly detection.

    Args:
        sparse_component (pd.DataFrame): The sparse component from Robust PCA (n_assets x T_observations).
        interval (str): The interval of the data (e.g., '1h', '1d').
        alpha (float): The proportionality constant for position sizing.
        ewma_span (int): Span for Exponentially Weighted Moving Average volatility estimation.
        quantile_threshold (float): Quantile threshold for anomaly detection (0.0 to 1.0).

    Returns:
        pd.DataFrame: DataFrame containing trading signals for all assets,
                      indexed by time and with columns as asset tickers.
    """
    all_positions = pd.DataFrame(index=sparse_component.columns)

    if interval.endswith('h'):
        hours_per_interval = int(interval[:-1])
        rows_per_day = 24 // hours_per_interval
        # Minimum data points for reliable estimation
        min_data_points = max(rows_per_day, 20)
    elif interval.endswith('d'):
        rows_per_day = int(interval[:-1])
        min_data_points = max(7 * rows_per_day, 20)
    elif interval.endswith('m'):
        minutes_per_interval = int(interval[:-1])
        rows_per_day = 1440 // minutes_per_interval
        min_data_points = max(rows_per_day // 7, 20)
    else:
        raise ValueError(
            "Unsupported interval format. Use 'Xh', 'Xd', or 'Xm'.")

    # Initialize positions DataFrame with time as index and assets as columns
    all_positions = pd.DataFrame(
        index=sparse_component.index, columns=sparse_component.columns)

    # Iterate through each asset (column) in the sparse_component
    # sparse_component has time as index and assets as columns
    for asset_ticker in sparse_component.columns:
        asset_sparse_component = sparse_component[asset_ticker]
        positions = []
        anomaly_thresholds = []  # Store dynamic thresholds for analysis

        # Calculate EWMA volatility for adaptive volatility estimation
        abs_anomalies = asset_sparse_component.abs()

        # Debug: Track if any trades are made for this asset
        trades_made = 0

        for T in range(len(asset_sparse_component)):
            if T < min_data_points:
                # Use simple std for initial period (avoid look-ahead)
                if T > 0:
                    current_volatility = asset_sparse_component.iloc[:T].std()
                else:
                    current_volatility = 10  # Default high value to avoid early trades
                threshold = 2 * current_volatility
            else:
                # Use both EWMA volatility and quantile threshold for more stable detection
                # EWMA provides smooth volatility estimation, quantile provides empirical bounds
                ewma_volatility = abs_anomalies.iloc[:T].ewm(
                    span=ewma_span).std().iloc[-1]
                historical_threshold = abs_anomalies.iloc[:T].quantile(
                    quantile_threshold)

                # Use a weighted average for more stable thresholds (70% quantile, 30% EWMA)
                threshold = 0.7 * historical_threshold + \
                    0.3 * (2 * ewma_volatility)

            anomaly_thresholds.append(threshold)

            # Generate trading signal
            current_anomaly = abs(asset_sparse_component.iloc[T])
            if current_anomaly > threshold:
                # Invert the signal: positive anomaly suggests short, negative suggests long
                # This is because unusually high returns may indicate overbought conditions
                # and unusually low returns may indicate oversold conditions
                position_size = -alpha * \
                    asset_sparse_component.iloc[T] / threshold

                # Clip position size to avoid over-leverage (max 20% per asset)
                max_position = 0.2  # Maximum 20% allocation per asset
                position_size = np.clip(
                    position_size, -max_position, max_position)

                trades_made += 1
                # Debug output for significant trades
                if current_anomaly > 0.01:  # Only show significant anomalies
                    print(
                        f"TRADE: {asset_ticker} at T={T}, anomaly={current_anomaly:.6f}, threshold={threshold:.6f}, position={position_size:.4f}")
            else:
                position_size = 0

            positions.append(position_size)

        # Debug output for each asset
        if trades_made > 0:
            print(f"Asset {asset_ticker}: {trades_made} trades made")
        else:
            max_anomaly = abs_anomalies.max()
            min_threshold = min(
                anomaly_thresholds) if anomaly_thresholds else 0
            print(
                f"Asset {asset_ticker}: No trades made (max anomaly: {max_anomaly:.6f}, min threshold: {min_threshold:.6f})")

        # Assign positions to the correct column in the DataFrame
        all_positions[asset_ticker] = positions

    return all_positions


# Additional function to analyze anomaly thresholds
def analyze_anomaly_thresholds(sparse_component, interval, ewma_span=20, quantile_threshold=0.99):
    """
    Analyzes and returns the dynamic anomaly thresholds for each asset.

    Args:
        sparse_component (pd.DataFrame): The sparse component from Robust PCA.
        interval (str): The data interval.
        ewma_span (int): Span for EWMA volatility estimation.
        quantile_threshold (float): Quantile threshold for anomaly detection.

    Returns:
        pd.DataFrame: DataFrame containing dynamic thresholds for each asset.
    """
    threshold_analysis = pd.DataFrame(index=sparse_component.columns)

    if interval.endswith('h'):
        hours_per_interval = int(interval[:-1])
        rows_per_day = 24 // hours_per_interval
        min_data_points = max(rows_per_day, 20)
    elif interval.endswith('d'):
        rows_per_day = int(interval[:-1])
        min_data_points = max(7 * rows_per_day, 20)
    elif interval.endswith('m'):
        minutes_per_interval = int(interval[:-1])
        rows_per_day = 1440 // minutes_per_interval
        min_data_points = max(rows_per_day // 7, 20)
    else:
        raise ValueError("Unsupported interval format.")

    for asset_ticker in sparse_component.index:
        asset_sparse_component = sparse_component.loc[asset_ticker]
        abs_anomalies = asset_sparse_component.abs()
        thresholds = []

        for T in range(len(asset_sparse_component)):
            if T < min_data_points:
                if T > 0:
                    current_volatility = asset_sparse_component.iloc[:T+1].std(
                    )
                else:
                    current_volatility = 10
                threshold = 2 * current_volatility
            else:
                ewma_volatility = abs_anomalies.iloc[:T +
                                                     1].ewm(span=ewma_span).std().iloc[-1]
                historical_threshold = abs_anomalies.iloc[:T+1].quantile(
                    quantile_threshold)
                threshold = max(2 * ewma_volatility, historical_threshold)

            thresholds.append(threshold)

        threshold_analysis[asset_ticker] = thresholds

    return threshold_analysis


def backtest_strategy(positions, returns_matrix, transaction_cost=0.003, initial_capital=10000):
    """
    Backtests the trading strategy and calculates PnL with transaction costs.

    Args:
        positions (pd.DataFrame): Trading positions from strategy function.
        returns_matrix (pd.DataFrame): Returns matrix (assets × time).
        transaction_cost (float): Transaction cost per trade (e.g., 0.003 for 0.3%).
        initial_capital (float): Initial capital for backtesting.

    Returns:
        tuple: (portfolio_values, cumulative_returns, trades_df)
    """
    # Ensure positions and returns have the same index and columns
    positions = positions.reindex(
        index=returns_matrix.index, columns=returns_matrix.columns, fill_value=0)

    # Calculate portfolio returns
    portfolio_returns = (positions.shift(1) * returns_matrix).sum(axis=1)

    # Calculate transaction costs
    position_changes = positions.diff().abs()
    transaction_costs = position_changes * transaction_cost

    # Net returns after transaction costs
    net_returns = portfolio_returns - transaction_costs.sum(axis=1)

    # Calculate cumulative returns
    cumulative_returns = (1 + net_returns.fillna(0)).cumprod() - 1

    # Calculate portfolio values
    portfolio_values = initial_capital * (1 + cumulative_returns)

    # Create trades dataframe for analysis
    trades_df = pd.DataFrame({
        'portfolio_returns': portfolio_returns,
        'transaction_costs': transaction_costs.sum(axis=1),
        'net_returns': net_returns,
        'cumulative_returns': cumulative_returns,
        'portfolio_value': portfolio_values
    })

    return portfolio_values, cumulative_returns, trades_df


def plot_pnl_chart(portfolio_values, cumulative_returns, trades_df, title="Strategy PnL"):
    """
    Plots the PnL chart for the backtested strategy.

    Args:
        portfolio_values (pd.Series): Portfolio values over time.
        cumulative_returns (pd.Series): Cumulative returns.
        trades_df (pd.DataFrame): Trades dataframe from backtest.
        title (str): Chart title.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot portfolio value
    ax1.plot(portfolio_values.index, portfolio_values,
             label='Portfolio Value', linewidth=2)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'{title} - Portfolio Value')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot cumulative returns
    ax2.plot(cumulative_returns.index, cumulative_returns *
             100, label='Cumulative Returns', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    ax2.set_ylabel('Cumulative Returns (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Cumulative Returns')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('strategy_pnl_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print performance statistics
    total_return = cumulative_returns.iloc[-1]
    annualized_return = (1 + total_return) ** (252 /
                                               len(cumulative_returns)) - 1
    volatility = trades_df['net_returns'].std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    print(f"\n--- Strategy Performance Statistics ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Final Portfolio Value: ${portfolio_values.iloc[-1]:.2f}")
    print(
        f"Maximum Drawdown: {calculate_max_drawdown(cumulative_returns):.2%}")
    print(f"Transaction Costs: ${trades_df['transaction_costs'].sum():.2f}")


def calculate_max_drawdown(cumulative_returns):
    """
    Calculates maximum drawdown from cumulative returns.
    """
    cumulative_values = 1 + cumulative_returns
    rolling_max = cumulative_values.expanding().max()
    drawdown = (cumulative_values - rolling_max) / rolling_max
    return drawdown.min()
