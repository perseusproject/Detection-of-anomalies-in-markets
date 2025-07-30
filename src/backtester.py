import pandas as pd
import numpy as np
import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib.pyplot as plt
import os

from price_downloader import get_hourly_prices
from returns import calculate_log_returns
from create_matrix import create_returns_matrix
from PCA import robust_pca, get_last_n_days_submatrix
plt.style.use("default")  # ggplot is also fine
plt.rcParams["figure.figsize"] = (15, 12)


# Existing strategy function
def strategy(sparse_component, interval, alpha=1.0):
    """
    Generates trading signals for all assets based on their sparse components.

    Args:
        sparse_component (pd.DataFrame): The sparse component from Robust PCA (n_assets x T_observations).
        interval (str): The interval of the data (e.g., '1h', '1d').
        alpha (float): The proportionality constant for position sizing.

    Returns:
        pd.DataFrame: DataFrame containing trading signals for all assets,
                      indexed by time and with columns as asset tickers.
    """
    all_positions = pd.DataFrame(index=sparse_component.columns)

    if interval.endswith('h'):
        hours_per_interval = int(interval[:-1])
        rows_per_day = 24 // hours_per_interval
        lookback_rows = rows_per_day  # 1 day lookback
    elif interval.endswith('d'):
        rows_per_day = int(interval[:-1])
        lookback_rows = 7 * rows_per_day  # 7 days lookback
    elif interval.endswith('m'):
        minutes_per_interval = int(interval[:-1])
        rows_per_day = 1440 // minutes_per_interval
        lookback_rows = rows_per_day // 7  # 1/7th of a day lookback
    else:
        raise ValueError(
            "Unsupported interval format. Use 'Xh', 'Xd', or 'Xm'.")

    # Iterate through each asset (row) in the sparse_component
    for asset_ticker in sparse_component.index:
        asset_sparse_component = sparse_component.loc[asset_ticker]
        positions = []

        for T in range(len(asset_sparse_component)):  # Iterate through time observations
            start_index = max(0, T - lookback_rows + 1)

            if T >= lookback_rows - 1:
                sigma = asset_sparse_component.iloc[start_index: T + 1].std()
            else:
                if T > 0:
                    sigma = asset_sparse_component.iloc[0: T + 1].std()
                else:
                    sigma = 10  # Default if not enough data, high enough to avoid trades

            if abs(asset_sparse_component.iloc[T]) > 2*sigma:
                position_size = alpha * asset_sparse_component.iloc[T]
            else:
                position_size = 0
            positions.append(position_size)

        all_positions[asset_ticker] = positions

    return all_positions  # Return with time as index and assets as columns
