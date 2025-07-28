import pandas as pd
import numpy as np


def strategy(matrix, sparse_component, j):
    # simple strategy : return long if indice is under L from more than 2\sigma, else short
    L = []
    # Access the row for the j-th asset
    asset_returns = matrix.iloc[j]
    asset_sparse_component = sparse_component.iloc[j]

    # Calculate std dev of returns for this asset
    sigma = np.std(asset_returns)
    for T in range(len(asset_returns)):  # Iterate through time observations
        # Access specific time point
        if asset_sparse_component.iloc[T] > .2*sigma:
            L.append(1)
        # Access specific time point
        elif asset_sparse_component.iloc[T] < -.2*sigma:
            L.append(-1)
        else:
            L.append(0)
    # Index by timestamps, name by ticker
    return pd.Series(L, index=matrix.columns, name=f"Strategy_{matrix.index[j]}")


# def calculate_pnl(returns_series, strategy_signals):
#     """
#     Calculates the Profit and Loss (PnL) based on log returns and strategy signals.

#     Args:
#         returns_series (pd.Series): Series of log returns for an asset.
#         strategy_signals (pd.Series): Series of trading signals (1 for long, -1 for short, 0 for neutral).

#     Returns:
#         pd.Series: Cumulative PnL.
#     """
#     # Ensure signals and returns are aligned by index
#     aligned_returns, aligned_signals = returns_series.align(
#         strategy_signals, join='inner')

#     # PnL for each period: signal * return
#     daily_pnl = aligned_signals * aligned_returns

#     # Cumulative PnL
#     cumulative_pnl = daily_pnl.cumsum()
#     return cumulative_pnl
