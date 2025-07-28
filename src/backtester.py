import pandas as pd
import numpy as np
import backtrader as bt
import backtrader.analyzers as btanalyzers
import matplotlib.pyplot as plt
import os

from price_downloader import get_hourly_prices
from returns import calculate_log_returns
from create_matrix import create_returns_matrix
from PCA import robust_pca


# Existing strategy function
def strategy(matrix, sparse_component, j, alpha=1.0):
    # Position size is proportional to the distance of the sparse component from equilibrium (alpha * z_t)
    positions = []
    # Access the row for the j-th asset
    asset_sparse_component = sparse_component.iloc[j]

    # Calculate standard deviation of the sparse component
    sigma = asset_sparse_component.std()
    for T in range(len(asset_sparse_component)):  # Iterate through time observations
        # Take position only if the sparse component is over 2 * sigma
        if abs(asset_sparse_component.iloc[T]) > 2 * sigma:
            position_size = alpha * asset_sparse_component.iloc[T]
        else:
            position_size = 0  # No position if condition is not met
        positions.append(position_size)
    return pd.Series(positions, index=matrix.columns, name=f"Strategy_{matrix.index[j]}")


class AnomalousStrategy(bt.Strategy):
    """
    Backtesting strategy for detecting anomalies in asset prices.
    This strategy uses the strategy function to generate buy/sell signals based on the sparse component
    of the returns matrix.
    """
    params = (('returns_matrix', None), ('sparse_component', None),
              ('asset_index', None), ('strategy_signals', None), ('alpha', 1.0),)

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.strategy_signals = self.p.strategy_signals
        self.alpha = self.p.alpha

    def next(self):
        if self.order:
            return

        current_timestamp = self.data.datetime.datetime()
        signal = self.strategy_signals.asof(current_timestamp)

        if pd.isna(signal):
            signal = 0.0  # Default to no position if no signal

        # Calculate target position size
        target_position_size = signal

        # Current position size
        current_position_size = self.position.size

        # Determine action based on target vs. current position
        if target_position_size > 0:  # Target is long
            if current_position_size < 0:  # Currently short, close short first
                self.close()
            if current_position_size < target_position_size:  # Need to buy more or open long
                self.buy(size=target_position_size - current_position_size)
        elif target_position_size < 0:  # Target is short
            if current_position_size > 0:  # Currently long, close long first
                self.close()
            if current_position_size > target_position_size:  # Need to sell more or open short
                self.sell(size=abs(target_position_size - current_position_size))
        else:  # Target position is 0 (flat)
            if current_position_size != 0:  # If any position is open, close it
                self.close()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy:
                # print(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                pass
            elif order.issell:
                # print(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                pass

            # Record the bar where the order was executed
            self.bar_executed = len(self)
            self.order = None  # Clear the order reference
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # print(f'Order Canceled/Margin/Rejected: {order.Status[order.status]}')
            self.order = None  # Clear the order reference


def run_backtest(ticker_symbol, period, interval, returns_matrix, sparse_component, asset_index, alpha=1.0):
    """
    Runs a backtest for a given ticker using the AnomalousStrategy.

    Args:
        ticker_symbol (str): The ticker symbol for the asset to backtest.
        period (str): The period for the backtest (used for plot filename).
        interval (str): The interval for the backtest (used for plot filename).
        returns_matrix (pd.DataFrame): The full returns matrix.
        sparse_component (pd.DataFrame): The sparse component from Robust PCA.
        asset_index (int): The integer index of the asset in the returns_matrix.
        alpha (float): The proportionality constant for position sizing.
    """
    print(f"\n--- Running Backtest for {ticker_symbol} ---")

    # 1. Load processed hourly data for the specific ticker
    processed_file_path = os.path.join(
        os.getcwd(), "price_downloader", f"{ticker_symbol}_processed_hourly_prices.csv")

    try:
        # Load the processed hourly data, assuming it's now standard with 'datetime' as index
        hourly_data = pd.read_csv(
            processed_file_path,
            index_col='datetime',  # 'datetime' is now the index column name
            parse_dates=True,
            # Corrected: Remove %z as data is now timezone-naive
            date_format='%Y-%m-%d %H:%M:%S'
        )

    except FileNotFoundError:
        print(
            f"Error: Processed hourly data file not found for {ticker_symbol} at {processed_file_path}. Cannot run backtest.")
        return
    except Exception as e:
        print(
            f"Error loading processed data for {ticker_symbol}: {e}. Cannot run backtest.")
        return

    if hourly_data.empty:
        print(
            f"Error: Processed hourly data for {ticker_symbol} is empty. Cannot run backtest.")
        return

    # Ensure 'Close' column is numeric and handle potential NaNs
    if 'Close' in hourly_data.columns:
        hourly_data['Close'] = pd.to_numeric(
            hourly_data['Close'], errors='coerce')
        hourly_data.dropna(subset=['Close'], inplace=True)
    else:
        print(
            f"Error: 'Close' column not found in processed data for {ticker_symbol}. Cannot run backtest.")
        return

    # Get the first date from the hourly_data index
    # Ensure it's timezone-naive if backtrader prefers it that way for fromdate
    first_date = hourly_data.index[0]
    if first_date.tz is not None:
        first_date = first_date.tz_localize(None)

    # Create a backtrader data feed
    data = bt.feeds.PandasData(
        dataname=hourly_data,
        datetime=None,  # Datetime is already the index
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=None,
        # Corrected: Use bt.TimeFrame.Minutes for hourly data
        timeframe=bt.TimeFrame.Minutes,
        compression=60,  # 60 minutes = 1 hour
        fromdate=first_date  # Start processing from the first date in the data
    )

    # Generate strategy signals for the specific asset
    signals = strategy(returns_matrix, sparse_component,
                       asset_index, alpha=alpha)

    # Initialize Cerebro
    cerebro = bt.Cerebro()

    # Add data feed
    cerebro.adddata(data)

    # Add the strategy
    cerebro.addstrategy(AnomalousStrategy,
                        returns_matrix=returns_matrix,
                        sparse_component=sparse_component,
                        asset_index=asset_index,
                        strategy_signals=signals,
                        alpha=alpha)

    # Set starting cash
    cerebro.broker.setcash(100000.0)

    # Add analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='tradeanalyzer')
    cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
    cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Run the backtest
    results = cerebro.run()

    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # Print analyzer results
    print("\n--- Backtest Results ---")
    if results and results[0].analyzers.sharpe:
        sharpe_ratio = results[0].analyzers.sharpe.get_analysis().get(
            'sharperatio')
        print(
            f"Sharpe Ratio: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "Sharpe Ratio: N/A")
    if results and results[0].analyzers.drawdown:
        max_drawdown = results[0].analyzers.drawdown.get_analysis().get(
            'max', {}).get('drawdown')
        print(
            f"Max Drawdown: {max_drawdown:.2f}%" if max_drawdown is not None else "Max Drawdown: N/A")
    if results and results[0].analyzers.returns:
        total_return = results[0].analyzers.returns.get_analysis().get('rtot')
        print(
            f"Total Return: {total_return:.2%}" if total_return is not None else "Total Return: N/A")
    if results and results[0].analyzers.sqn:
        sqn_value = results[0].analyzers.sqn.get_analysis().get('sqn')
        print(
            f"System Quality Number (SQN): {sqn_value:.2f}" if sqn_value is not None else "System Quality Number (SQN): N/A")

    # Plot the results
    print("\n--- Plotting Backtest Results ---")
    fig = cerebro.plot(style='candlestick', iplot=False, numfigs=1)[0][0]

    # # Define filename for PnL graph
    # pnl_filename = os.path.join("Figures", f"{period}{interval}PnL.png")

    # # Check if the PnL file already exists before saving
    # if not os.path.exists(pnl_filename):
    #     fig.savefig(pnl_filename)
    #     print(f"PnL graph saved to {pnl_filename}")
    # else:
    #     print(
    #         f"PnL graph file already exists at {pnl_filename}, skipping save.")
