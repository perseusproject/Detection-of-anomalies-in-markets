import pandas as pd
import numpy as np


def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std_dev: int = 2):
    """
    Calculates Bollinger Bands for a given price series.

    Args:
        series (pd.Series): The input price series.
        window (int): The rolling window for mean and standard deviation calculation.
        num_std_dev (int): The number of standard deviations for the bands.

    Returns:
        pd.DataFrame: A DataFrame with 'Middle Band', 'Upper Band', and 'Lower Band'.
    """
    if series.empty:
        print("Warning: Series is empty for Bollinger Band calculation.")
        return pd.DataFrame()
    if len(series) < window:
        print(
            f"Warning: Series length ({len(series)}) is less than rolling window ({window}). Cannot calculate Bollinger Bands.")
        return pd.DataFrame()

    middle_band = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = middle_band + (rolling_std * num_std_dev)
    lower_band = middle_band - (rolling_std * num_std_dev)

    bollinger_bands = pd.DataFrame({
        'Middle Band': middle_band,
        'Upper Band': upper_band,
        'Lower Band': lower_band
    }, index=series.index)

    return bollinger_bands


def generate_signals(prices: pd.Series, bollinger_bands: pd.DataFrame, num_std_dev: int = 2):
    """
    Generates trading signals based on Bollinger Band strategy.
    Long if price is below Lower Band by more than num_std_dev, Short if above Upper Band by more than num_std_dev.

    Args:
        prices (pd.Series): The price series.
        bollinger_bands (pd.DataFrame): DataFrame containing 'Upper Band' and 'Lower Band'.
        num_std_dev (int): The number of standard deviations for the signal threshold.

    Returns:
        pd.Series: Series of trading signals (1 for long, -1 for short, 0 for hold).
    """
    signals = pd.Series(0, index=prices.index, dtype=int)

    if bollinger_bands.empty or 'Upper Band' not in bollinger_bands.columns or 'Lower Band' not in bollinger_bands.columns:
        print("Warning: Bollinger Bands DataFrame is incomplete or empty. Cannot generate signals.")
        return signals

    # Calculate the threshold for signals based on the standard deviation of the price series itself
    # This is to interpret "2σ" in "LLL de plus de 2σ" as a deviation from the band, not the band's width.
    # A more precise interpretation might involve the rolling_std used for the bands, but for simplicity
    # and to match the "2σ" phrasing for deviation from the band, we'll use a simple rolling std of prices.
    # Let's re-evaluate this. The user said "LLL de plus de 2σ". This implies the price is below the LLL,
    # and that deviation itself is more than 2 standard deviations of the price's movement.
    # For simplicity, let's assume the "2σ" refers to the standard deviation used in the Bollinger Band calculation.
    # So, if price < (Lower Band - 2 * rolling_std), then long.
    # If price > (Upper Band + 2 * rolling_std), then short.

    # Re-calculate rolling std for the signal threshold, as it's not directly passed in bollinger_bands df
    # Use a reasonable window, e.g., half the BB window
    rolling_std_for_signal = prices.rolling(
        window=bollinger_bands.shape[0] // 2).std()

    # Ensure indices align
    prices, bollinger_bands = prices.align(bollinger_bands, join='inner')

    # Calculate the rolling standard deviation used for the bands
    # This is a more accurate interpretation of "2σ" relative to the band's construction
    # We need to re-calculate the rolling_std from the original series for this.
    # Assuming the window for BB is 20, we'll use that for the rolling_std here.
    # Assuming 20 is the window used for BB
    bb_rolling_std = prices.rolling(window=20).std()

    # Long signal: Price is below the Lower Band by more than 2 * rolling_std
    signals[prices < (bollinger_bands['Lower Band'] -
                      num_std_dev * bb_rolling_std)] = 1
    # Short signal: Price is above the Upper Band by more than 2 * rolling_std
    signals[prices > (bollinger_bands['Upper Band'] +
                      num_std_dev * bb_rolling_std)] = -1

    # Fill NaN signals with 0 (hold)
    signals = signals.fillna(0)

    return signals


def backtest_strategy(prices: pd.Series, signals: pd.Series):
    """
    Performs a simple backtest of the trading strategy.

    Args:
        prices (pd.Series): The price series.
        signals (pd.Series): The generated trading signals (1 for long, -1 for short, 0 for hold).

    Returns:
        pd.DataFrame: DataFrame with daily returns, strategy returns, and cumulative returns.
    """
    if prices.empty or signals.empty:
        print("Warning: Prices or signals series is empty for backtesting.")
        return pd.DataFrame()

    # Ensure indices align
    prices, signals = prices.align(signals, join='inner')

    # Calculate daily returns of the asset
    daily_returns = prices.pct_change().fillna(0)

    # Calculate strategy returns
    # Strategy return = signal (position) * asset's daily return
    # Shift signals to avoid look-ahead bias
    strategy_returns = signals.shift(1) * daily_returns

    # Calculate cumulative returns
    cumulative_asset_returns = (1 + daily_returns).cumprod() - 1
    cumulative_strategy_returns = (1 + strategy_returns).cumprod() - 1

    results = pd.DataFrame({
        'Asset Returns': daily_returns,
        'Strategy Returns': strategy_returns,
        'Cumulative Asset Returns': cumulative_asset_returns,
        'Cumulative Strategy Returns': cumulative_strategy_returns
    }, index=prices.index)

    return results
