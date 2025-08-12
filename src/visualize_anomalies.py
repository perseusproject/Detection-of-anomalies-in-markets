import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from create_matrix import create_returns_matrix
from PCA import perform_pca
from returns import normalize_series_rolling_zscore


def plot_common_factor_vs_residual_misalignment(tickers, period="7d", interval="1h"):
    """
    Generates a plot showing the common market factor and individual stock residuals
    after removing the market factor, highlighting anomaly zones.
    Also includes an inset bar chart of explained variance by principal components.

    Args:
        tickers (list): List of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA']).
        period (str): Data period for hourly prices (e.g., '7d' for a week).
        interval (str): Data interval (e.g., '1h').
    """
    print(
        f"--- Generating Common Factor vs Residual Misalignment Plot for {tickers} ---")

    # 1. Create returns matrix
    # The create_returns_matrix function returns X_t_matrix which is n x T (assets x observations)
    # For PCA, we typically want T observations x n features, so we'll use the transpose of the output
    # or ensure PCA function handles it correctly. Let's assume PCA expects features as columns.
    # create_returns_matrix already transposes to n x T, so for PCA where features are columns,
    # we need T x n. So we will transpose it back.
    returns_matrix_Xt = create_returns_matrix(
        tickers, period=period, interval=interval)

    if returns_matrix_Xt.empty:
        print("Could not create returns matrix. Exiting plot generation.")
        return

    # Transpose to get T observations x n assets (features as columns for PCA)
    returns_matrix = returns_matrix_Xt.T
    print(f"Returns matrix shape for PCA (T x n): {returns_matrix.shape}")

    # 2. Perform PCA
    # perform_pca returns (principal_components, scores, singular_values)
    # scores will be T x n_components
    # singular_values are related to explained variance
    # We need to ensure n_components is at least 4 for the user's request
    # Use min to avoid error if less than 4 stocks
    n_components_for_common_factor = min(3, returns_matrix.shape[1])

    principal_components, scores, singular_values = perform_pca(returns_matrix)

    # Calculate explained variance ratio
    explained_variance = (singular_values ** 2) / np.sum(singular_values ** 2)
    print(f"Explained variance by principal components: {explained_variance}")

    # 3. Extract the common factor (from first n_components)
    # Reconstruct the common factor's contribution from the first n_components
    # scores_k is T x k, pc_vectors_k is k x n
    # common_factor_contribution = scores[:, :n_components_for_common_factor] @ principal_components[:n_components_for_common_factor, :]
    # The common factor itself is the sum of the contributions of these components to each stock.
    # A simpler way to get the "common factor" as a single time series is to take the first PC's scores,
    # but if the user wants "more principal components for common factor", they likely mean
    # the combined effect of these components.
    # Let's define the common factor as the sum of the first n_components scores.
    # This is a common way to represent a multi-factor common component.
    common_factor_scores = scores.iloc[:,
                                       :n_components_for_common_factor].sum(axis=1)
    print(
        f"Common factor scores shape (sum of first {n_components_for_common_factor} PCs): {common_factor_scores.shape}")

    # 4. Calculate residuals
    # Reconstruct the full common factor contribution to the original data
    # This is scores[:, :k] @ principal_components[:k, :]
    common_factor_contribution_matrix = scores.iloc[:,
                                                    :n_components_for_common_factor] @ principal_components.iloc[:n_components_for_common_factor, :]
    common_factor_contribution_df = pd.DataFrame(
        common_factor_contribution_matrix, index=returns_matrix.index, columns=returns_matrix.columns)
    print(
        f"Common factor contribution matrix shape: {common_factor_contribution_df.shape}")

    # Residuals are original returns minus the common factor contribution
    residuals = returns_matrix - common_factor_contribution_df
    print(f"Residuals matrix shape: {residuals.shape}")

    # 5. Normalize the residuals and the common factor
    # Normalize each residual series and the common factor using rolling z-score
    normalized_residuals = pd.DataFrame(index=residuals.index)
    for col in residuals.columns:
        normalized_residuals[col] = normalize_series_rolling_zscore(
            residuals[col])

    normalized_common_factor = normalize_series_rolling_zscore(
        common_factor_scores)
    print(f"Normalized residuals shape: {normalized_residuals.shape}")
    print(f"Normalized common factor shape: {normalized_common_factor.shape}")

    # Drop NaN values introduced by rolling z-score calculation
    normalized_residuals = normalized_residuals.dropna()
    normalized_common_factor = normalized_common_factor.dropna()  # Corrected variable name

    # Align indices after dropping NaNs
    common_index = normalized_residuals.index.intersection(
        normalized_common_factor.index)  # Corrected variable name
    normalized_residuals = normalized_residuals.loc[common_index]
    # Corrected variable name
    normalized_common_factor = normalized_common_factor.loc[common_index]

    if normalized_residuals.empty or normalized_common_factor.empty:  # Corrected variable name
        print("Not enough data after normalization to plot. Exiting.")
        return

    # 6. Plotting
    # Removed 'seaborn-v0_8-darkgrid' style for a plain background
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Grey thick line: Common factor (sum of first 4 PCs)
    ax1.plot(normalized_common_factor.index, normalized_common_factor,
             color='grey', linewidth=3, label='Common Factor (Sum of first 4 PCs)')

    # Plot Thin colored lines: Actual z-scores of each stock’s returns after removing the market factor
    colors = sns.color_palette("tab10", len(tickers))
    for i, ticker in enumerate(tickers):
        if ticker in normalized_residuals.columns:
            ax1.plot(normalized_residuals.index, normalized_residuals[ticker],
                     color=colors[i], linewidth=1, label=f'{ticker} Residual Z-score')

            # Shading / markers: Highlight points where a given stock’s residual exceeds ±2σ
            # Assuming ±2σ is relative to the normalized z-score, so it's just ±2
            anomalies_pos = normalized_residuals[ticker][normalized_residuals[ticker] > 2]
            anomalies_neg = normalized_residuals[ticker][normalized_residuals[ticker] < -2]

            ax1.scatter(anomalies_pos.index, anomalies_pos,
                        color=colors[i], marker='^', s=50, zorder=5, label=f'{ticker} > +2σ' if not anomalies_neg.empty else "")
            ax1.scatter(anomalies_neg.index, anomalies_neg,
                        color=colors[i], marker='v', s=50, zorder=5, label=f'{ticker} < -2σ' if not anomalies_pos.empty else "")
        else:
            print(
                f"Warning: {ticker} not found in normalized residuals. Skipping plot for this ticker.")

    # Removed "Hourly over a Week"
    ax1.set_title('Common Factor vs Residual Misalignment')
    ax1.set_xlabel('')  # Removed x-axis label
    ax1.set_ylabel('Normalized Z-score of Returns')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, linestyle='--', alpha=0.7)
    # Removed x-axis ticks and rotation for date captions
    plt.tight_layout()

    # Removed the inset bar chart for explained variance

    plt.show()
    print("Plot generation complete.")


if __name__ == "__main__":
    # Example usage:
    stocks = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'ADBE', 'NFLX',
              'ORCL', 'CRM', 'INTC', 'AMD', 'QCOM', 'TXN', 'AVGO', 'CSCO', 'IBM', 'SHOP', 'PYPL']
    plot_common_factor_vs_residual_misalignment(stocks)
