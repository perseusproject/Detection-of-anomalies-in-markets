import pandas as pd
import numpy as np
from backtester import strategy, backtest_strategy
from sklearn.model_selection import TimeSeriesSplit
import itertools
from tqdm import tqdm


def optimize_parameters(sparse_component, returns_matrix, interval,
                        alpha_range=[0.1, 0.2, 0.3, 0.4, 0.5],
                        quantile_range=[0.95, 0.96, 0.97, 0.98, 0.99],
                        ewma_range=[10, 15, 20, 25, 30],
                        transaction_cost=0.001,
                        initial_capital=10000,
                        n_splits=5):
    """
    Optimize strategy parameters using time series cross-validation.

    Args:
        sparse_component (pd.DataFrame): Sparse component from Robust PCA
        returns_matrix (pd.DataFrame): Returns matrix for backtesting
        interval (str): Data interval ('1h', '1d', etc.)
        alpha_range (list): Range of alpha values to test
        quantile_range (list): Range of quantile thresholds to test
        ewma_range (list): Range of EWMA spans to test
        transaction_cost (float): Transaction cost per trade
        initial_capital (float): Initial capital for backtesting
        n_splits (int): Number of time series splits for cross-validation

    Returns:
        dict: Best parameters and optimization results
    """

    # Create time series cross-validation splits
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        alpha_range, quantile_range, ewma_range))

    results = []
    best_sharpe = -np.inf
    best_params = None

    print(
        f"Testing {len(param_combinations)} parameter combinations with {n_splits}-fold CV...")

    for alpha, quantile_threshold, ewma_span in tqdm(param_combinations, desc="Parameter Optimization"):
        fold_sharpes = []
        fold_returns = []

        for train_idx, test_idx in tscv.split(sparse_component):
            # Split data for this fold
            sparse_train = sparse_component.iloc[train_idx]
            sparse_test = sparse_component.iloc[test_idx]

            returns_train = returns_matrix.iloc[train_idx]
            returns_test = returns_matrix.iloc[test_idx]

            try:
                # Generate positions using training data
                positions = strategy(
                    sparse_train, interval,
                    alpha=alpha,
                    ewma_span=ewma_span,
                    quantile_threshold=quantile_threshold
                )

                # Backtest on test data
                portfolio_values, cumulative_returns, trades_df = backtest_strategy(
                    positions, returns_test,
                    transaction_cost=transaction_cost,
                    initial_capital=initial_capital
                )

                # Calculate performance metrics
                total_return = cumulative_returns.iloc[-1]
                if len(cumulative_returns) > 1:
                    annualized_return = (
                        1 + total_return) ** (252 / len(cumulative_returns)) - 1
                    volatility = trades_df['net_returns'].std() * np.sqrt(252)
                    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
                else:
                    sharpe_ratio = 0

                fold_sharpes.append(sharpe_ratio)
                fold_returns.append(total_return)

            except Exception as e:
                print(
                    f"Error with params alpha={alpha}, quantile={quantile_threshold}, ewma={ewma_span}: {e}")
                fold_sharpes.append(-np.inf)
                fold_returns.append(-1)

        # Calculate average performance across folds
        avg_sharpe = np.mean(fold_sharpes)
        avg_return = np.mean(fold_returns)

        results.append({
            'alpha': alpha,
            'quantile_threshold': quantile_threshold,
            'ewma_span': ewma_span,
            'avg_sharpe': avg_sharpe,
            'avg_return': avg_return,
            'fold_sharpes': fold_sharpes,
            'fold_returns': fold_returns
        })

        # Update best parameters
        if avg_sharpe > best_sharpe:
            best_sharpe = avg_sharpe
            best_params = {
                'alpha': alpha,
                'quantile_threshold': quantile_threshold,
                'ewma_span': ewma_span,
                'sharpe': avg_sharpe,
                'return': avg_return
            }

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)

    return {
        'best_params': best_params,
        'all_results': results_df,
        'param_combinations': param_combinations
    }


def grid_search_optimization(sparse_component, returns_matrix, interval,
                             alpha_range=[0.05, 0.1, 0.15, 0.2, 0.25],
                             quantile_range=[0.97, 0.98, 0.985, 0.99, 0.992],
                             ewma_range=[15, 20, 25, 30, 35],
                             min_trades_threshold=5):
    """
    Simple grid search optimization (faster but less robust than CV)
    """

    results = []
    best_sharpe = -np.inf
    best_params = None

    param_combinations = list(itertools.product(
        alpha_range, quantile_range, ewma_range))

    print(f"Grid search with {len(param_combinations)} combinations...")

    for alpha, quantile_threshold, ewma_span in tqdm(param_combinations, desc="Grid Search"):
        try:
            # Generate positions
            positions = strategy(
                sparse_component, interval,
                alpha=alpha,
                ewma_span=ewma_span,
                quantile_threshold=quantile_threshold
            )

            # Count number of trades (non-zero positions)
            total_trades = (positions != 0).sum().sum()

            # Skip parameter sets with too few trades
            if total_trades < min_trades_threshold:
                continue

            # Backtest
            portfolio_values, cumulative_returns, trades_df = backtest_strategy(
                positions, returns_matrix,
                transaction_cost=0.001,
                initial_capital=10000
            )

            # Calculate performance
            total_return = cumulative_returns.iloc[-1]
            if len(cumulative_returns) > 1:
                annualized_return = (
                    1 + total_return) ** (252 / len(cumulative_returns)) - 1
                volatility = trades_df['net_returns'].std() * np.sqrt(252)
                sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            else:
                sharpe_ratio = 0

            results.append({
                'alpha': alpha,
                'quantile_threshold': quantile_threshold,
                'ewma_span': ewma_span,
                'sharpe': sharpe_ratio,
                'total_return': total_return,
                'volatility': volatility if 'volatility' in locals() else 0,
                'total_trades': total_trades
            })

            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_params = {
                    'alpha': alpha,
                    'quantile_threshold': quantile_threshold,
                    'ewma_span': ewma_span,
                    'sharpe': sharpe_ratio,
                    'return': total_return,
                    'total_trades': total_trades
                }

        except Exception as e:
            print(
                f"Error with params alpha={alpha}, quantile={quantile_threshold}, ewma={ewma_span}: {e}")
            continue

    results_df = pd.DataFrame(results)

    return {
        'best_params': best_params,
        'all_results': results_df,
        'param_combinations': param_combinations
    }


def analyze_optimization_results(results):
    """
    Analyze and visualize optimization results
    """
    results_df = results['all_results']
    best_params = results['best_params']

    print("\n=== OPTIMIZATION RESULTS ===")
    print(
        f"Best parameters: alpha={best_params['alpha']}, quantile={best_params['quantile_threshold']}, ewma={best_params['ewma_span']}")
    print(f"Best Sharpe ratio: {best_params['sharpe']:.4f}")
    print(f"Best return: {best_params['return']:.2%}")
    print(f"Total trades: {best_params.get('total_trades', 'N/A')}")

    # Top 10 parameter combinations (include trade count)
    top_10 = results_df.nlargest(10, 'sharpe')
    print("\nTop 10 parameter combinations:")
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i}. alpha={row['alpha']}, quantile={row['quantile_threshold']}, ewma={row['ewma_span']}: "
              f"Sharpe={row['sharpe']:.4f}, Return={row['total_return']:.2%}, Trades={row.get('total_trades', 'N/A')}")

    return top_10


# Example usage:
if __name__ == "__main__":
    # This would be called from main.py after PCA analysis
    print("Parameter optimization module loaded")
