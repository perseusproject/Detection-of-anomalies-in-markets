"""
Optimized parameters from previous grid search optimization.
These parameters can be used directly without running optimization every time.
"""

# Best parameters from the most recent optimization
OPTIMIZED_PARAMETERS = {
    'alpha': 0.05,
    'quantile_threshold': 0.995,
    'ewma_span': 30
}

# Expanded parameter ranges for mean reversion optimization
EXPANDED_PARAMETER_RANGES = {
    'alpha_range': [0.05, 0.1, 0.15, 0.2, 0.25],
    'quantile_range': [0.97, 0.98, 0.985, 0.99, 0.992],
    'ewma_range': [15, 20, 25, 30, 35],
    'min_trades_threshold': 5
}

# Alternative parameter sets for different market conditions
PARAMETER_SETS = {
    'conservative': {
        'alpha': 0.05,
        'quantile_threshold': 0.995,
        'ewma_span': 30
    },
    'moderate': {
        'alpha': 0.1,
        'quantile_threshold': 0.99,
        'ewma_span': 25
    },
    'aggressive': {
        'alpha': 0.2,
        'quantile_threshold': 0.98,
        'ewma_span': 20
    }
}


def get_optimized_parameters(strategy_type='conservative'):
    """
    Get optimized parameters for the strategy.

    Args:
        strategy_type (str): Type of strategy - 'conservative', 'moderate', or 'aggressive'

    Returns:
        dict: Dictionary with optimized parameters
    """
    if strategy_type in PARAMETER_SETS:
        return PARAMETER_SETS[strategy_type]
    else:
        return OPTIMIZED_PARAMETERS


def save_optimized_parameters(parameters, filename='optimized_parameters.json'):
    """
    Save optimized parameters to a JSON file for future reference.

    Args:
        parameters (dict): Dictionary of optimized parameters
        filename (str): Name of the file to save parameters to
    """
    import json

    # Convert numpy types to native Python types for JSON serialization
    serializable_params = {}
    for key, value in parameters.items():
        if hasattr(value, 'item'):  # numpy types
            serializable_params[key] = value.item()
        else:
            serializable_params[key] = value

    with open(filename, 'w') as f:
        json.dump(serializable_params, f, indent=2)


def load_optimized_parameters(filename='optimized_parameters.json'):
    """
    Load optimized parameters from a JSON file.

    Args:
        filename (str): Name of the file to load parameters from

    Returns:
        dict: Dictionary with optimized parameters
    """
    import json
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return OPTIMIZED_PARAMETERS


def run_parameter_optimization(sparse_component, returns_matrix, interval):
    """
    Run parameter optimization and return best parameters.
    This function should be called from optimized_parameters.py to perform optimization.

    Args:
        sparse_component: Sparse component from PCA
        returns_matrix: Returns matrix for backtesting
        interval: Data interval

    Returns:
        dict: Optimized parameters
    """
    from parameter_optimizer import grid_search_optimization, analyze_optimization_results

    print("Running parameter optimization with expanded ranges...")

    optimization_results = grid_search_optimization(
        sparse_component,
        returns_matrix,
        interval,
        **EXPANDED_PARAMETER_RANGES
    )

    analyze_optimization_results(optimization_results)

    # Save the optimized parameters
    best_params = optimization_results['best_params']
    save_optimized_parameters(best_params)

    return best_params
