import pandas as pd
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet


def perform_pca(matrix: pd.DataFrame, n_components: int = None):
    """
    Performs PCA on a matrix using its SVD decomposition.

    Args:
        matrix (pd.DataFrame): The input matrix (e.g., returns matrix).
        n_components (int, optional): The number of principal components to keep.
                                      If None, all components are kept.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The principal components (eigenvectors).
            - pd.DataFrame: The transformed data (scores).
            - np.array: The singular values (related to explained variance).
    """
    # Center the data (PCA is typically performed on centered data)
    centered_matrix = matrix - matrix.mean(axis=0)

    # Perform SVD on the centered matrix
    U, S, Vt = svd(centered_matrix, full_matrices=False)

    # Principal Components (eigenvectors) are the rows of Vt
    # Each row of Vt corresponds to a principal component
    principal_components = pd.DataFrame(Vt, columns=matrix.columns)

    # Transformed data (scores) are U @ np.diag(S)
    # U has orthogonal columns, S are singular values
    scores = pd.DataFrame(U @ np.diag(S), index=matrix.index)

    if n_components is not None:
        principal_components = principal_components.iloc[:n_components, :]
        scores = scores.iloc[:, :n_components]
        S = S[:n_components]

    return principal_components, scores, S


def robust_pca(matrix: pd.DataFrame, n_components: int = None):
    """
    Performs robust PCA on a matrix using a robust estimator and splits the matrix into low-rank and sparse components.

    Args:
        matrix (pd.DataFrame): The input matrix (e.g., returns matrix).
        n_components (int, optional): The number of principal components to keep for the low-rank part.
                                       If None, all components are kept.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The low-rank component.
            - pd.DataFrame: The sparse component (residuals/anomalies).
            - np.array: The singular values from the PCA of the robustly scaled data.
    """
    # 1. Robustly scale the data
    scaler = RobustScaler()
    scaled_matrix_np = scaler.fit_transform(matrix)
    scaled_matrix = pd.DataFrame(
        scaled_matrix_np, index=matrix.index, columns=matrix.columns)

    # 2. Perform PCA on the robustly scaled data
    pca = PCA(n_components=n_components)

    # Fit PCA and transform the scaled data to get the scores (low-rank representation)
    scores_scaled_np = pca.fit_transform(scaled_matrix)

    # Reconstruct the low-rank component in the scaled space
    low_rank_scaled_np = pca.inverse_transform(scores_scaled_np)

    # Inverse transform to get the low-rank component in the original data scale
    low_rank_component = pd.DataFrame(scaler.inverse_transform(low_rank_scaled_np),
                                      index=matrix.index,
                                      columns=matrix.columns)

    # 3. Calculate the sparse component as the residual
    sparse_component = matrix - low_rank_component

    # Singular values from the PCA
    singular_values = pca.singular_values_

    return low_rank_component, sparse_component, singular_values


def get_last_n_days_submatrix(matrix: pd.DataFrame, n_days: int, interval: str):
    """
    Extracts a submatrix containing only the values from the last n days,
    considering the data interval.

    Args:
        matrix (pd.DataFrame): The input matrix (e.g., sparse component).
        n_days (int): The number of last days to extract.
        interval (str): The data interval (e.g., "1h", "1d").

    Returns:
        pd.DataFrame: The submatrix containing values from the last n days.
    """
    if interval.endswith('h'):
        hours_per_interval = int(interval[:-1])
        rows_per_day = 24 // hours_per_interval
    elif interval.endswith('d'):
        rows_per_day = int(interval[:-1])  # Assuming '1d' means 1 row per day
    else:
        raise ValueError("Unsupported interval format. Use 'Xh' or 'Xd'.")

    total_rows = n_days * rows_per_day
    return matrix.tail(total_rows)
