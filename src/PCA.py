import pandas as pd
import numpy as np
from numpy.linalg import svd


def perform_pca_on_svd(matrix: pd.DataFrame, n_components: int = None):
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

    # Transformed data (scores) are U * S
    # U has orthogonal columns, S are singular values
    scores = pd.DataFrame(U @ np.diag(S), index=matrix.index)

    if n_components is not None:
        principal_components = principal_components.iloc[:n_components, :]
        scores = scores.iloc[:, :n_components]
        S = S[:n_components]

    return principal_components, scores, S


def robust_pca(matrix: pd.DataFrame, n_components: int = None):
    """
    Performs robust PCA on a matrix using SVD. It means splitting the matrix into low-rank and sparse components.

    Args:
        matrix (pd.DataFrame): The input matrix (e.g., returns matrix).
        n_components (int, optional): The number of principal components to keep.
                                       If None, all components are kept.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The low-rank component (principal components).
            - pd.DataFrame: The sparse component (anomalies).
            - np.array: The singular values (related to explained variance).
    """
