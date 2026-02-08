"""
Affinity matrix construction and Laplacian normalization.

Implements the affinity matrix and Laplacian operations from:
"Automatic Determination of the Number of Clusters using Spectral Algorithms"
(Sanguinetti, Laidler, Lawrence, 2005)
"""

import numpy as np
from scipy.spatial.distance import cdist


def build_affinity_matrix(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    Construct RBF affinity matrix.
    
    Computes A_ij = exp(-||x_i - x_j||^2 / sigma^2) as defined in the paper.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data points
    sigma : float
        Scale parameter for RBF kernel. Controls the locality of the
        similarity measure.
        
    Returns
    -------
    A : ndarray of shape (n_samples, n_samples)
        Symmetric affinity matrix with values in (0, 1]. Diagonal
        entries are 1.0 (distance from point to itself is 0).
        
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 0], [0, 1]])
    >>> A = build_affinity_matrix(X, sigma=1.0)
    >>> A[0, 0]  # Diagonal should be 1.0
    1.0
    >>> A[0, 1] == A[1, 0]  # Should be symmetric
    True
    
    References
    ----------
    Paper Section 2, Algorithm 1 step 1
    MATLAB: SpectralCluster.m lines 28-34
    """
    # Compute pairwise squared Euclidean distances using scipy
    # cdist is optimized and handles the computation efficiently
    dists_sq = cdist(X, X, metric='sqeuclidean')
    
    # Apply RBF (Gaussian) kernel: exp(-d^2 / sigma^2)
    # Note: MATLAB code uses sigma directly, which is sigma^2 in standard notation
    A = np.exp(-dists_sq / (sigma ** 2))
    
    return A


def normalize_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Normalize affinity matrix to form L = D^(-1/2) * A * D^(-1/2).
    
    This is the symmetric normalized Laplacian used in spectral clustering.
    The normalization accounts for different cluster densities.
    
    Parameters
    ----------
    A : ndarray of shape (n, n)
        Affinity matrix (must be non-negative and symmetric)
        
    Returns
    -------
    L : ndarray of shape (n, n)
        Normalized Laplacian matrix. L is symmetric positive definite
        with eigenvalues in [0, 1].
        
    Notes
    -----
    The degree matrix D has diagonal entries D_ii = sum_j A_ij
    (sum of row i). The normalization L = D^(-1/2) * A * D^(-1/2)
    ensures that points in sparse clusters aren't penalized.
    
    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[1.0, 0.5, 0.0],
    ...               [0.5, 1.0, 0.5],
    ...               [0.0, 0.5, 1.0]])
    >>> L = normalize_laplacian(A)
    >>> np.allclose(L, L.T)  # Should be symmetric
    True
    >>> eigvals = np.linalg.eigvalsh(L)
    >>> np.all((eigvals >= 0) & (eigvals <= 1 + 1e-10))  # Eigenvalues in [0,1]
    True
    
    References
    ----------
    Paper Section 2, Algorithm 1 step 1
    MATLAB: SpectralCluster.m line 36
    """
    # Compute degree matrix D (sum of each row)
    D = np.sum(A, axis=1)
    
    # Compute D^(-1/2) on the diagonal
    # Add small epsilon for numerical stability in case of isolated points
    D_inv_sqrt = 1.0 / np.sqrt(D + 1e-10)
    
    # Create diagonal matrix efficiently (avoids full matrix construction)
    # L = D^(-1/2) * A * D^(-1/2)
    # This is equivalent to: L[i,j] = A[i,j] / (sqrt(D[i]) * sqrt(D[j]))
    # We can do this with broadcasting:
    L = D_inv_sqrt[:, np.newaxis] * A * D_inv_sqrt[np.newaxis, :]
    
    return L
