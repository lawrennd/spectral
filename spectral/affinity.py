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
    Construct RBF affinity matrix using standard exponentiated quadratic kernel.
    
    Computes A_ij = exp(-||x_i - x_j||^2 / (2*sigma^2)) using the standard
    exponentiated quadratic form.
    
    **NOTE ON MATLAB DIFFERENCE**: The original MATLAB implementation
    (SpectralCluster.m line 32) uses a non-standard parameterization:
    
        MATLAB:   exp(-||x_i - x_j||^2 / sigma)  
        Standard: exp(-||x_i - x_j||^2 / (2*sigma^2))  [CORRECT]
    
    Relationship between MATLAB and Python sigma values:
    - To match MATLAB sigma_m, use Python sigma = sqrt(sigma_m / 2)
    - Examples:
      - MATLAB sigma=0.05 → Python sigma=sqrt(0.05/2)≈0.158
      - MATLAB sigma=0.5  → Python sigma=sqrt(0.5/2)≈0.5
    
    The Python implementation uses the mathematically correct formulation
    where sigma represents the standard deviation of the Gaussian kernel,
    consistent with sklearn.metrics.pairwise.rbf_kernel and standard ML libraries.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data points
    sigma : float
        Standard deviation (bandwidth) for exponentiated quadratic kernel.
        Controls the locality of the similarity measure.
        Smaller values = more local, larger = more global.
        Typical values: 0.1 to 1.0 depending on data scale.
        
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
    MATLAB: SpectralCluster.m lines 28-34 (uses non-standard formula)
    sklearn.metrics.pairwise.rbf_kernel (uses standard formula)
    """
    # Compute pairwise squared Euclidean distances using scipy
    # cdist is optimized and handles the computation efficiently
    dists_sq = cdist(X, X, metric='sqeuclidean')
    
    # Apply standard RBF (Gaussian) kernel: exp(-d^2 / (2*sigma^2))
    # NOTE: MATLAB code uses exp(-d^2 / sigma) which is non-standard
    # The factor of 2 comes from the Gaussian probability density function
    A = np.exp(-dists_sq / (2.0 * sigma ** 2))
    
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
