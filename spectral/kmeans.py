"""
Elongated k-means clustering with Mahalanobis distance.

Implements the elongated k-means algorithm from:
"Automatic Determination of the Number of Clusters using Spectral Algorithms"
(Sanguinetti, Laidler, Lawrence, 2005)
"""

import numpy as np


def mahalanobis_distance(
    x: np.ndarray,
    center: np.ndarray,
    lambda_: float = 0.2,
    epsilon: float = 0.0001
) -> np.ndarray:
    """
    Compute elongated (Mahalanobis) distance from points to center.
    
    Implements the distance metric from equation (1) of the paper.
    For centers far from origin, uses elongated distance metric that
    downweights radial direction and upweights transverse direction.
    For centers near origin, uses standard Euclidean distance.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_dims)
        Data points
    center : ndarray of shape (n_dims,)
        Cluster center
    lambda_ : float, default=0.2
        Sharpness parameter controlling elongation. Smaller values create
        more elongated clusters (more downweighting of radial distances).
        The paper uses 0.2 for all experiments.
    epsilon : float, default=0.0001
        Threshold for determining if center is near origin. Centers with
        ||c||^2 < epsilon use Euclidean distance instead.
        
    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Squared distances from points to center
        
    Notes
    -----
    The elongated distance metric is:
        d(x, c) = (x - c)^T * M * (x - c)
    where:
        M = (1/λ) * (I - cc^T/||c||^2) + λ * (cc^T/||c||^2)
        
    This downweights distances along the radial direction (towards origin)
    and upweights distances in transverse directions, creating elongated
    clusters aligned radially.
    
    Examples
    --------
    >>> x = np.array([[1, 0], [2, 0], [0, 1]])
    >>> center = np.array([1.5, 0])
    >>> distances = mahalanobis_distance(x, center, lambda_=0.2)
    
    References
    ----------
    Paper: Section 3, Equation (1)
    MATLAB: mahDist2.m
    """
    # Check if center is far enough from origin
    center_norm_sq = np.dot(center, center)
    
    if center_norm_sq > epsilon:
        # Center is far from origin: use elongated distance
        n_dims = len(center)
        
        # Compute the radial projection matrix: cc^T / ||c||^2
        radial_proj = np.outer(center, center) / center_norm_sq
        
        # M = (1/lambda) * (I - radial_proj) + lambda * radial_proj
        # This downweights radial direction (I - radial_proj) by 1/lambda
        # and upweights radial direction (radial_proj) by lambda
        M = (1.0 / lambda_) * (np.eye(n_dims) - radial_proj) + lambda_ * radial_proj
        
        # Compute (x - c)^T * M * (x - c) for each point
        diff = x - center  # shape (n_samples, n_dims)
        # Use einsum for efficient computation
        distances = np.einsum('ij,jk,ik->i', diff, M, diff)
        
        # Handle rounding errors that may cause negative distances
        distances = np.maximum(distances, 0.0)
    else:
        # Center is near origin: use Euclidean distance
        diff = x - center
        distances = np.sum(diff ** 2, axis=1)
    
    return distances


class ElongatedKMeans:
    """
    K-means with elongated distance metric for spectral clustering.
    
    Based on the algorithm in "Automatic Determination of the Number
    of Clusters using Spectral Algorithms" (Sanguinetti et al., 2005).
    
    The elongated k-means uses a Mahalanobis distance that downweights
    distances along radial directions (towards the origin) and upweights
    transverse distances, making it suitable for clustering data with
    radial structure in eigenspace.
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    lambda_ : float, default=0.2
        Sharpness parameter controlling elongation. Smaller values
        create more elongated clusters.
    epsilon : float, default=0.0001
        Threshold for origin proximity. Centers with ||c||^2 < epsilon
        use Euclidean distance instead of elongated distance.
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence (maximum center movement)
        
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers
    labels_ : ndarray of shape (n_samples,)
        Labels of each point
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center
    n_iter_ : int
        Number of iterations run
        
    References
    ----------
    Paper: Section 3, "elongated K-means"
    MATLAB: mahKmeans.m, mahDist2.m
    """
    
    def __init__(self, n_clusters: int, lambda_: float = 0.2,
                 epsilon: float = 0.0001, max_iter: int = 100,
                 tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X: np.ndarray, init_centers: np.ndarray) -> 'ElongatedKMeans':
        """
        Fit elongated k-means on X with provided initial centers.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        init_centers : ndarray of shape (n_clusters, n_features)
            Initial cluster centers
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        n_samples = X.shape[0]
        centers = init_centers.copy()
        
        old_error = np.inf
        
        for iteration in range(self.max_iter):
            # Save old centers for convergence check
            old_centers = centers.copy()
            
            # Compute distances from all points to all centers
            distances = np.zeros((n_samples, self.n_clusters))
            for j in range(self.n_clusters):
                distances[:, j] = mahalanobis_distance(
                    X, centers[j], self.lambda_, self.epsilon
                )
            
            # Assign each point to nearest center
            labels = np.argmin(distances, axis=1)
            min_distances = np.min(distances, axis=1)
            
            # Update centers as mean of assigned points
            for j in range(self.n_clusters):
                points_in_cluster = X[labels == j]
                if len(points_in_cluster) > 0:
                    centers[j] = np.mean(points_in_cluster, axis=0)
                # If no points assigned, keep the center where it is
            
            # Compute error (sum of squared distances)
            error = np.sum(min_distances)
            
            # Check for convergence
            center_shift = np.max(np.abs(centers - old_centers))
            error_change = np.abs(error - old_error)
            
            if center_shift < self.tol and error_change < self.tol:
                break
            
            old_error = error
        
        # Store results
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = error
        self.n_iter_ = iteration + 1
        
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels (indices of closest centers)
        """
        # Compute distances to all centers
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        for j in range(self.n_clusters):
            distances[:, j] = mahalanobis_distance(
                X, self.cluster_centers_[j], self.lambda_, self.epsilon
            )
        
        # Return index of nearest center
        return np.argmin(distances, axis=1)
