"""
Main SpectralCluster class implementing the automatic cluster detection algorithm.
"""

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state

from .affinity import build_affinity_matrix, normalize_laplacian
from .kmeans import ElongatedKMeans


class SpectralCluster(BaseEstimator, ClusterMixin):
    """
    Spectral clustering with automatic cluster number determination.
    
    Implementation of the algorithm from "Automatic Determination of the Number
    of Clusters using Spectral Algorithms" (Sanguinetti, Laidler, Lawrence, 2005).
    
    Parameters
    ----------
    sigma : float
        Scale parameter for RBF kernel in affinity matrix
    lambda_ : float, default=0.2
        Elongation parameter for distance metric in k-means
    max_clusters : int, default=10
        Maximum number of eigenvectors to compute
    epsilon : float, default=0.0001
        Threshold for determining if center is near origin
    random_state : int or None, default=None
        Random seed for reproducibility
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample
    n_clusters_ : int
        Number of clusters found
    centers_ : ndarray
        Cluster centers in eigenspace
    eigenvectors_ : ndarray
        Eigenvectors used for clustering
    eigenvalues_ : ndarray
        Corresponding eigenvalues
    affinity_matrix_ : ndarray
        The affinity matrix A
    laplacian_matrix_ : ndarray
        The normalized Laplacian matrix L
        
    Examples
    --------
    >>> import numpy as np
    >>> from spectral import SpectralCluster
    >>> # Generate three concentric circles
    >>> theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    >>> X = np.vstack([
    ...     np.column_stack([np.cos(theta), np.sin(theta)]),
    ...     np.column_stack([2*np.cos(theta), 2*np.sin(theta)]),
    ...     np.column_stack([3*np.cos(theta), 3*np.sin(theta)])
    ... ])
    >>> clf = SpectralCluster(sigma=0.05)
    >>> clf.fit(X)
    >>> print(f"Found {clf.n_clusters_} clusters")
    
    References
    ----------
    Sanguinetti, G., Laidler, J., and Lawrence, N.D. (2005).
    "Automatic Determination of the Number of Clusters using Spectral Algorithms".
    Proceedings of the IEEE Workshop on Machine Learning for Signal Processing.
    """
    
    def __init__(self, sigma, lambda_=0.2, max_clusters=10, 
                 epsilon=0.0001, random_state=None):
        self.sigma = sigma
        self.lambda_ = lambda_
        self.max_clusters = max_clusters
        self.epsilon = epsilon
        self.random_state = random_state
        
    def fit(self, X, y=None):
        """
        Perform clustering on X.
        
        Implements Algorithm 2 (Cluster Detecting Algorithm) from the paper.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency by convention
            
        Returns
        -------
        self : object
            Fitted estimator
            
        Notes
        -----
        The algorithm:
        1. Forms normalized Laplacian L from affinity matrix
        2. Starts with q=2 eigenvectors
        3. Initializes q centers + 1 at origin
        4. Runs elongated k-means
        5. If origin cluster non-empty: increment q, repeat
        6. Otherwise: clustering complete
        
        References
        ----------
        Paper: Section 3, Algorithm 2 (Cluster Detecting Algorithm)
        MATLAB: SpectralCluster.m
        """
        # Convert to numpy array
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # Step 1: Form affinity matrix A and normalized Laplacian L
        self.affinity_matrix_ = build_affinity_matrix(X, self.sigma)
        self.laplacian_matrix_ = normalize_laplacian(self.affinity_matrix_)
        
        # Step 2: Compute eigenvectors with largest eigenvalues
        # Use scipy.linalg.eigh for symmetric matrices (more stable)
        eigvals, eigvecs = linalg.eigh(self.laplacian_matrix_)
        
        # Sort by eigenvalue in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take first max_clusters eigenvectors
        all_eigvecs = eigvecs[:, :self.max_clusters]
        self.eigenvalues_ = eigvals[:self.max_clusters]
        
        # Step 3: Start with Dim=2 (first 2 eigenvectors)
        Dim = 2
        ExtraCluster = False
        
        # Initialize with first Dim eigenvectors
        PcEig = all_eigvecs[:, :Dim]
        
        # Initialize first two centers
        # First center: point with largest norm (furthest from origin)
        norms = np.sum(PcEig ** 2, axis=1)
        idx_first = np.argmax(norms)
        centers = PcEig[[idx_first], :]
        
        # Second center: minimize (projection on first)^2 / norm
        # This finds point in different cluster (different radial direction)
        projections_sq = (PcEig @ centers[0]) ** 2
        # Avoid division by zero
        S = projections_sq / (norms + 1e-10)
        idx_second = np.argmin(S)
        centers = PcEig[[idx_first, idx_second], :]
        
        # Main loop: iteratively add dimensions
        kmeans = None  # Initialize to avoid UnboundLocalError
        
        while not ExtraCluster and Dim < self.max_clusters:
            # Add origin as (Dim+1)-th center
            centers_with_origin = np.vstack([centers, np.zeros(Dim)])
            
            # Run elongated k-means
            kmeans = ElongatedKMeans(
                n_clusters=Dim + 1,
                lambda_=self.lambda_,
                epsilon=self.epsilon,
                max_iter=100,
                tol=1e-4
            )
            kmeans.fit(PcEig, centers_with_origin)
            
            # Check if any points assigned to origin (last cluster)
            n_points_in_origin = np.sum(kmeans.labels_ == Dim)
            
            if n_points_in_origin > 0:
                # There's an extra cluster - expand dimensionality
                Dim += 1
                
                if Dim >= self.max_clusters:
                    # Hit max clusters - use current clustering without origin
                    # Dim was just incremented, so we have Dim-1 actual clusters
                    # Run one final k-means without origin
                    centers_final = kmeans.cluster_centers_[:-1]  # Remove origin
                    n_final_clusters = len(centers_final)
                    kmeans_final = ElongatedKMeans(
                        n_clusters=n_final_clusters,
                        lambda_=self.lambda_,
                        epsilon=self.epsilon,
                        max_iter=100,
                        tol=1e-4
                    )
                    kmeans_final.fit(PcEig, centers_final)
                    kmeans = kmeans_final  # Use final k-means result
                    centers = kmeans.cluster_centers_
                    Dim = n_final_clusters  # Update Dim to actual number of clusters
                    break
                
                # Take next eigenvector
                PcEig = all_eigvecs[:, :Dim]
                
                # Re-initialize centers: find points closest to previous centers
                centers = np.zeros((Dim, Dim))
                for i in range(Dim):
                    # Find point closest to i-th old center
                    # (using Euclidean distance in current eigenspace)
                    if i < len(kmeans.cluster_centers_) - 1:  # Skip origin
                        old_center = kmeans.cluster_centers_[i]
                        # Pad old center with 0 for new dimension
                        old_center_padded = np.pad(old_center, (0, 1), mode='constant')
                        distances = np.sum((PcEig - old_center_padded) ** 2, axis=1)
                        closest_point_idx = np.argmin(distances)
                        centers[i] = PcEig[closest_point_idx]
                    else:
                        # For additional center, take furthest from existing
                        if i == 0:
                            # Fallback for first center
                            centers[i] = PcEig[np.argmax(np.sum(PcEig ** 2, axis=1))]
                        else:
                            # Find point minimally represented
                            distances_to_all = np.zeros((n_samples, i))
                            for j in range(i):
                                distances_to_all[:, j] = np.sum((PcEig - centers[j]) ** 2, axis=1)
                            min_distances = np.min(distances_to_all, axis=1)
                            centers[i] = PcEig[np.argmax(min_distances)]
            else:
                # No points in origin cluster - found correct number
                ExtraCluster = True
                
                # Use the clustering without origin
                centers = kmeans.cluster_centers_[:-1]  # Remove origin
        
        # Store final results
        self.n_clusters_ = Dim
        self.eigenvectors_ = PcEig
        self.centers_ = centers
        
        # Store labels (kmeans is guaranteed to be defined now)
        if kmeans is not None:
            self.labels_ = kmeans.labels_
        else:
            # Fallback: shouldn't happen but handle gracefully
            self.labels_ = np.zeros(n_samples, dtype=int)
        
        return self
        
    def fit_predict(self, X, y=None):
        """
        Perform clustering on X and return cluster labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : Ignored
            Not used, present for API consistency by convention
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels
        """
        return self.fit(X).labels_
