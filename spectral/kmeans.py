"""
Elongated k-means clustering with Mahalanobis distance.
"""

import numpy as np


def mahalanobis_distance(x, center, lambda_=0.2, epsilon=0.0001):
    """
    Compute elongated (Mahalanobis) distance from points to center.
    
    Implements the distance metric from equation (1) of the paper.
    For centers far from origin, uses elongated distance metric that
    downweights radial direction. For centers near origin, uses
    Euclidean distance.
    
    Parameters
    ----------
    x : ndarray of shape (n_samples, n_dims)
        Data points
    center : ndarray of shape (n_dims,)
        Cluster center
    lambda_ : float, default=0.2
        Sharpness parameter controlling elongation
    epsilon : float, default=0.0001
        Threshold for determining if center is near origin
        
    Returns
    -------
    distances : ndarray of shape (n_samples,)
        Distances from points to center
    """
    # TODO: Implement Mahalanobis distance from paper equation (1)
    raise NotImplementedError("Mahalanobis distance implementation in progress")


class ElongatedKMeans:
    """
    K-means with elongated distance metric for spectral clustering.
    
    Based on the algorithm in "Automatic Determination of the Number
    of Clusters using Spectral Algorithms" (Sanguinetti et al., 2005).
    
    Parameters
    ----------
    n_clusters : int
        Number of clusters
    lambda_ : float, default=0.2
        Sharpness parameter controlling elongation
    epsilon : float, default=0.0001
        Threshold for origin proximity
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Tolerance for convergence
    """
    
    def __init__(self, n_clusters, lambda_=0.2, epsilon=0.0001, 
                 max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X, init_centers):
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
        # TODO: Implement elongated k-means iteration
        raise NotImplementedError("Elongated k-means implementation in progress")
        
    def predict(self, X):
        """
        Predict cluster labels for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels
        """
        # TODO: Implement prediction
        raise NotImplementedError("Prediction implementation in progress")
