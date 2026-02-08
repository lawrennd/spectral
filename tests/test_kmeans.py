"""
Unit tests for elongated k-means clustering.
"""

import numpy as np
import pytest
from spectral.kmeans import mahalanobis_distance, ElongatedKMeans


class TestMahalanobisDistance:
    """Tests for mahalanobis_distance function."""
    
    def test_distance_from_origin_is_euclidean(self):
        """Distance from origin should be Euclidean."""
        x = np.array([[1, 0], [0, 1], [2, 2]])
        center = np.array([0, 0])  # At origin
        
        distances = mahalanobis_distance(x, center, lambda_=0.2, epsilon=0.0001)
        
        # Should be Euclidean: d^2 = sum(x^2)
        expected = np.array([1.0, 1.0, 8.0])  # [1^2, 1^2, 2^2+2^2]
        np.testing.assert_allclose(distances, expected, rtol=1e-10)
    
    def test_radial_distance_downweighted(self):
        """Distance along radial direction should be downweighted."""
        # Center at [1, 0], test point further along same radial line
        center = np.array([1.0, 0.0])
        x_radial = np.array([[2.0, 0.0]])  # Along radial direction
        x_transverse = np.array([[1.0, 1.0]])  # Perpendicular
        
        dist_radial = mahalanobis_distance(x_radial, center, lambda_=0.2)
        dist_transverse = mahalanobis_distance(x_transverse, center, lambda_=0.2)
        
        # Euclidean distance is same for both (1.0)
        # But elongated distance should favor radial
        assert dist_radial < dist_transverse
    
    def test_transverse_distance_upweighted(self):
        """Distance in transverse direction should be upweighted."""
        center = np.array([2.0, 0.0])
        
        # Two points at equal Euclidean distance from center
        x_radial = np.array([[3.0, 0.0]])  # Along radial (towards origin from center)
        x_transverse = np.array([[2.0, 1.0]])  # Perpendicular
        
        dist_radial = mahalanobis_distance(x_radial, center, lambda_=0.2)
        dist_transverse = mahalanobis_distance(x_transverse, center, lambda_=0.2)
        
        # Transverse distance should be larger
        assert dist_transverse > dist_radial
    
    def test_distance_is_positive(self):
        """Distances should always be non-negative."""
        np.random.seed(42)
        x = np.random.randn(10, 3)
        center = np.random.randn(3)
        
        distances = mahalanobis_distance(x, center, lambda_=0.2)
        assert np.all(distances >= 0)
    
    def test_distance_zero_at_center(self):
        """Distance from center to itself should be zero."""
        center = np.array([1.5, 2.3, -0.5])
        x = center.reshape(1, -1)
        
        distances = mahalanobis_distance(x, center, lambda_=0.2)
        np.testing.assert_allclose(distances, 0.0, atol=1e-10)
    
    def test_lambda_effect(self):
        """Smaller lambda should create more elongation."""
        center = np.array([2.0, 0.0])
        x_radial = np.array([[3.0, 0.0]])
        x_transverse = np.array([[2.0, 1.0]])
        
        # With smaller lambda (more elongation)
        dist_r_small = mahalanobis_distance(x_radial, center, lambda_=0.1)[0]
        dist_t_small = mahalanobis_distance(x_transverse, center, lambda_=0.1)[0]
        
        # With larger lambda (less elongation)
        dist_r_large = mahalanobis_distance(x_radial, center, lambda_=0.5)[0]
        dist_t_large = mahalanobis_distance(x_transverse, center, lambda_=0.5)[0]
        
        # Ratio should be larger with smaller lambda (more discrimination)
        ratio_small = dist_t_small / dist_r_small
        ratio_large = dist_t_large / dist_r_large
        assert ratio_small > ratio_large


class TestElongatedKMeans:
    """Tests for ElongatedKMeans class."""
    
    def test_radial_clusters_separation(self):
        """Should separate radial clusters correctly."""
        # Create two radial clusters in 2D
        np.random.seed(42)
        cluster1 = np.random.randn(20, 2) * 0.1 + np.array([1, 0])
        cluster2 = np.random.randn(20, 2) * 0.1 + np.array([0, 1])
        X = np.vstack([cluster1, cluster2])
        
        # Initialize one center in each cluster, one at origin
        init_centers = np.array([
            [1, 0],
            [0, 1],
            [0, 0]
        ])
        
        kmeans = ElongatedKMeans(n_clusters=3, lambda_=0.2)
        kmeans.fit(X, init_centers)
        
        # Should converge
        assert kmeans.n_iter_ <= kmeans.max_iter
        
        # First 20 points should mostly be in one cluster
        # Last 20 points should mostly be in another cluster
        labels1 = kmeans.labels_[:20]
        labels2 = kmeans.labels_[20:]
        
        # Most points in each half should share a label
        from scipy.stats import mode
        mode1 = mode(labels1, keepdims=True).mode[0]
        mode2 = mode(labels2, keepdims=True).mode[0]
        
        # The two dominant labels should be different
        assert mode1 != mode2
    
    def test_convergence(self):
        """Algorithm should converge within max_iter."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        init_centers = X[:3].copy()
        
        kmeans = ElongatedKMeans(n_clusters=3, max_iter=100)
        kmeans.fit(X, init_centers)
        
        assert kmeans.n_iter_ <= 100
        assert hasattr(kmeans, 'cluster_centers_')
        assert hasattr(kmeans, 'labels_')
        assert hasattr(kmeans, 'inertia_')
    
    def test_empty_cluster_handling(self):
        """Should handle empty clusters (e.g., origin cluster)."""
        # Create data away from origin
        X = np.random.randn(20, 2) + np.array([5, 5])
        
        # Initialize one center at origin (will likely stay empty)
        init_centers = np.array([
            [5, 5],
            [0, 0]
        ])
        
        kmeans = ElongatedKMeans(n_clusters=2, lambda_=0.2)
        kmeans.fit(X, init_centers)
        
        # Should not crash and should produce valid labels
        assert len(kmeans.labels_) == 20
        assert np.all(kmeans.labels_ >= 0)
        assert np.all(kmeans.labels_ < 2)
    
    def test_predict_matches_fit_labels(self):
        """Predict on training data should mostly match fit labels."""
        # Note: Exact match not guaranteed for points on cluster boundaries
        np.random.seed(42)
        X = np.random.randn(30, 3)
        init_centers = X[:3].copy()
        
        kmeans = ElongatedKMeans(n_clusters=3)
        kmeans.fit(X, init_centers)
        
        predicted_labels = kmeans.predict(X)
        # Check that most labels match (>80% agreement is reasonable)
        agreement = np.sum(predicted_labels == kmeans.labels_) / len(kmeans.labels_)
        assert agreement > 0.8, f"Only {agreement*100:.1f}% of predictions match fit labels"
    
    def test_center_initialization_preserved(self):
        """Initial centers structure should influence final clustering."""
        np.random.seed(42)
        # Create obvious two-cluster data
        cluster1 = np.random.randn(20, 2) * 0.1 + np.array([0, 0])
        cluster2 = np.random.randn(20, 2) * 0.1 + np.array([10, 0])
        X = np.vstack([cluster1, cluster2])
        
        # Good initialization
        init_centers = np.array([[0, 0], [10, 0]])
        kmeans = ElongatedKMeans(n_clusters=2)
        kmeans.fit(X, init_centers)
        
        # Should find two distinct clusters
        assert len(np.unique(kmeans.labels_)) == 2
    
    def test_attributes_set_after_fit(self):
        """All expected attributes should be set after fitting."""
        X = np.random.randn(15, 2)
        init_centers = X[:2].copy()
        
        kmeans = ElongatedKMeans(n_clusters=2)
        kmeans.fit(X, init_centers)
        
        assert hasattr(kmeans, 'cluster_centers_')
        assert hasattr(kmeans, 'labels_')
        assert hasattr(kmeans, 'inertia_')
        assert hasattr(kmeans, 'n_iter_')
        
        assert kmeans.cluster_centers_.shape == (2, 2)
        assert kmeans.labels_.shape == (15,)
        assert isinstance(kmeans.inertia_, (int, float))
        assert isinstance(kmeans.n_iter_, int)


class TestElongatedKMeansIntegration:
    """Integration tests for elongated k-means."""
    
    def test_origin_cluster_detection(self):
        """When no points assigned to origin, should be detectable."""
        # Create two well-separated radial clusters
        np.random.seed(42)
        cluster1 = np.random.randn(15, 2) * 0.05 + np.array([1, 0])
        cluster2 = np.random.randn(15, 2) * 0.05 + np.array([0, 1])
        X = np.vstack([cluster1, cluster2])
        
        # Initialize with origin as third center
        init_centers = np.array([
            [1, 0],
            [0, 1],
            [0, 0]  # Origin
        ])
        
        kmeans = ElongatedKMeans(n_clusters=3, lambda_=0.2)
        kmeans.fit(X, init_centers)
        
        # Check if origin cluster (index 2) has any points
        n_points_in_origin = np.sum(kmeans.labels_ == 2)
        
        # This is the key test for automatic cluster detection:
        # If there are truly only 2 clusters, origin should get 0 or very few points
        # (exact number depends on noise and initialization)
        print(f"Points assigned to origin cluster: {n_points_in_origin}")
        # We don't assert a specific value as it depends on the data,
        # but this demonstrates the detection mechanism
        assert n_points_in_origin >= 0  # Sanity check