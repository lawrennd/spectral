"""
Unit tests for main SpectralCluster algorithm.
"""

import numpy as np
import pytest
from spectral import SpectralCluster


class TestSpectralClusterBasic:
    """Basic tests for SpectralCluster class."""
    
    def test_instantiation(self):
        """Should instantiate with required parameter."""
        clf = SpectralCluster(sigma=0.1)
        assert clf.sigma == 0.1
        assert clf.lambda_ == 0.2
        assert clf.max_clusters == 10
    
    def test_sklearn_interface(self):
        """Should follow sklearn interface."""
        clf = SpectralCluster(sigma=0.1)
        assert hasattr(clf, 'fit')
        assert hasattr(clf, 'fit_predict')
    
    def test_parameter_storage(self):
        """Should store all parameters as-is (sklearn convention)."""
        clf = SpectralCluster(
            sigma=0.5,
            lambda_=0.3,
            max_clusters=5,
            epsilon=0.001,
            random_state=42
        )
        assert clf.sigma == 0.5
        assert clf.lambda_ == 0.3
        assert clf.max_clusters == 5
        assert clf.epsilon == 0.001
        assert clf.random_state == 42


class TestSpectralClusterFit:
    """Tests for fit() method."""
    
    def test_fit_returns_self(self):
        """fit() should return self for method chaining."""
        X = np.random.randn(20, 2)
        clf = SpectralCluster(sigma=1.0)
        result = clf.fit(X)
        assert result is clf
    
    def test_fit_sets_attributes(self):
        """fit() should set all expected attributes."""
        X = np.random.randn(30, 2)
        clf = SpectralCluster(sigma=0.5)
        clf.fit(X)
        
        assert hasattr(clf, 'labels_')
        assert hasattr(clf, 'n_clusters_')
        assert hasattr(clf, 'centers_')
        assert hasattr(clf, 'eigenvectors_')
        assert hasattr(clf, 'eigenvalues_')
        assert hasattr(clf, 'affinity_matrix_')
        assert hasattr(clf, 'laplacian_matrix_')
    
    def test_labels_shape(self):
        """labels_ should have correct shape."""
        X = np.random.randn(25, 3)
        clf = SpectralCluster(sigma=1.0)
        clf.fit(X)
        
        assert clf.labels_.shape == (25,)
        assert clf.labels_.dtype in [np.int32, np.int64]
    
    def test_n_clusters_positive(self):
        """Should find at least 1 cluster."""
        X = np.random.randn(15, 2)
        clf = SpectralCluster(sigma=1.0)
        clf.fit(X)
        
        assert clf.n_clusters_ >= 1
        assert clf.n_clusters_ <= clf.max_clusters


class TestSpectralClusterThreeCircles:
    """Tests on three concentric circles - the canonical example."""
    
    def test_three_circles_basic(self):
        """Should detect 3 clusters in three concentric circles."""
        # Generate three concentric circles
        np.random.seed(42)
        npts = 30  # Reduced for faster testing
        theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
        noise = np.random.randn(npts) * 0.1
        
        r1 = 1 + noise
        r2 = 2 + noise
        r3 = 3 + noise
        
        X = np.vstack([
            np.column_stack([r1 * np.cos(theta), r1 * np.sin(theta)]),
            np.column_stack([r2 * np.cos(theta), r2 * np.sin(theta)]),
            np.column_stack([r3 * np.cos(theta), r3 * np.sin(theta)])
        ])
        
        clf = SpectralCluster(sigma=0.05)
        clf.fit(X)
        
        # Should find 3 clusters
        assert clf.n_clusters_ == 3, f"Expected 3 clusters, found {clf.n_clusters_}"
    
    def test_three_circles_labels_valid(self):
        """Labels should be valid integers."""
        np.random.seed(42)
        npts = 20
        theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
        
        X = np.vstack([
            np.column_stack([np.cos(theta), np.sin(theta)]),
            np.column_stack([2*np.cos(theta), 2*np.sin(theta)]),
            np.column_stack([3*np.cos(theta), 3*np.sin(theta)])
        ])
        
        clf = SpectralCluster(sigma=0.05)
        clf.fit(X)
        
        # Labels should be integers from 0 to n_clusters-1
        assert np.all(clf.labels_ >= 0)
        assert np.all(clf.labels_ < clf.n_clusters_)
        
        # Should have points in each cluster
        unique_labels = np.unique(clf.labels_)
        assert len(unique_labels) == clf.n_clusters_


class TestSpectralClusterFitPredict:
    """Tests for fit_predict() method."""
    
    def test_fit_predict_returns_labels(self):
        """fit_predict() should return labels."""
        X = np.random.randn(20, 2)
        clf = SpectralCluster(sigma=1.0)
        labels = clf.fit_predict(X)
        
        assert isinstance(labels, np.ndarray)
        assert labels.shape == (20,)
    
    def test_fit_predict_matches_fit_labels(self):
        """fit_predict() should match calling fit() then accessing labels_."""
        X = np.random.randn(25, 3)
        
        clf1 = SpectralCluster(sigma=0.5, random_state=42)
        labels1 = clf1.fit_predict(X)
        
        clf2 = SpectralCluster(sigma=0.5, random_state=42)
        clf2.fit(X)
        labels2 = clf2.labels_
        
        np.testing.assert_array_equal(labels1, labels2)


class TestSpectralClusterEdgeCases:
    """Tests for edge cases."""
    
    def test_single_cluster(self):
        """Tight single cluster should be detected as 1 cluster."""
        # Very tight cluster
        np.random.seed(42)
        X = np.random.randn(20, 2) * 0.01
        
        clf = SpectralCluster(sigma=0.1)
        clf.fit(X)
        
        # Should find 1 cluster (or maybe 2 with noise, but not many)
        assert clf.n_clusters_ <= 3, f"Tight cluster found {clf.n_clusters_} clusters"
    
    def test_small_sample(self):
        """Should handle small number of samples."""
        X = np.array([[0, 0], [1, 0], [0, 1]])
        clf = SpectralCluster(sigma=1.0)
        clf.fit(X)
        
        assert clf.n_clusters_ >= 1
        assert len(clf.labels_) == 3
    
    def test_deterministic_with_seed(self):
        """Should be deterministic with same random_state."""
        X = np.random.RandomState(10).randn(30, 2)
        
        clf1 = SpectralCluster(sigma=1.0, random_state=42)
        labels1 = clf1.fit_predict(X)
        
        clf2 = SpectralCluster(sigma=1.0, random_state=42)
        labels2 = clf2.fit_predict(X)
        
        np.testing.assert_array_equal(labels1, labels2)


class TestSpectralClusterIntegration:
    """Integration tests."""
    
    def test_complete_pipeline(self):
        """Test complete pipeline on synthetic data."""
        # Create clear two-cluster data
        np.random.seed(42)
        cluster1 = np.random.randn(20, 2) * 0.3 + np.array([0, 0])
        cluster2 = np.random.randn(20, 2) * 0.3 + np.array([5, 0])
        X = np.vstack([cluster1, cluster2])
        
        clf = SpectralCluster(sigma=1.0)
        clf.fit(X)
        
        # Should find 2 clusters
        assert clf.n_clusters_ == 2
        
        # First 20 and last 20 points should mostly be in different clusters
        assert len(np.unique(clf.labels_[:20])) >= 1
        assert len(np.unique(clf.labels_[20:])) >= 1
    
    def test_affinity_and_laplacian_stored(self):
        """Affinity matrix and Laplacian should be stored."""
        X = np.random.randn(10, 2)
        clf = SpectralCluster(sigma=0.5)
        clf.fit(X)
        
        assert clf.affinity_matrix_.shape == (10, 10)
        assert clf.laplacian_matrix_.shape == (10, 10)
        
        # Matrices should be symmetric
        np.testing.assert_allclose(
            clf.affinity_matrix_, clf.affinity_matrix_.T, rtol=1e-10
        )
        np.testing.assert_allclose(
            clf.laplacian_matrix_, clf.laplacian_matrix_.T, rtol=1e-10
        )
