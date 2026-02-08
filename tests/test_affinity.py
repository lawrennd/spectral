"""
Unit tests for affinity matrix construction and Laplacian normalization.
"""

import numpy as np
import pytest
from spectral.affinity import build_affinity_matrix, normalize_laplacian


class TestBuildAffinityMatrix:
    """Tests for build_affinity_matrix function."""
    
    def test_affinity_diagonal_is_one(self):
        """Diagonal entries should be 1.0 (distance from point to itself is 0)."""
        X = np.array([[0, 0], [1, 0], [0, 1]])
        A = build_affinity_matrix(X, sigma=1.0)
        np.testing.assert_allclose(np.diag(A), 1.0, rtol=1e-10)
    
    def test_affinity_is_symmetric(self):
        """Affinity matrix should be symmetric."""
        X = np.random.randn(5, 3)
        A = build_affinity_matrix(X, sigma=0.5)
        np.testing.assert_allclose(A, A.T, rtol=1e-10)
    
    def test_affinity_values_in_range(self):
        """Affinity values should be in (0, 1]."""
        X = np.random.randn(4, 2)
        A = build_affinity_matrix(X, sigma=1.0)
        assert np.all(A > 0)
        assert np.all(A <= 1.0)
    
    def test_affinity_decreases_with_distance(self):
        """Affinity should decrease as distance increases."""
        # Create three points: p0 at origin, p1 nearby, p2 far
        X = np.array([[0, 0], [0.1, 0], [10, 0]])
        A = build_affinity_matrix(X, sigma=1.0)
        # A[0,1] (close) should be larger than A[0,2] (far)
        assert A[0, 1] > A[0, 2]
    
    def test_affinity_sigma_effect(self):
        """Larger sigma should make distant points more similar."""
        X = np.array([[0, 0], [2, 0]])
        A_small = build_affinity_matrix(X, sigma=0.5)
        A_large = build_affinity_matrix(X, sigma=2.0)
        # With larger sigma, distant points should have higher affinity
        assert A_large[0, 1] > A_small[0, 1]
    
    def test_affinity_single_point(self):
        """Should handle single point correctly."""
        X = np.array([[1, 2]])
        A = build_affinity_matrix(X, sigma=1.0)
        assert A.shape == (1, 1)
        assert A[0, 0] == 1.0
    
    def test_affinity_duplicate_points(self):
        """Duplicate points should have affinity 1.0."""
        X = np.array([[0, 0], [0, 0], [1, 1]])
        A = build_affinity_matrix(X, sigma=1.0)
        # Points 0 and 1 are identical
        assert A[0, 1] == 1.0
        assert A[1, 0] == 1.0
    
    def test_affinity_known_values(self):
        """Test against hand-computed values."""
        # Two points at distance 1 apart
        X = np.array([[0, 0], [1, 0]])
        sigma = 1.0
        A = build_affinity_matrix(X, sigma=sigma)
        
        # Distance squared = 1^2 = 1
        # A[0,1] = exp(-1 / 1^2) = exp(-1) ≈ 0.3679
        expected = np.exp(-1.0)
        np.testing.assert_allclose(A[0, 1], expected, rtol=1e-10)
    
    def test_affinity_shape(self):
        """Output should have correct shape."""
        X = np.random.randn(7, 4)
        A = build_affinity_matrix(X, sigma=0.5)
        assert A.shape == (7, 7)


class TestNormalizeLaplacian:
    """Tests for normalize_laplacian function."""
    
    def test_laplacian_is_symmetric(self):
        """Normalized Laplacian should be symmetric."""
        A = np.array([[1.0, 0.5, 0.0],
                      [0.5, 1.0, 0.5],
                      [0.0, 0.5, 1.0]])
        L = normalize_laplacian(A)
        np.testing.assert_allclose(L, L.T, rtol=1e-10)
    
    def test_laplacian_eigenvalues_in_range(self):
        """Eigenvalues of L should be in [0, 1]."""
        # Create a random symmetric affinity matrix
        A = np.random.rand(5, 5)
        A = (A + A.T) / 2  # Make symmetric
        A = A + np.eye(5)  # Ensure positive diagonal
        
        L = normalize_laplacian(A)
        eigvals = np.linalg.eigvalsh(L)
        
        # Eigenvalues should be in [0, 1] (allow small numerical error)
        assert np.all(eigvals >= -1e-10)
        assert np.all(eigvals <= 1.0 + 1e-10)
    
    def test_laplacian_known_values(self):
        """Test against hand-computed values."""
        # Simple 2x2 case
        A = np.array([[1.0, 0.5],
                      [0.5, 1.0]])
        L = normalize_laplacian(A)
        
        # D = [1.5, 1.5]
        # D^(-1/2) = [1/sqrt(1.5), 1/sqrt(1.5)]
        # L[0,0] = 1.0 / 1.5 = 2/3
        # L[0,1] = 0.5 / 1.5 = 1/3
        np.testing.assert_allclose(L[0, 0], 2/3, rtol=1e-10)
        np.testing.assert_allclose(L[0, 1], 1/3, rtol=1e-10)
        np.testing.assert_allclose(L[1, 0], 1/3, rtol=1e-10)
        np.testing.assert_allclose(L[1, 1], 2/3, rtol=1e-10)
    
    def test_laplacian_shape(self):
        """Output should preserve shape."""
        A = np.random.rand(6, 6)
        A = (A + A.T) / 2
        L = normalize_laplacian(A)
        assert L.shape == (6, 6)
    
    def test_laplacian_with_affinity_output(self):
        """Should work correctly with affinity matrix output."""
        X = np.random.randn(4, 3)
        A = build_affinity_matrix(X, sigma=0.5)
        L = normalize_laplacian(A)
        
        # Basic sanity checks
        assert L.shape == (4, 4)
        np.testing.assert_allclose(L, L.T, rtol=1e-10)
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10)
        assert np.all(eigvals <= 1.0 + 1e-10)


class TestAffinityLaplacianIntegration:
    """Integration tests for affinity and Laplacian together."""
    
    def test_three_circles_properties(self):
        """Test on three concentric circles (will be used in examples)."""
        # Generate simplified three circles
        theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
        r1 = np.column_stack([np.cos(theta), np.sin(theta)])
        r2 = np.column_stack([2*np.cos(theta), 2*np.sin(theta)])
        r3 = np.column_stack([3*np.cos(theta), 3*np.sin(theta)])
        X = np.vstack([r1, r2, r3])
        
        # Build affinity and Laplacian with reasonable sigma
        A = build_affinity_matrix(X, sigma=0.05)
        L = normalize_laplacian(A)
        
        # Basic properties
        assert A.shape == (60, 60)
        assert L.shape == (60, 60)
        np.testing.assert_allclose(A, A.T, rtol=1e-10)
        np.testing.assert_allclose(L, L.T, rtol=1e-10)
        
        # Eigenvalues should be in range
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10)
        assert np.all(eigvals <= 1.0 + 1e-10)
    
    def test_pipeline(self):
        """Test the full affinity -> Laplacian pipeline."""
        X = np.random.randn(10, 2)
        
        # Should not raise any errors
        A = build_affinity_matrix(X, sigma=1.0)
        L = normalize_laplacian(A)
        
        # All checks should pass
        assert A.shape == (10, 10)
        assert L.shape == (10, 10)
        np.testing.assert_allclose(A, A.T)
        np.testing.assert_allclose(L, L.T)
