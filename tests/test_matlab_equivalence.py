"""
MATLAB equivalence tests for scientific reproducibility.

This module provides tests that validate the Python implementation
against the original MATLAB code. It requires MATLAB reference data
to be generated first (see generate_matlab_reference.m).

To generate MATLAB reference data:
1. Run MATLAB scripts in matlab/ directory
2. Use generate_matlab_reference.m to save outputs
3. Convert .mat files to .npz using scipy.io.loadmat

Without MATLAB reference data, this module provides:
- Internal consistency checks
- Known property validation
- Framework for future MATLAB comparison
"""

import numpy as np
import pytest
from pathlib import Path
from spectral import SpectralCluster
from spectral.affinity import build_affinity_matrix, normalize_laplacian
from spectral.kmeans import ElongatedKMeans


# Path to MATLAB reference data
FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


def generate_three_circles(n_points=100, noise=0.05, seed=42):
    """
    Generate three concentric circles (canonical test case).
    
    Parameters
    ----------
    n_points : int
        Points per circle
    noise : float
        Gaussian noise standard deviation
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (3*n_points, 2)
        Data points
    """
    np.random.seed(seed)
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    noise_vals = np.random.randn(n_points, 2) * noise
    
    # Three circles at radii 1, 2, 3
    circles = []
    for radius in [1, 2, 3]:
        x = radius * np.cos(theta) + noise_vals[:, 0]
        y = radius * np.sin(theta) + noise_vals[:, 1]
        circles.append(np.column_stack([x, y]))
    
    return np.vstack(circles)


class TestMATLABReferenceData:
    """Tests using MATLAB reference data (if available)."""
    
    @pytest.mark.skipif(
        not (FIXTURES_DIR / "matlab_three_circles.npz").exists(),
        reason="MATLAB reference data not available"
    )
    def test_three_circles_matches_matlab(self):
        """Compare Python output to MATLAB reference for three circles."""
        # Load MATLAB reference
        ref = np.load(FIXTURES_DIR / "matlab_three_circles.npz")
        
        # Run Python implementation
        clf = SpectralCluster(sigma=0.05)
        clf.fit(ref['x'])
        
        # Compare number of clusters
        matlab_n_clusters = ref['labels'].shape[1]
        assert clf.n_clusters_ == matlab_n_clusters, \
            f"Python found {clf.n_clusters_} clusters, MATLAB found {matlab_n_clusters}"
        
        # Note: Detailed label comparison requires handling permutation
        # This is complex and requires careful implementation
        print(f"Python clusters: {clf.n_clusters_}")
        print(f"MATLAB clusters: {matlab_n_clusters}")
    
    @pytest.mark.skipif(
        not (FIXTURES_DIR / "matlab_affinity.npz").exists(),
        reason="MATLAB reference data not available"
    )
    def test_affinity_matrix_matches_matlab(self):
        """Compare affinity matrix computation to MATLAB."""
        ref = np.load(FIXTURES_DIR / "matlab_affinity.npz")
        
        # Compute Python affinity matrix
        A_python = build_affinity_matrix(ref['x'], ref['sigma'].item())
        
        # Compare to MATLAB
        np.testing.assert_allclose(
            A_python, ref['A'],
            rtol=1e-5, atol=1e-8,
            err_msg="Affinity matrices differ from MATLAB"
        )
    
    @pytest.mark.skipif(
        not (FIXTURES_DIR / "matlab_laplacian.npz").exists(),
        reason="MATLAB reference data not available"
    )
    def test_laplacian_matches_matlab(self):
        """Compare Laplacian computation to MATLAB."""
        ref = np.load(FIXTURES_DIR / "matlab_laplacian.npz")
        
        # Compute Python Laplacian
        L_python = normalize_laplacian(ref['A'])
        
        # Compare to MATLAB
        np.testing.assert_allclose(
            L_python, ref['L'],
            rtol=1e-5, atol=1e-8,
            err_msg="Laplacian matrices differ from MATLAB"
        )


class TestInternalConsistency:
    """Test internal consistency without MATLAB (always runs)."""
    
    def test_three_circles_reproducibility(self):
        """Three circles should give reproducible results."""
        X = generate_three_circles(n_points=50, seed=42)
        
        clf1 = SpectralCluster(sigma=0.05, random_state=42)
        clf1.fit(X)
        
        clf2 = SpectralCluster(sigma=0.05, random_state=42)
        clf2.fit(X)
        
        # Should be exactly reproducible
        assert clf1.n_clusters_ == clf2.n_clusters_
        np.testing.assert_array_equal(clf1.labels_, clf2.labels_)
    
    def test_three_circles_detects_three_clusters(self):
        """Three circles example should detect 3 clusters."""
        X = generate_three_circles(n_points=50, noise=0.03, seed=42)
        
        clf = SpectralCluster(sigma=0.05)
        clf.fit(X)
        
        # Should find exactly 3 clusters
        assert clf.n_clusters_ == 3, \
            f"Expected 3 clusters in three circles, found {clf.n_clusters_}"
        
        # Each cluster should have roughly equal points (with noise)
        unique, counts = np.unique(clf.labels_, return_counts=True)
        assert len(unique) == 3
        # With 50 points per circle, each cluster should have ~50 points
        # Allow for some misclassification due to noise
        assert all(30 <= c <= 70 for c in counts), \
            f"Unbalanced cluster sizes: {counts}"
    
    def test_affinity_matrix_properties(self):
        """Affinity matrix should have expected properties."""
        X = generate_three_circles(n_points=30, seed=42)
        
        A = build_affinity_matrix(X, sigma=0.05)
        
        # Should be symmetric
        np.testing.assert_allclose(A, A.T, rtol=1e-10)
        
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(A), 1.0, rtol=1e-10)
        
        # All values in (0, 1]
        assert np.all(A > 0)
        assert np.all(A <= 1.0)
    
    def test_laplacian_eigenvalues(self):
        """Laplacian eigenvalues should be in [0, 1]."""
        X = generate_three_circles(n_points=30, seed=42)
        
        A = build_affinity_matrix(X, sigma=0.05)
        L = normalize_laplacian(A)
        
        eigvals = np.linalg.eigvalsh(L)
        
        # Eigenvalues should be in [0, 1] (allow numerical tolerance)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalues: {eigvals[eigvals < 0]}"
        assert np.all(eigvals <= 1.0 + 1e-10), f"Large eigenvalues: {eigvals[eigvals > 1]}"
    
    def test_clusters_are_radially_separated(self):
        """Three circles clusters should be radially separated."""
        X = generate_three_circles(n_points=50, noise=0.02, seed=42)
        
        clf = SpectralCluster(sigma=0.05)
        clf.fit(X)
        
        # Compute mean radius for each cluster
        radii = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        cluster_radii = []
        for i in range(clf.n_clusters_):
            cluster_points_radii = radii[clf.labels_ == i]
            cluster_radii.append(np.mean(cluster_points_radii))
        
        # Cluster mean radii should be approximately 1, 2, 3
        cluster_radii_sorted = sorted(cluster_radii)
        expected_radii = [1.0, 2.0, 3.0]
        
        for observed, expected in zip(cluster_radii_sorted, expected_radii):
            assert abs(observed - expected) < 0.3, \
                f"Cluster radius {observed} too far from expected {expected}"


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_identical_points(self):
        """Should handle identical points gracefully."""
        X = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
        
        clf = SpectralCluster(sigma=1.0)
        clf.fit(X)
        
        # Should not crash
        assert clf.n_clusters_ >= 1
        assert len(clf.labels_) == 4
    
    def test_large_sigma(self):
        """Large sigma should tend towards single cluster."""
        X = generate_three_circles(n_points=30, noise=0.01, seed=42)
        
        clf = SpectralCluster(sigma=10.0)  # Very large sigma
        clf.fit(X)
        
        # With very large sigma, all points are similar
        # Might detect 1 or 2 clusters
        assert clf.n_clusters_ <= 3
    
    def test_small_sigma(self):
        """Small sigma should detect structure if present."""
        X = generate_three_circles(n_points=30, noise=0.01, seed=42)
        
        clf = SpectralCluster(sigma=0.01)  # Very small sigma
        clf.fit(X)
        
        # With very small sigma, should detect 3 clusters
        # (or possibly more if too local)
        assert clf.n_clusters_ >= 2


class TestAlgorithmProperties:
    """Test algorithm-specific properties."""
    
    def test_increasing_dimensions(self):
        """Algorithm should increase dimensions when detecting clusters."""
        X = generate_three_circles(n_points=40, seed=42)
        
        clf = SpectralCluster(sigma=0.05, max_clusters=5)
        clf.fit(X)
        
        # For three circles, should use >= 2 dimensions
        assert clf.eigenvectors_.shape[1] >= 2
        
        # Final number of clusters should equal dimensions used
        assert clf.n_clusters_ == clf.eigenvectors_.shape[1]
    
    def test_max_clusters_respected(self):
        """Should not exceed max_clusters."""
        X = generate_three_circles(n_points=30, seed=42)
        
        clf = SpectralCluster(sigma=0.05, max_clusters=2)
        clf.fit(X)
        
        # Should not find more than max_clusters
        assert clf.n_clusters_ <= 2


# ===== Instructions for generating MATLAB reference data =====

MATLAB_GENERATION_SCRIPT = """
% generate_matlab_reference.m
% Generate reference data for Python validation

% Three circles example
cd('matlab');
x = demoCircles();
sigma = 0.05;

% Run clustering
[labels, PcEig, Centres] = SpectralCluster(x, sigma);

% Save outputs
save('../tests/fixtures/matlab_three_circles.mat', ...
     'x', 'sigma', 'labels', 'PcEig', 'Centres');

% Also save intermediate results for component testing
A = zeros(size(x,1), size(x,1));
for i=1:size(x,1)
    for j=1:size(x,1)
        A(i,j) = exp(-norm(x(i,:)-x(j,:))^2/sigma);
    end
end
D = sum(A,2);
L = inv(diag(sqrt(D)))*(A/diag(sqrt(D)));

save('../tests/fixtures/matlab_affinity.mat', 'x', 'sigma', 'A');
save('../tests/fixtures/matlab_laplacian.mat', 'A', 'L');

% Convert to NumPy (run in Python):
% from scipy.io import loadmat
% import numpy as np
% for name in ['matlab_three_circles', 'matlab_affinity', 'matlab_laplacian']:
%     data = loadmat(f'tests/fixtures/{name}.mat')
%     np.savez(f'tests/fixtures/{name}.npz', **data)
"""

# Write generation script to file
if __name__ == "__main__":
    with open(FIXTURES_DIR.parent / "generate_matlab_reference.m", "w") as f:
        f.write(MATLAB_GENERATION_SCRIPT)
    
    print("=" * 70)
    print("MATLAB validation framework created.")
    print("=" * 70)
    print("\nTo generate reference data:")
    print("1. Run generate_matlab_reference.m in MATLAB")
    print("2. Convert .mat files to .npz using scipy.io.loadmat")
    print("3. Re-run pytest to enable MATLAB comparison tests")
    print("\nWithout MATLAB data, internal consistency tests will run.")
    print("=" * 70)
