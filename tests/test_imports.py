"""
Test basic package imports.
"""

import pytest


def test_import_package():
    """Test that package can be imported."""
    import spectral
    assert spectral.__version__ == "0.1.0"


def test_import_spectral_cluster():
    """Test that SpectralCluster can be imported."""
    from spectral import SpectralCluster
    assert SpectralCluster is not None


def test_spectral_cluster_instantiation():
    """Test that SpectralCluster can be instantiated."""
    from spectral import SpectralCluster
    clf = SpectralCluster(sigma=0.1)
    assert clf.sigma == 0.1
    assert clf.lambda_ == 0.2
    assert clf.max_clusters == 10
