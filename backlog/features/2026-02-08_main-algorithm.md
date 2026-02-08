---
id: "2026-02-08_main-algorithm"
title: "Implement main SpectralCluster algorithm"
status: "Proposed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "features"
related_cips:
  - "0001"
owner: "Unassigned"
dependencies:
  - "2026-02-08_affinity-module"
  - "2026-02-08_elongated-kmeans"
tags:
- backlog
- core-algorithm
- main-class
---

# Task: Implement main SpectralCluster algorithm

> **Note**: This task implements the main clustering class from CIP-0001.

## Description

Implement the `SpectralCluster` class in `spectral/cluster.py` that orchestrates the complete automatic cluster detection algorithm following Algorithm 2 from the paper.

## Acceptance Criteria

- [ ] `SpectralCluster` class implemented
  - Inherits from `sklearn.base.BaseEstimator` and `ClusterMixin`
  - Constructor takes sigma, lambda_, max_clusters, epsilon, random_state
  - Stores all parameters as-is (sklearn convention)
- [ ] `fit(X, y=None)` method implemented
  - Builds affinity matrix
  - Normalizes to Laplacian
  - Iteratively expands eigenspace from q=2 to max_clusters
  - Performs elongated k-means with q+1 centers (one at origin)
  - Stops when origin cluster is empty
  - Stores labels_, n_clusters_, centers_, eigenvectors_, eigenvalues_
  - Returns self
- [ ] `fit_predict(X, y=None)` method implemented
- [ ] Center initialization strategy implemented
  - First center: point furthest from origin
  - Second center: maximize norm while minimizing dot product with first
- [ ] Integration tests pass:
  - Three circles dataset produces exactly 3 clusters
  - Single cluster detected for tight data
  - Deterministic with fixed random_state
- [ ] NumPy-style docstrings with full API documentation
- [ ] Type hints on all methods

## Implementation Notes

Follow Algorithm 2 from paper (Cluster Detecting Algorithm):
1. Form matrix L (use affinity module)
2. Set q=2
3. Compute q eigenvectors with largest eigenvalues
4. Perform elongated k-means with q+1 centres (q+1-th at origin)
5. If q+1-th cluster contains any points: set q=q+1, go to step 3
6. Otherwise: end, output clustered data

Key sklearn conventions:
- All constructor params stored as attributes
- Learned attributes have trailing underscore
- Return self from fit() for chaining
- Accept X as array-like, validate with sklearn utilities

## Related

- CIP: 0001 (Scikit-learn Compatible Package Architecture)
- Paper: Section 3, Algorithm 2 "Cluster Detecting Algorithm"
- MATLAB: SpectralCluster.m main loop (lines 56-81)

## Progress Updates

### 2026-02-08
Task created from CIP-0001 implementation plan.
