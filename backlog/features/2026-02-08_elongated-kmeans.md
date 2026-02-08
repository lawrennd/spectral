---
id: "2026-02-08_elongated-kmeans"
title: "Implement elongated k-means clustering"
status: "Proposed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "features"
related_cips:
  - "0001"
owner: "Unassigned"
dependencies:
  - "2026-02-08_package-structure-setup"
tags:
- backlog
- core-algorithm
- kmeans
---

# Task: Implement elongated k-means clustering

> **Note**: This task implements the elongated k-means component from CIP-0001.

## Description

Implement the elongated k-means clustering algorithm with Mahalanobis distance metric in `spectral/kmeans.py`. This is the key innovation that allows automatic cluster detection by exploiting radial structure in eigenspace.

## Acceptance Criteria

- [ ] `mahalanobis_distance(x, center, lambda_)` function implemented
  - Implements equation (1) from paper
  - Uses elongated distance for centers far from origin
  - Uses Euclidean distance for centers near origin (threshold epsilon)
  - Handles numerical edge cases
- [ ] `ElongatedKMeans` class or `elongated_kmeans()` function implemented
  - Takes initial centers as input
  - Iterates until convergence (max_iter, tolerance)
  - Returns final centers and labels
  - Compatible with standard k-means interface
- [ ] Unit tests pass:
  - Radial clusters are correctly separated
  - Distance along radial direction is downweighted
  - Distance in transverse direction is upweighted
  - Convergence within max iterations
  - Origin center behavior is correct
- [ ] NumPy-style docstrings
- [ ] Type hints

## Implementation Notes

Mahalanobis distance from paper equation (1):
```python
def mahalanobis_distance(
    x: np.ndarray, 
    center: np.ndarray, 
    lambda_: float = 0.2,
    epsilon: float = 0.0001
) -> np.ndarray:
    if np.dot(center, center) > epsilon:
        # Far from origin: use elongated distance
        c_norm_sq = np.dot(center, center)
        # M = (1/lambda)*(I - cc^T/||c||^2) + lambda*(cc^T/||c||^2)
        radial_proj = np.outer(center, center) / c_norm_sq
        M = (1/lambda_) * (np.eye(len(center)) - radial_proj) + lambda_ * radial_proj
        diff = x - center
        distances = np.einsum('ij,jk,ik->i', diff, M, diff)
        return distances
    else:
        # Near origin: use Euclidean
        return np.sum((x - center)**2, axis=1)
```

## Related

- CIP: 0001 (Scikit-learn Compatible Package Architecture)
- Paper: Section 3, equation (1), "elongated K-means"
- MATLAB: mahKmeans.m, mahDist2.m

## Progress Updates

### 2026-02-08
Task created from CIP-0001 implementation plan.
