---
id: "2026-02-08_matlab-validation"
title: "Validate Python implementation against MATLAB"
status: "Proposed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "features"
related_cips:
  - "0001"
owner: "Unassigned"
dependencies:
  - "2026-02-08_main-algorithm"
tags:
- backlog
- validation
- testing
---

# Task: Validate Python implementation against MATLAB

> **Note**: This task implements the validation strategy from CIP-0001.

## Description

Create regression tests that validate the Python implementation produces identical results to the MATLAB code within numerical tolerance. This proves scientific reproducibility.

## Acceptance Criteria

- [ ] MATLAB reference outputs generated and saved:
  - Three circles: labels, eigenvectors, centers
  - Simple test cases with known ground truth
  - Saved as .npz files in tests/fixtures/
- [ ] Regression test suite created:
  - `tests/test_matlab_equivalence.py`
  - Affinity matrices match (rtol=1e-5, atol=1e-8)
  - Eigenvectors span same subspace (handle rotational ambiguity)
  - Cluster labels match (handle label permutation)
  - Number of clusters matches exactly
- [ ] Documentation of any differences:
  - Numerical precision differences documented
  - Algorithm behavior differences explained
  - Tolerance values justified
- [ ] All tests pass
- [ ] Test data committed to repository

## Implementation Notes

Steps to generate MATLAB reference data:
1. Run MATLAB demoCircles.m
2. Save outputs:
```matlab
[labels, PcEig, Centres] = SpectralCluster(x, 0.05);
save('three_circles_reference.mat', 'labels', 'PcEig', 'Centres', 'x');
```
3. Convert to NumPy:
```python
from scipy.io import loadmat
data = loadmat('three_circles_reference.mat')
np.savez('tests/fixtures/matlab_three_circles.npz', **data)
```

Test structure:
```python
def test_three_circles_matches_matlab():
    # Load MATLAB reference
    ref = np.load('tests/fixtures/matlab_three_circles.npz')
    
    # Run Python implementation
    clf = SpectralCluster(sigma=0.05)
    clf.fit(ref['x'])
    
    # Compare (handle label permutation, eigenspace rotation)
    assert clf.n_clusters_ == ref['labels'].shape[1]
    # ... more comparisons
```

Handle inherent ambiguities:
- Cluster labels: arbitrary permutation invariance
- Eigenvectors: rotational ambiguity in eigenspace
- Sign flips: eigenvectors can have opposite signs

## Related

- CIP: 0001 (Scikit-learn Compatible Package Architecture)
- Tenet: scientific-reproducibility
- MATLAB: All demo scripts

## Progress Updates

### 2026-02-08
Task created from CIP-0001 implementation plan.
