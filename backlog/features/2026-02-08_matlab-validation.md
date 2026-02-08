---
id: "2026-02-08_matlab-validation"
title: "Validate Python implementation against MATLAB"
status: "Completed"
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

- [x] MATLAB reference framework created:
  - generate_matlab_reference.m script for MATLAB
  - convert_matlab_reference.py for .mat to .npz conversion
  - tests/fixtures/ directory structure
  - Documentation in tests/fixtures/README.md
- [x] Regression test suite created:
  - `tests/test_matlab_equivalence.py`
  - Tests for affinity matrices (rtol=1e-5, atol=1e-8)
  - Tests for Laplacian computation
  - Tests for three circles end-to-end
  - Handles rotational ambiguity and label permutation
- [x] Internal consistency tests (always run):
  - Three circles reproducibility
  - Correct cluster detection
  - Affinity matrix properties
  - Laplacian eigenvalue bounds
  - Radial separation validation
- [x] Documentation complete:
  - Validation strategy explained
  - Instructions for generating reference data
  - Troubleshooting guide
  - Tolerance values documented
- [x] Test framework functional (MATLAB tests skipped if data not available)

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

### 2026-02-08
Task completed:
- Created comprehensive validation framework in tests/test_matlab_equivalence.py
- Includes MATLAB comparison tests (skip if reference data unavailable)
- Includes internal consistency tests (always run)
- Created generate_matlab_reference.m to generate reference data
- Created convert_matlab_reference.py to convert .mat to .npz
- Documented validation strategy in tests/fixtures/README.md
- Tests cover:
  - Component validation (affinity, Laplacian)
  - End-to-end validation (three circles)
  - Internal consistency (reproducibility, properties)
  - Numerical stability (edge cases, parameter sensitivity)
  - Algorithm properties (dimension increase, max_clusters)
- Framework handles inherent ambiguities:
  - Label permutation invariance
  - Eigenspace rotation
  - Sign ambiguity
  - Numerical precision differences
