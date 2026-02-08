# Test Fixtures for MATLAB Validation

This directory contains reference data for validating the Python implementation against the original MATLAB code.

## Directory Structure

```
fixtures/
├── README.md                           # This file
├── matlab_three_circles.npz           # Three circles reference (if generated)
├── matlab_affinity.npz                # Affinity matrix reference (if generated)
└── matlab_laplacian.npz               # Laplacian reference (if generated)
```

## Generating Reference Data

To generate MATLAB reference data:

### Step 1: Run MATLAB Script

From the `tests/` directory:

```bash
matlab -batch "generate_matlab_reference"
```

Or from within MATLAB:

```matlab
cd tests
generate_matlab_reference
```

This will create `.mat` files in `tests/fixtures/`:
- `matlab_three_circles.mat`
- `matlab_affinity.mat`
- `matlab_laplacian.mat`

### Step 2: Convert to NumPy Format

From the repository root:

```bash
python tests/convert_matlab_reference.py
```

This will create corresponding `.npz` files that Python can read.

### Step 3: Run Validation Tests

```bash
pytest tests/test_matlab_equivalence.py -v
```

## Without MATLAB

If MATLAB is not available, the validation tests will skip MATLAB comparison tests but still run:
- Internal consistency checks
- Known property validation
- Algorithm behavior verification

These tests ensure correctness without requiring MATLAB reference data.

## Reference Data Contents

### matlab_three_circles.npz

Contains:
- `x`: Input data (N x 2 array)
- `sigma`: Scale parameter (scalar)
- `labels`: Cluster assignments (N x K binary matrix)
- `PcEig`: Eigenvectors used (N x K array)
- `Centres`: Final cluster centers (K x K array)

### matlab_affinity.npz

Contains:
- `x`: Input data (N x D array)
- `sigma`: Scale parameter (scalar)
- `A`: Affinity matrix (N x N array)

### matlab_laplacian.npz

Contains:
- `A`: Affinity matrix (N x N array)
- `L`: Normalized Laplacian (N x N array)

## Validation Strategy

The validation tests account for inherent ambiguities:

1. **Cluster Label Permutation**: Cluster indices are arbitrary, so labels `[0,1,2]` and `[2,0,1]` are equivalent. The tests handle this by comparing cluster memberships, not labels.

2. **Eigenspace Rotation**: Eigenvectors spanning the same subspace can be rotated. The tests compare subspace properties, not individual vectors.

3. **Sign Ambiguity**: Eigenvectors can have opposite signs (`v` and `-v` are both valid). The tests are sign-invariant.

4. **Numerical Precision**: MATLAB and NumPy may differ in floating-point precision. The tests use appropriate tolerances (`rtol=1e-5, atol=1e-8`).

## Test Categories

### 1. Component Tests

- `test_affinity_matrix_matches_matlab()`: Validates RBF kernel computation
- `test_laplacian_matches_matlab()`: Validates Laplacian normalization

### 2. End-to-End Tests

- `test_three_circles_matches_matlab()`: Validates complete algorithm on canonical example

### 3. Internal Consistency Tests (Always Run)

- `test_three_circles_reproducibility()`: Deterministic behavior
- `test_three_circles_detects_three_clusters()`: Correct cluster count
- `test_affinity_matrix_properties()`: Symmetry, range, diagonal
- `test_laplacian_eigenvalues()`: Eigenvalue bounds
- `test_clusters_are_radially_separated()`: Geometric properties

### 4. Numerical Stability Tests

- `test_identical_points()`: Edge case handling
- `test_large_sigma()`: Parameter sensitivity
- `test_small_sigma()`: Locality detection

## Adding New Test Cases

To add a new test case:

1. Add generation code to `generate_matlab_reference.m`
2. Save as `matlab_<testname>.mat`
3. Add to `MAT_FILES` in `convert_matlab_reference.py`
4. Add test function in `test_matlab_equivalence.py`

## Troubleshooting

**"MATLAB reference data not available" skip messages**
- This is expected if MATLAB hasn't been run yet
- Internal consistency tests will still run
- To enable MATLAB tests, follow steps above

**Conversion errors**
- Ensure MATLAB script completed successfully
- Check that `.mat` files exist in `fixtures/`
- Verify scipy is installed: `pip install scipy`

**Test failures**
- Check tolerance values (may need adjustment)
- Verify MATLAB and Python use same random seed
- Check for algorithm differences (e.g., eigenvalue ordering)
