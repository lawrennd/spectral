Spectral Clustering with Automatic Cluster Detection
=====================================================

Python implementation of the spectral clustering algorithm from:

> **"Automatic Determination of the Number of Clusters using Spectral Algorithms"**  
> Guido Sanguinetti, Jonathan Laidler, and Neil D. Lawrence (2005)  
> IEEE Workshop on Machine Learning for Signal Processing

This package automatically determines the number of clusters in a dataset using spectral methods.

## Features

- Automatic cluster number detection
- Handles non-convex cluster shapes
- Scikit-learn compatible API
- Faithful implementation of the 2005 algorithm

## Installation

### From source (development)

```bash
git clone https://github.com/lawrennd/spectral.git
cd spectral
pip install -e .
```

### Install with example dependencies

```bash
pip install -e ".[examples]"
```

## Quick Start

```python
import numpy as np
from spectral import SpectralCluster

# Generate three concentric circles
theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
X = np.vstack([
    np.column_stack([np.cos(theta), np.sin(theta)]),
    np.column_stack([2*np.cos(theta), 2*np.sin(theta)]),
    np.column_stack([3*np.cos(theta), 3*np.sin(theta)])
])

# Cluster with automatic number detection
clf = SpectralCluster(sigma=0.05)
clf.fit(X)
print(f"Found {clf.n_clusters_} clusters")  # Should find 3
```

## Examples

See the `examples/` directory for Jupyter notebooks reproducing the paper's experiments:
- Three concentric circles
- Non-convex shapes (ear, swirls)
- Image segmentation
- Spectrogram clustering

## Citation

If you use this software, please cite the original paper:

```bibtex
@inproceedings{sanguinetti2005automatic,
  title={Automatic Determination of the Number of Clusters using Spectral Algorithms},
  author={Sanguinetti, Guido and Laidler, Jonathan and Lawrence, Neil D},
  booktitle={IEEE Workshop on Machine Learning for Signal Processing},
  year={2005}
}
```

## Original MATLAB Code

The original MATLAB implementation is preserved in the `matlab/` directory for reference and validation.

## License

BSD 3-Clause License

Page last modified on Fri Jan 5 12:55:17 GMT 2007 (MATLAB version)  
Python version: 2026-02-08
