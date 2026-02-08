# Spectral Clustering Examples

This directory contains Jupyter notebooks demonstrating the automatic spectral clustering algorithm from:

> "Automatic Determination of the Number of Clusters using Spectral Algorithms"  
> Sanguinetti, G., Laidler, J., Lawrence, N.D. (2005)  
> NNSP 2005 Workshop

## Example Notebooks

### 1. Three Circles (`01_three_circles.ipynb`)

**Reproduces:** Paper Figure 1

Demonstrates the algorithm on three concentric circles - the canonical example of radial cluster structure. Shows:
- How the algorithm automatically detects 3 clusters
- Visualization in input space and eigenspace
- Effect of sigma parameter on clustering

**Key Concepts:**
- Radial cluster detection
- Eigenspace transformation
- Automatic cluster number determination

### 2. Non-Convex Shapes (`02_nonconvex_shapes.ipynb`)

**Reproduces:** Paper Figures 2-4

Applies clustering to non-convex shapes from images:
- Shapes (two interlocking spirals)
- Ear image (segmentation)
- Swirls (complex patterns)

**Key Concepts:**
- Image segmentation
- Non-convex cluster shapes
- Comparison with k-means

### 3. Parameter Exploration (`03_parameter_exploration.ipynb`)

**Reproduces:** Paper analysis of sigma parameter

Systematic exploration of how parameters affect clustering:
- Effect of sigma (RBF scale) on cluster detection
- Effect of lambda (elongation) on radial sensitivity
- Guidelines for parameter selection

**Key Concepts:**
- Parameter sensitivity
- Model selection
- Algorithm robustness

## Installation

To run the examples, install the package with example dependencies:

```bash
# From the repository root
pip install -e .
```

This installs:
- `spectral` - The main package
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `scikit-learn` - ML utilities (comparison)
- `matplotlib` - Visualization
- `scikit-image` - Image processing
- `jupyter` - Notebook interface

## Running the Notebooks

### Option 1: Jupyter Notebook

```bash
jupyter notebook examples/
```

This opens the Jupyter interface in your browser. Click on any `.ipynb` file to open it.

### Option 2: Jupyter Lab

```bash
jupyter lab examples/
```

Modern interface with more features.

### Option 3: VS Code

Open the `.ipynb` files directly in VS Code with the Jupyter extension.

## Data Files

The `data/` directory contains example datasets from the original MATLAB implementation:

- **shapes.bmp** - Two interlocking spirals (Paper Figure 2)
- **ear.bmp** - Ear image for segmentation (Paper Figure 3)
- **swirls.bmp** - Complex swirl patterns (Paper Figure 4)
- **spectrogram.txt** - Spectrogram data for time-series clustering

All data files are from the original MATLAB code (2005) and are included for reproducibility.

## Expected Outputs

Each notebook includes:
1. **Data generation/loading** - Synthetic or real data
2. **Clustering** - Apply SpectralCluster
3. **Visualization** - Results in input and eigenspace
4. **Analysis** - Interpretation and comparison with paper
5. **Validation** - Correctness checks

## Reproducing Paper Results

The notebooks are designed to reproduce the figures and results from the original paper. Minor differences may occur due to:
- Random initialization (set `random_state` for reproducibility)
- Numerical precision (NumPy vs MATLAB)
- Plotting style (modern matplotlib vs MATLAB 2005)

## Citation

If you use this code or the algorithm in your research, please cite the original paper:

```bibtex
@inproceedings{sanguinetti2005automatic,
  title={Automatic determination of the number of clusters using spectral algorithms},
  author={Sanguinetti, Guido and Laidler, Jonathan and Lawrence, Neil D},
  booktitle={IEEE Workshop on Machine Learning for Signal Processing},
  pages={55--60},
  year={2005},
  organization={IEEE}
}
```

## Troubleshooting

**Import error: "No module named spectral"**
- Make sure you installed the package: `pip install -e .`
- Run from the repository root, not the examples/ directory

**Matplotlib backend errors**
- Try: `%matplotlib inline` at the top of the notebook

**Out of memory**
- Reduce the number of data points in synthetic examples
- Close other applications

## Further Resources

- [Repository README](../README.md) - Installation and quick start
- [Paper (LaTeX source)](../tex/nnsp05/clusterNumber.tex) - Full algorithm description
- [MATLAB code](../matlab/) - Original implementation
- [Tests](../tests/) - Validation and examples

## Contributing

Found a bug or have a suggestion? Please open an issue on GitHub!
