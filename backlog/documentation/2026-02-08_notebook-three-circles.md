---
id: "2026-02-08_notebook-three-circles"
title: "Create three circles example notebook"
status: "Proposed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "documentation"
related_cips:
  - "0002"
owner: "Unassigned"
dependencies:
  - "2026-02-08_main-algorithm"
  - "2026-02-08_examples-data-setup"
tags:
- backlog
- notebooks
- examples
---

# Task: Create three circles example notebook

> **Note**: This task creates the primary example notebook from CIP-0002.

## Description

Create `examples/01_three_circles.ipynb` that demonstrates the algorithm on the canonical three concentric circles example, reproducing Figures 1-3 from the paper. This is the most important notebook - it serves as the template for others.

## Acceptance Criteria

- [ ] Notebook created with clear structure:
  - Title and overview
  - Data generation (three concentric circles with noise)
  - Visualization of data
  - Comparison with standard k-means (shows failure)
  - Apply SpectralCluster
  - Visualize results (original space and eigenspace)
  - Interpretation section
- [ ] Reproduces paper figures:
  - Figure 1: Original three circles data
  - Figure 2: 2D eigenspace showing radial structure
  - Figure 3: Final clustering result
- [ ] Algorithm finds exactly 3 clusters
- [ ] Notebook runs without errors
- [ ] Output cells saved (for GitHub preview)
- [ ] Explanatory markdown between code cells
- [ ] LaTeX equations for key formulas
- [ ] Links to paper sections

## Implementation Notes

Structure:
1. Generate data matching MATLAB demoCircles.m:
```python
np.random.seed(1)
npts = 100
theta = np.linspace(0, 2*np.pi, npts, endpoint=False)
noise = np.random.randn(npts) * 0.1

r1 = 1 + noise
r2 = 2 + noise  
r3 = 3 + noise

x = np.vstack([
    np.column_stack([r1*np.cos(theta), r1*np.sin(theta)]),
    np.column_stack([r2*np.cos(theta), r2*np.sin(theta)]),
    np.column_stack([r3*np.cos(theta), r3*np.sin(theta)])
])
```

2. Show k-means failure
3. Apply spectral clustering with sigma=0.05
4. Visualize eigenspace to show radial structure
5. Explain the key insight

## Related

- CIP: 0002 (Example Notebooks Structure and Content)
- Paper: Figures 1-3, Section 4
- MATLAB: demoCircles.m

## Progress Updates

### 2026-02-08
Task created from CIP-0002 implementation plan.
