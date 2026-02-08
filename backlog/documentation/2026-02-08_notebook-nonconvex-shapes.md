---
id: "2026-02-08_notebook-nonconvex-shapes"
title: "Create non-convex shapes example notebooks"
status: "Completed"
priority: "Medium"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "documentation"
related_cips:
  - "0002"
owner: "Unassigned"
dependencies:
  - "2026-02-08_notebook-three-circles"
  - "2026-02-08_examples-data-setup"
tags:
- backlog
- notebooks
- examples
---

# Task: Create non-convex shapes example notebooks

> **Note**: This task creates additional example notebooks from CIP-0002.

## Description

Create notebooks for the ear, swirls, shapes image, and spectrogram examples from the paper. These can be combined into 2-3 notebooks covering the remaining paper figures.

## Acceptance Criteria

- [ ] `examples/02_nonconvex_shapes.ipynb` created with:
  - Ear dataset (Figure 4a from paper)
  - Swirls dataset (Figure 4b from paper)
  - Both using appropriate sigma values
- [ ] `examples/03_image_segmentation.ipynb` created with:
  - Shapes.bmp image (Figure 5 from paper)
  - Two runs with different sigma (1 and 2)
  - Discussion of granularity control
- [ ] `examples/04_spectrogram_clustering.ipynb` created with:
  - Spectrogram data (Figure 6 from paper)
  - Clustering at different sigma values
  - Speech interpretation (consonant/vowel segmentation)
- [ ] All notebooks run without errors
- [ ] Visual outputs match paper figures
- [ ] Output cells saved
- [ ] Cross-references to paper sections

## Implementation Notes

Follow the template from notebook 1 (three circles).

For image examples, extract features as (x, y, intensity):
```python
import imageio
img = imageio.imread('data/shapes.bmp')
if img.ndim == 3:
    img = img.mean(axis=2)  # Convert to grayscale
    
# Normalize intensity
img = img / img.max()

# Create feature vectors
rows, cols = img.shape
x1 = np.repeat(np.arange(rows), cols)
x2 = np.tile(np.arange(cols), rows)
intensity = img.ravel()

# Weight intensity same as spatial
X = np.column_stack([x1, x2, intensity * (rows + cols)])
```

## Related

- CIP: 0002 (Example Notebooks Structure and Content)
- Paper: Figures 4, 5, 6
- MATLAB: demoEar.m, demoSwirls.m, demoShapes.m, demoSpectrogram.m

## Progress Updates

### 2026-02-08
Task created from CIP-0002 implementation plan.
