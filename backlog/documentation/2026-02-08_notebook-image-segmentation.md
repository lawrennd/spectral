---
id: "2026-02-08_notebook-image-segmentation"
title: "Create image segmentation example notebook"
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
- image-segmentation
---

# Task: Create image segmentation example notebook

> **Note**: This task creates notebook 3 from CIP-0002, reproducing paper Figures 5b-5c.

## Description

Create `examples/03_image_segmentation.ipynb` that demonstrates spectral clustering on the shapes image (shapes.bmp). This example shows how sigma affects segmentation granularity, reproducing Figures 5b and 5c from the paper.

## Acceptance Criteria

- [ ] Notebook loads shapes.bmp from examples/data/
- [ ] Creates feature vectors: (x, y, intensity)
- [ ] Runs clustering with sigma=1 (coarse segmentation)
- [ ] Runs clustering with sigma=2 (fine segmentation)
- [ ] Visualizes original image and both clusterings side-by-side
- [ ] Reproduces Figures 5b and 5c from paper
- [ ] Explains intensity weighting and spatial features
- [ ] Discusses sigma selection for image segmentation
- [ ] Compares number of clusters detected in each case
- [ ] Includes markdown explaining the segmentation principle
- [ ] Notebook runs without errors in fresh environment
- [ ] All cells execute in <2 minutes

## Implementation Notes

### Technical Approach

1. **Data Loading**:
   ```python
   from PIL import Image
   img = Image.open('data/shapes.bmp')
   img_array = np.array(img)
   ```

2. **Feature Construction**:
   - Extract (x, y) coordinates for each pixel
   - Normalize spatial coordinates
   - Include intensity as third feature
   - Optionally weight intensity vs. spatial distance

3. **Two Clustering Runs**:
   - sigma=1: Coarser segmentation (fewer clusters)
   - sigma=2: Finer segmentation (more clusters)

4. **Visualization**:
   - Original grayscale image
   - Segmentation with sigma=1 (color-coded clusters)
   - Segmentation with sigma=2 (color-coded clusters)
   - Side-by-side comparison

### Paper References

- Figure 5a: Original shapes image
- Figure 5b: Clustering with sigma=1
- Figure 5c: Clustering with sigma=2

### Data Requirements

- shapes.bmp must exist in examples/data/
- Should be copied from matlab/ directory

## Related

- **CIP**: 0002 (Example Notebooks Structure and Content)
- **Dependencies**: 
  - 2026-02-08_notebook-three-circles (template)
  - 2026-02-08_examples-data-setup (data files)
- **Related Tasks**:
  - 2026-02-08_notebook-nonconvex-shapes
  - 2026-02-08_notebook-spectrogram

## Progress Updates

### 2026-02-08
Task created based on CIP-0002 implementation plan.
