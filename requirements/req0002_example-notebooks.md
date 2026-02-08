---
id: "0002"
title: "Example Jupyter Notebooks Reproducing Paper Experiments"
status: "Proposed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
related_tenets:
  - scientific-reproducibility
  - python-ecosystem
stakeholders:
  - Researchers validating the implementation
  - New users learning the algorithm
  - Educators teaching spectral clustering
tags:
  - examples
  - validation
  - documentation
---

# REQ-0002: Example Jupyter Notebooks Reproducing Paper Experiments

## Description

Create Jupyter notebooks that demonstrate the algorithm on all examples from the 2005 paper, reproducing the key figures and results. These notebooks serve both as validation of the implementation and as educational material for users.

**Why this matters**: The notebooks provide concrete evidence that the Python implementation reproduces the paper's results, and they serve as starting points for users to understand and adapt the algorithm to their own problems.

**Who benefits**: 
- Users learning how to use the algorithm
- Researchers validating correctness against the paper
- Educators teaching spectral clustering concepts

## Acceptance Criteria

- [ ] Notebook 1: Three concentric circles example (reproduces Figures 1-3 from paper)
- [ ] Notebook 2: Non-convex shapes - ear and swirls datasets (reproduces Figure 4)
- [ ] Notebook 3: Image segmentation with shapes.bmp (reproduces Figure 5 with different sigma values)
- [ ] Notebook 4: Spectrogram clustering of "Aba" utterance (reproduces Figure 6)
- [ ] All notebooks run without errors on fresh Python environment
- [ ] Visual outputs match paper figures
- [ ] Notebooks include explanatory text connecting code to paper concepts
- [ ] Example data files from MATLAB code included in examples/data/

## Notes

Each notebook should:
- Start with explanation of the dataset and what makes it challenging
- Show step-by-step algorithm execution with intermediate visualizations
- Compare results with standard k-means or Ng et al. spectral clustering where relevant
- Discuss parameter selection (sigma, lambda)
- Reference specific equations and figures from the paper

## References

- **Related Tenets**: scientific-reproducibility
- **Paper Figures**: Figures 1-6 in Sanguinetti et al. (2005)
- **MATLAB Demos**: demoCircles.m, demoEar.m, demoShapes.m, demoSpectrogram.m
- **Depends On**: REQ-0001 (Python implementation must exist first)

## Progress Updates

### 2026-02-08
Requirement created.
