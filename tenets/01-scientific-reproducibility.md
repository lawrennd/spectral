---
id: "scientific-reproducibility"
title: "Scientific Reproducibility"
status: "Active"
created: "2026-02-08"
last_reviewed: "2026-02-08"
review_frequency: "Annual"
tags:
- tenet
- scientific
- reproducibility
---

# Scientific Reproducibility

## Tenet

**Description**: The Python implementation must faithfully reproduce the algorithm and results from the 2005 paper "Automatic Determination of the Number of Clusters using Spectral Algorithms" by Sanguinetti, Laidler, and Lawrence. This is a reference implementation of a published scientific algorithm where accuracy and reproducibility are paramount to maintain scientific validity.

**Quote**: *"Scientific code must produce results that can be independently verified and reproduced."*

**Examples**:
- Use exact mathematical formulas from the paper (affinity matrix, Laplacian normalization, elongated distance)
- Reproduce all paper figures in example notebooks (three circles, ear, swirls, shapes, spectrogram)
- Validate outputs match MATLAB implementation within numerical tolerance
- Default parameters match those in the paper (lambda=0.2, epsilon=0.0001)

**Counter-examples**:
- Changing algorithm steps for convenience without validation
- Using approximations that alter results without documenting impact
- Skipping validation against MATLAB outputs
