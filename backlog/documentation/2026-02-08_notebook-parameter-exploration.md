---
id: "2026-02-08_notebook-parameter-exploration"
title: "Create parameter exploration notebook"
status: "Completed"
priority: "Low"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "documentation"
related_cips:
  - "0002"
owner: "Unassigned"
dependencies:
  - "2026-02-08_notebook-three-circles"
tags:
- backlog
- notebooks
- examples
- education
---

# Task: Create parameter exploration notebook

> **Note**: This task creates the educational parameter exploration from CIP-0002.

## Description

Create `examples/05_parameter_exploration.ipynb` that provides interactive exploration of parameter effects and guidelines for parameter selection.

## Acceptance Criteria

- [ ] Notebook created with sections:
  - Effect of sigma (scale parameter)
  - Effect of lambda (elongation parameter)
  - Effect of epsilon (origin threshold)
  - Guidelines for parameter selection
  - When does the algorithm struggle?
- [ ] Interactive widgets for parameter exploration (optional but nice)
- [ ] Systematic study with visualizations
- [ ] Practical guidelines document
- [ ] Notebook runs without errors
- [ ] Output cells saved

## Implementation Notes

Use ipywidgets for interactivity (optional):
```python
from ipywidgets import interact, FloatSlider

@interact(sigma=FloatSlider(min=0.01, max=2.0, step=0.01, value=0.05))
def explore_sigma(sigma):
    clf = SpectralCluster(sigma=sigma)
    clf.fit(X)
    plot_clustering(X, clf.labels_)
    plt.title(f'Clusters: {clf.n_clusters_}, sigma: {sigma:.2f}')
```

Parameter selection guidelines:
- Sigma: roughly between intra-cluster and inter-cluster distance
- Start with median pairwise distance
- Lambda: default 0.2 works well, rarely needs tuning
- If too many clusters: increase sigma
- If too few clusters: decrease sigma

## Related

- CIP: 0002 (Example Notebooks Structure and Content)
- Paper: Section 4 discussion of parameter choices

## Progress Updates

### 2026-02-08
Task created from CIP-0002 implementation plan.
