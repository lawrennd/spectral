---
id: "2026-02-08_test-notebooks"
title: "Test all example notebooks"
status: "Proposed"
priority: "Medium"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "documentation"
related_cips:
  - "0002"
owner: "Unassigned"
dependencies:
  - "2026-02-08_notebook-three-circles"
  - "2026-02-08_notebook-nonconvex-shapes"
  - "2026-02-08_notebook-parameter-exploration"
tags:
- backlog
- testing
- notebooks
---

# Task: Test all example notebooks

> **Note**: This task validates the notebooks from CIP-0002.

## Description

Create an automated test that runs all example notebooks in a fresh environment to ensure they execute without errors and produce expected outputs.

## Acceptance Criteria

- [ ] Test script created (pytest-based or nbval)
- [ ] All notebooks execute without errors
- [ ] Execution time is reasonable (<2 minutes per notebook)
- [ ] Visual outputs are checked (at minimum, cells have output)
- [ ] Test runs in CI (optional but recommended)
- [ ] Documentation on how to run tests locally

## Implementation Notes

Using pytest-nbval:
```bash
pip install nbval
pytest --nbval examples/*.ipynb
```

Or create custom test:
```python
# tests/test_notebooks.py
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def test_all_notebooks():
    notebooks = [
        'examples/01_three_circles.ipynb',
        'examples/02_nonconvex_shapes.ipynb',
        # ...
    ]
    
    for nb_path in notebooks:
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': 'examples/'}})
```

Can also check:
- Number of clusters found in three circles is 3
- Figures are generated (check matplotlib calls)
- No error cells

## Related

- CIP: 0002 (Example Notebooks Structure and Content)
- All example notebooks

## Progress Updates

### 2026-02-08
Task created from CIP-0002 implementation plan.
