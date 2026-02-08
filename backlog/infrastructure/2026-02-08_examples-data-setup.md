---
id: "2026-02-08_examples-data-setup"
title: "Set up examples data directory"
status: "Completed"
priority: "High"
created: "2026-02-08"
last_updated: "2026-02-08"
category: "infrastructure"
related_cips:
  - "0002"
owner: "Unassigned"
dependencies:
  - "2026-02-08_package-structure-setup"
tags:
- backlog
- examples
- data
---

# Task: Set up examples data directory

> **Note**: This task sets up the data infrastructure for CIP-0002.

## Description

Copy example data files from the MATLAB directory to a proper examples/data/ structure and create a README explaining the datasets.

## Acceptance Criteria

- [x] examples/data/ directory created
- [x] Data files copied from matlab/ to examples/data/:
  - shapes.bmp (1.2K)
  - ear.bmp (29K)
  - swirls.bmp (29K)
  - spectrogram.txt (31K)
- [x] examples/README.md created explaining:
  - Overview of all 3 example notebooks
  - Installation instructions
  - How to run notebooks (Jupyter, JupyterLab, VS Code)
  - Data sources and paper figure mapping
  - Troubleshooting guide
  - Citation information
- [x] Data files tracked in git (all under 32KB)
- [x] examples/.gitignore created for generated outputs

## Implementation Notes

Copy commands:
```bash
mkdir -p examples/data
cp matlab/shapes.bmp examples/data/
cp matlab/ear.bmp examples/data/
cp matlab/swirls.bmp examples/data/
cp matlab/spectrogram.txt examples/data/
```

examples/README.md should include:
- Brief description of each example
- Installation: `pip install -e .[examples]`
- How to run: `jupyter notebook examples/`
- References to paper figures

## Related

- CIP: 0002 (Example Notebooks Structure and Content)
- MATLAB data: matlab/*.bmp, matlab/spectrogram.txt

## Progress Updates

### 2026-02-08
Task created from CIP-0002 implementation plan.

### 2026-02-08
Task completed:
- Created examples/data/ directory structure
- Copied all 4 data files from matlab/ directory
- Created comprehensive examples/README.md:
  - Overview of each notebook with paper figure references
  - Installation instructions with all dependencies
  - Multiple ways to run (Jupyter, JupyterLab, VS Code)
  - Data file descriptions with size info
  - Citation information
  - Troubleshooting guide
- Created examples/.gitignore to exclude generated outputs
- All data files small enough to commit (largest is 31KB)
