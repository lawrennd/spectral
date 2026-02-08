---
id: "2026-02-08_examples-data-setup"
title: "Set up examples data directory"
status: "Proposed"
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

- [ ] examples/data/ directory created
- [ ] Data files copied from matlab/ to examples/data/:
  - shapes.bmp
  - ear.bmp
  - swirls.bmp
  - spectrogram.txt
- [ ] examples/README.md created explaining:
  - Overview of example notebooks
  - Data sources (from MATLAB code)
  - How to run the notebooks
- [ ] Data files tracked in git (small enough)
- [ ] examples/.gitignore if needed for generated outputs

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
