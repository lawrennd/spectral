---
id: "python-ecosystem"
title: "Python Scientific Computing Ecosystem"
status: "Active"
created: "2026-02-08"
last_reviewed: "2026-02-08"
review_frequency: "Annual"
tags:
- tenet
- python
- ecosystem
---

# Python Scientific Computing Ecosystem

## Tenet

**Description**: Use standard Python scientific computing libraries (NumPy, SciPy, scikit-learn) and follow their conventions. Leveraging established libraries ensures reliability, performance, and familiarity for users in the scientific Python community.

**Quote**: *"Build on the shoulders of giants - use proven scientific Python libraries."*

**Examples**:
- Use NumPy for array operations instead of nested loops
- Use SciPy's eigendecomposition routines instead of reimplementing
- Follow scikit-learn API patterns (fit/predict, trailing underscore for learned attributes)
- Standard packaging with pyproject.toml
- Type hints for better IDE support and error detection

**Counter-examples**:
- Reimplementing standard linear algebra routines
- Using camelCase instead of snake_case naming
- Custom package installation scripts instead of standard tools
