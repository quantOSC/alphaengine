---
name: Method request
about: Propose a published quantitative or statistical method for alphaengine.core
title: "[Method]: "
labels: ["enhancement", "methods"]
assignees: []
---

## Published reference

Everything in `alphaengine.core` comes from peer-reviewed literature. Please cite the paper or textbook:

- **Citation (authors, year, title, venue):**
- **DOI / URL:**

## What question does it answer?

Describe the statistical or risk question this primitive answers and how it fits into the validation/research loop.

## Proposed signature and inputs/outputs

```python
def proposed_method(returns, ...):
    ...
```

## Dependencies

Does this method run on `numpy` + `scipy` alone, or does it require an optional extra (e.g. `statsmodels`)?
