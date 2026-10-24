---
name: Data shape / loader issue
about: Report a CSV, Parquet, or interop data-shape edge case
title: "[Data shape]: "
labels: ["data-shape"]
assignees: []
---

## Data source / format

Which loader or flag was used (`--data`, `load_panel`, `alphaengine.io.from_quantconnect`, `alphaengine.io.from_vectorbt`, etc.)?

## Anonymized header and sample rows

Please paste 3–5 anonymized rows showing the header and column types (never paste proprietary price history):

```csv
date,symbol,close
2026-01-02,AAA,100.00
```

## Current behavior / error message

Paste the `DataShapeError` or unexpected behavior observed:
