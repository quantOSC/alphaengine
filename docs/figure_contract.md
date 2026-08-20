# Figure contract

A figure that travels is a **named kind**, not a series. The executor attaches
`charts: [{kind, key, title}]` hints. The portal maps `kind` + `key` to a
component. Old portals ignore `charts` and still have the raw keys
(`ic_by_period`, `best_curve`, …).

No hint carries a nested series. Titles are one line, no em dash.

Study schema **1.1** may also store `charts` and `process` on the artifact.
Both are optional. A 1.0 study still loads.

## Kinds

| Kind | Payload | Draws as |
|---|---|---|
| `curve` | `{i, v}` | line |
| `period` | `{p, ic}` or `{p, lambda}` | CS time series |
| `band` | `{i, lo, mid, hi}` | fan / Monte Carlo quantile |
| `hist` | `{edges, counts}` | histogram |
| `cost` | `{bps, sharpe}` | cost ladder |
| `scatter` | `{x, y}` | overlap |
| `triangle` | `{a, b, v}` | covariance heatmap |
| `rows` | bounded dict rows | table |
| `quantiles` | `[float]` | bar |

`i` is a position index into the original series, never a timestamp. Timestamps
stay on the machine that holds the data.

Lists longer than 512 are a series, not a figure, and the wire guard refuses
them. Sketches (`path_sketch`, `vol_path`, `mean_path`, `band`) are bucketed
below that cap before they travel.

## Who emits what

| Op | Typical hints |
|---|---|
| `compute.ou_calibrate` | `curve` / `path_sketch` |
| `compute.ou_simulate` | `curve` / `mean_path`, `band` / `band` |
| `compute.garch` | `curve` / `vol_path` |
| `compute.dgp_stress` | `hist` / `sharpe_hist` |
| `compute.denoise_cov` | `triangle` / `triangle`, `rows` / `variance_explained` |
| `compute.detone_cov` | same |
| `compute.grinold_alpha` | scalars only; the alpha vector stays in the workspace |

The Python helper is `alphaengine.charts.chart(kind, key, title)`.
