# Changelog

Every entry that changes a computed value says so in the first line, because a
saved study has to reproduce years after it was written. See `_version.py` for
the versioning rule: while the leading digit is 0, a changed figure costs a
minor bump.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-07-30

First public release. No figures changed, because there is nothing yet to
change them from.

### Added

- `sweep()`, which runs a parameter grid through the caller's own backtest
  function and derives the trial count from the grid rather than accepting it as
  an argument. The signature has no `n_trials` parameter and a test pins that
  absence, since a supplied count makes the deflation self-reported.
- `SweepResult.verdict()`, reporting the deflated Sharpe ratio, PSR, minimum
  track record length and a performance summary. Probability of backtest
  overfitting is reported beside them under `selection` and is deliberately not
  the gate: measured on a null-versus-edge fixture it does not discriminate
  where the deflated Sharpe does, so gating on it would reject real results.
- `SweepResult.surface()`, classifying the parameter neighbourhood as a plateau,
  a ridge or a knife edge, with the robust region per parameter when parameters
  are being stored.
- The `Study` artifact and its versioned JSON schema. `load()` refuses an
  unknown major version rather than half-parsing it, and tolerates unknown
  fields within a known major so a newer writer stays readable by an older
  build. A study holds derived figures and hashes, never a series.
- `alphaengine.core`: deflated Sharpe, PSR, PBO by CSCV, minimum track record
  length, a performance report, value at risk and conditional value at risk,
  factor decomposition, cointegration tests and a signal backtest. Each cites
  its source in the docstring.

### Notes on defaults

- Parameter values are **not** stored unless `store_params=True`. A parameter
  grid is frequently larger intellectual property than the return series it
  produced. Hashes are always kept, so two runs stay comparable without the
  values leaving the machine.
- Data identity is a content hash of the series, never a caller-supplied label,
  so renaming an experiment does not detach it from its history.
- Importing the package makes no network call and requires no account.
