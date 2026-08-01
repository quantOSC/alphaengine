# Changelog

Every entry that changes a computed value says so in the first line, because a
saved study has to reproduce years after it was written. See `_version.py` for
the versioning rule: while the leading digit is 0, a changed figure costs a
minor bump.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.2] - 2026-07-31

**No computed value changed.** Identical figures to 0.1.0 and 0.1.1.

### Fixed

- `report()` used `try`/`except`/`pass` to read an error body, which ruff's
  SIM105 correctly objects to. It is `contextlib.suppress` now. The intent was
  always total suppression: a server answering with HTML, an empty body or
  nothing at all still has to produce a usable error rather than a second
  exception raised on top of the first.

### Changed

- `DEFAULT_BASE_URL` points at the live platform. 0.1.1 shipped with a
  placeholder host, so every `report()` call that did not pass `base_url` or set
  `QUANTOS_API_URL` resolved to a domain that does not answer. Reporting was
  effectively unusable at its default in that release.

## [0.1.1] - 2026-07-31

**No computed value changed.** Every figure this version produces is identical
to 0.1.0; the goldens are untouched. A study saved by 0.1.0 reproduces here.

### Added

- `report()`, which sends a study to a QuantOS workspace. The offline half is
  unchanged and always will be: `save()` still needs no account, no key and no
  network. What it could not do was reach the person who has to act on the
  work, so a run done in a notebook stayed in the notebook. This is that
  crossing, and nothing else about the package assumes you will ever call it.

  Available as `alphaengine.report(study, ...)` and as `Study.report()`. A key
  comes from `QUANTOS_API_KEY` unless passed explicitly, because a key pasted
  into a notebook is a key committed to git. `QUANTOS_API_URL` points it at a
  self-hosted or VPC deployment.

- A client-side series guard on the reported payload, keyed on LENGTH rather
  than field name, so renaming a key cannot smuggle a return series past it.
  The same check runs on the server. The duplication is deliberate: a check
  that runs only on the client is not a check, and one that runs only on the
  server tells you too late and without naming the field.

  What crosses is an explicit allowlist — the trial count and how it was
  obtained, the data hash, the verdict, the surface, the performance figures. A
  field added to `Study` later does not start leaving the machine because
  somebody forgot to exclude it. `best_params` is absent on purpose: the grid
  is frequently bigger intellectual property than the returns.

### Unchanged, and enforced

- `import alphaengine` still makes no network call and pulls in no HTTP client.
  Reporting is the one thing here that touches a network, so it is the one
  thing not imported until you ask for it — `report` resolves through a module
  `__getattr__` rather than at import time. Two tests hold that line: one
  asserts no HTTP client lands in `sys.modules`, the other that the top level
  never drags in the client subpackage.

- Still two runtime dependencies. Reporting uses `urllib` from the standard
  library rather than `requests`, because a third dependency would be a cost
  paid by every user of the offline half, who did not ask for it.

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
