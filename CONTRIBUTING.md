# Contributing to AlphaEngine

Thank you for your interest in contributing to AlphaEngine.

## Core design constraints

Before opening a pull request, please keep these architectural rules in mind:

1. **Two runtime dependencies (`numpy` and `scipy`).** Optional extras (`factors`, `agents`, `connectors`, `tui`) are lazy-imported so `import alphaengine` never requires network clients, LLM SDKs, or UI frameworks.
2. **Goldens are a public contract.** Computed values in `tests/test_goldens.py` are pinned so saved studies remain reproducible. Never modify a golden value simply to make a test pass; deliberate mathematical corrections require a version bump and a `CHANGELOG.md` entry explaining what changed and why.
3. **Data boundary.** Computation runs locally. Without explicit opt-in (`--share-series` or `ALPHAENGINE_SHARE_SERIES=1`), no list longer than 512 elements may leave the machine.
4. **Trial-count honesty.** Trial counts must be derived from the executed grid (`derived_from_grid`), explicitly marked (`asserted`), or `not_recorded` (which refuses an `edge` verdict).

## Local development setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,factors]"
```

## Required checks before submitting

Run all five CI gates locally before pushing:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src
python -m pytest
python -m pytest tests -m golden -q
python scripts/gen_docs.py --check
```

If you add or modify CLI commands or flags in `src/alphaengine/commands.py`, regenerate the README command table with:

```bash
python scripts/gen_docs.py --write
```
