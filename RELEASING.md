# Release Runbook (`alphaengine` 2.0)

This document describes the release engineering procedure for `alphaengine`.

> [!IMPORTANT]
> **Branching & Authorization Rules:**
> 1. All ongoing work must remain on `development`. Never commit directly to `main` or create side feature branches.
> 2. Merging `development` into `main`, creating git release tags, or publishing to PyPI requires **explicit authorization from William**.

---

## 1. Pre-Release Verification Gates

Before cutting a release candidate or final release, ensure every verification gate passes locally:

```bash
# 1. Version and documentation sync
python scripts/check_version_sync.py
python scripts/gen_docs.py --check

# 2. Linting, formatting, and typing
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy src

# 3. Unit test suite
python -m pytest

# 4. Golden test suite
python -m pytest tests -m golden -q

# 5. Wheel-installed isolated golden runner
python scripts/golden_wheel.py
```

All commands must exit 0 with zero warnings or errors.

---

## 2. Version Bump Procedure

1. Update `src/alphaengine/_version.py`:
   ```python
   __version__ = "2.0.0"  # or "2.0.0rc1", "2.1.0", etc.
   ```
2. Update `CHANGELOG.md`:
   - Ensure the topmost header matches the target version: `## [2.0.0] - YYYY-MM-DD`.
   - Ensure all output changes and breaking modifications are documented under `### 2.0.0: output changes`.
3. Validate version synchronization:
   ```bash
   python scripts/check_version_sync.py
   ```

---

## 3. Wheel Build & Verification

Verify the distribution package builds cleanly without network access:

```bash
python -m build --wheel --sdist --no-isolation --outdir dist/
```

Check the built artifacts:
```bash
python -m twine check dist/*
```

### Canonical 2.0.0 Built Artifact Digests (`QOS-173`, `QOS-175`)

- **Wheel (`dist/alphaengine-2.0.0-py3-none-any.whl`):**
  `sha256:cc83fb5f3fbaf97df613bdcc585b478e59573060776ee28c9de310218507ed1d`
- **Source Distribution (`dist/alphaengine-2.0.0.tar.gz`):**
  `sha256:9554a47959d55b340b7ae51d81e3ef6240baf3bbf04d7b366b1651effb8a579f`

These digests are pinned in `constraints/engine.txt` and verified across backend, runner, and MCP consumers via `scripts/check_engine_version.py`.

---

## 4. Release Promotion Gate (William's Approval Required)

Upon receiving William's explicit sign-off:

1. Fast-forward merge `development` into `main`:
   ```bash
   git checkout main
   git merge --ff-only development
   ```
2. Tag the release commit:
   ```bash
   git tag -a v2.0.0 -m "AlphaEngine 2.0.0 release"
   ```
3. Push to upstream:
   ```bash
   git push origin main
   git push origin v2.0.0
   ```
4. CI on GitHub Actions will run the test matrix, golden wheel verification, and publish the release to PyPI.
