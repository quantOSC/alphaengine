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
   __version__ = "2.0.1"  # or "2.0.1rc1", "2.1.0", etc.
   ```
2. Update `CHANGELOG.md`:
   - Ensure the topmost header matches the target version: `## [2.0.1] - YYYY-MM-DD`.
   - Ensure all output changes and breaking modifications are documented under `### 2.0.1: output changes`.
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

### Canonical 2.0.4 Built Artifact Digests (`QOS-173`, `QOS-175`)

- **Wheel (`dist/alphaengine-2.0.4-py3-none-any.whl`):**
  `sha256:3a0ac8e5cd4af2b8652d2c3e9b79483d294d4fe989647e6f7dc566a5ef2ca8dd`
- **Source Distribution (`dist/alphaengine-2.0.4.tar.gz`):**
  `sha256:0853b39416a6e5141680a3bd37d2726ffb61aec64f4a4fb33faa444240f11176`

### Canonical 2.0.3 Built Artifact Digests (`QOS-173`, `QOS-175`)

- **Wheel (`dist/alphaengine-2.0.3-py3-none-any.whl`):**
  `sha256:4d2ed5af5bdc49e0e22497020c0cca5fb5b2567635949694a04bfce92ca8c958`
- **Source Distribution (`dist/alphaengine-2.0.3.tar.gz`):**
  `sha256:68eab7be4f9893175e6c3a8846e831740c1855f6696b72fd9467e2be1b41e194`

### Canonical 2.0.2 Built Artifact Digests (`QOS-173`, `QOS-175`)

- **Wheel (`dist/alphaengine-2.0.2-py3-none-any.whl`):**
  `sha256:40b34e44f92234a766c1d21d67c54401a63aba07f287c85c6c4d2631f2a1c52a`
- **Source Distribution (`dist/alphaengine-2.0.2.tar.gz`):**
  `sha256:a503f85da27350a60954d77db3322b82366387d701772dadd279f60a00e18f90`

### Canonical 2.0.1 Built Artifact Digests (`QOS-173`, `QOS-175`)

- **Wheel (`dist/alphaengine-2.0.1-py3-none-any.whl`):**
  `sha256:e1d1fed6380dbbd65e32efde71c82474300a25d633920a41af312a17be8178bf`
- **Source Distribution (`dist/alphaengine-2.0.1.tar.gz`):**
  `sha256:afe2b95f5a118293ef79f2fddb7ab4dbbae3066eb589878fdee9a4091c5076f7`

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
   git tag -a v2.0.1 -m "AlphaEngine 2.0.1 release"
   ```
3. Push to upstream:
   ```bash
   git push origin main
   git push origin v2.0.1
   ```
4. CI on GitHub Actions will run the test matrix, golden wheel verification, and publish the release to PyPI.
