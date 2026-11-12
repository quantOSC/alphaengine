"""Global pytest isolation fixtures for AlphaEngine."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_ceilings() -> str:
    ceilings = [str(REPO_ROOT.resolve()), str(Path.home().resolve())]
    existing = os.environ.get("ALPHAENGINE_CEILING_DIRS", "").strip()
    if existing:
        for part in existing.split(os.pathsep):
            if part and part not in ceilings:
                ceilings.append(part)
    return os.pathsep.join(ceilings)


def pytest_configure(config: pytest.Config) -> None:
    """Ensure module/session-scoped fixtures (e.g. wheel goldens) never touch ambient `.alphaengine/`."""
    _ = config
    os.environ["ALPHAENGINE_CEILING_DIRS"] = _default_ceilings()


@pytest.fixture(autouse=True)
def _isolate_ambient_alphaengine_project(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep `ALPHAENGINE_CEILING_DIRS` active across per-test `monkeypatch` environments."""
    monkeypatch.setenv("ALPHAENGINE_CEILING_DIRS", _default_ceilings())
