"""The study: what was tried, what came back, and on what.

The artifact that crosses the boundary between the person who ran the research
and the person who has to decide on it. It carries the claim WITH its
qualifiers, the trial count, the deflation, the data identity, because the
documented failure mode is a headline Sharpe outliving its caveats into a deck
where the footnotes did not follow.

Writes to disk by default. No account, no upload, no network. The hosted
platform makes a study shareable and durable across a firm; it is not what makes
one exist.
"""

from typing import Any

from .schema import (
    SCHEMA_VERSION,
    SEAL_VERSION,
    Study,
    canonical_json,
    canonical_sha256,
    compute_content_sha256,
    load,
    save,
    seal_study,
    sealed_study_projection,
    verify_study,
    verify_study_mode,
)

__all__ = [
    "report",
    "ReportError",
    "Study",
    "save",
    "load",
    "seal_study",
    "sealed_study_projection",
    "verify_study",
    "verify_study_mode",
    "canonical_json",
    "canonical_sha256",
    "compute_content_sha256",
    "SCHEMA_VERSION",
    "SEAL_VERSION",
]


def __getattr__(name: str) -> Any:  # PEP 562
    """`report` and `ReportError` load on first use."""
    if name in ("report", "ReportError"):
        import importlib

        mod = importlib.import_module(".report", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
