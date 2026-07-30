"""The study schema.

VERSIONING RULE
    `schema_version` is MAJOR.MINOR. A minor bump adds optional fields and older
    readers must keep working. A major bump means a field changed meaning or was
    removed, and a reader is entitled to refuse.

    `load()` refuses a major version it does not know rather than guessing. A
    study that silently half-parses is worse than one that fails loudly: the
    figures in it get quoted to investors.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import __version__

SCHEMA_VERSION = "1.0"

__all__ = ["Study", "save", "load", "SCHEMA_VERSION"]


@dataclass
class Study:
    """One piece of research, with everything needed to judge it.

    Field order below is the order a reader needs them in: what was studied,
    how hard it was searched, what came back, and what it can be checked
    against.
    """

    # ── what this is ────────────────────────────────────────────────────────
    label: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ── the search ──────────────────────────────────────────────────────────
    # n_trials is the count of configurations actually executed. `n_trials_source`
    # records HOW it was obtained, because a derived count and an asserted one
    # are not the same evidence and a reader is entitled to know which they hold.
    n_trials: int = 0
    n_trials_source: str = "derived_from_grid"
    grid_keys: list[str] = field(default_factory=list)
    n_failed_trials: int = 0

    # ── identity of the inputs ──────────────────────────────────────────────
    # A content hash, not a name. A label can be changed to escape a history;
    # an array cannot. Two studies over the same data are recognisably the same
    # segment however they were titled.
    data_hash: str = ""
    data_description: str = ""

    # ── what came back ──────────────────────────────────────────────────────
    verdict: str | None = None
    deflated_sharpe: float | None = None
    psr_vs_zero: float | None = None
    sr0_expected_max: float | None = None
    min_track_record_length: dict[str, Any] | None = None
    performance: dict[str, Any] = field(default_factory=dict)

    # Reported, deliberately not part of the verdict: PBO answers whether the
    # choice AMONG configurations was informative, which is a different question
    # from whether the edge is real.
    selection: dict[str, Any] | None = None

    # ── the shape of the neighbourhood ──────────────────────────────────────
    surface: dict[str, Any] = field(default_factory=dict)

    # ── optional, off by default ────────────────────────────────────────────
    # The parameter grid is frequently bigger IP than the return series.
    best_params: dict[str, Any] | None = None

    # ── provenance ──────────────────────────────────────────────────────────
    schema_version: str = SCHEMA_VERSION
    engine_version: str = __version__
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_sweep(
        cls,
        result: Any,
        *,
        label: str = "",
        data_description: str = "",
        notes: str = "",
        risk_free_rate: float = 0.0,
    ) -> Study:
        """Build a study from a SweepResult.

        Kept here rather than as a method on SweepResult so the sweep module has
        no opinion about serialisation, and a study can be assembled from
        something other than a sweep later.
        """
        v = result.verdict(risk_free_rate=risk_free_rate)
        return cls(
            label=label,
            n_trials=v["n_trials"],
            n_trials_source=v["n_trials_source"],
            grid_keys=list(result.grid_keys),
            n_failed_trials=sum(1 for t in result.trials if t.failed is not None),
            data_hash=v["data_hash"],
            data_description=data_description,
            verdict=v.get("verdict"),
            deflated_sharpe=v.get("deflated_sharpe"),
            psr_vs_zero=v.get("psr_vs_zero"),
            sr0_expected_max=v.get("sr0_expected_max"),
            min_track_record_length=v.get("min_track_record_length"),
            performance=v.get("performance", {}),
            selection=v.get("selection"),
            surface=result.surface(),
            best_params=v.get("best_params"),
            notes=notes,
        )


def save(study: Study, path: str | Path) -> Path:
    """Write a study to disk. Local file, no network, no account."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(study.to_dict(), indent=2, default=str), encoding="utf8")
    return p


def load(path: str | Path) -> Study:
    """Read a study, refusing a schema major version this build does not know."""
    raw = json.loads(Path(path).read_text(encoding="utf8"))

    got = str(raw.get("schema_version", "0"))
    if got.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"study schema version {got} cannot be read by alphaengine {__version__}, "
            f"which understands {SCHEMA_VERSION}. Refusing rather than partially parsing: "
            f"the figures in a study get quoted to investors."
        )

    known = {f for f in Study.__dataclass_fields__}
    # Unknown keys are dropped, not fatal, that is what makes a minor bump
    # forward-compatible for an older reader.
    return Study(**{k: v for k, v in raw.items() if k in known})
