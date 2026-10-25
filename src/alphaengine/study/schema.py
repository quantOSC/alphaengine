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

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import __version__

SCHEMA_VERSION = "1.2"

__all__ = [
    "Study",
    "save",
    "load",
    "seal_study",
    "verify_study",
    "verify_study_mode",
    "canonical_json",
    "canonical_sha256",
    "compute_content_sha256",
    "SCHEMA_VERSION",
]


def _normalize_canonical(node: Any) -> Any:
    """Normalize nested values with sorted keys and fixed 8-decimal float formatting for hashing."""
    if isinstance(node, dict):
        return {str(k): _normalize_canonical(node[k]) for k in sorted(node.keys(), key=str)}
    if isinstance(node, (list, tuple)):
        return [_normalize_canonical(v) for v in node]
    if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
        return node
    if isinstance(node, float):
        if not math.isfinite(node):
            return None
        val = 0.0 if node == 0.0 else node
        return f"{val:.8f}"
    return str(node)


def _normalize_legacy_round10(node: Any) -> Any:
    """Legacy normalization (round(float, 10) as numeric JSON) for backward-compatible verification."""
    if isinstance(node, dict):
        return {str(k): _normalize_legacy_round10(node[k]) for k in sorted(node.keys(), key=str)}
    if isinstance(node, (list, tuple)):
        return [_normalize_legacy_round10(v) for v in node]
    if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
        return node
    if isinstance(node, float):
        if not math.isfinite(node):
            return None
        return 0.0 if node == 0.0 else round(node, 10)
    return str(node)


def _strip_content_sha(payload: Any) -> Any:
    if isinstance(payload, Study):
        raw: Any = dict(payload.to_dict())
    elif isinstance(payload, dict):
        raw = dict(payload)
    else:
        return payload
    raw.pop("content_sha256", None)
    return raw


def canonical_json(study: Study | dict[str, Any] | Any) -> str:
    """Deterministic canonical JSON over a Study or arbitrary JSON-serializable payload."""
    raw = _strip_content_sha(study)
    normalized = _normalize_canonical(raw)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(payload: Study | dict[str, Any] | Any) -> str:
    """Single canonical 64-hex SHA-256 hash function used for content_sha256, seals, and CAS objects."""
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def compute_content_sha256(study: Study | dict[str, Any]) -> str:
    """Full 64-hex SHA-256 content hash over canonical JSON of the Study."""
    return canonical_sha256(study)


def _legacy_hashes(study: Study | dict[str, Any]) -> set[str]:
    """Compute legacy study hashes so studies sealed under pre-1.1.1 formats still verify as 'legacy'."""
    raw = _strip_content_sha(study)
    candidates: set[str] = set()
    # 1. Legacy 10-decimal numeric float normalization
    leg10 = json.dumps(
        _normalize_legacy_round10(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    candidates.add(hashlib.sha256(leg10.encode("utf-8")).hexdigest())
    # 2. Legacy plain sort_keys JSON (compact and default separators)
    for sep in ((",", ":"), (", ", ": ")):
        plain = json.dumps(raw, sort_keys=True, separators=sep, default=str, ensure_ascii=False)
        candidates.add(hashlib.sha256(plain.encode("utf-8")).hexdigest())
    return candidates


def verify_study_mode(study: Study | dict[str, Any]) -> str | None:
    """Return `'canonical'` if `content_sha256` matches, `'legacy'` for legacy seals, or `None`."""
    claimed = (
        study.content_sha256
        if isinstance(study, Study)
        else (study.get("content_sha256") if isinstance(study, dict) else None)
    )
    if not claimed or not isinstance(claimed, str):
        return None
    clean = claimed.strip().lower()
    if clean == canonical_sha256(study):
        return "canonical"
    if clean in _legacy_hashes(study):
        return "legacy"
    return None


def seal_study(study: Study) -> str:
    """Compute and store ``content_sha256`` on ``study``, returning the 64-hex digest."""
    digest = canonical_sha256(study)
    study.content_sha256 = digest
    return digest


def verify_study(study: Study) -> bool:
    """True when ``study.content_sha256`` is present and matches canonical or legacy hash."""
    return verify_study_mode(study) is not None


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

    # ── optional, 1.1 ────────────────────────────────────────────────────────
    # A named data-generating process and the chart hints a portal draws from
    # figures. Absent on studies that never asked; older readers drop both.
    process: dict[str, Any] | None = None
    charts: list[dict[str, Any]] | None = None

    # ── optional, 1.2 ────────────────────────────────────────────────────────
    # Per-trial ledger (parameter hash + Sharpe per trial, plus failed trials)
    # and the canonical SHA-256 seal over the study content.
    trial_ledger: list[dict[str, Any]] | None = None
    content_sha256: str | None = None

    # ── provenance ──────────────────────────────────────────────────────────
    schema_version: str = SCHEMA_VERSION
    engine_version: str = __version__
    notes: str = ""
    assignment_id: str | None = None
    thesis_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def seal(self) -> str:
        """Seal this study in place and return its 64-hex ``content_sha256``."""
        return seal_study(self)

    def verify(self) -> bool:
        """Verify that ``content_sha256`` matches this study's canonical content."""
        return verify_study(self)

    def report(self, **kw: Any) -> dict[str, Any]:
        """Send this study to a QuantOS workspace. See `alphaengine.report`.

        Imported inside the method so that `import alphaengine` still touches no
        networking code at all — the offline half must stay offline by
        construction, not by good intentions.
        """
        from .report import report as _report

        return _report(self, **kw)

    @classmethod
    def from_sweep(
        cls,
        result: Any,
        *,
        label: str = "",
        data_description: str = "",
        notes: str = "",
        risk_free_rate: float | None = None,
        assignment_id: str | None = None,
        thesis_id: str | None = None,
        include_ledger: bool = False,
    ) -> Study:
        """Build a study from a SweepResult.

        Kept here rather than as a method on SweepResult so the sweep module has
        no opinion about serialisation, and a study can be assembled from
        something other than a sweep later.
        """
        v = result.verdict(risk_free_rate=risk_free_rate)
        ledger = result.ledger() if (include_ledger and hasattr(result, "ledger")) else None
        study = cls(
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
            trial_ledger=ledger,
            notes=notes,
            assignment_id=assignment_id,
            thesis_id=thesis_id,
        )
        study.seal()
        return study


def save(study: Study, path: str | Path) -> Path:
    """Write a study to disk. Local file, no network, no account."""
    if study.content_sha256 is None:
        study.seal()
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
