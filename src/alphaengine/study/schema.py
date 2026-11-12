"""The study schema.

VERSIONING RULE
    `schema_version` is MAJOR.MINOR. A minor bump adds optional fields and older
    readers must keep working. A major bump means a field changed meaning or was
    removed, and a reader is entitled to refuse.

    `load()` refuses a major version it does not know rather than guessing. A
    study that silently half-parses is worse than one that fails loudly: the
    figures in it get quoted to investors.

SEAL PROJECTION (`seal_version = 2`, `QOS-161`)
    `Study.seal()` hashes `sealed_study_projection(study)` — the exact
    deterministic allowlisted wire projection (excluding local-only `best_params`)
    with explicit keys, sorted dictionaries, and 8-decimal float formatting.
    Because `ae push` and `study.report()` transmit that exact projection,
    client-side `Study.verify()` and server-side seal verification always compute
    the identical 64-hex `content_sha256`. Studies sealed under 1.x (`seal_version`
    absent or `1`) remain verifiable via `verify_study_mode()` (`"legacy_v1"` /
    `"legacy"`).
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._version import current_engine_version

SCHEMA_VERSION = "1.2"
SEAL_VERSION = 2
_MAX_TRIAL_LEDGER = 512

__all__ = [
    "SEAL_VERSION",
    "SCHEMA_VERSION",
    "Study",
    "canonical_json",
    "canonical_sha256",
    "compute_content_sha256",
    "load",
    "save",
    "seal_study",
    "sealed_study_projection",
    "verify_study",
    "verify_study_mode",
]


def _unwrap_scalar_or_array(node: Any) -> Any:
    """Unwrap NumPy scalars and arrays to native Python primitives without importing numpy."""
    if hasattr(node, "dtype") and hasattr(node, "tolist") and callable(node.tolist):
        return node.tolist()
    return node


def _sanitize_study_json(node: Any) -> Any:
    """Convert non-finite floats to None and unwrap NumPy scalars/arrays for RFC 8259 JSON."""
    node = _unwrap_scalar_or_array(node)
    if isinstance(node, Mapping):
        return {str(k): _sanitize_study_json(node[k]) for k in node}
    if isinstance(node, (list, tuple)):
        return [_sanitize_study_json(v) for v in node]
    if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
        return node
    if isinstance(node, numbers.Integral):
        return int(node)
    if isinstance(node, (float, numbers.Real)):
        fval = float(node)
        return fval if math.isfinite(fval) else None
    return node


def _normalize_canonical(node: Any) -> Any:
    """Normalize nested values with sorted keys and fixed 8-decimal float formatting for hashing."""
    node = _unwrap_scalar_or_array(node)
    if isinstance(node, Mapping):
        return {str(k): _normalize_canonical(node[k]) for k in sorted(node.keys(), key=str)}
    if isinstance(node, (list, tuple)):
        return [_normalize_canonical(v) for v in node]
    if isinstance(node, bool) or node is None or isinstance(node, (int, str)):
        return node
    if isinstance(node, numbers.Integral):
        return int(node)
    if isinstance(node, (float, numbers.Real)):
        fval = float(node)
        if not math.isfinite(fval):
            return None
        val = 0.0 if fval == 0.0 else fval
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


def sealed_study_projection(study: Study | Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical `seal_version = 2` wire projection of a Study (`QOS-161`).

    Every field in this projection travels unchanged in `ae push` and
    `study.report()`. Local-only `best_params` is intentionally excluded so
    parameter grids never leave the researcher's machine while client and server
    compute the exact same `content_sha256`.
    """
    src: Mapping[str, Any] = study.to_dict() if isinstance(study, Study) else study
    raw_ledger = src.get("trial_ledger")
    bounded_ledger = list(raw_ledger)[:_MAX_TRIAL_LEDGER] if isinstance(raw_ledger, list) else None
    dsr_prob = src.get("dsr_probability")
    def_sr = src.get("deflated_sharpe")
    if dsr_prob is None and isinstance(def_sr, (int, float)) and not isinstance(def_sr, bool):
        dsr_prob = float(def_sr)
    if def_sr is None and isinstance(dsr_prob, (int, float)) and not isinstance(dsr_prob, bool):
        def_sr = float(dsr_prob)

    proj = {
        "schema_version": str(src.get("schema_version") or SCHEMA_VERSION),
        "seal_version": SEAL_VERSION,
        "engine_version": str(src.get("engine_version") or current_engine_version()),
        "created_at": str(src.get("created_at") or ""),
        "label": str(src.get("label") or ""),
        "n_trials": int(src.get("n_trials") or 0),
        "n_trials_source": str(src.get("n_trials_source") or "derived_from_grid"),
        "grid_keys": [str(k) for k in (src.get("grid_keys") or [])],
        "n_failed_trials": int(src.get("n_failed_trials") or 0),
        "data_hash": str(src.get("data_hash") or ""),
        "data_description": str(src.get("data_description") or ""),
        "verdict": src.get("verdict"),
        "dsr_probability": dsr_prob,
        "deflation_applied": src.get("deflation_applied"),
        "deflated_sharpe": def_sr,
        "psr_vs_zero": src.get("psr_vs_zero"),
        "sr0_expected_max": src.get("sr0_expected_max"),
        "min_track_record_length": src.get("min_track_record_length"),
        "selection": src.get("selection"),
        "surface": dict(src.get("surface") or {}),
        "performance": dict(src.get("performance") or {}),
        "process": src.get("process"),
        "charts": list(src["charts"]) if isinstance(src.get("charts"), list) else None,
        "trial_ledger": bounded_ledger,
        "manifest": dict(src["manifest"]) if isinstance(src.get("manifest"), Mapping) else None,
        "notes": str(src.get("notes") or ""),
        "assignment_id": src.get("assignment_id"),
        "thesis_id": src.get("thesis_id"),
    }
    res: dict[str, Any] = _sanitize_study_json(proj)
    return res


def canonical_json(study: Study | dict[str, Any] | Any) -> str:
    """Deterministic canonical JSON over a Study or arbitrary JSON-serializable payload."""
    if isinstance(study, Study):
        raw: Any = sealed_study_projection(study)
    else:
        raw = _strip_content_sha(study)
    normalized = _normalize_canonical(raw)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(study_or_object: Study | dict[str, Any] | Any) -> str:
    """Single canonical 64-hex SHA-256 hash function used for content_sha256, seals, and CAS objects."""
    # codeql[py/weak-sensitive-data-hashing] Content integrity digest for non-password CAS objects/studies
    if isinstance(study_or_object, (bytes, bytearray)):
        # codeql[py/weak-sensitive-data-hashing]
        return hashlib.sha256(study_or_object).hexdigest()
    # codeql[py/weak-sensitive-data-hashing]
    return hashlib.sha256(canonical_json(study_or_object).encode("utf-8")).hexdigest()


def compute_content_sha256(study: Study | Mapping[str, Any]) -> str:
    """Full 64-hex SHA-256 content hash over the canonical `seal_version = 2` Study projection."""
    return canonical_sha256(sealed_study_projection(study))


def _legacy_v1_dict(study: Study | Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the 1.x full-Study dict (prior to `seal_version = 2` fields) for legacy verification."""
    raw = dict(study.to_dict()) if isinstance(study, Study) else dict(study)
    raw.pop("content_sha256", None)
    if "seal_version" not in study.__dict__ if isinstance(study, Study) else "seal_version" not in study:
        raw.pop("seal_version", None)
    # Strip 2.0-only optional fields when absent in a 1.x file or Study(seal_version=1)
    is_legacy_study = isinstance(study, Study) and int(getattr(study, "seal_version", SEAL_VERSION) or 1) < 2
    for v2_key in ("seal_version", "dsr_probability", "deflation_applied", "manifest"):
        if (isinstance(study, Mapping) and v2_key not in study) or (
            is_legacy_study and (v2_key == "seal_version" or raw.get(v2_key) is None)
        ):
            raw.pop(v2_key, None)
    return raw


def _legacy_hashes(study: Study | Mapping[str, Any]) -> tuple[str, set[str]]:
    """Compute legacy v1 canonical hash and pre-1.1.1 round10/plain hashes."""
    raw = _legacy_v1_dict(study)
    # codeql[py/weak-sensitive-data-hashing] Legacy non-password study verification hash
    v1_canonical = hashlib.sha256(
        json.dumps(
            _normalize_canonical(raw),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    older: set[str] = set()
    leg10 = json.dumps(
        _normalize_legacy_round10(raw), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    # codeql[py/weak-sensitive-data-hashing] Legacy non-password study verification hash
    older.add(hashlib.sha256(leg10.encode("utf-8")).hexdigest())
    for sep in ((",", ":"), (", ", ": ")):
        plain = json.dumps(raw, sort_keys=True, separators=sep, default=str, ensure_ascii=False)
        # codeql[py/weak-sensitive-data-hashing] Legacy non-password study verification hash
        older.add(hashlib.sha256(plain.encode("utf-8")).hexdigest())
    return v1_canonical, older


def verify_study_mode(study: Study | Mapping[str, Any]) -> str | None:
    """Return `'canonical'` (`seal_version=2`), `'legacy_v1'` (`1.1.x`), `'legacy'` (`<=1.1.0`), or `None`."""
    claimed = (
        study.content_sha256
        if isinstance(study, Study)
        else (study.get("content_sha256") if isinstance(study, Mapping) else None)
    )
    if not claimed or not isinstance(claimed, str):
        return None
    clean = claimed.strip().lower()
    if clean == compute_content_sha256(study):
        return "canonical"
    v1_hash, older_hashes = _legacy_hashes(study)
    if clean == v1_hash:
        return "legacy_v1"
    if clean in older_hashes:
        return "legacy"
    return None


def seal_study(study: Study) -> str:
    """Compute and store `seal_version = 2` ``content_sha256`` on ``study``."""
    study.seal_version = SEAL_VERSION
    if study.dsr_probability is None and study.deflated_sharpe is not None:
        study.dsr_probability = study.deflated_sharpe
    if study.deflated_sharpe is None and study.dsr_probability is not None:
        study.deflated_sharpe = study.dsr_probability
    digest = compute_content_sha256(study)
    study.content_sha256 = digest
    return digest


def verify_study(study: Study | Mapping[str, Any]) -> bool:
    """True when ``content_sha256`` is present and matches canonical or legacy hash."""
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
    n_trials: int = 0
    n_trials_source: str = "derived_from_grid"
    grid_keys: list[str] = field(default_factory=list)
    n_failed_trials: int = 0

    # ── identity of the inputs ──────────────────────────────────────────────
    data_hash: str = ""
    data_description: str = ""

    # ── what came back ──────────────────────────────────────────────────────
    verdict: str | None = None
    dsr_probability: float | None = None
    deflated_sharpe: float | None = None
    deflation_applied: bool | None = None
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

    # ── optional, off by default (NEVER included in `sealed_study_projection`) ──
    best_params: dict[str, Any] | None = None

    # ── optional, 1.1 ────────────────────────────────────────────────────────
    process: dict[str, Any] | None = None
    charts: list[dict[str, Any]] | None = None

    # ── optional, 1.2 / 2.0 ─────────────────────────────────────────────────
    trial_ledger: list[dict[str, Any]] | None = None
    manifest: dict[str, Any] | None = None
    seal_version: int = SEAL_VERSION
    content_sha256: str | None = None

    # ── provenance ──────────────────────────────────────────────────────────
    schema_version: str = SCHEMA_VERSION
    engine_version: str = field(default_factory=current_engine_version)
    notes: str = ""
    assignment_id: str | None = None
    thesis_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_sealed_projection(self) -> dict[str, Any]:
        """Return the exact `seal_version = 2` wire projection hashed by `seal()`."""
        return sealed_study_projection(self)

    def seal(self) -> str:
        """Seal this study in place and return its 64-hex ``content_sha256``."""
        return seal_study(self)

    def verify(self) -> bool:
        """Verify that ``content_sha256`` matches this study's canonical or legacy seal."""
        return verify_study(self)

    def report(self, **kw: Any) -> dict[str, Any]:
        """Send this study to a QuantOS workspace. See `alphaengine.report`."""
        from .report import report as _report

        return _report(self, **kw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Study:
        """Construct a `Study` from a deserialized dictionary, validating major schema version."""
        ver = current_engine_version()
        got = str(raw.get("schema_version", SCHEMA_VERSION))
        if got.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise ValueError(
                f"study schema version {got} cannot be read by alphaengine {ver}, "
                f"which understands {SCHEMA_VERSION}. Refusing rather than partially parsing: "
                f"the figures in a study get quoted to investors."
            )
        known = set(cls.__dataclass_fields__)
        filtered: dict[str, Any] = {k: v for k, v in raw.items() if k in known}
        if "seal_version" not in raw:
            filtered["seal_version"] = 1
        return cls(**filtered)

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
        """Build and seal a Study from a `SweepResult`."""
        v = result.verdict(risk_free_rate=risk_free_rate)
        ledger = (
            list(result.ledger())[:_MAX_TRIAL_LEDGER]
            if (include_ledger and hasattr(result, "ledger"))
            else None
        )
        study = cls(
            label=label,
            n_trials=v["n_trials"],
            n_trials_source=v["n_trials_source"],
            grid_keys=list(result.grid_keys),
            n_failed_trials=sum(1 for t in result.trials if t.failed is not None),
            data_hash=v["data_hash"],
            data_description=data_description,
            verdict=v.get("verdict"),
            dsr_probability=v.get("dsr_probability"),
            deflated_sharpe=v.get("deflated_sharpe"),
            deflation_applied=v.get("deflation_applied"),
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
    sanitized = _sanitize_study_json(study.to_dict())
    p.write_text(
        json.dumps(sanitized, indent=2, default=str, allow_nan=False),
        encoding="utf8",
    )
    return p


def load(path: str | Path) -> Study:
    """Read a study, refusing a schema major version this build does not know."""
    raw = json.loads(Path(path).read_text(encoding="utf8"))
    if "schema_version" not in raw:
        raw["schema_version"] = "0"
    return Study.from_dict(raw)
