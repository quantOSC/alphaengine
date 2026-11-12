r"""Shared Sync Contract v2 (`sync_contract = 2`) for `alphaengine` <-> `quantOS` (`QOS-152`).

This module is the single source of truth for the wire schema exchanged between
`ae push` / `ae pull` (`POST /api/workspace/sync` and `GET /api/workspace/pull`)
and the `quantOS` backend (`workspace_sync`).

CANONICAL V2 FIELD MAPPING & REMOVED V1 ALIASES
===============================================

| Object        | Canonical v2 Field | Type                       | Removed / Legacy v1 Aliases          |
|---------------|--------------------|----------------------------|--------------------------------------|
| Envelope      | `sync_contract`    | `int` (`2`)                | *(absent in v1)*                     |
| Envelope      | `engine_version`   | `str`                      | `project.engine_version` only        |
| Study         | `seal_version`     | `int` (`2`)                | *(implicit full-dict hash in v1)*    |
| Study         | `content_sha256`   | `str` (64-hex)             | *(mismatched `_payload` projection)* |
| Study         | `dsr_probability`  | `float \| None`            | `deflated_sharpe` (kept as alias)    |
| Run           | `run_id`           | `str`                      | `id`                                 |
| Run           | `share_series`     | `bool` (default `False`)   | *(global-only in v1)*                |
| Run           | `metrics`          | `dict[str, float \| None]` | *(ad-hoc / `1.20` server fallback)*  |
| Run           | `series`           | `dict[str, list[float]] \| None` | *(inline unguarded lists)*     |
| Run           | *(removed)*        | —                          | `cumulative_trials`, `project_trials`|
| Idea          | `idea_id`          | `str`                      | `id`                                 |
| Idea          | `kill_rule`        | `str`                      | `kill_if`                            |
| Idea          | `thesis`           | `str`                      | `statement`, `name`                  |
| TrialEvent    | `attempt_id`       | `str` (UUID)               | *(replaced cumulative integers)*     |
| TrialEvent    | `status`           | `"ok" \| "failed"`         | `failed`                             |
| TrialEvent    | `executed_at`      | `str` (ISO-8601 UTC)       | `timestamp`                          |
| PushResponse  | `items`            | `list[PushItemAck]`        | `n_studies`, `n_runs`, `n_ideas`     |
| PullRequest   | `since`, `limit`   | `str \| None`, `int`       | `GET /api/workspace/sync`            |
| PullResponse  | `next_cursor`      | `str \| None`              | *(unpaginated in v1)*                |

STRUCTURED CLI EXIT CODES (`QOS-167`)
=====================================
- `0` (`EXIT_OK`): all submitted items `accepted` or `duplicate` (or clean pull)
- `2` (`EXIT_PARTIAL_REJECTED`): at least one item `rejected` alongside accepted/duplicate items
- `3` (`EXIT_ALL_REJECTED`): every submitted item (`>= 1`) was `rejected`
- `4` (`EXIT_AUTH_ERROR`): authentication or entitlement error (`401`/`403` or missing `ae_live_` key)
- `5` (`EXIT_NETWORK_OR_SERVER_ERROR`): network failure, timeout, `5xx`, or unexpected endpoint error
- `6` (`EXIT_CONTRACT_MISMATCH`): `sync_contract` mismatch (`409`/`426` or incompatible wire version)
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._version import SYNC_CONTRACT_VERSION, current_engine_version
from ..core.conventions import normalize_verdict
from ..project import validate_safe_id
from ..study.report import MAX_FIGURE_LIST, ReportError
from ..study.schema import SEAL_VERSION, Study, sealed_study_projection, verify_study_mode

__all__ = [
    "SYNC_CONTRACT_VERSION",
    "EXIT_OK",
    "EXIT_PARTIAL_REJECTED",
    "EXIT_ALL_REJECTED",
    "EXIT_AUTH_ERROR",
    "EXIT_NETWORK_OR_SERVER_ERROR",
    "EXIT_CONTRACT_MISMATCH",
    "CANONICAL_METRIC_KEYS",
    "SyncError",
    "SyncValidationError",
    "SyncAuthError",
    "SyncNetworkError",
    "SyncServerError",
    "SyncContractMismatchError",
    "SealedStudyWire",
    "SyncRunPayload",
    "SyncIdeaPayload",
    "SyncTrialEventPayload",
    "PushRequest",
    "PushItemAck",
    "PushResponse",
    "PullRequest",
    "PullResponse",
    "compute_push_exit_code",
    "extract_optional_metrics",
    "export_json_schemas",
]

EXIT_OK = 0
EXIT_PARTIAL_REJECTED = 2
EXIT_ALL_REJECTED = 3
EXIT_AUTH_ERROR = 4
EXIT_NETWORK_OR_SERVER_ERROR = 5
EXIT_CONTRACT_MISMATCH = 6

CANONICAL_METRIC_KEYS: tuple[str, ...] = (
    "sharpe_annualized",
    "cagr",
    "max_drawdown",
    "total_return",
    "volatility_annualized",
    "dsr_probability",
    "psr_vs_zero",
    "pbo",
    "turnover",
)

_FORBIDDEN_WIRE_KEYS = frozenset(
    {
        "raw_rows",
        "dataset_rows",
        "prices",
        "closes",
        "equity_curve",
        "weights_path",
        "best_params",
        "matrix",
        "repro_spec",
        "cumulative_trials",
        "project_trials",
        "kill_if",
    }
)
_HEX64_RE_WIRE = re.compile(r"^[0-9a-fA-F]{64}$")

_VALID_ITEM_TYPES = frozenset({"study", "run", "idea", "trial_event"})
_VALID_ACK_STATUSES = frozenset({"accepted", "duplicate", "rejected"})
_VALID_TRIAL_STATUSES = frozenset({"ok", "failed"})


class SyncError(ReportError):
    """Base error for workspace sync operations carrying a structured CLI `exit_code`."""

    exit_code: int = EXIT_NETWORK_OR_SERVER_ERROR

    def __init__(self, message: str, *, exit_code: int | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = int(exit_code)
        self.status_code = status_code


class SyncValidationError(ValueError):
    """Raised when a sync v2 wire payload violates schema invariants locally."""


class SyncAuthError(SyncError):
    """Raised on `401`/`403` or missing `ae_live_` firm credential (`exit_code = 4`)."""

    exit_code = EXIT_AUTH_ERROR


class SyncNetworkError(SyncError):
    """Raised when the remote host cannot be reached or times out (`exit_code = 5`)."""

    exit_code = EXIT_NETWORK_OR_SERVER_ERROR


class SyncServerError(SyncError):
    """Raised when the remote returns `5xx` or an unexpected `4xx` (`exit_code = 5`)."""

    exit_code = EXIT_NETWORK_OR_SERVER_ERROR


class SyncContractMismatchError(SyncError):
    """Raised when client and server `sync_contract` versions disagree (`exit_code = 6`)."""

    exit_code = EXIT_CONTRACT_MISMATCH


def _opt_float(value: Any) -> float | None:
    """Coerce a finite scalar to `float`, or `None` when absent/non-finite (`QOS-165`)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    return None


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def extract_optional_metrics(figures: Mapping[str, Any] | None) -> dict[str, float | None]:
    """Extract canonical scalar metrics from `figures` without inventing fallback numbers (`QOS-165`).

    Every absent metric is explicitly `None` (`null` on JSON wire), never `0.0` or `1.20`.
    """
    src = figures or {}
    perf = src.get("performance") if isinstance(src.get("performance"), Mapping) else {}
    perf_map: Mapping[str, Any] = perf if isinstance(perf, Mapping) else {}

    def _pick(*keys: str) -> float | None:
        for key in keys:
            if key in src and src[key] is not None:
                got = _opt_float(src[key])
                if got is not None:
                    return got
            if key in perf_map and perf_map[key] is not None:
                got = _opt_float(perf_map[key])
                if got is not None:
                    return got
        return None

    return {
        "sharpe_annualized": _pick("sharpe_annualized", "sharpe"),
        "cagr": _pick("cagr"),
        "max_drawdown": _pick("max_drawdown", "max_drawdown_pct"),
        "total_return": _pick("total_return", "total_return_pct"),
        "volatility_annualized": _pick("volatility_annualized", "volatility_annualized_pct", "vol_ann"),
        "dsr_probability": _pick("dsr_probability", "deflated_sharpe"),
        "psr_vs_zero": _pick("psr_vs_zero"),
        "pbo": _pick("pbo"),
        "turnover": _pick("turnover", "avg_turnover"),
    }


def _check_no_unshared_series(node: Any, *, share_series: bool, path: str) -> None:
    if isinstance(node, Mapping):
        for k, v in node.items():
            key_str = str(k)
            if key_str.lower() in _FORBIDDEN_WIRE_KEYS:
                raise SyncValidationError(f"Forbidden field {path}.{key_str} is not permitted in sync v2.")
            _check_no_unshared_series(v, share_series=share_series, path=f"{path}.{key_str}")
    elif isinstance(node, (list, tuple)):
        if not share_series and len(node) > MAX_FIGURE_LIST:
            raise SyncValidationError(
                f"Refusing {path}: {len(node)} elements exceeds MAX_FIGURE_LIST ({MAX_FIGURE_LIST}) "
                "while share_series=False."
            )
        for idx, item in enumerate(node):
            _check_no_unshared_series(item, share_series=share_series, path=f"{path}[{idx}]")


@dataclass(frozen=True)
class SealedStudyWire:
    """Wire wrapper for a sealed Study using the exact `sealed_study_projection` (`QOS-161`)."""

    projection: dict[str, Any]
    content_sha256: str
    seal_version: int = SEAL_VERSION

    @classmethod
    def from_study(cls, study: Study) -> SealedStudyWire:
        if not study.content_sha256:
            study.seal()
        mode = verify_study_mode(study)
        if mode is None:
            raise SyncValidationError(
                f"Refusing to serialize Study {study.label or '(unlabeled)'!r}: content_sha256 seal mismatch."
            )
        if int(getattr(study, "seal_version", 1) or 1) != SEAL_VERSION or mode != "canonical":
            # Re-seal legacy v1 studies onto seal_version=2 projection so wire verification is deterministic,
            # UNLESS caller tampered with content (caught above by mode is None).
            study.seal()
        proj = sealed_study_projection(study)
        sha = str(study.content_sha256 or "")
        return cls(projection=proj, content_sha256=sha, seal_version=SEAL_VERSION)

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.projection)
        out["seal_version"] = int(self.seal_version)
        out["content_sha256"] = self.content_sha256
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SealedStudyWire:
        sha = str(raw.get("content_sha256") or "").strip()
        if not _HEX64_RE_WIRE.fullmatch(sha):
            raise SyncValidationError("SealedStudyWire requires a 64-hex content_sha256.")
        sha = sha.lower()
        sv = int(raw.get("seal_version") or SEAL_VERSION)
        st = Study.from_dict({**dict(raw), "content_sha256": sha})
        mode = verify_study_mode(st)
        if mode is None:
            raise SyncValidationError("SealedStudyWire failed content_sha256 verification.")
        proj = sealed_study_projection(st)
        return cls(projection=proj, content_sha256=sha, seal_version=sv)


@dataclass
class SyncRunPayload:
    """Canonical v2 wire payload for a single recorded run (`QOS-152`, `QOS-165`, `QOS-166`)."""

    run_id: str
    timestamp: str
    command: str
    idea_id: str | None = None
    family: str = "ad_hoc"
    params_hash: str = ""
    dataset_hashes: list[str] = field(default_factory=list)
    n_trials: int | None = None
    run_n_trials: int | None = None
    n_trials_source: str | None = None
    verdict: str | None = None
    engine_version: str = field(default_factory=current_engine_version)
    object_sha256: str | None = None
    manifest: dict[str, Any] | None = None
    share_series: bool = False
    metrics: dict[str, float | None] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    series: dict[str, list[float]] | None = None

    def validate(self) -> SyncRunPayload:
        if not self.run_id or not str(self.run_id).strip():
            raise SyncValidationError("SyncRunPayload.run_id must be non-empty.")
        try:
            self.run_id = validate_safe_id(str(self.run_id), "run_id")
            if self.idea_id is not None and str(self.idea_id).strip():
                self.idea_id = validate_safe_id(str(self.idea_id), "idea_id")
        except ValueError as exc:
            raise SyncValidationError(str(exc)) from exc
        if not self.command or not str(self.command).strip():
            raise SyncValidationError(f"SyncRunPayload({self.run_id}).command must be non-empty.")
        if not isinstance(self.share_series, bool):
            raise SyncValidationError(f"SyncRunPayload({self.run_id}).share_series must be a bool.")
        if not self.share_series and self.series:
            raise SyncValidationError(
                f"SyncRunPayload({self.run_id}) carries non-empty `series` while share_series=False."
            )
        for mk, mv in self.metrics.items():
            if mv is not None and (
                isinstance(mv, bool) or not isinstance(mv, (int, float)) or not math.isfinite(mv)
            ):
                raise SyncValidationError(
                    f"SyncRunPayload({self.run_id}).metrics[{mk!r}] must be finite float or None."
                )
        _check_no_unshared_series(
            self.figures,
            share_series=self.share_series,
            path=f"runs[{self.run_id}].figures",
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        norm_verdict: str | None = None
        if self.verdict:
            try:
                norm_verdict = normalize_verdict(self.verdict)
            except ValueError:
                norm_verdict = str(self.verdict)
        metrics_clean: dict[str, float | None] = {
            k: _opt_float(self.metrics.get(k)) for k in CANONICAL_METRIC_KEYS
        }
        for extra_k, extra_v in self.metrics.items():
            if extra_k not in metrics_clean:
                metrics_clean[str(extra_k)] = _opt_float(extra_v)
        return {
            "run_id": str(self.run_id),
            "timestamp": str(self.timestamp or ""),
            "command": str(self.command),
            "idea_id": str(self.idea_id) if self.idea_id else None,
            "family": str(self.family or "ad_hoc"),
            "params_hash": str(self.params_hash or ""),
            "dataset_hashes": [str(h) for h in self.dataset_hashes if h],
            "n_trials": _opt_int(self.n_trials),
            "run_n_trials": _opt_int(self.run_n_trials),
            "n_trials_source": str(self.n_trials_source) if self.n_trials_source else None,
            "verdict": norm_verdict,
            "engine_version": str(self.engine_version or current_engine_version()),
            "object_sha256": str(self.object_sha256) if self.object_sha256 else None,
            "manifest": dict(self.manifest) if isinstance(self.manifest, Mapping) else None,
            "share_series": bool(self.share_series),
            "metrics": metrics_clean,
            "figures": dict(self.figures),
            "series": dict(self.series) if (self.share_series and self.series) else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SyncRunPayload:
        for bad in ("cumulative_trials", "project_trials"):
            if bad in raw:
                raise SyncValidationError(
                    f"Removed v1 field {bad!r} is not allowed on SyncRunPayload; send `trial_events` instead."
                )
        raw_metrics = raw.get("metrics")
        metrics_map: dict[str, float | None] = {}
        if isinstance(raw_metrics, Mapping):
            for k, v in raw_metrics.items():
                metrics_map[str(k)] = _opt_float(v)
        else:
            figs_in = raw.get("figures")
            metrics_map = extract_optional_metrics(figs_in if isinstance(figs_in, Mapping) else None)
        raw_series = raw.get("series")
        series_map: dict[str, list[float]] | None = None
        if isinstance(raw_series, Mapping):
            series_map = {
                str(k): [float(x) for x in v]
                for k, v in raw_series.items()
                if isinstance(v, Sequence) and not isinstance(v, (str, bytes))
            }
        share_flag = bool(raw.get("share_series", False))
        inst = cls(
            run_id=str(raw.get("run_id") or "").strip(),
            timestamp=str(raw.get("timestamp") or ""),
            command=str(raw.get("command") or "").strip(),
            idea_id=str(raw["idea_id"]) if raw.get("idea_id") else None,
            family=str(raw.get("family") or "ad_hoc"),
            params_hash=str(raw.get("params_hash") or ""),
            dataset_hashes=[str(h) for h in (raw.get("dataset_hashes") or []) if h],
            n_trials=_opt_int(raw.get("n_trials")),
            run_n_trials=_opt_int(raw.get("run_n_trials")),
            n_trials_source=str(raw["n_trials_source"]) if raw.get("n_trials_source") else None,
            verdict=str(raw["verdict"]) if raw.get("verdict") else None,
            engine_version=str(raw.get("engine_version") or current_engine_version()),
            object_sha256=str(raw["object_sha256"]) if raw.get("object_sha256") else None,
            manifest=dict(raw["manifest"]) if isinstance(raw.get("manifest"), Mapping) else None,
            share_series=share_flag,
            metrics=metrics_map,
            figures=dict(raw["figures"]) if isinstance(raw.get("figures"), Mapping) else {},
            series=series_map,
        )
        return inst.validate()


@dataclass
class SyncIdeaPayload:
    """Canonical v2 wire payload for a research hypothesis (`QOS-163`).

    Uses `idea_id` and `kill_rule` exclusively on the wire (legacy `id` and `kill_if`
    are rejected on v2 push payloads).
    """

    idea_id: str
    thesis: str
    kill_rule: str = ""
    family: str = "general"
    status: str = "open"
    created_at: str = ""
    updated_at: str = ""
    object_sha256: str | None = None

    def validate(self) -> SyncIdeaPayload:
        if not self.idea_id or not str(self.idea_id).strip():
            raise SyncValidationError("SyncIdeaPayload.idea_id must be non-empty.")
        try:
            self.idea_id = validate_safe_id(str(self.idea_id), "idea_id")
        except ValueError as exc:
            raise SyncValidationError(str(exc)) from exc
        if not self.thesis or not str(self.thesis).strip():
            raise SyncValidationError(f"SyncIdeaPayload({self.idea_id}).thesis must be non-empty.")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "idea_id": str(self.idea_id).strip(),
            "thesis": str(self.thesis).strip(),
            "kill_rule": str(self.kill_rule or "").strip(),
            "family": str(self.family or "general").strip(),
            "status": str(self.status or "open").strip(),
            "created_at": str(self.created_at or ""),
            "updated_at": str(self.updated_at or ""),
            "object_sha256": str(self.object_sha256) if self.object_sha256 else None,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, strict_wire: bool = True) -> SyncIdeaPayload:
        if strict_wire:
            if "kill_if" in raw:
                raise SyncValidationError(
                    "Legacy field `kill_if` is not valid in sync_contract=2; use `kill_rule`."
                )
            if "id" in raw and "idea_id" not in raw:
                raise SyncValidationError("Legacy field `id` is not valid in sync_contract=2; use `idea_id`.")
        iid = str(raw.get("idea_id") if not strict_wire else (raw.get("idea_id") or "")).strip()
        if not iid and not strict_wire:
            iid = str(raw.get("id") or "").strip()
        thesis = str(
            raw.get("thesis") or ("" if strict_wire else (raw.get("statement") or raw.get("name") or ""))
        ).strip()
        kr = str(
            raw.get("kill_rule")
            if raw.get("kill_rule") is not None
            else ("" if strict_wire else (raw.get("kill_if") or ""))
        ).strip()
        return cls(
            idea_id=iid,
            thesis=thesis,
            kill_rule=kr,
            family=str(raw.get("family") or "general").strip(),
            status=str(raw.get("status") or "open").strip(),
            created_at=str(raw.get("created_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
            object_sha256=str(raw["object_sha256"]) if raw.get("object_sha256") else None,
        ).validate()


@dataclass
class SyncTrialEventPayload:
    """Idempotent per-attempt trial event replacing pushed cumulative counters (`QOS-164`)."""

    attempt_id: str
    run_id: str
    family: str
    idea_id: str | None = None
    params_hash: str = ""
    status: str = "ok"
    executed_at: str = ""

    def validate(self) -> SyncTrialEventPayload:
        if not self.attempt_id or not str(self.attempt_id).strip():
            raise SyncValidationError("SyncTrialEventPayload.attempt_id must be non-empty.")
        if not self.run_id or not str(self.run_id).strip():
            raise SyncValidationError(f"SyncTrialEventPayload({self.attempt_id}).run_id must be non-empty.")
        try:
            self.attempt_id = validate_safe_id(str(self.attempt_id), "attempt_id")
            self.run_id = validate_safe_id(str(self.run_id), "run_id")
            if self.idea_id is not None and str(self.idea_id).strip():
                self.idea_id = validate_safe_id(str(self.idea_id), "idea_id")
        except ValueError as exc:
            raise SyncValidationError(str(exc)) from exc
        if not self.family or not str(self.family).strip():
            raise SyncValidationError(f"SyncTrialEventPayload({self.attempt_id}).family must be non-empty.")
        if self.status not in _VALID_TRIAL_STATUSES:
            allowed_st = sorted(_VALID_TRIAL_STATUSES)
            raise SyncValidationError(
                f"SyncTrialEventPayload({self.attempt_id}).status must be one of {allowed_st}."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "attempt_id": str(self.attempt_id).strip(),
            "run_id": str(self.run_id).strip(),
            "family": str(self.family).strip(),
            "idea_id": str(self.idea_id).strip() if self.idea_id else None,
            "params_hash": str(self.params_hash or ""),
            "status": str(self.status),
            "executed_at": str(self.executed_at or ""),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SyncTrialEventPayload:
        return cls(
            attempt_id=str(raw.get("attempt_id") or "").strip(),
            run_id=str(raw.get("run_id") or "").strip(),
            family=str(raw.get("family") or "").strip(),
            idea_id=str(raw["idea_id"]).strip() if raw.get("idea_id") else None,
            params_hash=str(raw.get("params_hash") or ""),
            status=str(raw.get("status") or "ok").strip(),
            executed_at=str(raw.get("executed_at") or ""),
        ).validate()


@dataclass
class PushRequest:
    """Top-level `POST /api/workspace/sync` v2 request envelope."""

    project: dict[str, Any]
    studies: list[dict[str, Any]] = field(default_factory=list)
    runs: list[SyncRunPayload] = field(default_factory=list)
    ideas: list[SyncIdeaPayload] = field(default_factory=list)
    trial_events: list[SyncTrialEventPayload] = field(default_factory=list)
    sync_contract: int = SYNC_CONTRACT_VERSION
    engine_version: str = field(default_factory=current_engine_version)

    def validate(self) -> PushRequest:
        if int(self.sync_contract) != SYNC_CONTRACT_VERSION:
            raise SyncContractMismatchError(
                f"Unsupported sync_contract={self.sync_contract}; alphaengine {self.engine_version} "
                f"requires sync_contract={SYNC_CONTRACT_VERSION}."
            )
        for st in self.studies:
            SealedStudyWire.from_dict(st)
        for r in self.runs:
            r.validate()
        for idea in self.ideas:
            idea.validate()
        seen_attempts: set[str] = set()
        for ev in self.trial_events:
            ev.validate()
            if ev.attempt_id in seen_attempts:
                raise SyncValidationError(
                    f"Duplicate attempt_id {ev.attempt_id!r} in PushRequest.trial_events."
                )
            seen_attempts.add(ev.attempt_id)
        return self

    @property
    def total_items(self) -> int:
        return len(self.studies) + len(self.runs) + len(self.ideas) + len(self.trial_events)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "sync_contract": int(self.sync_contract),
            "engine_version": str(self.engine_version or current_engine_version()),
            "project": dict(self.project),
            "studies": [dict(s) for s in self.studies],
            "runs": [r.to_dict() for r in self.runs],
            "ideas": [i.to_dict() for i in self.ideas],
            "trial_events": [ev.to_dict() for ev in self.trial_events],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PushRequest:
        sc = int(raw.get("sync_contract") or 0)
        if sc != SYNC_CONTRACT_VERSION:
            raise SyncContractMismatchError(
                f"This payload speaks sync_contract={sc}, and this engine requires "
                f"sync_contract={SYNC_CONTRACT_VERSION}."
            )
        proj = dict(raw["project"]) if isinstance(raw.get("project"), Mapping) else {}
        studies = [dict(s) for s in (raw.get("studies") or []) if isinstance(s, Mapping)]
        runs = [SyncRunPayload.from_dict(r) for r in (raw.get("runs") or []) if isinstance(r, Mapping)]
        ideas = [SyncIdeaPayload.from_dict(i) for i in (raw.get("ideas") or []) if isinstance(i, Mapping)]
        events = [
            SyncTrialEventPayload.from_dict(ev)
            for ev in (raw.get("trial_events") or [])
            if isinstance(ev, Mapping)
        ]
        return cls(
            project=proj,
            studies=studies,
            runs=runs,
            ideas=ideas,
            trial_events=events,
            sync_contract=sc,
            engine_version=str(raw.get("engine_version") or current_engine_version()),
        ).validate()


@dataclass(frozen=True)
class PushItemAck:
    """Per-item acknowledgement returned by `POST /api/workspace/sync` (`QOS-167`)."""

    type: str
    local_id: str
    status: str
    reason: str | None = None
    server_id: str | None = None

    def validate(self) -> PushItemAck:
        if self.type not in _VALID_ITEM_TYPES:
            raise SyncValidationError(
                f"PushItemAck.type must be one of {sorted(_VALID_ITEM_TYPES)}, got {self.type!r}."
            )
        if self.status not in _VALID_ACK_STATUSES:
            raise SyncValidationError(
                f"PushItemAck.status must be one of {sorted(_VALID_ACK_STATUSES)}, got {self.status!r}."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "type": self.type,
            "local_id": self.local_id,
            "status": self.status,
            "reason": self.reason,
            "server_id": self.server_id,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PushItemAck:
        local_id = str(raw.get("local_id") or raw.get("id") or "").strip()
        raw_reason = raw.get("reason")
        code = str(raw["code"]).strip() if raw.get("code") else ""
        detail = str(raw["detail"]).strip() if raw.get("detail") else ""
        if raw_reason:
            reason: str | None = str(raw_reason).strip()
        elif code and detail:
            reason = f"{code}: {detail}"
        elif detail or code:
            reason = detail or code
        else:
            reason = None
        return cls(
            type=str(raw.get("type") or "").strip(),
            local_id=local_id,
            status=str(raw.get("status") or "").strip(),
            reason=reason,
            server_id=str(raw["server_id"]) if raw.get("server_id") else None,
        ).validate()


def compute_push_exit_code(items: Sequence[PushItemAck], *, submitted_count: int) -> int:
    """Compute structured `ae push` exit code (`0`, `2`, `3`) from per-item acknowledgements (`QOS-167`)."""
    if submitted_count <= 0 and not items:
        return EXIT_OK
    n_rejected = sum(1 for item in items if item.status == "rejected")
    n_ok_or_dup = sum(1 for item in items if item.status in ("accepted", "duplicate"))
    if n_rejected == 0:
        return EXIT_OK
    if n_ok_or_dup == 0:
        return EXIT_ALL_REJECTED
    return EXIT_PARTIAL_REJECTED


@dataclass
class PushResponse:
    """Structured `POST /api/workspace/sync` v2 response (`QOS-167`)."""

    items: list[PushItemAck] = field(default_factory=list)
    sync_contract: int = SYNC_CONTRACT_VERSION
    engine_version: str = field(default_factory=current_engine_version)
    submitted_count: int = 0

    @property
    def counts(self) -> dict[str, int]:
        accepted = sum(1 for i in self.items if i.status == "accepted")
        duplicate = sum(1 for i in self.items if i.status == "duplicate")
        rejected = sum(1 for i in self.items if i.status == "rejected")
        return {
            "accepted": accepted,
            "duplicate": duplicate,
            "rejected": rejected,
            "total": len(self.items),
        }

    @property
    def exit_code(self) -> int:
        return compute_push_exit_code(self.items, submitted_count=max(self.submitted_count, len(self.items)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sync_contract": int(self.sync_contract),
            "engine_version": str(self.engine_version),
            "items": [i.to_dict() for i in self.items],
            "counts": self.counts,
            "exit_code": self.exit_code,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, submitted_count: int = 0) -> PushResponse:
        if "sync_contract" not in raw:
            raise SyncContractMismatchError(
                "This response has no sync contract, so the URL is not the quantOS sync API. "
                "The portal site is not that API. Reset with `ae remote add quantos` (no URL)."
            )
        sc = int(raw.get("sync_contract") or 0)
        if sc != SYNC_CONTRACT_VERSION:
            raise SyncContractMismatchError(
                f"This engine speaks sync contract {SYNC_CONTRACT_VERSION}. The server speaks {sc}. "
                "Upgrade the quantOS backend to contract 2, or install the alphaengine build "
                "that server expects (`ae doctor`)."
            )
        raw_items = raw.get("items")
        if not isinstance(raw_items, list):
            raise SyncServerError("Server sync v2 response is missing per-item `items` list.")
        acks = [PushItemAck.from_dict(it) for it in raw_items if isinstance(it, Mapping)]
        return cls(
            items=acks,
            sync_contract=sc,
            engine_version=str(raw.get("engine_version") or "unknown"),
            submitted_count=submitted_count,
        )


@dataclass(frozen=True)
class PullRequest:
    """Query parameters for `GET /api/workspace/pull` (`QOS-162`)."""

    since: str | None = None
    limit: int = 100

    def to_query_params(self) -> dict[str, str]:
        if self.limit < 1 or self.limit > 1000:
            raise SyncValidationError("PullRequest.limit must be between 1 and 1000.")
        out: dict[str, str] = {"limit": str(int(self.limit))}
        if self.since and self.since.strip():
            out["since"] = self.since.strip()
        return out


@dataclass
class PullResponse:
    """Paginated `GET /api/workspace/pull` v2 response (`QOS-162`)."""

    runs: list[dict[str, Any]] = field(default_factory=list)
    ideas: list[dict[str, Any]] = field(default_factory=list)
    studies: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: str | None = None
    sync_contract: int = SYNC_CONTRACT_VERSION
    engine_version: str = field(default_factory=current_engine_version)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PullResponse:
        if "sync_contract" not in raw:
            raise SyncContractMismatchError(
                "This pull response has no sync contract, so the URL is not the quantOS sync API. "
                "The portal site is not that API. Reset with `ae remote add quantos` (no URL)."
            )
        sc = int(raw.get("sync_contract") or 0)
        if sc != SYNC_CONTRACT_VERSION:
            raise SyncContractMismatchError(
                f"This engine speaks sync contract {SYNC_CONTRACT_VERSION}. The server speaks {sc}. "
                "Upgrade the quantOS backend to contract 2, or install the alphaengine build "
                "that server expects (`ae doctor`)."
            )
        nxt = raw.get("next_cursor")
        cursor_str = str(nxt).strip() if isinstance(nxt, str) and nxt.strip() else None
        return cls(
            runs=[dict(r) for r in (raw.get("runs") or []) if isinstance(r, Mapping)],
            ideas=[dict(i) for i in (raw.get("ideas") or []) if isinstance(i, Mapping)],
            studies=[dict(s) for s in (raw.get("studies") or []) if isinstance(s, Mapping)],
            next_cursor=cursor_str,
            sync_contract=sc,
            engine_version=str(raw.get("engine_version") or "unknown"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sync_contract": int(self.sync_contract),
            "engine_version": str(self.engine_version),
            "runs": [dict(r) for r in self.runs],
            "ideas": [dict(i) for i in self.ideas],
            "studies": [dict(s) for s in self.studies],
            "next_cursor": self.next_cursor,
        }


def export_json_schemas() -> dict[str, dict[str, Any]]:
    """Export JSON Schema (Draft 2020-12) definitions for contract v2 CI parity checks (`QOS-152`)."""
    nullable_num: dict[str, Any] = {"type": ["number", "null"]}
    nullable_str: dict[str, Any] = {"type": ["string", "null"]}
    nullable_int: dict[str, Any] = {"type": ["integer", "null"]}
    return {
        "SyncRunPayload": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "SyncRunPayload",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "run_id",
                "timestamp",
                "command",
                "family",
                "params_hash",
                "dataset_hashes",
                "engine_version",
                "share_series",
                "metrics",
                "figures",
            ],
            "properties": {
                "run_id": {"type": "string", "minLength": 1},
                "timestamp": {"type": "string"},
                "command": {"type": "string", "minLength": 1},
                "idea_id": nullable_str,
                "family": {"type": "string", "minLength": 1},
                "params_hash": {"type": "string"},
                "dataset_hashes": {"type": "array", "items": {"type": "string"}},
                "n_trials": nullable_int,
                "run_n_trials": nullable_int,
                "n_trials_source": nullable_str,
                "verdict": nullable_str,
                "engine_version": {"type": "string", "minLength": 1},
                "object_sha256": nullable_str,
                "manifest": {"type": ["object", "null"]},
                "share_series": {"type": "boolean"},
                "metrics": {
                    "type": "object",
                    "properties": {k: nullable_num for k in CANONICAL_METRIC_KEYS},
                    "additionalProperties": nullable_num,
                },
                "figures": {"type": "object"},
                "series": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "array", "items": {"type": "number"}},
                },
            },
        },
        "SyncIdeaPayload": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "SyncIdeaPayload",
            "type": "object",
            "additionalProperties": False,
            "required": ["idea_id", "thesis", "kill_rule", "family", "status", "created_at", "updated_at"],
            "properties": {
                "idea_id": {"type": "string", "minLength": 1},
                "thesis": {"type": "string", "minLength": 1},
                "kill_rule": {"type": "string"},
                "family": {"type": "string", "minLength": 1},
                "status": {"type": "string", "minLength": 1},
                "created_at": {"type": "string"},
                "updated_at": {"type": "string"},
                "object_sha256": nullable_str,
            },
        },
        "SyncTrialEventPayload": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "SyncTrialEventPayload",
            "type": "object",
            "additionalProperties": False,
            "required": ["attempt_id", "run_id", "family", "params_hash", "status", "executed_at"],
            "properties": {
                "attempt_id": {"type": "string", "minLength": 1},
                "run_id": {"type": "string", "minLength": 1},
                "family": {"type": "string", "minLength": 1},
                "idea_id": nullable_str,
                "params_hash": {"type": "string"},
                "status": {"type": "string", "enum": ["ok", "failed"]},
                "executed_at": {"type": "string"},
            },
        },
        "PushRequest": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "PushRequest",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "sync_contract",
                "engine_version",
                "project",
                "studies",
                "runs",
                "ideas",
                "trial_events",
            ],
            "properties": {
                "sync_contract": {"type": "integer", "const": SYNC_CONTRACT_VERSION},
                "engine_version": {"type": "string", "minLength": 1},
                "project": {"type": "object"},
                "studies": {"type": "array", "items": {"type": "object"}},
                "runs": {"type": "array", "items": {"$ref": "#/definitions/SyncRunPayload"}},
                "ideas": {"type": "array", "items": {"$ref": "#/definitions/SyncIdeaPayload"}},
                "trial_events": {"type": "array", "items": {"$ref": "#/definitions/SyncTrialEventPayload"}},
            },
        },
        "PushResponse": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "PushResponse",
            "type": "object",
            "required": ["sync_contract", "engine_version", "items"],
            "properties": {
                "sync_contract": {"type": "integer", "const": SYNC_CONTRACT_VERSION},
                "engine_version": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["type", "local_id", "status"],
                        "properties": {
                            "type": {"type": "string", "enum": ["study", "run", "idea", "trial_event"]},
                            "local_id": {"type": "string"},
                            "status": {"type": "string", "enum": ["accepted", "duplicate", "rejected"]},
                            "reason": nullable_str,
                            "server_id": nullable_str,
                        },
                    },
                },
                "counts": {"type": "object"},
            },
        },
        "PullRequest": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "PullRequest",
            "type": "object",
            "properties": {
                "since": nullable_str,
                "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
            },
        },
        "PullResponse": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "PullResponse",
            "type": "object",
            "required": ["sync_contract", "engine_version", "runs", "ideas"],
            "properties": {
                "sync_contract": {"type": "integer", "const": SYNC_CONTRACT_VERSION},
                "engine_version": {"type": "string"},
                "runs": {"type": "array", "items": {"type": "object"}},
                "ideas": {"type": "array", "items": {"type": "object"}},
                "studies": {"type": "array", "items": {"type": "object"}},
                "next_cursor": nullable_str,
            },
        },
    }
