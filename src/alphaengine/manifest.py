"""Versioned reproducibility manifest (`RunManifest` v1) for local, hosted, and MCP runs.

Captures exact runtime provenance — engine version, wheel digest, Python and
core numerical library versions, container image digest (when hosted), user
callable/module SHA-256 (never source code text), parameter hash, dataset
SHA-256s, evaluation horizon/as-of, and stated conventions (costs, calendar,
annualization, risk-free, RNG seeds, sync contract).
"""

from __future__ import annotations

import hashlib
import inspect
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from ._version import (
    MANIFEST_VERSION,
    SYNC_CONTRACT_VERSION,
    _discover_wheel_sha256,
    current_engine_version,
)
from .core.conventions import (
    build_annualization_spec,
    build_costs_model_spec,
    build_risk_free_spec,
)
from .study.schema import canonical_sha256

__all__ = [
    "MANIFEST_VERSION",
    "RunManifest",
    "build_run_manifest",
    "check_manifest_provenance",
    " compute_dependency_lock_sha256".strip(),
    "hash_callable_or_module",
    "hash_source_text",
]


def hash_source_text(source: str | bytes) -> str:
    """Return 64-hex SHA-256 of normalized UTF-8 source text (never stores source)."""
    if isinstance(source, bytes):
        normalized = source.replace(b"\r\n", b"\n")
    else:
        normalized = source.replace("\r\n", "\n").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def hash_callable_or_module(target: Any) -> str | None:
    """Compute a deterministic 64-hex SHA-256 of a callable, module, or file path.

    Returns ``None`` for built-in/C callables or dynamically generated objects
    whose source cannot be inspected. Never returns source text.
    """
    if target is None:
        return None
    if isinstance(target, Path):
        if target.is_file():
            return hash_source_text(target.read_bytes())
        return None
    if isinstance(target, str):
        p = Path(target)
        if p.is_file():
            return hash_source_text(p.read_bytes())
        return None
    try:
        src_file = inspect.getsourcefile(target)
        if src_file:
            fp = Path(src_file)
            if fp.is_file():
                return hash_source_text(fp.read_bytes())
    except (OSError, TypeError):
        pass
    try:
        src = inspect.getsource(target)
        return hash_source_text(src)
    except (OSError, TypeError):
        return None


def compute_dependency_lock_sha256() -> str:
    """Compute canonical SHA-256 over active core runtime package versions."""
    deps: dict[str, str | None] = {}
    for pkg in ("numpy", "scipy", "statsmodels", "pyarrow"):
        try:
            deps[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            deps[pkg] = None
    return canonical_sha256(deps)


@dataclass(frozen=True)
class RunManifest:
    """Immutable v1 provenance manifest attached to every recorded run (`QOS-158`)."""

    manifest_version: int = MANIFEST_VERSION
    engine_version: str = ""
    wheel_sha256: str | None = None
    python_version: str = ""
    dependency_lock_sha256: str | None = None
    image_digest: str | None = None
    code_sha256: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    params_hash: str = ""
    dataset_hashes: list[str] = field(default_factory=list)
    as_of: str | None = None
    horizon: dict[str, Any] | None = None
    costs_model: dict[str, Any] = field(default_factory=dict)
    calendar: str = "trading_252"
    annualization: dict[str, Any] = field(default_factory=dict)
    risk_free: dict[str, Any] = field(default_factory=dict)
    rng_seeds: list[int] = field(default_factory=list)
    sync_contract: int = SYNC_CONTRACT_VERSION

    def to_cas_dict(self) -> dict[str, Any]:
        """Full local content-addressed dictionary (includes ``params`` locally)."""
        return {
            "manifest_version": self.manifest_version,
            "engine_version": self.engine_version,
            "wheel_sha256": self.wheel_sha256,
            "python_version": self.python_version,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "image_digest": self.image_digest,
            "code_sha256": self.code_sha256,
            "params": dict(self.params),
            "params_hash": self.params_hash,
            "dataset_hashes": list(self.dataset_hashes),
            "as_of": self.as_of,
            "horizon": dict(self.horizon) if self.horizon is not None else None,
            "costs_model": dict(self.costs_model),
            "calendar": self.calendar,
            "annualization": dict(self.annualization),
            "risk_free": dict(self.risk_free),
            "rng_seeds": list(self.rng_seeds),
            "sync_contract": self.sync_contract,
        }

    def to_wire_summary(self) -> dict[str, Any]:
        """Wire-safe manifest summary (omits parameter values; keeps hashes & conventions)."""
        return {
            "manifest_version": self.manifest_version,
            "engine_version": self.engine_version,
            "wheel_sha256": self.wheel_sha256,
            "python_version": self.python_version,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "image_digest": self.image_digest,
            "code_sha256": self.code_sha256,
            "params_hash": self.params_hash,
            "dataset_hashes": list(self.dataset_hashes),
            "as_of": self.as_of,
            "horizon": dict(self.horizon) if self.horizon is not None else None,
            "costs_model": dict(self.costs_model),
            "calendar": self.calendar,
            "annualization": dict(self.annualization),
            "risk_free": dict(self.risk_free),
            "rng_seeds": list(self.rng_seeds),
            "sync_contract": self.sync_contract,
        }

    def manifest_sha256(self) -> str:
        """Return 64-hex canonical SHA-256 of the local CAS manifest."""
        return canonical_sha256(self.to_cas_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        """Reconstruct a ``RunManifest`` from a CAS or wire summary mapping."""
        horizon_raw = data.get("horizon")
        return cls(
            manifest_version=int(data.get("manifest_version") or MANIFEST_VERSION),
            engine_version=str(data.get("engine_version") or ""),
            wheel_sha256=str(data["wheel_sha256"]) if data.get("wheel_sha256") else None,
            python_version=str(data.get("python_version") or ""),
            dependency_lock_sha256=(
                str(data["dependency_lock_sha256"]) if data.get("dependency_lock_sha256") else None
            ),
            image_digest=str(data["image_digest"]) if data.get("image_digest") else None,
            code_sha256=str(data["code_sha256"]) if data.get("code_sha256") else None,
            params=dict(data.get("params") or {}),
            params_hash=str(data.get("params_hash") or ""),
            dataset_hashes=[str(h) for h in (data.get("dataset_hashes") or []) if h],
            as_of=str(data["as_of"]) if data.get("as_of") is not None else None,
            horizon=dict(horizon_raw) if isinstance(horizon_raw, Mapping) else None,
            costs_model=dict(data.get("costs_model") or {}),
            calendar=str(data.get("calendar") or "trading_252"),
            annualization=dict(data.get("annualization") or {}),
            risk_free=dict(data.get("risk_free") or {}),
            rng_seeds=[int(s) for s in (data.get("rng_seeds") or []) if isinstance(s, int)],
            sync_contract=int(data.get("sync_contract") or SYNC_CONTRACT_VERSION),
        )


def build_run_manifest(
    *,
    params: Mapping[str, Any] | None = None,
    dataset_hashes: Sequence[str] | None = None,
    code_target: Any = None,
    code_sha256: str | None = None,
    as_of: str | None = None,
    horizon: Mapping[str, Any] | None = None,
    costs_model: Mapping[str, Any] | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float | None = None,
    calendar: str | None = None,
    rng_seeds: Sequence[int] | None = None,
    image_digest: str | None = None,
) -> RunManifest:
    """Construct a complete ``RunManifest`` from live runtime environment + run inputs."""
    ppy = max(int(periods_per_year), 1)
    clean_params = dict(params or {})
    p_hash = canonical_sha256(clean_params)
    resolved_code_sha = code_sha256 or hash_callable_or_module(code_target)
    resolved_image = image_digest or (os.environ.get("ALPHAENGINE_IMAGE_DIGEST", "").strip() or None)
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    resolved_costs = (
        dict(costs_model) if costs_model is not None else build_costs_model_spec(mode="pre_netted_returns")
    )
    resolved_cal = calendar or (f"trading_{ppy}" if ppy == 252 else f"periods_{ppy}")
    seeds: list[int] = []
    if rng_seeds is not None:
        for s in rng_seeds:
            if isinstance(s, int) and not isinstance(s, bool):
                seeds.append(int(s))
    elif isinstance(clean_params.get("seed"), int) and not isinstance(clean_params.get("seed"), bool):
        seeds.append(int(clean_params["seed"]))

    return RunManifest(
        manifest_version=MANIFEST_VERSION,
        engine_version=current_engine_version(),
        wheel_sha256=_discover_wheel_sha256(),
        python_version=py_ver,
        dependency_lock_sha256=compute_dependency_lock_sha256(),
        image_digest=resolved_image,
        code_sha256=resolved_code_sha,
        params=clean_params,
        params_hash=p_hash,
        dataset_hashes=[str(h) for h in (dataset_hashes or []) if h],
        as_of=as_of,
        horizon=dict(horizon) if horizon is not None else None,
        costs_model=resolved_costs,
        calendar=resolved_cal,
        annualization=build_annualization_spec(ppy),
        risk_free=build_risk_free_spec(risk_free_rate, ppy),
        rng_seeds=seeds,
        sync_contract=SYNC_CONTRACT_VERSION,
    )


def check_manifest_provenance(
    recorded: Mapping[str, Any] | None,
    *,
    current: RunManifest | Mapping[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Compare a recorded ``RunManifest`` against the active runtime environment.

    Returns ``(provenance_verified, diagnostics)``. When ``recorded`` is ``None``
    (for example a 1.x study or ledger entry), ``provenance_verified`` is
    ``False`` with ``reason="manifest_not_recorded"``.
    """
    if not recorded or not isinstance(recorded, Mapping):
        return False, {"reason": "manifest_not_recorded", "drift": {}}

    cur_dict = (
        current.to_wire_summary()
        if isinstance(current, RunManifest)
        else dict(current)
        if isinstance(current, Mapping)
        else build_run_manifest().to_wire_summary()
    )

    drift: dict[str, dict[str, Any]] = {}
    rec_ver = recorded.get("engine_version")
    cur_ver = cur_dict.get("engine_version")
    if rec_ver != cur_ver:
        drift["engine_version"] = {"recorded": rec_ver, "current": cur_ver}

    rec_wheel = recorded.get("wheel_sha256")
    cur_wheel = cur_dict.get("wheel_sha256")
    if rec_wheel is not None and rec_wheel != cur_wheel:
        drift["wheel_sha256"] = {"recorded": rec_wheel, "current": cur_wheel}

    rec_code = recorded.get("code_sha256")
    cur_code = cur_dict.get("code_sha256")
    if rec_code is not None and cur_code is not None and rec_code != cur_code:
        drift["code_sha256"] = {"recorded": rec_code, "current": cur_code}

    rec_ds = list(recorded.get("dataset_hashes") or [])
    cur_ds = list(cur_dict.get("dataset_hashes") or [])
    if rec_ds and cur_ds and rec_ds != cur_ds:
        drift["dataset_hashes"] = {"recorded": rec_ds, "current": cur_ds}

    for conv_key in ("annualization", "risk_free", "costs_model"):
        rec_conv = recorded.get(conv_key)
        cur_conv = cur_dict.get(conv_key)
        if (
            isinstance(rec_conv, Mapping)
            and isinstance(cur_conv, Mapping)
            and current is not None
            and dict(rec_conv) != dict(cur_conv)
        ):
            drift[conv_key] = {"recorded": dict(rec_conv), "current": dict(cur_conv)}

    advisories: dict[str, dict[str, Any]] = {}
    for adv_key in ("python_version", "dependency_lock_sha256", "image_digest"):
        rv = recorded.get(adv_key)
        cv = cur_dict.get(adv_key)
        if rv is not None and cv is not None and rv != cv:
            advisories[adv_key] = {"recorded": rv, "current": cv}

    ok = len(drift) == 0
    return ok, {
        "manifest_version": recorded.get("manifest_version"),
        "drift": drift,
        "environment_advisories": advisories,
    }
