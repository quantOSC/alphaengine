"""Project-wide experiment ledger: run records, cumulative trial counts, diff, and rerun.

Every command or sweep executed inside a `.alphaengine/` project writes an
append-only run record to `.alphaengine/ledger/runs.jsonl` (and per-trial
records to `.alphaengine/ledger/trials.jsonl`), backed by content-addressed
SHA-256 objects in `.alphaengine/objects/`.

Trial counts follow William's rule (`QOS-153`): **every executed attempt counts**,
including identical reruns and failed grid points. Deduplication occurs ONLY on
exact `attempt_id` / `run_id` transport redelivery. Any ledger read or write
failure raises `LedgerError` (fail-closed) rather than silently returning `N=1`.
"""

from __future__ import annotations

import contextvars
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ._version import current_engine_version
from .core.conventions import Verdict
from .manifest import (
    RunManifest,
    build_run_manifest,
    check_manifest_provenance,
)
from .project import Project, sha256_hex

__all__ = [
    "LedgerError",
    "record_run",
    "record_trials",
    "cumulative_trials_for",
    "list_runs",
    "resolve_run",
    "diff_runs",
    "rerun_record",
    "rerun_record_report",
    "execute_local_workflow",
    "suppress_auto_sweep_record",
]


class LedgerError(RuntimeError):
    """Raised when reading or writing `.alphaengine/ledger/` fails (fail-closed)."""


_SUPPRESS_SWEEP_RECORD: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "alphaengine_suppress_sweep_record", default=False
)

_LEDGER_META_KEYS = frozenset(
    {
        "n_trials",
        "run_n_trials",
        "run_deflated_sharpe",
        "run_dsr_probability",
        "deflated_sharpe",
        "dsr_probability",
        "deflation_applied",
        "sr0_expected_max",
        "verdict",
        "cumulative_trials",
        "family_trials",
        "project_trials",
        "duplicate_of",
        "cumulative_deflated_sharpe",
        "cumulative_dsr_probability",
        "cumulative_sr0_expected_max",
        "cumulative_verdict",
        "engine_version",
        "risk_free_rate",
        "periods_per_year",
    }
)


class suppress_auto_sweep_record:
    """Context manager so a CLI `validate` run records one enclosing run rather than two."""

    def __init__(self) -> None:
        self._token: contextvars.Token[bool] | None = None

    def __enter__(self) -> None:
        self._token = _SUPPRESS_SWEEP_RECORD.set(True)

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _SUPPRESS_SWEEP_RECORD.reset(self._token)


def is_auto_sweep_suppressed() -> bool:
    return _SUPPRESS_SWEEP_RECORD.get()


def _read_ledger_jsonl(proj: Project, rel_path: str) -> list[dict[str, Any]]:
    try:
        return proj.read_jsonl(rel_path, strict=True)
    except (OSError, ValueError) as exc:
        raise LedgerError(f"Failed reading ledger {proj.dot_dir / rel_path}: {exc}") from exc


def _append_ledger_jsonl(proj: Project, rel_path: str, record: dict[str, Any]) -> str:
    try:
        return proj.append_jsonl(rel_path, record)
    except (OSError, ValueError) as exc:
        raise LedgerError(f"Failed writing ledger {proj.dot_dir / rel_path}: {exc}") from exc


def hash_dataset_any(data: Any) -> list[str]:
    """Compute 64-hex SHA-256 dataset hashes for arbitrary in-memory data via `canonical_sha256`."""
    if data is None:
        return []
    try:
        arr = np.asarray(data, dtype=float)
        return [sha256_hex(arr.tolist())]
    except (TypeError, ValueError):
        return [sha256_hex(data)]


def _run_config_signature(
    command: str,
    params_hash: str,
    dataset_hashes: list[str],
) -> str:
    return sha256_hex(
        {
            "command": command,
            "params_hash": params_hash,
            "dataset_hashes": sorted(str(h) for h in dataset_hashes if h),
        }
    )


def cumulative_trials_for(
    proj: Project,
    *,
    idea_id: str | None = None,
    family: str | None = None,
) -> tuple[int, int]:
    """Return `(family_or_idea_trials, project_cumulative_trials)` across all executed attempts.

    Per William's trial counting rule (`QOS-153`), **every executed attempt counts**,
    including identical reruns and failed grid points. Deduplication occurs ONLY
    on exact `attempt_id` / `run_id` transport redelivery:
    - ``monte_carlo`` runs add 0 to both family and project trial counts.
    - ``derived_from_universe`` (universe screening) counts toward the screened
      idea or family only, not toward ``project_trials``.
    """
    runs = _read_ledger_jsonl(proj, "ledger/runs.jsonl")
    family_total = 0
    project_total = 0
    target_idea = (idea_id or "").strip() or None
    target_family = (family or "").strip() or None
    seen_run_ids: set[str] = set()
    seen_attempt_ids: set[str] = set()

    for row in runs:
        rid = str(row.get("run_id") or "").strip()
        if rid:
            if rid in seen_run_ids:
                continue
            seen_run_ids.add(rid)
        aid = str(row.get("attempt_id") or "").strip()
        if aid:
            if aid in seen_attempt_ids:
                continue
            seen_attempt_ids.add(aid)

        src = str(row.get("n_trials_source") or "")
        if src == "monte_carlo":
            continue
        n = row.get("run_n_trials") if row.get("run_n_trials") is not None else row.get("n_trials")
        count = int(n) if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 0
        if src != "derived_from_universe":
            project_total += count
        if (
            target_idea
            and row.get("idea_id") == target_idea
            or not target_idea
            and target_family
            and row.get("family") == target_family
        ):
            family_total += count
    return family_total, project_total


def record_trials(
    proj: Project,
    *,
    run_id: str,
    trials: list[dict[str, Any]],
    idea_id: str | None = None,
    family: str = "default",
    timestamp: str | None = None,
) -> list[dict[str, Any]]:
    """Append per-attempt trial events to `.alphaengine/ledger/trials.jsonl` (`QOS-153`, `QOS-164`).

    Idempotent by `attempt_id`: re-recording an already-persisted `attempt_id`
    is skipped, whereas every new attempt (including identical parameter hashes
    and failed executions) is appended and counted.
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    existing = _read_ledger_jsonl(proj, "ledger/trials.jsonl")
    seen_attempts = {str(r.get("attempt_id")) for r in existing if r.get("attempt_id")}
    written: list[dict[str, Any]] = []

    for t in trials:
        aid = str(t.get("attempt_id") or uuid.uuid4())
        if aid in seen_attempts:
            continue
        seen_attempts.add(aid)
        failed_msg = t.get("failed")
        status = "failed" if failed_msg is not None else str(t.get("status") or "ok")
        exec_at = str(t.get("executed_at") or ts)
        entry: dict[str, Any] = {
            "attempt_id": aid,
            "run_id": run_id,
            "timestamp": ts,
            "executed_at": exec_at,
            "idea_id": idea_id,
            "family": family,
            "index": t.get("index"),
            "params_hash": t.get("params_hash"),
            "status": status,
            "n_obs": t.get("n_obs"),
            "sharpe": t.get("sharpe"),
            "sharpe_annualized": t.get("sharpe_annualized"),
            "total_return_pct": t.get("total_return_pct"),
            "failed": failed_msg,
        }
        _append_ledger_jsonl(proj, "ledger/trials.jsonl", entry)
        written.append(entry)
    return written


def record_run(
    proj: Project,
    *,
    command: str,
    params: dict[str, Any] | None = None,
    dataset_hashes: list[str] | None = None,
    n_trials: int | None = None,
    n_trials_source: str = "not_recorded",
    figures: dict[str, Any] | None = None,
    verdict: str | None = None,
    idea_id: str | None = None,
    family: str | None = None,
    repro_spec: dict[str, Any] | None = None,
    returns_for_deflation: list[float] | None = None,
    trials_sharpe_std: float | None = None,
    risk_free_rate: float | None = None,
    periods_per_year: int | None = None,
    run_id: str | None = None,
    attempt_id: str | None = None,
    already_counted_in_ledger: bool = False,
    code_target: Any = None,
    code_sha256: str | None = None,
    costs_model: Mapping[str, Any] | None = None,
    manifest: RunManifest | None = None,
) -> dict[str, Any]:
    """Write a run record (and CAS `RunManifest`) to `.alphaengine/ledger/runs.jsonl`."""
    ts = datetime.now(timezone.utc).isoformat()
    rid = run_id or f"run_{uuid.uuid4().hex[:12]}"
    aid = attempt_id or str(uuid.uuid4())
    clean_params = dict(params or {})
    params_hash = sha256_hex(clean_params)
    ds_hashes = list(dataset_hashes or [])

    resolved_idea_id = (idea_id or "").strip() or None
    idea_family: str | None = None
    if resolved_idea_id:
        try:
            from .ideas import resolve_idea, update_idea_status

            idea_obj = resolve_idea(proj, resolved_idea_id)
            resolved_idea_id = str(idea_obj.get("idea_id") or resolved_idea_id)
            idea_family = str(idea_obj.get("family") or "") or None
            if idea_obj.get("status") == "open":
                update_idea_status(proj, resolved_idea_id, "testing")
        except Exception:  # noqa: BLE001
            pass

    resolved_family = (family or "").strip() or idea_family or (resolved_idea_id or "") or command
    idea_id = resolved_idea_id

    config_sha = _run_config_signature(command, params_hash, ds_hashes)
    existing_runs = _read_ledger_jsonl(proj, "ledger/runs.jsonl")

    # `duplicate_of` records prior-work config lineage for researcher visibility,
    # while STILL counting every executed attempt per William's rule (`QOS-153`).
    duplicate_of: str | None = None
    already_persisted: dict[str, Any] | None = None
    for prev in existing_runs:
        if prev.get("run_id") == rid or (prev.get("attempt_id") and prev.get("attempt_id") == aid):
            already_persisted = prev
            break
        prev_sig = str(prev.get("config_sha256") or "")
        if not prev_sig:
            prev_sig = _run_config_signature(
                str(prev.get("command") or ""),
                str(prev.get("params_hash") or ""),
                list(prev.get("dataset_hashes") or []),
            )
        if prev_sig == config_sha and duplicate_of is None:
            duplicate_of = str(prev.get("run_id") or "")

    if already_persisted is not None:
        return already_persisted

    prior_fam, prior_proj = cumulative_trials_for(
        proj,
        idea_id=idea_id,
        family=resolved_family,
    )
    this_trials = (
        int(n_trials) if isinstance(n_trials, int) and not isinstance(n_trials, bool) and n_trials > 0 else 0
    )
    if already_counted_in_ledger or n_trials_source == "monte_carlo":
        added_fam = 0
        added_proj = 0
    elif n_trials_source == "derived_from_universe":
        added_fam = this_trials
        added_proj = 0
    else:
        added_fam = this_trials
        added_proj = this_trials

    proj_trials = prior_proj + added_proj
    fam_trials = max(
        prior_fam + added_fam,
        this_trials if (already_counted_in_ledger and n_trials_source != "monte_carlo") else 0,
    )
    cum_trials = max(proj_trials, fam_trials)

    figs = dict(figures or {})
    if returns_for_deflation and len(returns_for_deflation) >= 8 and trials_sharpe_std is None:
        raw_std = figs.get("trials_sharpe_std")
        if isinstance(raw_std, (int, float)) and not isinstance(raw_std, bool) and float(raw_std) > 0:
            trials_sharpe_std = float(raw_std)

    rf_eff = risk_free_rate
    if rf_eff is None and "risk_free_rate" in clean_params and clean_params["risk_free_rate"] is not None:
        rf_eff = float(clean_params["risk_free_rate"])
    elif rf_eff is None and "risk_free_rate" in figs and figs["risk_free_rate"] is not None:
        rf_eff = float(figs["risk_free_rate"])

    ppy_eff = periods_per_year
    if ppy_eff is None and clean_params.get("periods_per_year") is not None:
        ppy_eff = int(clean_params["periods_per_year"])
    elif ppy_eff is None and figs.get("periods_per_year") is not None:
        ppy_eff = int(figs["periods_per_year"])
    if ppy_eff is None:
        ppy_eff = 252

    if n_trials_source == "not_recorded":
        if verdict in ("edge", Verdict.ROBUST.value):
            verdict = Verdict.UNVERIFIED.value
        figs["n_trials"] = None
        figs["run_n_trials"] = None
        figs["n_trials_source"] = "not_recorded"
    else:
        figs["run_n_trials"] = this_trials
        if "deflated_sharpe" in figs:
            figs["run_deflated_sharpe"] = figs["deflated_sharpe"]
        if "dsr_probability" in figs:
            figs["run_dsr_probability"] = figs["dsr_probability"]
        figs["n_trials"] = cum_trials
        figs["n_trials_source"] = n_trials_source

    figs["cumulative_trials"] = cum_trials
    figs["family_trials"] = fam_trials
    figs["project_trials"] = proj_trials
    figs["duplicate_of"] = duplicate_of

    if (
        returns_for_deflation
        and len(returns_for_deflation) >= 8
        and cum_trials > 0
        and n_trials_source not in ("not_recorded", "monte_carlo")
    ):
        from .core import deflated_sharpe

        cum_dsr = deflated_sharpe(
            returns_for_deflation,
            n_trials=cum_trials,
            trials_sharpe_std=trials_sharpe_std,
            risk_free_rate=rf_eff,
            periods_per_year=ppy_eff,
        )
        if "deflated_sharpe" in cum_dsr:
            figs["dsr_probability"] = cum_dsr.get("dsr_probability")
            figs["deflated_sharpe"] = cum_dsr["deflated_sharpe"]
            figs["deflation_applied"] = cum_dsr.get("deflation_applied")
            figs["sr0_expected_max"] = cum_dsr["sr0_expected_max"]
            figs["verdict"] = cum_dsr["verdict"]
            figs["risk_free_rate"] = cum_dsr.get("risk_free_rate")
            figs["periods_per_year"] = cum_dsr.get("periods_per_year")
            figs["cumulative_dsr_probability"] = cum_dsr.get("dsr_probability")
            figs["cumulative_deflated_sharpe"] = cum_dsr["deflated_sharpe"]
            figs["cumulative_sr0_expected_max"] = cum_dsr["sr0_expected_max"]
            figs["cumulative_verdict"] = cum_dsr["verdict"]
            verdict = str(cum_dsr["verdict"])

    manifest_obj = (
        manifest
        if isinstance(manifest, RunManifest)
        else build_run_manifest(
            params=clean_params,
            dataset_hashes=ds_hashes,
            code_target=code_target,
            code_sha256=code_sha256,
            costs_model=costs_model,
            periods_per_year=ppy_eff,
            risk_free_rate=rf_eff,
        )
    )
    try:
        manifest_sha = proj.put_object(manifest_obj.to_cas_dict())
    except OSError as exc:
        raise LedgerError(f"Failed storing RunManifest in {proj.objects_dir}: {exc}") from exc

    eng_ver = current_engine_version()
    record: dict[str, Any] = {
        "run_id": rid,
        "attempt_id": aid,
        "timestamp": ts,
        "command": command,
        "idea_id": idea_id,
        "family": resolved_family,
        "params": clean_params,
        "params_hash": params_hash,
        "config_sha256": config_sha,
        "duplicate_of": duplicate_of,
        "dataset_hashes": ds_hashes,
        "run_n_trials": this_trials if n_trials_source != "not_recorded" else None,
        "n_trials": cum_trials if n_trials_source != "not_recorded" else None,
        "n_trials_source": n_trials_source,
        "cumulative_trials": cum_trials,
        "family_trials": fam_trials,
        "project_trials": proj_trials,
        "figures": figs,
        "verdict": verdict,
        "engine_version": eng_ver,
        "manifest_sha256": manifest_sha,
        "manifest": manifest_obj.to_wire_summary(),
    }
    if repro_spec is not None:
        record["repro_spec"] = repro_spec

    obj_sha = _append_ledger_jsonl(proj, "ledger/runs.jsonl", record)
    record["object_sha256"] = obj_sha
    return record


def list_runs(
    proj: Project,
    *,
    idea_id: str | None = None,
    family: str | None = None,
    since: str | None = None,
    verdict: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter and return project runs (newest first)."""
    rows = list(reversed(_read_ledger_jsonl(proj, "ledger/runs.jsonl")))
    out: list[dict[str, Any]] = []
    for row in rows:
        if idea_id and str(row.get("idea_id") or "") != idea_id:
            continue
        if family and str(row.get("family") or "") != family:
            continue
        if verdict and str(row.get("verdict") or "") != verdict:
            continue
        if since and str(row.get("timestamp") or "")[: len(since)] < since:
            continue
        out.append(row)
        if limit and limit > 0 and len(out) >= limit:
            break
    return out


def resolve_run(proj: Project, run_ref: str) -> dict[str, Any]:
    """Find a run in `proj` by exact `run_id`, prefix, or `HEAD` / `latest`."""
    ref = (run_ref or "").strip()
    rows = _read_ledger_jsonl(proj, "ledger/runs.jsonl")
    if not rows:
        raise KeyError("No runs recorded in this project yet.")
    if ref.lower() in ("head", "latest", "last"):
        return rows[-1]
    exact = [r for r in rows if str(r.get("run_id")) == ref]
    if len(exact) == 1:
        return exact[0]
    prefix = [r for r in rows if str(r.get("run_id") or "").startswith(ref)]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        ids = ", ".join(str(r.get("run_id")) for r in prefix[:5])
        raise KeyError(f"Ambiguous run prefix {ref!r}; matches: {ids}")
    raise KeyError(f"No run matching {ref!r} in {proj.dot_dir / 'ledger/runs.jsonl'}")


def diff_runs(run_a: dict[str, Any], run_b: dict[str, Any]) -> dict[str, Any]:
    """Compute structured differences (data, parameters, trials, figures, verdict) between two runs."""
    ds_a = list(run_a.get("dataset_hashes") or [])
    ds_b = list(run_b.get("dataset_hashes") or [])
    params_a = dict(run_a.get("params") or {})
    params_b = dict(run_b.get("params") or {})
    all_param_keys = sorted(set(params_a) | set(params_b))
    param_changes: dict[str, dict[str, Any]] = {}
    for k in all_param_keys:
        if params_a.get(k) != params_b.get(k):
            param_changes[k] = {"a": params_a.get(k), "b": params_b.get(k)}

    figs_a = _flatten_scalars(run_a.get("figures") or {})
    figs_b = _flatten_scalars(run_b.get("figures") or {})
    all_fig_keys = sorted(set(figs_a) | set(figs_b))
    figure_changes: dict[str, dict[str, Any]] = {}
    for k in all_fig_keys:
        va = figs_a.get(k)
        vb = figs_b.get(k)
        if va != vb:
            delta: float | None = None
            if (
                isinstance(va, (int, float))
                and not isinstance(va, bool)
                and isinstance(vb, (int, float))
                and not isinstance(vb, bool)
            ):
                delta = round(float(vb) - float(va), 6)
            figure_changes[k] = {"a": va, "b": vb, "delta": delta}

    return {
        "run_a": run_a.get("run_id"),
        "run_b": run_b.get("run_id"),
        "command_a": run_a.get("command"),
        "command_b": run_b.get("command"),
        "data_changed": ds_a != ds_b,
        "dataset_hashes_a": ds_a,
        "dataset_hashes_b": ds_b,
        "params_changed": run_a.get("params_hash") != run_b.get("params_hash"),
        "params_hash_a": run_a.get("params_hash"),
        "params_hash_b": run_b.get("params_hash"),
        "param_changes": param_changes,
        "n_trials_a": run_a.get("n_trials"),
        "n_trials_b": run_b.get("n_trials"),
        "cumulative_trials_a": run_a.get("cumulative_trials"),
        "cumulative_trials_b": run_b.get("cumulative_trials"),
        "verdict_a": run_a.get("verdict"),
        "verdict_b": run_b.get("verdict"),
        "figure_changes": figure_changes,
    }


def _flatten_scalars(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in sorted(d.items(), key=lambda item: str(item[0])):
        if k in _LEDGER_META_KEYS:
            continue
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_scalars(v, key))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[key] = v
    return out


def _flatten_figures(node: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten both scalars and list/curve figures so `rerun_record` verifies every recorded figure."""
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in sorted(node.items(), key=lambda item: str(item[0])):
            if not prefix and k in _LEDGER_META_KEYS:
                continue
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list, tuple)):
                out.update(_flatten_figures(v, key))
            elif isinstance(v, (int, float, str, bool)) or v is None:
                out[key] = v
    elif isinstance(node, (list, tuple)):
        out[f"{prefix}.__len__"] = len(node)
        for idx, item in enumerate(node):
            key = f"{prefix}[{idx}]"
            if isinstance(item, (dict, list, tuple)):
                out.update(_flatten_figures(item, key))
            elif isinstance(item, (int, float, str, bool)) or item is None:
                out[key] = item
    return out


def execute_local_workflow(
    workflow: str,
    *,
    data: Any,
    backtest_fn: Any = None,
    grid: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None, int | None, str, list[float] | None, list[dict[str, Any]]]:
    """Execute a workflow locally via `StepExecutor` and return its figures, verdict, and trials."""
    from .client.executor import StepExecutor

    raw_inp = dict(inputs or {})
    inp = {k: v for k, v in raw_inp.items() if k not in ("grid", "label")}
    no_trials_inp = {k: v for k, v in inp.items() if k != "n_trials"}
    ex = StepExecutor(data=data, backtest_fn=backtest_fn)
    wf = workflow.lower().strip()
    figures: dict[str, Any] = {}
    verdict: str | None = None
    n_trials: int | None = None
    source = "not_recorded"
    returns_col: list[float] | None = None
    trial_rows: list[dict[str, Any]] = []

    if wf in ("diagnose", "diagnose_data"):
        figures = ex.execute("data.profile", no_trials_inp)
        n_bad = sum(
            int(figures.get(k) or 0)
            for k in (
                "n_spike_names",
                "n_stale_names",
                "n_unreadable",
                "n_ragged",
                "n_nonpositive",
                "n_ends_early",
            )
        )
        verdict = "clean" if n_bad == 0 else "issues_found"
        n_trials = 1
        source = "diagnostic"
    elif wf in ("screen", "screen_universe"):
        figures = ex.execute("compute.screen", no_trials_inp)
        rows = figures.get("rows") or []
        verdict = "ranked" if rows else "empty"
        n_trials = int(figures.get("n_evaluated") or len(rows) or 1)
        source = "derived_from_universe"
    elif wf in ("signal", "evaluate_signal"):
        ic = ex.execute("compute.signal_ic", no_trials_inp)
        qr = ex.execute("compute.signal_quantiles", no_trials_inp)
        dec = ex.execute("compute.signal_decay", no_trials_inp)
        figures = {**ic, **qr, **dec}
        t_stat = abs(float(ic.get("ic_t_stat") or 0.0))
        verdict = "informative" if t_stat >= 2.0 else "marginal" if t_stat >= 1.5 else "likely_noise"
        n_trials = int(inp.get("n_trials") or 1)
        source = "asserted" if "n_trials" in inp else "single_signal"
    elif wf in ("validate", "validate_study"):
        with suppress_auto_sweep_record():
            if backtest_fn is not None and (grid or raw_inp.get("grid")):
                g = grid or raw_inp.get("grid") or {}
                sw = ex.execute("compute.sweep", {"grid": g})
                dsr = ex.execute("compute.deflated_sharpe", no_trials_inp)
                pbo = ex.execute("compute.pbo_cscv", no_trials_inp)
                mintrl = ex.execute("compute.min_track_record_length", no_trials_inp)
                perf = ex.execute("compute.performance_report", no_trials_inp)
                figures = {
                    **perf,
                    **sw,
                    **dsr,
                    "pbo": pbo.get("pbo"),
                    "min_track_record_length": mintrl.get("min_track_record_length"),
                }
                verdict = dsr.get("verdict")
                n_trials = int(dsr.get("n_trials") or sw.get("n_trials") or 0)
                source = "derived_from_grid"
                sweep_res = ex.workspace.get("sweep")
                if sweep_res is not None:
                    best = sweep_res.best
                    ok_trials = [t for t in sweep_res.trials if t.failed is None]
                    returns_col = sweep_res.matrix[:, ok_trials.index(best)].tolist()
                    trial_rows = sweep_res.ledger()
                    sv = sweep_res.verdict()
                    figures["n_eff_participation"] = sv.get("n_eff_participation")
                    figures["deflated_sharpe_at_n_eff"] = sv.get("deflated_sharpe_at_n_eff")
                    figures["n_eff_diagnostic_note"] = sv.get("n_eff_diagnostic_note")
                    # William's rule (`QOS-153`): count all attempted grid combos, including failures
                    n_trials = int(sweep_res.n_trials)
            else:
                perf = ex.execute("compute.performance_report", no_trials_inp)
                mintrl = ex.execute("compute.min_track_record_length", no_trials_inp)
                dsr = ex.execute("compute.deflated_sharpe", inp)
                figures = {**perf, **dsr, "min_track_record_length": mintrl.get("min_track_record_length")}
                verdict = dsr.get("verdict")
                n_trials = dsr.get("n_trials")
                source = str(dsr.get("n_trials_source") or "not_recorded")
                try:
                    returns_col = ex._returns_column(ex.workspace)
                except Exception:  # noqa: BLE001
                    returns_col = None
    elif wf in ("stress", "stress_study"):
        cl = ex.execute("compute.cost_ladder", no_trials_inp)
        sub = ex.execute("compute.subperiods", no_trials_inp)
        dd = ex.execute("compute.drawdown_anatomy", no_trials_inp)
        figures = {**cl, **sub, **dd}
        worst_sr = float(sub.get("worst_segment_sharpe") or 0.0)
        verdict = Verdict.MARGINAL.value if worst_sr > 0 else Verdict.LIKELY_NOISE.value
        n_trials = int(inp.get("n_trials") or 1)
        source = "asserted" if "n_trials" in inp else "stress_test"
    elif wf in ("overlap", "check_overlap"):
        figures = ex.execute("compute.overlap", no_trials_inp)
        verdict = str(figures.get("verdict") or "related")
        n_trials = 1
        source = "overlap_check"
    elif wf in ("size", "size_position"):
        perf = ex.execute("compute.performance_report", no_trials_inp)
        var = ex.execute("compute.compute_var_cvar", no_trials_inp)
        mintrl = ex.execute("compute.min_track_record_length", no_trials_inp)
        vt = ex.execute("compute.vol_target", no_trials_inp)
        sufficient = bool(mintrl.get("sufficient", False))
        cvar_val = float(
            (var.get("cvar") or {}).get("cvar_pct")
            or (var.get("historical") or {}).get("var_pct")
            or (var.get("parametric") or {}).get("var_pct")
            or 2.0
        )
        var_val = float(
            (var.get("historical") or {}).get("var_pct")
            or (var.get("parametric") or {}).get("var_pct")
            or cvar_val
        )
        target_w = round(min(1.0, max(0.0, 2.0 / max(cvar_val, 0.25))), 4) if sufficient else None
        figures = {
            **perf,
            **var,
            "var_pct": var_val,
            "cvar_pct": cvar_val,
            "min_track_record_length": mintrl.get("min_track_record_length"),
            "sufficient_track_record": sufficient,
            "target_weight": target_w,
            "mean_leverage": vt.get("mean_leverage"),
        }
        verdict = "sized" if sufficient else "refused_short_track_record"
        n_trials = int(inp.get("n_trials") or 1)
        source = "asserted" if "n_trials" in inp else "sizing"
    elif wf in ("monitor", "monitor_sleeve"):
        mon_inp = {k: v for k, v in no_trials_inp.items() if k != "tolerances"}
        perf = ex.execute("compute.performance_report", mon_inp)
        var = ex.execute("compute.compute_var_cvar", mon_inp)
        mon_var = (var.get("historical") or {}).get("var_pct") or (var.get("parametric") or {}).get("var_pct")
        mon_cvar = (var.get("cvar") or {}).get("cvar_pct")
        base_mon = {**perf, **var, "var_pct": mon_var, "cvar_pct": mon_cvar}
        tols = inp.get("tolerances")
        if not isinstance(tols, dict) or not tols:
            figures = {**base_mon, "status": "unchecked", "n_checked": 0, "breaches": []}
            verdict = "unchecked"
        else:
            breaches = []
            for k, lim in tols.items():
                obs_v = base_mon.get(k)
                if (
                    isinstance(obs_v, (int, float))
                    and isinstance(lim, (int, float))
                    and float(obs_v) > float(lim)
                ):
                    breaches.append({"limit": k, "observed": obs_v, "tolerance": lim, "direction": "above"})
            status = "breached" if breaches else "ok"
            figures = {**base_mon, "status": status, "n_checked": len(tols), "breaches": breaches}
            verdict = status
        n_trials = 1
        source = "monitor"
    else:
        raise ValueError(f"Unknown local workflow {workflow!r}")

    return figures, verdict, n_trials, source, returns_col, trial_rows


def rerun_record_report(proj: Project, run_rec: dict[str, Any]) -> dict[str, Any]:
    """Re-execute a recorded run and evaluate all 3 `QOS-158` verification statuses:

    1. ``hash_verified``: CAS object integrity (`object_sha256` & `manifest_sha256`).
    2. ``figures_reproduced``: Re-executes from inputs and compares EVERY recorded
       scalar and curve/list figure within `1e-6`, failing immediately if any
       recorded key is missing in the reproduced output.
    3. ``provenance_verified``: Compares recorded `RunManifest` (`engine_version`,
       `wheel_sha256`, `code_sha256`, `dataset_hashes`, conventions) against the
       active runtime environment.
    """
    if run_rec.get("read_only"):
        raise ValueError(
            "Pulled remote runs are read-only summaries without local code or data "
            "and cannot be rerun locally."
        )

    hash_ok = True
    hash_errors: list[str] = []
    obj_sha = run_rec.get("object_sha256")
    if isinstance(obj_sha, str) and obj_sha:
        try:
            proj.get_object(obj_sha)
        except (FileNotFoundError, ValueError, OSError) as exc:
            hash_ok = False
            hash_errors.append(f"object_sha256: {exc}")

    man_sha = run_rec.get("manifest_sha256")
    if isinstance(man_sha, str) and man_sha:
        try:
            proj.get_object(man_sha)
        except (FileNotFoundError, ValueError, OSError) as exc:
            hash_ok = False
            hash_errors.append(f"manifest_sha256: {exc}")

    spec = run_rec.get("repro_spec") or {}
    kind = str(spec.get("kind") or run_rec.get("command") or "")
    data: Any = None
    backtest_fn: Any = None
    code_target: Any = None

    if spec.get("data_object_sha256"):
        data = proj.get_object(str(spec["data_object_sha256"]))
    elif spec.get("data_path"):
        from .connectors import load_panel

        dp = Path(str(spec["data_path"]))
        if not dp.is_absolute():
            dp = proj.root / dp
        data = load_panel(dp)

    if spec.get("project_module"):
        from .cli import load_project, project_grid

        mod_data, backtest_fn = load_project(str(spec["project_module"]))
        code_target = backtest_fn
        if data is None:
            data = mod_data
        grid = spec.get("grid") or project_grid()
    else:
        grid = spec.get("grid")

    if kind == "demo":
        from . import demo
        from .sweep import sweep

        code_target = demo.backtest_fn
        with suppress_auto_sweep_record():
            res = sweep(demo.backtest_fn, demo.GRID, data=demo.data)
            v = res.verdict()
            repro_figs: dict[str, Any] = {
                "n_trials": v["n_trials"],
                "n_ok_trials": v.get("n_ok_trials"),
                "n_failed_trials": v.get("n_failed_trials"),
                "n_trials_source": v["n_trials_source"],
                "n_eff_participation": v.get("n_eff_participation"),
                "deflated_sharpe_at_n_eff": v.get("deflated_sharpe_at_n_eff"),
                "n_eff_diagnostic_note": v.get("n_eff_diagnostic_note"),
                "dsr_probability": v.get("dsr_probability"),
                "deflated_sharpe": v.get("deflated_sharpe"),
                "psr_vs_zero": v.get("psr_vs_zero"),
                "sr0_expected_max": v.get("sr0_expected_max"),
                "performance": v.get("performance", {}),
                "surface": res.surface(),
            }
            ds_hashes = hash_dataset_any(demo.data)
    elif kind == "process":
        from .cli import _process_series
        from .core.process import dgp_stress

        code_target = dgp_stress
        if data is None:
            from . import demo

            data = demo.data
        series = _process_series(data)
        dgp = str(spec.get("dgp") or "gbm")
        repro_figs = dgp_stress(
            series, dgp=dgp, n_paths=int(spec.get("n_paths") or 80), seed=int(spec.get("seed") or 7)
        )
        ds_hashes = hash_dataset_any(series)
    elif kind == "sweep" and spec.get("matrix_object_sha256"):
        from .sweep.runner import SweepResult, Trial

        saved_obj = proj.get_object(str(spec["matrix_object_sha256"]))
        trials = [Trial(**t) for t in saved_obj["trials"]]
        matrix = np.asarray(saved_obj["matrix"], dtype=float)
        sr = SweepResult(
            trials=trials,
            matrix=matrix,
            data_hash=str(saved_obj.get("data_hash") or ""),
            grid_keys=list(saved_obj.get("grid_keys") or []),
            store_params=bool(saved_obj.get("store_params", False)),
            risk_free_rate=saved_obj.get("risk_free_rate"),
            periods_per_year=int(saved_obj.get("periods_per_year") or 252),
        )
        v = sr.verdict(
            risk_free_rate=spec.get("risk_free_rate", saved_obj.get("risk_free_rate")),
            periods_per_year=spec.get("periods_per_year", saved_obj.get("periods_per_year")),
        )
        repro_figs = {
            "n_trials": v["n_trials"],
            "n_ok_trials": v.get("n_ok_trials"),
            "n_failed_trials": v.get("n_failed_trials"),
            "n_trials_source": v["n_trials_source"],
            "n_eff_participation": v.get("n_eff_participation"),
            "deflated_sharpe_at_n_eff": v.get("deflated_sharpe_at_n_eff"),
            "n_eff_diagnostic_note": v.get("n_eff_diagnostic_note"),
            "dsr_probability": v.get("dsr_probability"),
            "deflated_sharpe": v.get("deflated_sharpe"),
            "psr_vs_zero": v.get("psr_vs_zero"),
            "sr0_expected_max": v.get("sr0_expected_max"),
            "performance": v.get("performance", {}),
            "surface": sr.surface(),
        }
        ds_hashes = [sr.data_hash] if sr.data_hash else []
    elif kind == "coded_signal":
        from .coded import run_signal

        sig_target = spec.get("signal_target")
        if not sig_target:
            raise ValueError("coded_signal repro_spec is missing `signal_target`.")
        code_target = sig_target
        with suppress_auto_sweep_record():
            coded_prices = data.get("prices") if isinstance(data, dict) and "prices" in data else data
            coded_res = run_signal(
                sig_target,
                prices=coded_prices,
                data=data,
                params=spec.get("params"),
                grid=spec.get("grid"),
                slippage_bps=float(spec.get("slippage_bps", 5.0)),
                commission_bps=float(spec.get("commission_bps", 1.0)),
                borrow_bps=float(spec.get("borrow_bps", 0.0)),
                record_in_project=False,
            )
            repro_figs = dict(coded_res.get("figures") or {})
            ds_hashes = list(coded_res.get("dataset_hashes") or [])
    else:
        wf = str(spec.get("workflow") or kind)
        repro_figs, _verdict, _nt, _src, _rets, _tr = execute_local_workflow(
            wf,
            data=data,
            backtest_fn=backtest_fn,
            grid=grid,
            inputs=spec.get("inputs"),
        )
        if _src is not None and "n_trials_source" not in repro_figs:
            repro_figs["n_trials_source"] = _src
        if _nt is not None and "run_n_trials" not in repro_figs:
            repro_figs["run_n_trials"] = _nt
        if "n_trials" in (run_rec.get("figures") or {}) and "n_trials" not in repro_figs:
            repro_figs["n_trials"] = (run_rec.get("figures") or {}).get("n_trials")
        ds_hashes = hash_dataset_any(data)

    ledger_meta_keys = {
        "cumulative_trials",
        "family_trials",
        "project_trials",
        "duplicate_of",
        "run_id",
        "attempt_id",
        "timestamp",
    }
    orig_flat = _flatten_figures(run_rec.get("figures") or {})
    repro_flat = _flatten_figures(repro_figs)
    mismatches: dict[str, dict[str, Any]] = {}
    for k, orig_v in orig_flat.items():
        if k in ledger_meta_keys:
            continue
        if k not in repro_flat:
            mismatches[k] = {
                "recorded": orig_v,
                "reproduced": None,
                "reason": "missing_in_reproduction",
            }
            continue
        new_v = repro_flat[k]
        if (
            isinstance(orig_v, (int, float))
            and isinstance(new_v, (int, float))
            and not isinstance(orig_v, bool)
            and not isinstance(new_v, bool)
        ):
            if abs(float(orig_v) - float(new_v)) > 1e-6:
                mismatches[k] = {"recorded": orig_v, "reproduced": new_v}
        elif orig_v != new_v:
            mismatches[k] = {"recorded": orig_v, "reproduced": new_v}

    figures_ok = len(mismatches) == 0

    rec_manifest = run_rec.get("manifest") if isinstance(run_rec.get("manifest"), Mapping) else None
    rec_rf = (rec_manifest or {}).get("risk_free") or {}
    rf_arg = rec_rf.get("annual_rate") if rec_rf.get("source") == "caller" else None
    cur_manifest = build_run_manifest(
        params=dict(run_rec.get("params") or {}),
        dataset_hashes=ds_hashes or list(run_rec.get("dataset_hashes") or []),
        code_target=code_target,
        costs_model=(rec_manifest.get("costs_model") if isinstance(rec_manifest, Mapping) else None),
        periods_per_year=int(
            ((rec_manifest or {}).get("annualization") or {}).get("periods_per_year") or 252
        ),
        risk_free_rate=rf_arg,
    )
    prov_ok, prov_diag = check_manifest_provenance(rec_manifest, current=cur_manifest)

    return {
        "run_id": run_rec.get("run_id"),
        "command": run_rec.get("command"),
        "hash_verified": hash_ok,
        "hash_errors": hash_errors,
        "figures_reproduced": figures_ok,
        "provenance_verified": prov_ok,
        "provenance": prov_diag,
        "reproduced_figures": repro_figs,
        "mismatches": mismatches,
    }


def rerun_record(proj: Project, run_rec: dict[str, Any]) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    """Re-execute a recorded run from its `repro_spec` and compare all recorded figures.

    Returns `(matched, reproduced_figures, diff_map)` where `matched` requires
    both `hash_verified` and `figures_reproduced`. Use `rerun_record_report()`
    for the full 3-status (`hash_verified`, `figures_reproduced`,
    `provenance_verified`) dictionary.
    """
    rep = rerun_record_report(proj, run_rec)
    matched = bool(rep["hash_verified"] and rep["figures_reproduced"])
    return matched, dict(rep["reproduced_figures"]), dict(rep["mismatches"])
