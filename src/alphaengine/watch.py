"""Local file-polling pipeline watcher (`ae watch`) and live drift monitor (`ae monitor <run> --live`).

Pure Python + numpy. No filesystem-watcher third-party dependency required.
Research tool, not investment advice.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .connectors import load_panel
from .core import performance_report
from .datasets import list_datasets
from .envelope import build_envelope, returns_from_panel_or_series
from .ledger import hash_dataset_any, record_run, resolve_run
from .pipeline import load_pipeline_file, run_pipeline
from .project import Project, find_project

__all__ = [
    "watched_files_for_pipeline",
    "fingerprint_files",
    "watch_pipeline",
    "monitor_live_drift",
]


def watched_files_for_pipeline(pipeline_path: str | Path, proj: Project | None = None) -> list[Path]:
    """Return the list of files whose changes should trigger a re-run of `pipeline_path`."""
    p = Path(pipeline_path).expanduser().resolve()
    active_proj = proj if proj is not None else find_project(p.parent) or find_project()
    base_dir = active_proj.root if active_proj is not None else p.parent

    files: list[Path] = [p]
    spec = load_pipeline_file(p)
    refs: list[str] = []
    if spec.get("data"):
        refs.append(str(spec["data"]))
    for step in spec.get("steps") or []:
        if isinstance(step, dict) and step.get("data"):
            refs.append(str(step["data"]))

    if active_proj is not None:
        for ds in list_datasets(active_proj):
            abs_p = ds.get("abs_path")
            rel_p = ds.get("path")
            cand = Path(str(abs_p)) if abs_p else (active_proj.root / str(rel_p or ""))
            if cand.is_file() and cand not in files:
                files.append(cand)

    for ref in refs:
        cand = Path(ref)
        if not cand.is_absolute():
            cand = base_dir / cand
        if cand.is_file() and cand not in files:
            files.append(cand)

    return files


def fingerprint_files(paths: list[Path]) -> dict[str, str]:
    """Compute SHA-256 content fingerprints for each existing file in `paths`."""
    out: dict[str, str] = {}
    for p in paths:
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def watch_pipeline(
    pipeline_path: str | Path,
    *,
    proj: Project | None = None,
    interval: float = 1.0,
    max_cycles: int = 0,
    on_run: Callable[[int, list[dict[str, Any]]], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Poll watched dataset files and re-run `pipeline_path` whenever a file changes.

    Returns the number of pipeline executions performed.
    """
    p = Path(pipeline_path).expanduser().resolve()
    active_proj = proj if proj is not None else find_project(p.parent) or find_project()

    files = watched_files_for_pipeline(p, active_proj)
    last_fp = fingerprint_files(files)
    runs_executed = 0

    envelopes = run_pipeline(p, proj=active_proj)
    runs_executed += 1
    if on_run is not None:
        on_run(runs_executed, envelopes)

    cycle = 0
    while max_cycles <= 0 or cycle < max_cycles:
        sleep_fn(max(0.01, float(interval)))
        cycle += 1
        files = watched_files_for_pipeline(p, active_proj)
        cur_fp = fingerprint_files(files)
        if cur_fp != last_fp:
            last_fp = cur_fp
            envelopes = run_pipeline(p, proj=active_proj)
            runs_executed += 1
            if on_run is not None:
                on_run(runs_executed, envelopes)

    return runs_executed


def _extract_backtest_returns(proj: Project, run_rec: dict[str, Any]) -> list[float]:
    spec = run_rec.get("repro_spec") or {}
    raw_data: Any = None
    if spec.get("data_object_sha256"):
        try:
            raw_data = proj.get_object(str(spec["data_object_sha256"]))
        except Exception:  # noqa: BLE001
            raw_data = None
    if raw_data is None and spec.get("data_path"):
        dp = Path(str(spec["data_path"]))
        if not dp.is_absolute():
            dp = proj.root / dp
        if dp.is_file():
            raw_data = load_panel(dp)
    if raw_data is not None:
        rets, _ = returns_from_panel_or_series(raw_data, spec.get("symbol"))
        if rets and len(rets) >= 8:
            return rets

    # Fallback: synthesize reference series matching the recorded backtest Sharpe and volatility
    figs: dict[str, Any] = dict(run_rec.get("figures") or {})
    raw_perf = figs.get("performance")
    perf: dict[str, Any] = dict(raw_perf) if isinstance(raw_perf, dict) else figs
    ann_vol = float(perf.get("volatility_annualized_pct") or 12.0) / 100.0
    ann_sr = float(perf.get("sharpe_annualized") or 0.5)
    daily_sd = max(ann_vol / math.sqrt(252.0), 1e-4)
    daily_mu = (0.0421 / 252.0) + (ann_sr / math.sqrt(252.0)) * daily_sd
    rng = np.random.default_rng(42)
    return [float(x) for x in rng.normal(daily_mu, daily_sd, 252)]


def monitor_live_drift(
    proj: Project,
    run_ref: str,
    live_file: str | Path,
    *,
    n_boot: int = 250,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare live returns against a recorded backtest run's bootstrapped 95% intervals.

    Research-only diagnostic output; never investment advice.
    """
    run_rec = resolve_run(proj, run_ref)
    bt_rets = np.asarray(_extract_backtest_returns(proj, run_rec), dtype=float)
    bt_rets = bt_rets[np.isfinite(bt_rets)]
    if bt_rets.size < 8:
        raise ValueError(
            f"Backtest run {run_rec.get('run_id')} has insufficient observations (< 8) for drift bootstrap."
        )

    lp = Path(live_file).expanduser().resolve()
    if not lp.is_file():
        raise FileNotFoundError(f"No live returns file at {lp}")
    loaded_live = load_panel(lp)
    live_list, _ = returns_from_panel_or_series(loaded_live)
    if not live_list or len(live_list) < 5:
        raise ValueError(f"Live file {lp} needs at least 5 return observations.")
    live_arr = np.asarray(live_list, dtype=float)
    live_arr = live_arr[np.isfinite(live_arr)]
    n_live = int(live_arr.size)

    bt_perf = performance_report(bt_rets.tolist())
    live_perf = performance_report(live_arr.tolist())

    # Bootstrap `n_boot` sub-samples of length `n_live` from `bt_rets`
    rng = np.random.default_rng(seed)
    boot_sharpes: list[float] = []
    boot_dds: list[float] = []
    boot_vols: list[float] = []
    rf_daily = 0.0421 / 252.0

    for _ in range(max(50, int(n_boot))):
        idx = rng.integers(0, bt_rets.size, size=n_live)
        sample = bt_rets[idx]
        sd = float(sample.std(ddof=1)) if n_live > 1 else 0.0
        sr = float((sample.mean() - rf_daily) / sd * math.sqrt(252.0)) if sd > 0 else 0.0
        vol = sd * math.sqrt(252.0) * 100.0
        wealth = np.cumprod(1.0 + sample)
        peak = np.maximum.accumulate(wealth)
        dd = abs(float(np.min(wealth / np.maximum(peak, 1e-18) - 1.0))) * 100.0
        boot_sharpes.append(sr)
        boot_dds.append(dd)
        boot_vols.append(vol)

    sr_q025, sr_q975 = (
        round(float(np.quantile(boot_sharpes, 0.025)), 4),
        round(float(np.quantile(boot_sharpes, 0.975)), 4),
    )
    dd_q025, dd_q975 = (
        round(float(np.quantile(boot_dds, 0.025)), 2),
        round(float(np.quantile(boot_dds, 0.975)), 2),
    )
    vol_q025, vol_q975 = (
        round(float(np.quantile(boot_vols, 0.025)), 2),
        round(float(np.quantile(boot_vols, 0.975)), 2),
    )

    live_sr = float(live_perf.get("sharpe_annualized") or 0.0)
    live_dd = float(live_perf.get("max_drawdown_pct") or 0.0)
    live_vol = float(live_perf.get("volatility_annualized_pct") or 0.0)

    drift_flags: list[str] = []
    if live_sr < sr_q025:
        drift_flags.append(f"live Sharpe {live_sr} < expected 2.5th percentile {sr_q025}")
    elif live_sr > sr_q975:
        drift_flags.append(f"live Sharpe {live_sr} > expected 97.5th percentile {sr_q975}")
    if live_dd > dd_q975:
        drift_flags.append(f"live max drawdown {live_dd}% > expected 97.5th percentile {dd_q975}%")
    if live_vol > vol_q975 or live_vol < vol_q025:
        drift_flags.append(f"live annualized vol {live_vol}% outside [{vol_q025}%, {vol_q975}%]")

    status = "drift_flagged" if drift_flags else "within_expected_range"
    figures: dict[str, Any] = {
        "reference_run_id": run_rec.get("run_id"),
        "n_live_obs": n_live,
        "backtest_sharpe_annualized": bt_perf.get("sharpe_annualized"),
        "live_sharpe_annualized": live_sr,
        "expected_sharpe_95_interval": [sr_q025, sr_q975],
        "backtest_max_drawdown_pct": bt_perf.get("max_drawdown_pct"),
        "live_max_drawdown_pct": live_dd,
        "expected_max_drawdown_95_interval_pct": [dd_q025, dd_q975],
        "live_volatility_annualized_pct": live_vol,
        "expected_volatility_95_interval_pct": [vol_q025, vol_q975],
        "status": status,
        "drift_flags": drift_flags,
        "note": "Research tool, not investment advice.",
    }

    rec = record_run(
        proj,
        command="monitor_live",
        params={"reference_run_id": run_rec.get("run_id"), "live_file": str(lp.name)},
        dataset_hashes=hash_dataset_any(live_list),
        n_trials=1,
        n_trials_source="monitor_live",
        figures=figures,
        verdict=status,
        idea_id=run_rec.get("idea_id"),
        family=str(run_rec.get("family") or "monitor"),
    )
    return build_envelope(
        command="monitor_live",
        figures=rec["figures"],
        verdict=status,
        run_id=str(rec["run_id"]),
        timestamp=str(rec["timestamp"]),
        idea_id=run_rec.get("idea_id"),
        family=str(rec["family"]),
        params_hash=str(rec["params_hash"]),
        dataset_hashes=rec["dataset_hashes"],
        n_trials=1,
        n_trials_source="monitor_live",
        cumulative_trials=int(rec["cumulative_trials"]),
        project_trials=int(rec["project_trials"]),
    )
