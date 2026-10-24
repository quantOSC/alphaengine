"""Versioned JSON result envelope (`v1.0`) for CLI `--json` output and `--stdin` pipes.

Every AlphaEngine command can emit a deterministic JSON envelope (`--json`) and
consume an upstream envelope from standard input (`--stdin`), enabling Unix-style
pipelines such as:

    ae screen --data prices.csv --json | ae validate --stdin --json | ae size --stdin --json
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from ._version import __version__

ENVELOPE_SCHEMA_VERSION = "1.0"

__all__ = [
    "ENVELOPE_SCHEMA_VERSION",
    "build_envelope",
    "read_stdin_envelope",
    "emit_json",
    "returns_from_panel_or_series",
]


def build_envelope(
    *,
    command: str,
    figures: dict[str, Any] | None = None,
    verdict: str | None = None,
    run_id: str | None = None,
    timestamp: str | None = None,
    idea_id: str | None = None,
    family: str | None = None,
    params_hash: str | None = None,
    dataset_hashes: list[str] | None = None,
    n_trials: int | None = None,
    n_trials_source: str = "not_recorded",
    cumulative_trials: int | None = None,
    project_trials: int | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct a versioned `v1.0` result envelope."""
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "engine_version": __version__,
        "run_id": run_id or f"run_{uuid.uuid4().hex[:12]}",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "command": command,
        "idea_id": idea_id,
        "family": family or command,
        "params_hash": params_hash or "",
        "dataset_hashes": list(dataset_hashes or []),
        "n_trials": n_trials,
        "n_trials_source": n_trials_source,
        "cumulative_trials": cumulative_trials if cumulative_trials is not None else (n_trials or 0),
        "project_trials": project_trials if project_trials is not None else (n_trials or 0),
        "verdict": verdict,
        "figures": dict(figures or {}),
        "context": dict(context or {}),
    }


def read_stdin_envelope(stream: Any = None) -> dict[str, Any]:
    """Read and validate a JSON result envelope from stdin."""
    src = stream if stream is not None else sys.stdin
    raw = src.read().strip()
    if not raw:
        raise ValueError("--stdin expected a JSON result envelope on standard input, got empty input.")
    # If narration lines preceded the JSON object, extract from first '{' to last '}'
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Could not parse JSON result envelope from stdin.")
    parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object envelope on stdin.")
    return parsed


def emit_json(envelope: dict[str, Any]) -> None:
    """Print the JSON result envelope to stdout."""
    sys.stdout.write(json.dumps(envelope, indent=2, sort_keys=True, default=str) + "\n")
    sys.stdout.flush()


def returns_from_panel_or_series(
    data: Any, symbol: str | None = None
) -> tuple[list[float] | None, str | None]:
    """Extract a 1-D return series (and chosen symbol name) from loaded data or a universe panel."""
    from .core.series_shapes import series_values

    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        vals = [float(x) for x in data]
        if any(abs(v) > 2.0 for v in vals[:5]) and len(vals) >= 2:
            rets = [(vals[i + 1] - vals[i]) / vals[i] for i in range(len(vals) - 1) if vals[i] != 0]
            return rets, symbol
        return vals, symbol

    if isinstance(data, dict):
        for k in ("returns", "pnl"):
            if isinstance(data.get(k), list) and data[k]:
                return [float(x) for x in data[k]], symbol
        for k in ("close", "series", "spread"):
            if isinstance(data.get(k), list) and len(data[k]) >= 2:
                vals = [float(x) for x in data[k]]
                rets = [(vals[i + 1] - vals[i]) / vals[i] for i in range(len(vals) - 1) if vals[i] != 0]
                return rets, k

        # Universe dict `{symbol: closes}`
        chosen_sym = None
        if symbol:
            by_upper = {str(k).upper(): str(k) for k in data}
            chosen_sym = by_upper.get(symbol.upper())
        if chosen_sym is None and data:
            chosen_sym = str(next(iter(data.keys())))
        if chosen_sym and chosen_sym in data:
            closes_arr, _ = series_values(data[chosen_sym])
            closes = [float(x) for x in closes_arr.tolist()]
            if len(closes) >= 2:
                rets = [
                    (closes[i + 1] - closes[i]) / closes[i] for i in range(len(closes) - 1) if closes[i] != 0
                ]
                return rets, chosen_sym
    return None, None
