"""The figures a downloaded report is allowed to draw.

Computed here, on this machine, from the series the session already holds.
Nothing in this module is sent anywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def study_for(trace: Any, *, deep: bool = False) -> dict[str, Any]:
    """Paths, books, scenarios, and, when asked, a Monte Carlo of the constructed book."""
    data = _panel(getattr(trace, "panel", None))
    paths = _paths(data)
    books = _books(data)
    primary = next((book for book in books if book["method"] == "hrp"), None)
    if primary is None and books:
        primary = books[0]
    scenarios = _scenarios(primary) if primary is not None else []
    monte = _monte(primary["returns"]) if deep and primary is not None else None
    held = _bakeoff(getattr(trace, "bakeoff", None))
    findings = _findings(books, scenarios, held, _stack_lines(trace))
    return {
        "paths": paths,
        "correlation": _correlation(data),
        "books": books,
        "scenarios": scenarios,
        "monte": monte,
        "bakeoff": held,
        "boards": [row for row in (getattr(trace, "boards", None) or []) if isinstance(row, dict)],
        "findings": findings,
    }


def findings_for(trace: Any) -> str:
    return str(study_for(trace, deep=False).get("findings") or "")


def _panel(raw: Any) -> dict[str, list[float]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[float]] = {}
    for key, values in raw.items():
        nums = _nums(values)
        if len(nums) >= 8:
            out[str(key)] = nums[-2500:]
        if len(out) == 12:
            break
    return out


def _nums(values: Any) -> list[float]:
    if isinstance(values, dict):
        seq = list(values.values())
    elif isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        return []
    return [float(v) for v in seq if isinstance(v, (int, float)) and not isinstance(v, bool)]


def _paths(data: dict[str, list[float]]) -> dict[str, list[float]]:
    if not data:
        return {}
    try:
        from ..core.relations import as_returns
    except Exception:  # noqa: BLE001
        return {}
    out: dict[str, list[float]] = {}
    for name, series in list(as_returns(data).items())[:12]:
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 8:
            continue
        out[name] = np.cumprod(1.0 + arr).tolist()
    return out


def _correlation(data: dict[str, list[float]]) -> dict[str, Any] | None:
    if len(data) < 2:
        return None
    try:
        from ..core.relations import correlation

        raw = correlation(data)
    except (ValueError, Exception):  # noqa: BLE001 — a short book still gets a report
        return None
    names = [str(name) for name in (raw.get("names") or [])][:12]
    values = raw.get("values")
    if len(names) < 2 or not isinstance(values, list):
        return None
    return {"names": names, "values": values[: len(names)], "pairs": list(raw.get("pairs") or [])[:12]}


def _books(data: dict[str, list[float]]) -> list[dict[str, Any]]:
    if len(data) < 2:
        return []
    try:
        from ..core.panel import align_panel
        from ..core.portfolio import build_portfolio
        from ..core.relations import as_returns
    except Exception:  # noqa: BLE001
        return []
    try:
        aligned, names, _skipped = align_panel(as_returns(data), need=8)
    except Exception:  # noqa: BLE001
        return []
    if aligned.ndim != 2 or aligned.shape[1] < 2 or aligned.shape[0] < 8:
        return []
    built: dict[str, dict[str, float]] = {}
    for method in ("hrp", "risk_parity"):
        try:
            result = build_portfolio(data, method=method)
        except (ValueError, Exception):  # noqa: BLE001
            continue
        weights = result.get("weights") or {}
        if isinstance(weights, dict):
            built[method] = {str(k): float(v) for k, v in weights.items()}
    books: list[dict[str, Any]] = []
    equal = np.ones(aligned.shape[1]) / aligned.shape[1]
    books.append(_pack("equal_weight", names, equal, aligned))
    for method, weights in built.items():
        vector = np.array([float(weights.get(name, 0.0)) for name in names], dtype=float)
        books.append(_pack(method, names, vector, aligned))
    return books


def _pack(method: str, names: list[str], weights: Any, aligned: Any) -> dict[str, Any]:
    vector = np.asarray(weights, dtype=float)
    total = float(vector.sum())
    vector = np.ones(len(names)) / len(names) if total <= 0 else vector / total
    rets = np.asarray(aligned @ vector, dtype=float)
    wealth = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / np.maximum(peak, 1e-18) - 1.0
    pairs = sorted(
        ((names[i], round(float(vector[i]), 4)) for i in range(len(names))),
        key=lambda item: item[1],
        reverse=True,
    )
    series = [float(x) for x in rets]
    return {
        "method": method,
        "weights": pairs,
        "wealth": wealth.tolist(),
        "drawdown": drawdown.tolist(),
        "returns": series,
        "stats": _stats(series),
        "var": _var(series),
    }


def _stats(series: list[float]) -> dict[str, Any]:
    if len(series) < 30:
        return {}
    try:
        from ..core.performance import performance_report

        raw = performance_report(series)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict) or raw.get("error"):
        return {}
    keep = (
        "sharpe_annualized",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown_pct",
        "total_return_pct",
        "annualized_return_pct",
        "volatility_annualized_pct",
        "n_obs",
    )
    return {key: raw.get(key) for key in keep if isinstance(raw.get(key), (int, float))}


def _var(series: list[float]) -> dict[str, Any]:
    if len(series) < 20:
        return {}
    try:
        from ..core.risk import compute_var_cvar

        raw = compute_var_cvar(series, bootstrap_samples=200)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict) or raw.get("error"):
        return {}
    historical = _dict(raw.get("historical"))
    cvar = _dict(raw.get("cvar"))
    parametric = _dict(raw.get("parametric"))
    return {
        "var_pct": historical.get("var_pct"),
        "cvar_pct": cvar.get("cvar_pct"),
        "parametric_var_pct": parametric.get("var_pct"),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _scenarios(book: dict[str, Any]) -> list[dict[str, Any]]:
    rets = np.asarray(book.get("returns") or [], dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size < 20:
        return []
    rows = [_scene("full sample", rets), _scene("volatility x1.5", rets * 1.5)]
    mid = int(rets.size // 2)
    if mid >= 20 and rets.size - mid >= 20:
        rows.append(_scene("first half", rets[:mid]))
        rows.append(_scene("second half", rets[mid:]))
    if rets.size >= 21:
        wealth = np.cumprod(1.0 + rets)
        window = wealth[21:] / np.maximum(wealth[:-21], 1e-18) - 1.0
        rows.append(
            {
                "name": "worst 21 observations",
                "total_return_pct": round(float(np.min(window)) * 100, 2),
                "max_drawdown_pct": None,
                "sharpe_annualized": None,
            }
        )
    return rows


def _scene(name: str, rets: Any) -> dict[str, Any]:
    arr = np.asarray(rets, dtype=float)
    total = float(np.prod(1.0 + arr) - 1.0) if arr.size else 0.0
    wealth = np.cumprod(1.0 + arr) if arr.size else np.asarray([1.0])
    peak = np.maximum.accumulate(wealth)
    drawdown = float(np.min(wealth / np.maximum(peak, 1e-18) - 1.0))
    stats = _stats([float(x) for x in arr])
    return {
        "name": name,
        "total_return_pct": stats.get("total_return_pct", round(total * 100, 2)),
        "max_drawdown_pct": stats.get("max_drawdown_pct", round(abs(drawdown) * 100, 2)),
        "sharpe_annualized": stats.get("sharpe_annualized"),
    }


def _monte(series: list[float]) -> dict[str, Any] | None:
    if len(series) < 30:
        return None
    try:
        from ..core.process import dgp_stress

        raw = dgp_stress(series, dgp="gbm", n_paths=64, seed=7)
    except Exception:  # noqa: BLE001
        return None
    hist = _dict(raw.get("sharpe_hist"))
    edges = hist.get("edges") or []
    counts = hist.get("counts") or []
    if not edges or not counts:
        return None
    return {
        "dgp": raw.get("dgp") or "gbm",
        "n_paths": raw.get("n_paths"),
        "sharpe_q10": raw.get("sharpe_q10"),
        "sharpe_q50": raw.get("sharpe_q50"),
        "sharpe_q90": raw.get("sharpe_q90"),
        "frac_positive": raw.get("frac_positive"),
        "edges": edges,
        "counts": counts,
    }


def _bakeoff(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    curves = []
    for row in raw.get("curves") or []:
        if not isinstance(row, dict):
            continue
        growth = row.get("growth")
        if not isinstance(growth, list) or len(growth) < 8:
            continue
        curves.append({"label": str(row.get("label") or ""), "growth": growth, "sharpe": row.get("sharpe")})
    trials = [str(line) for line in (raw.get("trials") or [])]
    if not raw.get("survivor") and not trials and not curves:
        return None
    return {
        "survivor": raw.get("survivor"),
        "survivor_sharpe": raw.get("survivor_sharpe"),
        "attacked_sharpe": raw.get("attacked_sharpe"),
        "deflated_sharpe": raw.get("deflated_sharpe"),
        "verdict": raw.get("verdict"),
        "n_trials": raw.get("n_trials"),
        "attacks": [str(attack) for attack in (raw.get("attacks") or [])],
        "trials": trials,
        "curves": curves[:6],
    }


def _stack_lines(trace: Any) -> str:
    lines = []
    for call in getattr(trace, "calls", []) or []:
        note = str(getattr(call, "note", "") or getattr(call, "error", "") or "")
        lines.append(f"{getattr(call, 'tool', '')}  {note}".rstrip())
    return "\n".join(lines)


def _findings(
    books: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    bakeoff: dict[str, Any] | None,
    stack: str,
) -> str:
    lines: list[str] = []
    primary = next((book for book in books if book["method"] == "hrp"), None)
    if primary is None and books:
        primary = books[0]
    if primary is not None and primary.get("stats"):
        stats = primary["stats"]
        label = "Hierarchical risk parity" if primary["method"] == "hrp" else primary["method"]
        lines.append(
            f"{label} is the constructed book. "
            f"Annualized Sharpe {stats.get('sharpe_annualized')}. "
            f"Max drawdown {stats.get('max_drawdown_pct')} percent. "
            f"Annualized volatility {stats.get('volatility_annualized_pct')} percent. "
            f"Total return {stats.get('total_return_pct')} percent."
        )
        var = primary.get("var") or {}
        if var.get("var_pct") is not None:
            lines.append(
                f"Historical one-day VaR is {var.get('var_pct')} percent. "
                f"CVaR is {var.get('cvar_pct')} percent."
            )
        top = primary.get("weights") or []
        if top:
            held = ", ".join(f"{name} {weight}" for name, weight in top[:4])
            lines.append(f"Largest weights: {held}.")
    shock = next((row for row in scenarios if row["name"] == "volatility x1.5"), None)
    worst = next((row for row in scenarios if str(row["name"]).startswith("worst")), None)
    bits = []
    if shock is not None:
        bits.append(
            f"Scaling volatility by 1.5 takes the drawdown to {shock.get('max_drawdown_pct')} percent."
        )
    if worst is not None:
        bits.append(f"The worst 21-observation window returns {worst.get('total_return_pct')} percent.")
    if bits:
        lines.append(" ".join(bits))
    if bakeoff and bakeoff.get("survivor"):
        lines.append(
            f"Bakeoff survivor {bakeoff.get('survivor')}. "
            f"Sharpe {bakeoff.get('survivor_sharpe')}. "
            f"After attacks {bakeoff.get('attacked_sharpe')}. "
            f"Verdict {bakeoff.get('verdict')}. "
            f"Deflated Sharpe {bakeoff.get('deflated_sharpe')}."
        )
    if not lines and stack:
        lines.append(stack)
    return "\n\n".join(lines)
