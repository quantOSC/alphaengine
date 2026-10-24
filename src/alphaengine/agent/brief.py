"""The figures a downloaded report is allowed to draw.

Computed here, on this machine, from the series the session already holds.
Nothing in this module is sent anywhere.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def study_for(trace: Any, *, deep: bool = False) -> dict[str, Any]:
    """Paths, books, scenarios, Monte Carlo fan, and bakeoff tournament for the report."""
    question = str(getattr(trace, "question", "") or "")
    data = _panel(getattr(trace, "panel", None))
    focus_names = _focus_names(question, trace, data)
    paths = _paths(data, focus_names)
    books = _books(data, question)
    primary = _primary_book(books, question)
    assets = _asset_table(data, books, primary, focus_names)
    scenarios = _scenarios(primary) if primary is not None else []
    raw_monte = getattr(trace, "monte", None)
    if isinstance(raw_monte, dict) and raw_monte.get("edges"):
        monte: dict[str, Any] | None = raw_monte
    elif deep and primary is not None:
        monte = run_monte_carlo(primary["returns"], dgp="gbm", n_paths=64, seed=7)
    else:
        monte = None
    held = _bakeoff(getattr(trace, "bakeoff", None), data)
    corr = _correlation(data, focus_names, primary)
    findings = _findings(
        books=books,
        primary=primary,
        assets=assets,
        focus_names=focus_names,
        scenarios=scenarios,
        monte=monte,
        bakeoff=held,
        stack=_stack_lines(trace),
    )
    narrate_figures = _narrate_figures(
        data=data,
        books=books,
        primary=primary,
        assets=assets,
        focus_names=focus_names,
        scenarios=scenarios,
        monte=monte,
        bakeoff=held,
    )
    return {
        "n_universe": len(data),
        "focus_names": focus_names,
        "paths": paths,
        "correlation": corr,
        "books": books,
        "primary": primary,
        "assets": assets,
        "scenarios": scenarios,
        "monte": monte,
        "bakeoff": held,
        "boards": [row for row in (getattr(trace, "boards", None) or []) if isinstance(row, dict)],
        "findings": findings,
        "narrate_figures": narrate_figures,
    }


def findings_for(trace: Any) -> str:
    cached = getattr(trace, "study", None)
    if isinstance(cached, dict) and cached.get("findings"):
        return str(cached["findings"])
    return str(study_for(trace, deep=False).get("findings") or "")


def _panel(raw: Any) -> dict[str, list[float]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[float]] = {}
    for key, values in raw.items():
        nums = _nums(values)
        if len(nums) >= 8:
            out[str(key)] = nums[-2500:]
        if len(out) == 64:
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


def _focus_names(question: str, trace: Any, data: dict[str, list[float]]) -> list[str]:
    """Tickers from the loaded book that the user or tool stack explicitly named."""
    if not data:
        return []
    by_upper = {name.upper(): name for name in data}
    tokens = [tok.upper() for tok in re.findall(r"[A-Za-z0-9._-]+", question)]
    for call in getattr(trace, "calls", []) or []:
        args = getattr(call, "args", None)
        if isinstance(args, dict):
            for val in args.values():
                if isinstance(val, str):
                    tokens.extend(tok.upper() for tok in re.findall(r"[A-Za-z0-9._-]+", val))
                elif isinstance(val, list):
                    tokens.extend(str(item).upper() for item in val)
    found: list[str] = []
    for tok in tokens:
        matched = by_upper.get(tok)
        if matched and matched not in found:
            found.append(matched)
    return found[:8]


def _paths(data: dict[str, list[float]], focus_names: list[str]) -> dict[str, list[float]]:
    if not data:
        return {}
    try:
        from ..core.relations import as_returns

        rets_map = as_returns(data)
    except Exception:  # noqa: BLE001
        return {}
    ordered: list[str] = []
    for name in focus_names:
        if name in rets_map and name not in ordered:
            ordered.append(name)
    # Rank remaining names by total return spread so the chart shows leaders and laggards
    scored: list[tuple[float, str]] = []
    for name, series in rets_map.items():
        if name in ordered:
            continue
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size >= 8:
            scored.append((float(np.prod(1.0 + arr)), name))
    scored.sort(reverse=True)
    if scored:
        top_half = [name for _, name in scored[:6]]
        bot_half = [name for _, name in scored[-4:]]
        for name in top_half + bot_half:
            if name not in ordered:
                ordered.append(name)
    out: dict[str, list[float]] = {}
    for name in ordered[:10]:
        arr = np.asarray(rets_map.get(name) or [], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 8:
            continue
        out[name] = np.cumprod(1.0 + arr).tolist()
    return out


def _correlation(
    data: dict[str, list[float]],
    focus_names: list[str],
    primary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if len(data) < 2:
        return None
    selected: list[str] = []
    for name in focus_names:
        if name in data and name not in selected:
            selected.append(name)
    if primary is not None:
        for name, _w in primary.get("weights") or []:
            if name in data and name not in selected:
                selected.append(name)
            if len(selected) >= 14:
                break
    for name in data:
        if name not in selected:
            selected.append(name)
        if len(selected) >= 14:
            break
    sub = {name: data[name] for name in selected if name in data}
    try:
        from ..core.relations import correlation

        raw = correlation(sub)
    except (ValueError, Exception):  # noqa: BLE001 — a short book still gets a report
        return None
    names = [str(name) for name in (raw.get("names") or [])][:14]
    values = raw.get("values")
    if len(names) < 2 or not isinstance(values, list):
        return None
    return {
        "names": names,
        "values": [row[: len(names)] for row in values[: len(names)]],
        "pairs": list(raw.get("pairs") or [])[:12],
        "n_total": len(data),
    }


def _books(data: dict[str, list[float]], question: str = "") -> list[dict[str, Any]]:
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
    built: dict[str, Any] = {}
    for method in ("hrp", "risk_parity"):
        try:
            result = build_portfolio(data, method=method)
        except (ValueError, Exception):  # noqa: BLE001
            continue
        weights = result.get("weights") or {}
        if isinstance(weights, dict):
            built[method] = np.array([float(weights.get(name, 0.0)) for name in names], dtype=float)
    # VaR-sorted / inverse-tail-loss allocation across the aligned universe
    q05 = np.array(
        [max(abs(float(np.quantile(aligned[:, j], 0.05))), 1e-4) for j in range(aligned.shape[1])],
        dtype=float,
    )
    inv_var = (1.0 / q05) / float((1.0 / q05).sum())
    built["var_sorted"] = inv_var

    books: list[dict[str, Any]] = []
    equal = np.ones(aligned.shape[1]) / aligned.shape[1]
    books.append(_pack("equal_weight", names, equal, aligned))
    for method in ("hrp", "risk_parity", "var_sorted"):
        if method in built:
            books.append(_pack(method, names, built[method], aligned))
    return books


def _primary_book(books: list[dict[str, Any]], question: str) -> dict[str, Any] | None:
    if not books:
        return None
    low = question.lower()
    if "var" in low and ("sort" in low or "book" in low or "construct" in low):
        picked = next((book for book in books if book["method"] == "var_sorted"), None)
        if picked is not None:
            return picked
    if "risk parity" in low and "hierarchical" not in low:
        picked = next((book for book in books if book["method"] == "risk_parity"), None)
        if picked is not None:
            return picked
    return next((book for book in books if book["method"] == "hrp"), books[0])


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
    hhi = float(np.sum(vector**2))
    eff_n = round(1.0 / hhi, 2) if hhi > 0 else float(len(names))
    series = [float(x) for x in rets]
    return {
        "method": method,
        "weights": pairs,
        "weight_map": {names[i]: round(float(vector[i]), 4) for i in range(len(names))},
        "max_weight": round(float(vector.max()), 4),
        "min_weight": round(float(vector.min()), 4),
        "effective_n": eff_n,
        "wealth": wealth.tolist(),
        "drawdown": drawdown.tolist(),
        "returns": series,
        "stats": _stats(series),
        "var": _var(series),
    }


def _asset_table(
    data: dict[str, list[float]],
    books: list[dict[str, Any]],
    primary: dict[str, Any] | None,
    focus_names: list[str],
) -> list[dict[str, Any]]:
    """Per-asset return, vol, Sharpe, drawdown, 95% VaR, and weights across methods."""
    if not data:
        return []
    try:
        from ..core.relations import as_returns

        panel = as_returns(data)
    except Exception:  # noqa: BLE001
        return []
    by_method: dict[str, dict[str, float]] = {}
    for book in books:
        by_method[book["method"]] = dict(book.get("weight_map") or {})
    rows: list[dict[str, Any]] = []
    for name, series in panel.items():
        arr = np.asarray(series, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 8:
            continue
        wealth = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(wealth)
        dd = float(np.min(wealth / np.maximum(peak, 1e-18) - 1.0))
        total_ret = float(wealth[-1] - 1.0)
        ann_ret = float(wealth[-1] ** (252.0 / arr.size) - 1.0) if wealth[-1] > 0 else total_ret
        vol_ann = float(arr.std(ddof=1) * math.sqrt(252.0)) if arr.size > 1 else 0.0
        sharpe = (
            float((arr.mean() - 0.0421 / 252.0) / arr.std(ddof=1) * math.sqrt(252.0)) if vol_ann > 0 else 0.0
        )
        var95 = abs(float(np.quantile(arr, 0.05))) * 100.0
        tail = arr[arr <= np.quantile(arr, 0.05)]
        cvar95 = abs(float(tail.mean())) * 100.0 if tail.size else var95
        rows.append(
            {
                "name": name,
                "focused": name in focus_names,
                "hrp_weight": round(by_method.get("hrp", {}).get(name, 0.0), 4),
                "rp_weight": round(by_method.get("risk_parity", {}).get(name, 0.0), 4),
                "var_weight": round(by_method.get("var_sorted", {}).get(name, 0.0), 4),
                "eq_weight": round(by_method.get("equal_weight", {}).get(name, 0.0), 4),
                "total_return_pct": round(total_ret * 100.0, 2),
                "annualized_return_pct": round(ann_ret * 100.0, 2),
                "volatility_annualized_pct": round(vol_ann * 100.0, 2),
                "sharpe_annualized": round(sharpe, 4),
                "max_drawdown_pct": round(abs(dd) * 100.0, 2),
                "var_95_pct": round(var95, 2),
                "cvar_95_pct": round(cvar95, 2),
            }
        )
    primary_key = "var_weight" if (primary and primary.get("method") == "var_sorted") else "hrp_weight"
    rows.sort(key=lambda r: (not bool(r["focused"]), -float(r[primary_key]), -float(r["sharpe_annualized"])))
    return rows


def _stats(series: list[float]) -> dict[str, Any]:
    if len(series) < 30:
        return _short_stats(series)
    try:
        from ..core.performance import performance_report

        raw = performance_report(series)
    except Exception:  # noqa: BLE001
        return _short_stats(series)
    if not isinstance(raw, dict) or raw.get("error"):
        return _short_stats(series)
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


def _short_stats(series: list[float]) -> dict[str, Any]:
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {}
    wealth = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(wealth)
    dd = float(np.min(wealth / np.maximum(peak, 1e-18) - 1.0))
    total = float(wealth[-1] - 1.0)
    ann = float(wealth[-1] ** (252.0 / arr.size) - 1.0) if wealth[-1] > 0 else total
    sd = float(arr.std(ddof=1))
    vol = sd * math.sqrt(252.0)
    sharpe = float((arr.mean() - 0.0421 / 252.0) / sd * math.sqrt(252.0)) if sd > 0 else 0.0
    downside = np.minimum(arr - 0.0421 / 252.0, 0.0)
    dd_dev = float(np.sqrt(np.mean(downside**2)))
    sortino = float((arr.mean() - 0.0421 / 252.0) / dd_dev * math.sqrt(252.0)) if dd_dev > 0 else 0.0
    calmar = float(ann / abs(dd)) if abs(dd) > 1e-9 else 0.0
    return {
        "sharpe_annualized": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown_pct": round(abs(dd) * 100.0, 2),
        "total_return_pct": round(total * 100.0, 2),
        "annualized_return_pct": round(ann * 100.0, 2),
        "volatility_annualized_pct": round(vol * 100.0, 2),
        "n_obs": int(arr.size),
    }


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
    cf = _dict(raw.get("cornish_fisher"))
    return {
        "var_pct": historical.get("var_pct"),
        "cvar_pct": cvar.get("cvar_pct"),
        "parametric_var_pct": parametric.get("var_pct"),
        "cornish_fisher_var_pct": cf.get("var_pct"),
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _scenarios(book: dict[str, Any]) -> list[dict[str, Any]]:
    rets = np.asarray(book.get("returns") or [], dtype=float)
    rets = rets[np.isfinite(rets)]
    if rets.size < 20:
        return []
    mu = float(rets.mean())
    centered = rets - mu
    rate_hike = (mu - 0.01 / 252.0) + centered * 1.25
    stagflation = (mu - 0.02 / 252.0) + centered * 1.50
    rows = [
        _scene("full sample", rets),
        _scene("rate hike (+25 bps/qtr, 1.25x vol)", rate_hike),
        _scene("stagflation & conflict (1.5x vol)", stagflation),
        _scene("volatility x1.5", rets * 1.5),
        _scene("volatility x2.0 (crisis)", rets * 2.0),
    ]
    abs_r = np.abs(centered)
    high_vol_days = rets[abs_r >= float(np.median(abs_r))]
    if high_vol_days.size >= 10:
        rows.append(_scene("high-volatility regime days", high_vol_days))
    mid = int(rets.size // 2)
    if mid >= 20 and rets.size - mid >= 20:
        rows.append(_scene("first half", rets[:mid]))
        rows.append(_scene("second half", rets[mid:]))
    if rets.size >= 21:
        wealth = np.concatenate((np.array([1.0]), np.cumprod(1.0 + rets)))
        windows = wealth[21:] / np.maximum(wealth[:-21], 1e-18) - 1.0
        worst_i = int(np.argmin(windows))
        worst_slice = rets[worst_i : worst_i + 21]
        rows.append(_scene("worst 21 observations", worst_slice))
    return rows


def _scene(name: str, rets: Any) -> dict[str, Any]:
    arr = np.asarray(rets, dtype=float)
    arr = arr[np.isfinite(arr)]
    stats = _stats([float(x) for x in arr])
    var95 = round(abs(float(np.quantile(arr, 0.05))) * 100.0, 2) if arr.size >= 5 else 0.0
    return {
        "name": name,
        "total_return_pct": stats.get("total_return_pct", 0.0),
        "annualized_return_pct": stats.get("annualized_return_pct", 0.0),
        "volatility_annualized_pct": stats.get("volatility_annualized_pct", 0.0),
        "max_drawdown_pct": stats.get("max_drawdown_pct", 0.0),
        "sharpe_annualized": stats.get("sharpe_annualized", 0.0),
        "var_95_pct": var95,
    }


def run_monte_carlo(
    series: list[float],
    *,
    dgp: str = "gbm",
    n_paths: int = 64,
    horizon: int | None = None,
    seed: int = 7,
) -> dict[str, Any] | None:
    """Calibrate `dgp` to `series` and simulate both Sharpe and cumulative wealth fan paths."""
    if len(series) < 20:
        return None
    try:
        from ..core.process import dgp_stress, histogram
    except Exception:  # noqa: BLE001
        return None
    p = max(16, min(int(n_paths), 512))
    steps = max(20, min(int(horizon or len(series)), 252))
    try:
        raw = dgp_stress(series, dgp=dgp, n_paths=p, seed=seed)
    except Exception:  # noqa: BLE001
        return None
    hist = _dict(raw.get("sharpe_hist"))
    edges = hist.get("edges") or []
    counts = hist.get("counts") or []
    if not edges or not counts:
        return None

    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    mu = float(arr.mean())
    sigma = max(float(arr.std(ddof=1)), 1e-6)
    rng = np.random.default_rng(int(seed))
    draws = rng.normal(mu, sigma, size=(p, steps))
    wealth_paths = np.ones((p, steps + 1), dtype=float)
    wealth_paths[:, 1:] = np.cumprod(1.0 + draws, axis=1)

    q05, q25, q50, q75, q95 = np.quantile(wealth_paths, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    terminals = (wealth_paths[:, -1] - 1.0) * 100.0
    term_q = np.quantile(terminals, [0.05, 0.25, 0.50, 0.75, 0.95])
    peaks = np.maximum.accumulate(wealth_paths, axis=1)
    dds = np.abs(np.min(wealth_paths / np.maximum(peaks, 1e-18) - 1.0, axis=1)) * 100.0
    dd_q = np.quantile(dds, [0.50, 0.90, 0.95])
    term_hist = histogram(terminals, bins=16)

    frac_pos = raw.get("frac_positive")
    pos_pct = round(float(frac_pos) * 100.0, 1) if isinstance(frac_pos, (int, float)) else None
    return {
        "dgp": str(raw.get("dgp") or dgp),
        "n_paths": int(raw.get("n_paths") or p),
        "horizon_days": steps,
        "sharpe_q10": raw.get("sharpe_q10"),
        "sharpe_q50": raw.get("sharpe_q50"),
        "sharpe_q90": raw.get("sharpe_q90"),
        "frac_positive": round(float(frac_pos), 4) if isinstance(frac_pos, (int, float)) else None,
        "positive_pct": pos_pct,
        "terminal_q05_pct": round(float(term_q[0]), 2),
        "terminal_q25_pct": round(float(term_q[1]), 2),
        "terminal_q50_pct": round(float(term_q[2]), 2),
        "terminal_q75_pct": round(float(term_q[3]), 2),
        "terminal_q95_pct": round(float(term_q[4]), 2),
        "drawdown_q50_pct": round(float(dd_q[0]), 2),
        "drawdown_q90_pct": round(float(dd_q[1]), 2),
        "drawdown_q95_pct": round(float(dd_q[2]), 2),
        "edges": edges,
        "counts": counts,
        "terminal_edges": term_hist.get("edges") or [],
        "terminal_counts": term_hist.get("counts") or [],
        "fan": {
            "q05": [round(float(x), 4) for x in q05],
            "q25": [round(float(x), 4) for x in q25],
            "q50": [round(float(x), 4) for x in q50],
            "q75": [round(float(x), 4) for x in q75],
            "q95": [round(float(x), 4) for x in q95],
            "sample_paths": [[round(float(x), 4) for x in wealth_paths[i]] for i in range(min(12, p))],
        },
    }


def _bakeoff(raw: Any, data: dict[str, list[float]]) -> dict[str, Any] | None:
    if not isinstance(raw, dict) and len(data) >= 2:
        raw = _default_bakeoff(data)
    if not isinstance(raw, dict):
        return None
    curves = []
    for idx, row in enumerate(raw.get("curves") or []):
        if not isinstance(row, dict):
            continue
        growth = row.get("growth")
        if not isinstance(growth, list) or len(growth) < 8:
            continue
        arr = np.asarray(growth, dtype=float)
        rets = np.diff(np.concatenate((np.array([1.0]), arr))) / np.concatenate((np.array([1.0]), arr[:-1]))
        stats = _stats([float(x) for x in rets])
        curves.append(
            {
                "round": int(row.get("round") or (idx // 2 + 1)),
                "seat": int(row.get("seat") or (idx % 2 + 1)),
                "label": str(row.get("label") or ""),
                "growth": growth,
                "sharpe": row.get("sharpe"),
                "standing": row.get("standing"),
                "worst_attack": row.get("worst_attack") or "in_sample",
                "attack_scores": row.get("attack_scores") or {},
                "annualized_sharpe": stats.get("sharpe_annualized"),
                "total_return_pct": stats.get("total_return_pct"),
                "max_drawdown_pct": stats.get("max_drawdown_pct"),
                "volatility_annualized_pct": stats.get("volatility_annualized_pct"),
            }
        )
    trials = [str(line) for line in (raw.get("trials") or [])]
    top_cfg = raw.get("top_ranked_config") or raw.get("survivor")
    top_sh = (
        raw.get("top_ranked_sharpe")
        if raw.get("top_ranked_sharpe") is not None
        else raw.get("survivor_sharpe")
    )
    weights = raw.get("model_weights") or raw.get("survivor_weights") or {}
    if not top_cfg and not trials and not curves:
        return None
    passes = [p for p in (raw.get("passes") or []) if isinstance(p, dict)]
    return {
        "top_ranked_config": top_cfg,
        "top_ranked_sharpe": top_sh,
        "model_weights": weights,
        # Deprecated aliases kept for API backward compatibility:
        "survivor": top_cfg,
        "survivor_sharpe": top_sh,
        "survivor_weights": weights,
        "attacked_sharpe": raw.get("attacked_sharpe"),
        "deflated_sharpe": raw.get("deflated_sharpe"),
        "verdict": raw.get("verdict"),
        "n_trials": raw.get("n_trials"),
        "n_passes": raw.get("n_passes") or max(len(passes), 4),
        "attacks": [str(attack) for attack in (raw.get("attacks") or [])],
        "trials": trials,
        "passes": passes,
        "curves": curves[:8],
    }


def _default_bakeoff(data: dict[str, list[float]]) -> dict[str, Any] | None:
    """Run the 4-pass referee tournament across all eight rules when the session did not run one."""
    try:
        from ..core.bakeoff import referee, trial_paths
    except Exception:  # noqa: BLE001
        return None
    submissions: list[dict[str, Any]] = [
        {"rule": "hrp", "_round": 1, "_seat": 1},
        {"rule": "risk_parity", "_round": 1, "_seat": 2},
        {"rule": "inverse_vol", "_round": 2, "_seat": 1},
        {"rule": "var_sorted", "_round": 2, "_seat": 2},
        {"rule": "low_vol", "_round": 3, "_seat": 1},
        {"rule": "equal", "_round": 3, "_seat": 2},
        {"rule": "momentum", "lookback": 20, "_round": 4, "_seat": 1},
        {"rule": "reversion", "lookback": 20, "_round": 4, "_seat": 2},
    ]
    pass_attacks = ["last_half", "high_vol", "tail_stress", "rate_shock"]
    try:
        passes: list[dict[str, Any]] = []
        for p_idx, atk in enumerate(pass_attacks, start=1):
            sub_slice = submissions[: p_idx * 2]
            atk_slice = pass_attacks[:p_idx]
            step = referee(data, sub_slice, atk_slice)
            passes.append(
                {
                    "pass": p_idx,
                    "attack": atk,
                    "leader": step.get("top_ranked_config") or step.get("survivor"),
                    "leader_sharpe": step.get("top_ranked_sharpe")
                    if step.get("top_ranked_sharpe") is not None
                    else step.get("survivor_sharpe"),
                    "leader_standing": step.get("attacked_sharpe"),
                    "n_trials": len(sub_slice),
                }
            )
        figures = referee(data, submissions, pass_attacks)
        curves = trial_paths(data, submissions, pass_attacks)
    except Exception:  # noqa: BLE001
        return None
    top_cfg = figures.get("top_ranked_config") or figures.get("survivor")
    top_sh = (
        figures.get("top_ranked_sharpe")
        if figures.get("top_ranked_sharpe") is not None
        else figures.get("survivor_sharpe")
    )
    if not top_cfg and not curves:
        return None
    top_label = str(top_cfg or "")
    top_curve = next((c for c in curves if c.get("label") == top_label), None)
    weights = (top_curve or {}).get("weights") or {}
    return {
        "top_ranked_config": top_cfg,
        "top_ranked_sharpe": top_sh,
        "model_weights": weights,
        # Deprecated aliases kept for API backward compatibility:
        "survivor": top_cfg,
        "survivor_sharpe": top_sh,
        "survivor_weights": weights,
        "attacked_sharpe": figures.get("attacked_sharpe"),
        "deflated_sharpe": figures.get("deflated_sharpe"),
        "verdict": figures.get("verdict"),
        "n_trials": figures.get("n_trials"),
        "n_passes": 4,
        "attacks": list(figures.get("attacks") or []),
        "trials": list(figures.get("trials") or []),
        "passes": passes,
        "curves": curves,
    }


def _stack_lines(trace: Any) -> str:
    lines = []
    for call in getattr(trace, "calls", []) or []:
        note = str(getattr(call, "note", "") or getattr(call, "error", "") or "")
        lines.append(f"{getattr(call, 'tool', '')}  {note}".rstrip())
    return "\n".join(lines)


def _method_title(method: str) -> str:
    if method == "hrp":
        return "Hierarchical risk parity"
    if method == "risk_parity":
        return "Equal risk contribution (risk parity)"
    if method == "var_sorted":
        return "VaR-sorted inverse-tail book"
    if method == "equal_weight":
        return "Equal weight"
    return method


def _findings(
    *,
    books: list[dict[str, Any]],
    primary: dict[str, Any] | None,
    assets: list[dict[str, Any]],
    focus_names: list[str],
    scenarios: list[dict[str, Any]],
    monte: dict[str, Any] | None,
    bakeoff: dict[str, Any] | None,
    stack: str,
) -> str:
    lines: list[str] = []
    if primary is not None and primary.get("stats"):
        stats = primary["stats"]
        label = _method_title(str(primary["method"]))
        lines.append(
            f"{label} is the constructed book ({len(primary.get('weights') or [])} assets, "
            f"effective breadth {primary.get('effective_n')} names). "
            f"Annualized Sharpe {stats.get('sharpe_annualized')}. "
            f"Sortino {stats.get('sortino_ratio')}. "
            f"Max drawdown {stats.get('max_drawdown_pct')} percent. "
            f"Annualized volatility {stats.get('volatility_annualized_pct')} percent. "
            f"Total return {stats.get('total_return_pct')} percent "
            f"(annualized {stats.get('annualized_return_pct')} percent)."
        )
        var = primary.get("var") or {}
        if var.get("var_pct") is not None:
            lines.append(
                f"One-day 95% historical VaR is {var.get('var_pct')} percent "
                f"(parametric {var.get('parametric_var_pct')} percent, "
                f"CVaR {var.get('cvar_pct')} percent)."
            )
        top = primary.get("weights") or []
        if top:
            held = ", ".join(f"{name} {weight}" for name, weight in top[:6])
            lines.append(f"Largest weights: {held}.")
    if len(books) > 1:
        comp = []
        for book in books:
            st = book.get("stats") or {}
            if st.get("sharpe_annualized") is not None:
                comp.append(
                    f"{book['method']} (Sharpe {st.get('sharpe_annualized')}, "
                    f"drawdown {st.get('max_drawdown_pct')}%, "
                    f"vol {st.get('volatility_annualized_pct')}%)"
                )
        if comp:
            lines.append("Book comparison: " + "; ".join(comp) + ".")
    if focus_names and assets:
        by_name = {str(row["name"]): row for row in assets}
        focus_bits = []
        for name in focus_names:
            row = by_name.get(name)
            if row is None:
                continue
            focus_bits.append(
                f"{name} (HRP weight {row['hrp_weight']}, VaR-book weight {row['var_weight']}, "
                f"return {row['total_return_pct']}%, vol {row['volatility_annualized_pct']}%, "
                f"Sharpe {row['sharpe_annualized']}, 95% VaR {row['var_95_pct']}%)"
            )
        if focus_bits:
            lines.append("Named sleeves in focus: " + "; ".join(focus_bits) + ".")
    rate_row = next((row for row in scenarios if str(row["name"]).startswith("rate hike")), None)
    stag_row = next((row for row in scenarios if str(row["name"]).startswith("stagflation")), None)
    shock = next((row for row in scenarios if row["name"] == "volatility x1.5"), None)
    worst = next((row for row in scenarios if str(row["name"]).startswith("worst")), None)
    bits = []
    if rate_row is not None:
        bits.append(
            f"Under the quarterly rate-hike and inflation drift scenario, total return is "
            f"{rate_row.get('total_return_pct')} percent with a {rate_row.get('max_drawdown_pct')} "
            f"percent max drawdown (Sharpe {rate_row.get('sharpe_annualized')})."
        )
    if stag_row is not None:
        bits.append(
            f"Stagflation and conflict stress takes the drawdown to "
            f"{stag_row.get('max_drawdown_pct')} percent."
        )
    if shock is not None:
        bits.append(
            f"Scaling volatility by 1.5 takes the drawdown to {shock.get('max_drawdown_pct')} percent."
        )
    if worst is not None:
        bits.append(
            f"The worst 21-observation window returns {worst.get('total_return_pct')} percent "
            f"with a {worst.get('max_drawdown_pct')} percent drawdown."
        )
    if bits:
        lines.append(" ".join(bits))
    if monte is not None:
        lines.append(
            f"Monte Carlo ({monte.get('dgp')}, {monte.get('n_paths')} paths, "
            f"{monte.get('horizon_days')} observations): median terminal return "
            f"{monte.get('terminal_q50_pct')} percent (90% cone {monte.get('terminal_q05_pct')} "
            f"to {monte.get('terminal_q95_pct')} percent), median max drawdown "
            f"{monte.get('drawdown_q50_pct')} percent, Sharpe q50 {monte.get('sharpe_q50')} "
            f"(q10 {monte.get('sharpe_q10')}, q90 {monte.get('sharpe_q90')}, "
            f"positive paths {monte.get('positive_pct')}%)."
        )
    if bakeoff and (bakeoff.get("top_ranked_config") or bakeoff.get("survivor")):
        top_cfg = bakeoff.get("top_ranked_config") or bakeoff.get("survivor")
        top_sh = (
            bakeoff.get("top_ranked_sharpe")
            if bakeoff.get("top_ranked_sharpe") is not None
            else bakeoff.get("survivor_sharpe")
        )
        lines.append(
            f"Bakeoff referee top-ranked config: {top_cfg} across "
            f"{bakeoff.get('n_trials')} candidate rules. "
            f"In-sample period Sharpe {top_sh}, "
            f"after adversarial attacks {bakeoff.get('attacked_sharpe')}, "
            f"deflated Sharpe {bakeoff.get('deflated_sharpe')} ({bakeoff.get('verdict')})."
        )
    if not lines and stack:
        lines.append(stack)
    return "\n\n".join(lines)


def _narrate_figures(
    *,
    data: dict[str, list[float]],
    books: list[dict[str, Any]],
    primary: dict[str, Any] | None,
    assets: list[dict[str, Any]],
    focus_names: list[str],
    scenarios: list[dict[str, Any]],
    monte: dict[str, Any] | None,
    bakeoff: dict[str, Any] | None,
) -> dict[str, float]:
    """Exact numbers from the study so the model can cite them under the citation guard."""
    out: dict[str, float] = {}
    if data:
        out["universe.n_names"] = float(len(data))
    for book in books:
        prefix = str(book["method"])
        out[f"{prefix}.effective_n"] = float(book.get("effective_n") or 0.0)
        out[f"{prefix}.max_weight"] = float(book.get("max_weight") or 0.0)
        out[f"{prefix}.min_weight"] = float(book.get("min_weight") or 0.0)
        for k, v in (book.get("stats") or {}).items():
            if isinstance(v, (int, float)):
                out[f"{prefix}.{k}"] = float(v)
        for k, v in (book.get("var") or {}).items():
            if isinstance(v, (int, float)):
                out[f"{prefix}.{k}"] = float(v)
        for name, weight in (book.get("weights") or [])[:8]:
            out[f"{prefix}.weight.{name}"] = float(weight)
    if primary is not None:
        for name, weight in (primary.get("weights") or [])[:12]:
            out[f"book.weight.{name}"] = float(weight)
    by_name = {str(row["name"]): row for row in assets}
    for name in focus_names:
        row = by_name.get(name)
        if row is None:
            continue
        for key in (
            "hrp_weight",
            "rp_weight",
            "var_weight",
            "total_return_pct",
            "annualized_return_pct",
            "volatility_annualized_pct",
            "sharpe_annualized",
            "max_drawdown_pct",
            "var_95_pct",
            "cvar_95_pct",
        ):
            val = row.get(key)
            if isinstance(val, (int, float)):
                out[f"asset.{name}.{key}"] = float(val)
    for row in assets[:6]:
        name = str(row["name"])
        for key in (
            "hrp_weight",
            "var_weight",
            "total_return_pct",
            "volatility_annualized_pct",
            "var_95_pct",
        ):
            val = row.get(key)
            if isinstance(val, (int, float)):
                out[f"asset.{name}.{key}"] = float(val)
    for scene in scenarios:
        tag = str(scene.get("name") or "").split()[0]
        for key in ("total_return_pct", "max_drawdown_pct", "sharpe_annualized", "var_95_pct"):
            val = scene.get(key)
            if isinstance(val, (int, float)):
                out[f"scenario.{tag}.{key}"] = float(val)
    if monte is not None:
        for key in (
            "n_paths",
            "horizon_days",
            "sharpe_q10",
            "sharpe_q50",
            "sharpe_q90",
            "frac_positive",
            "positive_pct",
            "terminal_q05_pct",
            "terminal_q25_pct",
            "terminal_q50_pct",
            "terminal_q75_pct",
            "terminal_q95_pct",
            "drawdown_q50_pct",
            "drawdown_q90_pct",
            "drawdown_q95_pct",
        ):
            val = monte.get(key)
            if isinstance(val, (int, float)):
                out[f"monte_carlo.{key}"] = float(val)
    if bakeoff is not None:
        for key in (
            "n_trials",
            "top_ranked_sharpe",
            "survivor_sharpe",
            "attacked_sharpe",
            "deflated_sharpe",
        ):
            val = bakeoff.get(key)
            if isinstance(val, (int, float)):
                out[f"bakeoff.{key}"] = float(val)
    return out
