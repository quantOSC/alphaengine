"""The bakeoff referee. Generators propose. An adversary attacks. This names the survivor.

A proposal is a rule the engine can run on the loaded book. A return series
supplied by a model is not a proposal: the model would be grading its own
homework. Failures stay in the trial count. The source of that count is
``bakeoff``, which is a different claim from a grid and from a number the
caller typed.

The standing of a proposal is the worst of its full-sample Sharpe and every
attack the adversary actually submitted. The highest standing survives. A
field named ``winner`` on a submission is ignored.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .allocate import hrp_weights, risk_parity_weights
from .covariance import ledoit_wolf_cov
from .panel import align_panel
from .rates import cash_rate
from .relations import as_returns
from .validation import deflated_sharpe

RULES = ("equal", "hrp", "risk_parity", "inverse_vol", "momentum", "reversion")
ATTACKS = ("last_half", "first_half", "drop_best", "high_vol")

_MIN_OBS = 40


def choose_survivor(rows: list[dict[str, Any]]) -> int | None:
    """Index of the row with the highest standing. Failed rows cannot win."""
    best_i: int | None = None
    best_key: tuple[float, float, int] | None = None
    for row in rows:
        if row.get("failed"):
            continue
        standing = row.get("standing")
        sharpe = row.get("sharpe")
        if not isinstance(standing, (int, float)) or not isinstance(sharpe, (int, float)):
            continue
        if not math.isfinite(float(standing)) or not math.isfinite(float(sharpe)):
            continue
        key = (float(standing), float(sharpe), -int(row["index"]))
        if best_key is None or key > best_key:
            best_key = key
            best_i = int(row["index"])
    return best_i


def referee(
    data: Any,
    submissions: list[dict[str, Any]],
    attacks: list[str] | None = None,
) -> dict[str, Any]:
    """Score every submission. ``n_trials`` is how many were submitted, including the ones that broke."""
    chosen = [attack for attack in (attacks or []) if attack in ATTACKS]
    panel = _panel(data)
    rows: list[dict[str, Any]] = []
    series: list[np.ndarray | None] = []
    sharpes: list[float] = []
    for index, submission in enumerate(submissions):
        label, failed, earned, per = _materialize(panel, submission, index)
        series.append(earned)
        row: dict[str, Any] = {
            "index": index,
            "label": label,
            "failed": failed,
            "sharpe": None if per is None else round(per, 4),
            "standing": None,
        }
        rows.append(row)
        if per is not None:
            sharpes.append(per)
    _apply_attacks(rows, series, panel, chosen)
    winner = choose_survivor(rows)
    survivor = rows[winner]["label"] if winner is not None else None
    out: dict[str, Any] = {
        "mode": "bakeoff",
        "n_trials": len(submissions),
        "n_trials_source": "bakeoff",
        "n_failed": sum(1 for row in rows if row["failed"]),
        "survivor": survivor,
        "survivor_sharpe": None if winner is None else rows[winner]["sharpe"],
        "attacked_sharpe": None if winner is None else rows[winner]["standing"],
        "attacks": chosen,
        "rules": [row["label"] for row in rows],
        "trials": [
            f"{row['index']:02d} {row['label']}  {row['failed'] or 'sharpe ' + str(row['sharpe'])}"
            for row in rows
        ],
        "verdict": "no_survivor",
    }
    if winner is None or series[winner] is None:
        return out
    dispersion = float(np.std(sharpes, ddof=1)) if len(sharpes) >= 2 else None
    dsr = deflated_sharpe(
        series[winner].tolist(),
        n_trials=len(submissions),
        trials_sharpe_std=dispersion,
    )
    out["deflated_sharpe"] = dsr.get("deflated_sharpe")
    out["verdict"] = dsr.get("verdict") or "unscored"
    out["n_trials"] = len(submissions)
    return out


def trial_paths(data: Any, submissions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Growth of 1 for each proposal. Local to the report. Not a referee figure."""
    panel = _panel(data)
    out: list[dict[str, Any]] = []
    for index, submission in enumerate(submissions):
        label, failed, earned, per = _materialize(panel, submission, index)
        growth = None if earned is None else np.cumprod(1.0 + earned).tolist()
        out.append(
            {
                "label": label,
                "failed": failed,
                "sharpe": None if per is None else round(float(per), 4),
                "growth": growth,
            }
        )
    return out


def _apply_attacks(
    rows: list[dict[str, Any]],
    series: list[np.ndarray | None],
    panel: tuple[np.ndarray, list[str]] | None,
    attacks: list[str],
) -> None:
    market = None if panel is None else panel[0].mean(axis=1)
    for row, earned in zip(rows, series, strict=True):
        if earned is None or row["sharpe"] is None:
            continue
        standing = float(row["sharpe"])
        window = market[-earned.size :] if market is not None and market.size >= earned.size else None
        for attack in attacks:
            attacked = _attack(earned, attack, window)
            score = None if attacked is None else _sharpe(attacked)
            if score is None:
                continue
            standing = min(standing, score)
        row["standing"] = round(standing, 4)


def _attack(earned: np.ndarray, attack: str, market: np.ndarray | None) -> np.ndarray | None:
    if attack == "first_half":
        return earned[: earned.size // 2]
    if attack == "last_half":
        return earned[earned.size // 2 :]
    if attack == "drop_best":
        if earned.size < 20:
            return None
        cut = float(np.quantile(earned, 0.95))
        return earned[earned <= cut]
    if attack == "high_vol" and market is not None and market.size == earned.size:
        scale = np.abs(market)
        return earned[scale >= float(np.median(scale))]
    return None


def _materialize(
    panel: tuple[np.ndarray, list[str]] | None,
    submission: dict[str, Any],
    index: int,
) -> tuple[str, str | None, np.ndarray | None, float | None]:
    rule = str(submission.get("rule") or "").strip().lower().replace("-", "_").replace(" ", "_")
    lookback = _lookback(submission.get("lookback"))
    label = f"{rule}/{lookback}#{index}" if rule in {"momentum", "reversion"} else f"{rule}#{index}"
    if panel is None:
        return label, "need at least two named series", None, None
    if rule not in RULES:
        return label, "that rule is not in the bakeoff", None, None
    returns, names = _subset(panel, submission.get("names"))
    if returns is None or names is None:
        return label, "those names are not in the book", None, None
    try:
        earned = _earn(returns, rule, lookback)
    except (ValueError, FloatingPointError):
        return label, "the rule did not run", None, None
    score = None if earned is None else _sharpe(earned)
    if earned is None or score is None:
        return label, "not enough observations", None, None
    return label, None, earned, score


def _earn(returns: np.ndarray, rule: str, lookback: int) -> np.ndarray | None:
    if rule == "momentum":
        return _timed(returns, lookback, sign=1.0)
    if rule == "reversion":
        return _timed(returns, lookback, sign=-1.0)
    fit = returns[: returns.shape[0] // 2]
    if fit.shape[0] < 8:
        return None
    weights = _weights(fit, rule)
    if weights is None:
        return None
    return returns @ weights


def _weights(fit: np.ndarray, rule: str) -> np.ndarray | None:
    n = fit.shape[1]
    if rule == "equal":
        return np.ones(n) / n
    if rule == "inverse_vol":
        vol = np.clip(fit.std(axis=0, ddof=1), 1e-8, None)
        raw = 1.0 / vol
        return raw / raw.sum()
    cov, _shrink, _mu = ledoit_wolf_cov(fit)
    labels = [str(i) for i in range(n)]
    built = (
        risk_parity_weights(cov, names=labels) if rule == "risk_parity" else hrp_weights(cov, names=labels)
    )
    raw = np.array([float(built["weights"][label]) for label in labels], dtype=float)
    if raw.sum() <= 0:
        return None
    return raw / raw.sum()


def _timed(returns: np.ndarray, lookback: int, *, sign: float) -> np.ndarray | None:
    t, n = returns.shape
    if t <= lookback + 8:
        return None
    earned: list[float] = []
    for i in range(lookback, t):
        score = sign * returns[i - lookback : i].sum(axis=0)
        shifted = score - float(score.min())
        total = float(shifted.sum())
        weights = np.ones(n) / n if total <= 0 else shifted / total
        earned.append(float(returns[i] @ weights))
    return np.asarray(earned, dtype=float)


def _subset(
    panel: tuple[np.ndarray, list[str]], names: Any
) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    returns, known = panel
    if not isinstance(names, (list, tuple)) or not names:
        return returns, known
    wanted = {str(name) for name in names}
    keep = [i for i, name in enumerate(known) if name in wanted]
    if len(keep) < 2:
        return None, None
    return returns[:, keep], [known[i] for i in keep]


def _panel(data: Any) -> tuple[np.ndarray, list[str]] | None:
    panel = as_returns(data)
    if len(panel) < 2:
        return None
    aligned, names, _skipped = align_panel(panel, need=_MIN_OBS)
    if len(names) < 2 or aligned.shape[0] < _MIN_OBS:
        return None
    return np.asarray(aligned, dtype=float), names


def _lookback(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 20
    return min(60, max(5, number))


def _sharpe(earned: np.ndarray) -> float | None:
    clean = np.asarray(earned, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 8:
        return None
    annual, _source, _as_of = cash_rate(None)
    scale = float(clean.std(ddof=1))
    if scale <= 0 or not math.isfinite(scale):
        return None
    return float((clean.mean() - annual / 252.0) / scale)
