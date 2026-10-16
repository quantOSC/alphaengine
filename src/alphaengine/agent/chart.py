"""A local research note. Matplotlib stays inside this module.

`import alphaengine` does not import it. The series used for the figures never
leave the machine: this writer only reads what the trace already holds.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .._version import __version__

_COLORS = (
    "#F8766D",
    "#00BFC4",
    "#7CAE00",
    "#C77CFF",
    "#00B0F6",
    "#FF61CC",
    "#E58700",
    "#00BA38",
)


def write_report(trace: Any, path: Path) -> None:
    """Letter-size pages: the stack, then the charts the local series support."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    panel = _panel(getattr(trace, "panel", None))
    matrix = _matrix(getattr(trace, "matrix", None))
    scalars = _scalars(trace)
    rc = {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "#E5E5E5",
        "axes.edgecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "white",
        "grid.linewidth": 1.2,
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "legend.frameon": False,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with plt.rc_context(rc), PdfPages(path) as pdf:
        info = pdf.infodict()
        info["Title"] = str(getattr(trace, "question", "") or "alphaengine")[:120]
        info["Creator"] = f"alphaengine {__version__}"
        _front(plt, pdf, trace, stamp, scalars)
        if panel:
            _paths(plt, pdf, trace, panel, stamp)
            _wealth(plt, pdf, trace, panel, stamp)
        if matrix is not None:
            _dependence(plt, pdf, trace, matrix, stamp)


def _panel(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, values in raw.items():
        arr = _as_float(values)
        if arr.size >= 8:
            out[str(key)] = arr[-2500:]
        if len(out) == 8:
            break
    return out


def _matrix(raw: Any) -> tuple[list[str], Any] | None:
    if not isinstance(raw, dict):
        return None
    names = [str(name) for name in (raw.get("names") or [])][:24]
    values = raw.get("values")
    if len(names) < 2 or not isinstance(values, list):
        return None
    rows = []
    for row in values[: len(names)]:
        if not isinstance(row, list):
            return None
        rows.append([_num(cell) for cell in row[: len(names)]])
    if len(rows) != len(names):
        return None
    grid = np.asarray(rows, dtype=float)
    if grid.shape != (len(names), len(names)) or not np.isfinite(grid).any():
        return None
    return names, grid


def _scalars(trace: Any) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    for call in getattr(trace, "calls", []) or []:
        tool = str(getattr(call, "tool", "") or "")
        figures = getattr(call, "figures", {}) or {}
        if not isinstance(figures, dict):
            continue
        for key, value in figures.items():
            number = _num(value)
            if number is None or not math.isfinite(number):
                continue
            found.append((f"{tool}  {key}", float(number)))
            if len(found) == 16:
                return found
    return found


def _as_float(values: Any) -> Any:
    if isinstance(values, dict):
        seq = list(values.values())
    elif isinstance(values, (list, tuple)):
        seq = list(values)
    else:
        return np.asarray([], dtype=float)
    nums = [float(v) for v in seq if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return np.asarray(nums, dtype=float)


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _returns(level: Any) -> Any:
    arr = level[np.isfinite(level)]
    if arr.size < 8:
        return np.asarray([], dtype=float)
    if float(np.min(arr)) > 0.0 and float(np.median(np.abs(arr))) > 1.5:
        prev = arr[:-1]
        nxt = arr[1:]
        ok = prev != 0
        return (nxt[ok] - prev[ok]) / prev[ok]
    return arr


def _growth(level: Any) -> Any:
    rets = _returns(level)
    if rets.size < 8:
        return np.asarray([], dtype=float)
    return np.cumprod(1.0 + rets)


def _book(panel: dict[str, Any]) -> Any:
    paths = [_returns(series) for series in panel.values()]
    paths = [path for path in paths if path.size >= 8]
    if not paths:
        return np.asarray([], dtype=float)
    width = min(path.size for path in paths)
    stacked = np.vstack([path[-width:] for path in paths])
    return np.nanmean(stacked, axis=0)


def _front(plt: Any, pdf: Any, trace: Any, stamp: str, scalars: list[tuple[str, float]]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("alphaengine", x=0.08, ha="left", fontsize=11, color="#444444", fontweight="medium")
    question = str(getattr(trace, "question", "") or "").strip() or "untitled"
    fig.text(0.08, 0.90, _wrap(question, 78), fontsize=14, color="#111111", va="top")
    answer = str(getattr(trace, "answer", "") or "").strip()
    if answer:
        fig.text(0.08, 0.78, _wrap(answer, 92), fontsize=10, color="#222222", va="top")
    calls = list(getattr(trace, "calls", []) or [])
    rows = [
        [f"{i:02d}", str(call.tool), _clip(str(call.note or call.error or ""), 64)]
        for i, call in enumerate(calls[:12], start=1)
    ]
    if not rows:
        rows = [["-", "none", "no tools ran"]]
    ax = fig.add_axes((0.08, 0.36, 0.84, 0.28))
    ax.set_title("Stack", loc="left", color="#333333")
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["#", "tool", "note"], loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    if scalars:
        lines = [f"{name:<44} {value: .6g}" for name, value in scalars[:12]]
        fig.text(0.08, 0.32, "\n".join(lines), fontsize=8, family="monospace", color="#222222", va="top")
    _foot(fig, stamp, "The figures below are computed on this machine.")
    pdf.savefig(fig)
    plt.close(fig)


def _paths(plt: Any, pdf: Any, trace: Any, panel: dict[str, Any], stamp: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.1, right=0.72, top=0.88, bottom=0.10)
    ax.set_title("Growth of 1, by name", loc="left")
    drawn = 0
    for index, (name, series) in enumerate(panel.items()):
        path = _growth(series)
        if path.size < 8:
            continue
        ax.plot(path, color=_COLORS[index % len(_COLORS)], lw=1.4, label=name)
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return
    ax.set_xlabel("observation")
    ax.set_ylabel("growth of 1")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.text(
        0.1,
        0.045,
        "Price levels become simple returns first. A return series is left as it is.",
        fontsize=8,
        color="#444444",
    )
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _wealth(plt: Any, pdf: Any, trace: Any, panel: dict[str, Any], stamp: str) -> None:
    rets = _book(panel)
    if rets.size < 8:
        return
    wealth = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / np.maximum(peak, 1e-18) - 1.0
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 11), sharex=False)
    fig.subplots_adjust(left=0.1, right=0.94, top=0.90, bottom=0.10, hspace=0.38)
    axes[0].set_title("Equal-weight book", loc="left")
    axes[0].plot(wealth, color="#00BFC4", lw=1.5)
    axes[0].set_ylabel("growth of 1")
    axes[1].fill_between(np.arange(drawdown.size), drawdown, 0.0, color="#F8766D", alpha=0.85)
    axes[1].set_ylabel("drawdown")
    axes[2].hist(rets, bins=min(40, max(8, rets.size // 5)), color="#7CAE00", edgecolor="white")
    sd = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    if sd > 0:
        xs = np.linspace(float(np.min(rets)), float(np.max(rets)), 200)
        mu = float(np.mean(rets))
        ys = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))
        width = (float(np.max(rets)) - float(np.min(rets))) / max(8, min(40, rets.size // 5))
        axes[2].plot(xs, ys * rets.size * width, color="#222222", lw=1.0)
    axes[2].set_xlabel("period return")
    axes[2].set_ylabel("count")
    fig.text(
        0.1,
        0.045,
        "Equal weight, no costs, no borrow. The curve is a gaussian with the sample mean and variance.",
        fontsize=8,
        color="#444444",
    )
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _dependence(
    plt: Any,
    pdf: Any,
    trace: Any,
    matrix: tuple[list[str], Any],
    stamp: str,
) -> None:
    names, grid = matrix
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.16, right=0.88, top=0.88, bottom=0.12)
    finite = grid[np.isfinite(grid)]
    lo = float(np.min(finite)) if finite.size else -1.0
    hi = float(np.max(finite)) if finite.size else 1.0
    diag = np.diag(grid)
    corr = lo >= -1.05 and hi <= 1.05 and diag.size > 0 and bool(np.nanmax(np.abs(diag - 1.0)) < 0.05)
    if corr or lo < 0 < hi:
        lo, hi = (-1.0, 1.0) if corr else (-max(abs(lo), abs(hi)), max(abs(lo), abs(hi)))
    image = ax.imshow(grid, cmap="coolwarm", vmin=lo, vmax=hi, aspect="equal")
    ax.set_title("Dependence", loc="left")
    ax.grid(False)
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if len(names) <= 12:
        for i in range(len(names)):
            for j in range(len(names)):
                val = grid[i, j]
                if math.isfinite(float(val)):
                    ink = "white" if abs(float(val)) >= 0.55 else "#111111"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=ink)
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _foot(fig: Any, stamp: str, note: str) -> None:
    fig.text(0.08, 0.015, f"alphaengine {__version__}    {stamp}", fontsize=8, color="#666666")
    if note:
        fig.text(0.08, 0.955, _clip(note, 90), fontsize=8, color="#666666")


def _question(trace: Any) -> str:
    return str(getattr(trace, "question", "") or "")


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def _wrap(text: str, width: int) -> str:
    words = text.split()
    if not words:
        return ""
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        if len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}"
    lines.append(cur)
    return "\n".join(lines[:8])
