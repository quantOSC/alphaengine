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
    """Letter-size report: the reading, the stack, and the books built from it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from .brief import study_for

    study = study_for(trace, deep=True)
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
        info["Title"] = str(getattr(trace, "question", "") or "report")[:120]
        info["Creator"] = f"alphaengine {__version__}"
        _document(plt, pdf, trace, study, stamp, path.stem)


def _document(plt: Any, pdf: Any, trace: Any, study: dict[str, Any], stamp: str, stem: str) -> None:
    _prose(plt, pdf, trace, study, stamp, stem)
    _stack(plt, pdf, trace, stamp)
    if study.get("paths"):
        _growth(plt, pdf, trace, study["paths"], stamp)
    if study.get("correlation"):
        _heat(plt, pdf, trace, study["correlation"], stamp)
    books = list(study.get("books") or [])
    if books:
        primary = next((book for book in books if book["method"] == "hrp"), books[0])
        _weights(plt, pdf, trace, books, stamp)
        _wealth(plt, pdf, trace, books, stamp)
        _risk(plt, pdf, trace, primary, stamp)
        _rolling(plt, pdf, trace, books, stamp)
    if study.get("scenarios"):
        _scenes(plt, pdf, trace, study["scenarios"], stamp)
    if study.get("monte"):
        _monte(plt, pdf, trace, study["monte"], stamp)
    if study.get("bakeoff"):
        _bakeoff(plt, pdf, trace, study["bakeoff"], stamp)
    for board in study.get("boards") or []:
        if board.get("kind") == "cointegration" and board.get("rows"):
            _pairs(plt, pdf, trace, board, stamp)


def _prose(plt: Any, pdf: Any, trace: Any, study: dict[str, Any], stamp: str, stem: str) -> None:
    blocks: list[tuple[str, str]] = [("Question", str(getattr(trace, "question", "") or "report"))]
    answer = str(getattr(trace, "answer", "") or "").strip()
    findings = str(getattr(trace, "findings", "") or study.get("findings") or "").strip()
    if answer:
        blocks.append(("Reading", answer))
    if findings and findings != answer:
        blocks.append(("Findings", findings))
    fig: Any = None
    y = 0.0

    def flush() -> None:
        nonlocal fig
        if fig is None:
            return
        pdf.savefig(fig)
        plt.close(fig)
        fig = None

    def page() -> None:
        nonlocal fig, y
        flush()
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.08, 0.93, f"Report    {stem}", fontsize=11, color="#333333")
        _foot(fig, stamp, "")
        y = 0.88

    page()
    for title, body in blocks:
        wrapped = _wrap(body, 92).splitlines() or [""]
        need = 0.035 + 0.028 * len(wrapped)
        if y - need < 0.08:
            page()
        fig.text(0.08, y, title, fontsize=11, color="#111111")
        y -= 0.03
        for line in wrapped:
            if y < 0.08:
                page()
            fig.text(0.08, y, line, fontsize=9, color="#222222")
            y -= 0.024
        y -= 0.02
    flush()


def _stack(plt: Any, pdf: Any, trace: Any, stamp: str) -> None:
    calls = list(getattr(trace, "calls", []) or [])
    if not calls:
        return
    rows: list[list[str]] = []
    for index, call in enumerate(calls, start=1):
        figures = getattr(call, "figures", {}) or {}
        detail = getattr(call, "note", "") or getattr(call, "error", "") or ""
        if isinstance(figures, dict):
            bits = [
                f"{key} {value}"
                for key, value in figures.items()
                if isinstance(value, (int, float, str)) and not isinstance(value, bool)
            ]
            if bits:
                detail = "   ".join(bits)
        rows.append([f"{index:02d}", str(getattr(call, "tool", "")), _clip(str(detail), 88)])
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes((0.06, 0.08, 0.88, 0.82))
    ax.set_title("Stack", loc="left")
    ax.axis("off")
    table = ax.table(cellText=rows[:18], colLabels=["#", "tool", "figures"], loc="upper left", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.45)
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _growth(plt: Any, pdf: Any, trace: Any, paths: dict[str, list[float]], stamp: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.1, right=0.74, top=0.88, bottom=0.12)
    ax.set_title("Growth of 1, by name", loc="left")
    for index, (name, series) in enumerate(paths.items()):
        ax.plot(series, color=_COLORS[index % len(_COLORS)], lw=1.3, label=name)
    ax.set_xlabel("observation")
    ax.set_ylabel("growth of 1")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.text(
        0.1,
        0.05,
        "Simple returns from the loaded levels, then the cumulative product.",
        fontsize=8,
        color="#444444",
    )
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _heat(plt: Any, pdf: Any, trace: Any, matrix: dict[str, Any], stamp: str) -> None:
    names = list(matrix.get("names") or [])
    grid = np.asarray(
        [
            [_float(cell) for cell in (row or [])[: len(names)]]
            for row in (matrix.get("values") or [])[: len(names)]
        ],
        dtype=float,
    )
    if grid.shape != (len(names), len(names)) or len(names) < 2:
        return
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.16, right=0.86, top=0.88, bottom=0.14)
    image = ax.imshow(grid, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_title("Correlation", loc="left")
    ax.grid(False)
    ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(names)), names)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if len(names) <= 10:
        for i in range(len(names)):
            for j in range(len(names)):
                val = grid[i, j]
                if math.isfinite(val):
                    ink = "white" if abs(val) >= 0.55 else "#111111"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=ink)
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _weights(plt: Any, pdf: Any, trace: Any, books: list[dict[str, Any]], stamp: str) -> None:
    names: list[str] = []
    for book in books:
        for name, _weight in book.get("weights") or []:
            if name not in names:
                names.append(name)
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.18, right=0.94, top=0.88, bottom=0.12)
    offset = np.arange(len(names))
    width = 0.8 / max(len(books), 1)
    for index, book in enumerate(books):
        held = {name: weight for name, weight in book.get("weights") or []}
        values = [held.get(name, 0.0) for name in names]
        ax.barh(
            offset + index * width,
            values,
            height=width * 0.9,
            color=_COLORS[index % len(_COLORS)],
            label=book["method"],
        )
    ax.set_yticks(offset + width * (len(books) - 1) / 2, names)
    ax.set_title("Weights", loc="left")
    ax.set_xlabel("weight")
    ax.legend(frameon=False)
    fig.text(
        0.18,
        0.05,
        "Equal weight, hierarchical risk parity, and risk parity. Each column sums to 1.",
        fontsize=8,
        color="#444444",
    )
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _wealth(plt: Any, pdf: Any, trace: Any, books: list[dict[str, Any]], stamp: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.94, top=0.88, bottom=0.12, hspace=0.28)
    axes[0].set_title("Constructed books", loc="left")
    for index, book in enumerate(books):
        color = _COLORS[index % len(_COLORS)]
        axes[0].plot(book.get("wealth") or [], color=color, lw=1.4, label=book["method"])
        axes[1].plot(book.get("drawdown") or [], color=color, lw=1.2, label=book["method"])
    axes[0].set_ylabel("growth of 1")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("drawdown")
    axes[1].set_xlabel("observation")
    fig.text(
        0.1,
        0.05,
        "No costs and no borrow. The same aligned history is used for every book.",
        fontsize=8,
        color="#444444",
    )
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _risk(plt: Any, pdf: Any, trace: Any, book: dict[str, Any], stamp: str) -> None:
    rets = np.asarray(book.get("returns") or [], dtype=float)
    if rets.size < 8:
        return
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.1, right=0.94, top=0.88, bottom=0.12)
    ax.hist(rets, bins=min(40, max(8, rets.size // 5)), color="#7CAE00", edgecolor="white")
    var = book.get("var") or {}
    if isinstance(var.get("var_pct"), (int, float)):
        ax.axvline(-float(var["var_pct"]) / 100.0, color="#F8766D", lw=1.2, label="historical VaR")
    if isinstance(var.get("cvar_pct"), (int, float)):
        ax.axvline(-float(var["cvar_pct"]) / 100.0, color="#C77CFF", lw=1.2, label="CVaR")
    ax.set_title(f"Return distribution, {book['method']}", loc="left")
    ax.set_xlabel("period return")
    ax.set_ylabel("count")
    ax.legend(frameon=False)
    stats = book.get("stats") or {}
    caption = (
        f"Sharpe {stats.get('sharpe_annualized')}    "
        f"drawdown {stats.get('max_drawdown_pct')}%    "
        f"vol {stats.get('volatility_annualized_pct')}%"
    )
    fig.text(0.1, 0.05, caption, fontsize=8, color="#444444")
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _rolling(plt: Any, pdf: Any, trace: Any, books: list[dict[str, Any]], stamp: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.1, right=0.94, top=0.88, bottom=0.12)
    drawn = 0
    window = 63
    for index, book in enumerate(books):
        rets = np.asarray(book.get("returns") or [], dtype=float)
        if rets.size <= window:
            continue
        vol = np.array(
            [float(np.std(rets[i - window : i], ddof=1) * math.sqrt(252)) for i in range(window, rets.size)],
            dtype=float,
        )
        ax.plot(vol, color=_COLORS[index % len(_COLORS)], lw=1.3, label=book["method"])
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        return
    ax.set_title("Rolling 63-observation volatility, annualized", loc="left")
    ax.set_ylabel("volatility")
    ax.set_xlabel("observation")
    ax.legend(frameon=False, fontsize=8)
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _scenes(plt: Any, pdf: Any, trace: Any, rows: list[dict[str, Any]], stamp: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.28, right=0.94, top=0.88, bottom=0.28)
    drawn = [row for row in rows if isinstance(row.get("max_drawdown_pct"), (int, float))]
    labels = [str(row["name"]) for row in drawn][::-1]
    values = [float(row["max_drawdown_pct"]) for row in drawn][::-1]
    if labels:
        ax.barh(np.arange(len(values)), values, color="#F8766D")
        ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Scenario drawdowns", loc="left")
    ax.set_xlabel("max drawdown, percent")
    lines = []
    for row in rows:
        lines.append(
            f"{row['name']}: return {_shown(row.get('total_return_pct'))}%   "
            f"drawdown {_shown(row.get('max_drawdown_pct'))}%   "
            f"Sharpe {_shown(row.get('sharpe_annualized'))}"
        )
    fig.text(0.08, 0.045, "\n".join(lines[:6]), fontsize=8, color="#333333", va="bottom")
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _monte(plt: Any, pdf: Any, trace: Any, monte: dict[str, Any], stamp: str) -> None:
    edges = [float(x) for x in monte.get("edges") or []]
    counts = [float(x) for x in monte.get("counts") or []]
    if len(edges) < 2 or not counts:
        return
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    width = edges[1] - edges[0]
    fig, ax = plt.subplots(figsize=(8.5, 11))
    fig.subplots_adjust(left=0.1, right=0.94, top=0.88, bottom=0.12)
    ax.bar(centers[: len(counts)], counts, width=width * 0.9, color="#00BFC4", edgecolor="white")
    for mark, label in (
        (monte.get("sharpe_q10"), "q10"),
        (monte.get("sharpe_q50"), "q50"),
        (monte.get("sharpe_q90"), "q90"),
    ):
        if isinstance(mark, (int, float)):
            ax.axvline(float(mark), color="#333333", lw=0.8)
            ax.text(float(mark), ax.get_ylim()[1] * 0.9, label, fontsize=8, rotation=90, va="top")
    ax.set_title(f"Simulated Sharpe, {monte.get('dgp')} , {monte.get('n_paths')} paths", loc="left")
    ax.set_xlabel("Sharpe of a simulated path")
    ax.set_ylabel("count")
    fig.text(
        0.1,
        0.05,
        f"Fraction of paths with a positive Sharpe: {monte.get('frac_positive')}. Seed 7.",
        fontsize=8,
        color="#444444",
    )
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _bakeoff(plt: Any, pdf: Any, trace: Any, bakeoff: dict[str, Any], stamp: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.90, "Bakeoff", fontsize=14, color="#111111")
    summary = (
        f"Survivor {bakeoff.get('survivor')}\n"
        f"Sharpe {bakeoff.get('survivor_sharpe')}    after attacks {bakeoff.get('attacked_sharpe')}\n"
        f"Verdict {bakeoff.get('verdict')}    deflated Sharpe {bakeoff.get('deflated_sharpe')}\n"
        f"Trials {bakeoff.get('n_trials')}    attacks {', '.join(bakeoff.get('attacks') or []) or 'none'}"
    )
    fig.text(0.08, 0.78, summary, fontsize=10, color="#222222", va="top")
    trials = list(bakeoff.get("trials") or [])
    if trials:
        fig.text(
            0.08, 0.62, "\n".join(trials[:12]), fontsize=8, family="monospace", color="#222222", va="top"
        )
    curves = list(bakeoff.get("curves") or [])
    if curves:
        ax = fig.add_axes((0.1, 0.08, 0.8, 0.28))
        ax.set_title("Growth of 1, proposals", loc="left", fontsize=10)
        for index, curve in enumerate(curves):
            ax.plot(
                curve.get("growth") or [],
                color=_COLORS[index % len(_COLORS)],
                lw=1.2,
                label=curve.get("label"),
            )
        ax.legend(frameon=False, fontsize=7)
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _pairs(plt: Any, pdf: Any, trace: Any, board: dict[str, Any], stamp: str) -> None:
    rows = []
    for row in list(board.get("rows") or [])[:16]:
        if not isinstance(row, dict):
            continue
        rows.append(
            [
                str(row.get("a") or ""),
                str(row.get("b") or ""),
                "" if row.get("p") is None else str(row.get("p")),
                "" if row.get("half_life") is None else str(row.get("half_life")),
                "yes" if row.get("cointegrated") else "",
            ]
        )
    if not rows:
        return
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes((0.08, 0.2, 0.84, 0.68))
    ax.set_title(
        f"Cointegration    {board.get('n_cointegrated')} of {board.get('n_pairs')} pairs",
        loc="left",
    )
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["a", "b", "p", "half-life", "survives"],
        loc="upper left",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    _foot(fig, stamp, _question(trace))
    pdf.savefig(fig)
    plt.close(fig)


def _shown(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


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
