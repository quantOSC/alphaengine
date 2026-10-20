"""A local research note. Matplotlib stays inside this module.

`import alphaengine` does not import it. The series used for the figures never
leave the machine: this writer only reads what the trace already holds.
"""

from __future__ import annotations

import math
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from .._version import __version__

_COLORS = (
    "#D95F02",
    "#0072B2",
    "#009E73",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
    "#1B9E77",
    "#56B4E9",
)


def write_report(trace: Any, path: Path) -> None:
    """Letter-size report: the reading, the stack, and the books built from it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from .brief import study_for

    cached = getattr(trace, "study", None)
    study = cached if isinstance(cached, dict) and cached.get("monte") else study_for(trace, deep=True)
    rc: dict[str, Any] = {
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "#F8FAFC",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.8,
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#E2E8F0",
        "legend.framealpha": 0.92,
        "text.color": "#0F172A",
        "axes.labelcolor": "#1E293B",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with cast(Any, plt).rc_context(rc), PdfPages(path) as pdf:
        info = pdf.infodict()
        info["Title"] = str(getattr(trace, "question", "") or "report")[:120]
        info["Creator"] = f"alphaengine {__version__}"
        _document(plt, pdf, trace, study, stamp, path.stem)


def _document(plt: Any, pdf: Any, trace: Any, study: dict[str, Any], stamp: str, stem: str) -> None:
    _prose(plt, pdf, trace, study, stamp, stem)
    _stack(plt, pdf, trace, stamp, stem)
    books = list(study.get("books") or [])
    primary = study.get("primary")
    if not isinstance(primary, dict) and books:
        primary = next((book for book in books if book["method"] == "hrp"), books[0])
    assets = list(study.get("assets") or [])
    if study.get("paths"):
        _growth(plt, pdf, trace, study["paths"], primary, assets, stamp, stem)
    if study.get("correlation"):
        _heat(plt, pdf, trace, study["correlation"], stamp, stem)
    if books:
        _weights(plt, pdf, trace, books, assets, stamp, stem)
        _wealth(plt, pdf, trace, books, stamp, stem)
        if isinstance(primary, dict):
            _risk(plt, pdf, trace, primary, assets, stamp, stem)
    if study.get("scenarios"):
        _scenes(plt, pdf, trace, study["scenarios"], stamp, stem)
    if study.get("monte"):
        _monte(plt, pdf, trace, study["monte"], stamp, stem)
    if study.get("bakeoff"):
        _bakeoff(plt, pdf, trace, study["bakeoff"], stamp, stem)
    for board in study.get("boards") or []:
        if board.get("kind") == "cointegration" and board.get("rows"):
            _pairs(plt, pdf, trace, board, stamp, stem)


def _prose(plt: Any, pdf: Any, trace: Any, study: dict[str, Any], stamp: str, stem: str) -> None:
    question = str(getattr(trace, "question", "") or "report").strip()
    reading = str(getattr(trace, "reading", "") or "").strip()
    answer = str(getattr(trace, "answer", "") or "").strip()
    findings = str(getattr(trace, "findings", "") or study.get("findings") or "").strip()

    blocks: list[tuple[str, str]] = [("Research Mandate & Question", question)]
    if reading:
        blocks.append(("Agentic Portfolio & Risk Reading", reading))
    elif answer and answer != findings:
        blocks.append(("Agentic Portfolio & Risk Reading", answer))
    if findings and findings != reading:
        blocks.append(("Constructed Book, Scenario & Simulation Synthesis", findings))

    books = list(study.get("books") or [])
    fig: Any = None
    y = 0.0

    def flush() -> None:
        nonlocal fig
        if fig is None:
            return
        pdf.savefig(fig)
        plt.close(fig)
        fig = None

    def new_page(first: bool = False) -> None:
        nonlocal fig, y
        flush()
        fig = plt.figure(figsize=(8.5, 11))
        sub = "Executive Summary & Portfolio Brief" if first else "Portfolio Brief (continued)"
        _chrome(fig, sub, stamp, stem, "")
        y = 0.895
        if first and books:
            y = _scorecard_strip(fig, books, y)

    new_page(first=True)
    for title, body in blocks:
        paragraphs = _paragraphs(body, width=88)
        total_lines = sum(len(p) for p in paragraphs) + max(0, len(paragraphs) - 1)
        if y - (0.04 + min(total_lines, 4) * 0.021) < 0.085:
            new_page(first=False)
        fig.text(0.075, y, title, fontsize=10.5, fontweight="bold", color="#0F172A")
        y -= 0.025
        for p_idx, lines in enumerate(paragraphs):
            for line in lines:
                if y < 0.075:
                    new_page(first=False)
                fig.text(0.075, y, line, fontsize=8.8, color="#1E293B")
                y -= 0.0195
            if p_idx + 1 < len(paragraphs):
                y -= 0.008
        y -= 0.018
    flush()


def _scorecard_strip(fig: Any, books: list[dict[str, Any]], top_y: float) -> float:
    """Compact comparison table of constructed books at the top of Page 1."""
    rows: list[list[str]] = []
    for book in books:
        st = book.get("stats") or {}
        var = book.get("var") or {}
        rows.append(
            [
                str(book.get("method") or ""),
                _fmt_pct(st.get("total_return_pct")),
                _fmt_pct(st.get("annualized_return_pct")),
                _fmt_pct(st.get("volatility_annualized_pct")),
                _shown(st.get("sharpe_annualized")),
                _shown(st.get("sortino_ratio")),
                _fmt_pct(st.get("max_drawdown_pct")),
                _fmt_pct(var.get("var_pct")),
                _fmt_pct(var.get("cvar_pct")),
                _shown(book.get("effective_n")),
            ]
        )
    if not rows:
        return top_y
    height = 0.032 * (len(rows) + 1)
    ax = fig.add_axes((0.075, top_y - height - 0.01, 0.85, height))
    ax.axis("off")
    headers = [
        "Book",
        "Tot Ret",
        "Ann Ret",
        "Ann Vol",
        "Sharpe",
        "Sortino",
        "Max DD",
        "95% VaR",
        "95% CVaR",
        "Eff N",
    ]
    widths = [0.14, 0.095, 0.095, 0.095, 0.09, 0.09, 0.095, 0.10, 0.11, 0.09]
    _styled_table(ax, headers, rows, widths, fontsize=7.6)
    return top_y - height - 0.035


def _stack(plt: Any, pdf: Any, trace: Any, stamp: str, stem: str) -> None:
    calls = list(getattr(trace, "calls", []) or [])
    if not calls:
        return
    rows: list[list[str]] = []
    for index, call in enumerate(calls[:18], start=1):
        figures = getattr(call, "figures", {}) or {}
        detail = str(getattr(call, "note", "") or getattr(call, "error", "") or "")
        if isinstance(figures, dict):
            bits = [
                f"{key}: {value}"
                for key, value in figures.items()
                if isinstance(value, (int, float, str)) and not isinstance(value, bool)
            ]
            if bits:
                detail = "   ".join(bits)
        wrapped_detail = "\n".join(textwrap.wrap(detail, width=72, break_long_words=False)) or "-"
        rows.append([f"{index:02d}", str(getattr(call, "tool", "")), wrapped_detail])
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(fig, "Execution Stack & Deterministic Tool Ledger", stamp, stem, _question(trace))
    ax = fig.add_axes((0.075, 0.08, 0.85, 0.81))
    ax.axis("off")
    _styled_table(
        ax,
        ["#", "Tool", "Computed Figures & Diagnostics"],
        rows,
        [0.06, 0.16, 0.78],
        fontsize=7.8,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _growth(
    plt: Any,
    pdf: Any,
    trace: Any,
    paths: dict[str, list[float]],
    primary: dict[str, Any] | None,
    assets: list[dict[str, Any]],
    stamp: str,
    stem: str,
) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(fig, "Constituent Performance & Cross-Sectional Risk-Return Map", stamp, stem, _question(trace))
    ax1 = fig.add_axes((0.11, 0.54, 0.80, 0.34))
    ax1.set_title("Cumulative Wealth (Growth of 1) — Key Sleeves vs. Constructed Book", loc="left")
    for index, (name, series) in enumerate(list(paths.items())[:10]):
        ax1.plot(series, color=_COLORS[index % len(_COLORS)], lw=1.2, alpha=0.88, label=name)
    if isinstance(primary, dict) and primary.get("wealth"):
        ax1.plot(
            primary["wealth"],
            color="#0F172A",
            lw=2.2,
            label=f"book ({primary.get('method')})",
        )
    ax1.axhline(1.0, color="#64748B", lw=0.7, ls="--")
    ax1.set_xlabel("trading observation")
    ax1.set_ylabel("growth of 1")
    ax1.legend(loc="upper left", ncol=4, fontsize=7.2)

    if assets:
        ax2 = fig.add_axes((0.11, 0.10, 0.80, 0.35))
        vols = [float(row["volatility_annualized_pct"]) for row in assets]
        rets = [float(row["annualized_return_pct"]) for row in assets]
        sharpes = [float(row["sharpe_annualized"]) for row in assets]
        weights = [max(float(row["hrp_weight"]), 0.005) * 2200.0 for row in assets]
        sc = ax2.scatter(
            vols,
            rets,
            s=weights,
            c=sharpes,
            cmap="viridis",
            alpha=0.82,
            edgecolors="#0F172A",
            linewidths=0.6,
        )
        fig.colorbar(sc, ax=ax2, pad=0.02, label="Annualized Sharpe")
        for row in assets[:12]:
            ax2.annotate(
                str(row["name"]),
                (float(row["volatility_annualized_pct"]), float(row["annualized_return_pct"])),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.2,
                fontweight="bold" if row.get("focused") else "normal",
                color="#B91C1C" if row.get("focused") else "#0F172A",
            )
        ax2.axhline(0.0, color="#64748B", lw=0.7, ls="--")
        ax2.set_title(
            f"Cross-Sectional Risk vs. Return ({len(assets)} Assets, Sized by HRP Weight)",
            loc="left",
        )
        ax2.set_xlabel("annualized volatility (%)")
        ax2.set_ylabel("annualized return (%)")
    pdf.savefig(fig)
    plt.close(fig)


def _heat(plt: Any, pdf: Any, trace: Any, matrix: dict[str, Any], stamp: str, stem: str) -> None:
    names = [str(n) for n in (matrix.get("names") or [])][:14]
    grid = np.asarray(
        [
            [_float(cell) for cell in (row or [])[: len(names)]]
            for row in (matrix.get("values") or [])[: len(names)]
        ],
        dtype=float,
    )
    if grid.shape != (len(names), len(names)) or len(names) < 2:
        return
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(fig, "Cross-Asset Correlation Structure & Pairwise Extremes", stamp, stem, _question(trace))
    ax = fig.add_axes((0.14, 0.37, 0.72, 0.49))
    image = ax.imshow(grid, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="equal")
    ax.set_title(f"Return Correlation Matrix ({len(names)} Selected Sleeves)", loc="left")
    ax.grid(False)
    ax.set_xticks(np.arange(len(names)), names, rotation=40, ha="right", fontsize=7.8)
    ax.set_yticks(np.arange(len(names)), names, fontsize=7.8)
    fig.colorbar(image, ax=ax, fraction=0.042, pad=0.03)
    if len(names) <= 14:
        font_sz = 6.5 if len(names) > 10 else 7.5
        for i in range(len(names)):
            for j in range(len(names)):
                val = grid[i, j]
                if math.isfinite(val):
                    ink = "white" if abs(val) >= 0.55 else "#0F172A"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=font_sz, color=ink)

    pairs: list[tuple[float, str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if math.isfinite(grid[i, j]):
                pairs.append((float(grid[i, j]), names[i], names[j]))
    pairs.sort(reverse=True)
    if pairs:
        top_p = pairs[:4]
        low_p = list(reversed(pairs[-4:]))
        pair_rows: list[list[str]] = []
        for idx in range(max(len(top_p), len(low_p))):
            hi = top_p[idx] if idx < len(top_p) else None
            lo = low_p[idx] if idx < len(low_p) else None
            pair_rows.append(
                [
                    f"{hi[1]} / {hi[2]}" if hi else "-",
                    f"{hi[0]:+.4f}" if hi else "-",
                    f"{lo[1]} / {lo[2]}" if lo else "-",
                    f"{lo[0]:+.4f}" if lo else "-",
                ]
            )
        ax_tbl = fig.add_axes((0.08, 0.08, 0.84, 0.21))
        ax_tbl.axis("off")
        ax_tbl.set_title("Highest Co-Movement vs. Strongest Diversifying Pairs", loc="left", fontsize=9.5)
        _styled_table(
            ax_tbl,
            ["Most Correlated Pair", "Correlation", "Lowest / Diversifying Pair", "Correlation"],
            pair_rows,
            [0.32, 0.18, 0.32, 0.18],
            fontsize=8.0,
        )
    pdf.savefig(fig)
    plt.close(fig)


def _weights(
    plt: Any,
    pdf: Any,
    trace: Any,
    books: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    stamp: str,
    stem: str,
) -> None:
    if assets:
        names = [str(row["name"]) for row in assets[:14]]
    else:
        names = []
        for book in books:
            for name, _weight in book.get("weights") or []:
                if name not in names:
                    names.append(name)
                if len(names) == 14:
                    break
    if not names:
        return
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(
        fig,
        "Portfolio Construction & Multi-Method Weight Allocation",
        stamp,
        stem,
        _question(trace),
    )
    ax = fig.add_axes((0.14, 0.46, 0.79, 0.42))
    offset = np.arange(len(names))
    width = 0.82 / max(len(books), 1)
    for index, book in enumerate(books):
        held = dict(book.get("weight_map") or book.get("weights") or {})
        values = [float(held.get(name, 0.0)) for name in names][::-1]
        ax.barh(
            offset + index * width,
            values,
            height=width * 0.9,
            color=_COLORS[index % len(_COLORS)],
            label=book["method"],
        )
    ax.set_yticks(offset + width * (len(books) - 1) / 2, names[::-1], fontsize=8.0)
    ax.set_title("Allocation Weights Across Construction Methods (Top Sleeves & Focused Tickers)", loc="left")
    ax.set_xlabel("portfolio weight (fraction of book)")
    ax.legend(loc="lower right", ncol=2, fontsize=7.8)

    if assets:
        rows: list[list[str]] = []
        for row in assets[:12]:
            tag = f"{row['name']} *" if row.get("focused") else str(row["name"])
            rows.append(
                [
                    tag,
                    f"{float(row['hrp_weight']):.4f}",
                    f"{float(row['rp_weight']):.4f}",
                    f"{float(row['var_weight']):.4f}",
                    _fmt_pct(row.get("annualized_return_pct")),
                    _fmt_pct(row.get("volatility_annualized_pct")),
                    _shown(row.get("sharpe_annualized")),
                    _fmt_pct(row.get("max_drawdown_pct")),
                    _fmt_pct(row.get("var_95_pct")),
                ]
            )
        ax_tbl = fig.add_axes((0.075, 0.07, 0.85, 0.32))
        ax_tbl.axis("off")
        ax_tbl.set_title(
            "Top Holdings & Single-Name Risk Attribution (* = Named in Query)",
            loc="left",
            fontsize=9.5,
        )
        _styled_table(
            ax_tbl,
            [
                "Ticker",
                "HRP Wt",
                "RiskPar Wt",
                "VaR-Sort Wt",
                "Ann Ret",
                "Ann Vol",
                "Sharpe",
                "Max DD",
                "95% VaR",
            ],
            rows,
            [0.12, 0.11, 0.11, 0.12, 0.11, 0.11, 0.10, 0.11, 0.11],
            fontsize=7.4,
        )
    pdf.savefig(fig)
    plt.close(fig)


def _wealth(plt: Any, pdf: Any, trace: Any, books: list[dict[str, Any]], stamp: str, stem: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(
        fig,
        "Constructed Portfolios: Cumulative Wealth, Drawdown & Rolling Risk",
        stamp,
        stem,
        _question(trace),
    )
    ax1 = fig.add_axes((0.11, 0.65, 0.82, 0.23))
    ax2 = fig.add_axes((0.11, 0.37, 0.82, 0.21))
    ax3 = fig.add_axes((0.11, 0.09, 0.82, 0.21))

    ax1.set_title("Cumulative Wealth (Growth of 1)", loc="left")
    ax2.set_title("Underwater Drawdown (%)", loc="left")
    roll_drawn = 0
    for index, book in enumerate(books):
        color = _COLORS[index % len(_COLORS)]
        lw = 2.0 if book["method"] in {"hrp", "var_sorted"} else 1.3
        ax1.plot(book.get("wealth") or [], color=color, lw=lw, label=book["method"])
        dd_pct = [float(x) * 100.0 for x in (book.get("drawdown") or [])]
        ax2.plot(dd_pct, color=color, lw=1.3, label=book["method"])
        rets = np.asarray(book.get("returns") or [], dtype=float)
        win = 63 if rets.size > 75 else max(10, rets.size // 3)
        if rets.size > win:
            vol = [
                float(np.std(rets[i - win : i], ddof=1) * math.sqrt(252.0) * 100.0)
                for i in range(win, rets.size)
            ]
            ax3.plot(range(win, rets.size), vol, color=color, lw=1.3, label=book["method"])
            roll_drawn += 1

    ax1.axhline(1.0, color="#64748B", lw=0.7, ls="--")
    ax1.set_ylabel("growth of 1")
    ax1.legend(loc="upper left", ncol=4, fontsize=7.6)

    ax2.axhline(0.0, color="#64748B", lw=0.7, ls="--")
    ax2.set_ylabel("drawdown (%)")

    ax3.set_title("Rolling Annualized Volatility (%)", loc="left")
    ax3.set_ylabel("annualized vol (%)")
    ax3.set_xlabel("trading observation")
    if roll_drawn == 0:
        ax3.text(0.5, 0.5, "Insufficient window for rolling volatility", ha="center", va="center")
    pdf.savefig(fig)
    plt.close(fig)


def _risk(
    plt: Any,
    pdf: Any,
    trace: Any,
    book: dict[str, Any],
    assets: list[dict[str, Any]],
    stamp: str,
    stem: str,
) -> None:
    rets = np.asarray(book.get("returns") or [], dtype=float) * 100.0
    if rets.size < 8:
        return
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(
        fig,
        f"Tail Risk, VaR / CVaR Distribution & VaR-Sorted Ranking ({book['method']})",
        stamp,
        stem,
        _question(trace),
    )
    ax = fig.add_axes((0.11, 0.50, 0.80, 0.38))
    bins = min(42, max(10, rets.size // 6))
    ax.hist(rets, bins=bins, color="#0072B2", alpha=0.72, edgecolor="white", density=True, label="realized")
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
    if sd > 0:
        xs = np.linspace(float(rets.min()), float(rets.max()), 200)
        pdf_norm = (1.0 / (sd * math.sqrt(2.0 * math.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
        ax.plot(xs, pdf_norm, color="#0F172A", lw=1.4, ls="--", label="Gaussian fit")

    var = book.get("var") or {}
    for key, label, col, style in (
        ("var_pct", "95% Historical VaR", "#D95F02", "-"),
        ("parametric_var_pct", "95% Parametric VaR", "#009E73", "--"),
        ("cornish_fisher_var_pct", "95% Cornish-Fisher VaR", "#E6AB02", ":"),
        ("cvar_pct", "95% CVaR (Expected Shortfall)", "#7570B3", "-"),
    ):
        val = var.get(key)
        if isinstance(val, (int, float)):
            ax.axvline(-float(val), color=col, lw=1.6, ls=style, label=f"{label}: -{val}%")

    ax.set_title(f"Daily Return Density & Tail-Loss Thresholds ({book['method']})", loc="left")
    ax.set_xlabel("daily return (%)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=7.6)

    if assets:
        by_var = sorted(assets, key=lambda r: float(r.get("var_95_pct") or 0.0))
        picks = by_var[:6] + [r for r in by_var[-6:] if r not in by_var[:6]]
        rows: list[list[str]] = []
        for rank_i, row in enumerate(picks[:12], start=1):
            rows.append(
                [
                    f"{rank_i:02d}",
                    f"{row['name']} *" if row.get("focused") else str(row["name"]),
                    _fmt_pct(row.get("var_95_pct")),
                    _fmt_pct(row.get("cvar_95_pct")),
                    f"{float(row['var_weight']):.4f}",
                    f"{float(row['hrp_weight']):.4f}",
                    _fmt_pct(row.get("volatility_annualized_pct")),
                    _fmt_pct(row.get("max_drawdown_pct")),
                ]
            )
        ax_tbl = fig.add_axes((0.075, 0.07, 0.85, 0.34))
        ax_tbl.axis("off")
        ax_tbl.set_title(
            "Constituents Sorted by 95% Daily VaR (Lowest-Risk Sleeves vs. Highest-Tail-Risk Sleeves)",
            loc="left",
            fontsize=9.5,
        )
        _styled_table(
            ax_tbl,
            ["#", "Ticker", "95% VaR", "95% CVaR", "VaR-Book Wt", "HRP Wt", "Ann Vol", "Max DD"],
            rows,
            [0.06, 0.14, 0.12, 0.12, 0.14, 0.14, 0.14, 0.14],
            fontsize=7.6,
        )
    pdf.savefig(fig)
    plt.close(fig)


def _scenes(
    plt: Any,
    pdf: Any,
    trace: Any,
    rows: list[dict[str, Any]],
    stamp: str,
    stem: str,
) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(
        fig,
        "Macro, Inflation/Rate & Volatility Scenario Stress Analysis",
        stamp,
        stem,
        _question(trace),
    )
    ax1 = fig.add_axes((0.26, 0.50, 0.29, 0.38))
    ax2 = fig.add_axes((0.61, 0.50, 0.29, 0.38))

    labels = [str(row["name"]) for row in rows][::-1]
    dds = [float(row.get("max_drawdown_pct") or 0.0) for row in rows][::-1]
    rets = [float(row.get("total_return_pct") or 0.0) for row in rows][::-1]
    y_pos = np.arange(len(labels))

    ax1.barh(y_pos, dds, color="#D95F02", height=0.68)
    ax1.set_yticks(y_pos, labels, fontsize=7.6)
    ax1.set_title("Max Drawdown (%)", loc="left", fontsize=9.8)
    ax1.set_xlabel("drawdown (%)")

    ret_colors = ["#009E73" if r >= 0 else "#D95F02" for r in rets]
    ax2.barh(y_pos, rets, color=ret_colors, height=0.68)
    ax2.set_yticks(y_pos, [""] * len(labels))
    ax2.axvline(0.0, color="#475569", lw=0.8)
    ax2.set_title("Total Return (%)", loc="left", fontsize=9.8)
    ax2.set_xlabel("total return (%)")

    tbl_rows: list[list[str]] = []
    for row in rows:
        tbl_rows.append(
            [
                str(row.get("name") or ""),
                _fmt_pct(row.get("total_return_pct")),
                _fmt_pct(row.get("annualized_return_pct")),
                _fmt_pct(row.get("volatility_annualized_pct")),
                _fmt_pct(row.get("max_drawdown_pct")),
                _shown(row.get("sharpe_annualized")),
                _fmt_pct(row.get("var_95_pct")),
            ]
        )
    ax_tbl = fig.add_axes((0.075, 0.07, 0.85, 0.34))
    ax_tbl.axis("off")
    ax_tbl.set_title("Scenario Stress Scorecard (Constructed Book)", loc="left", fontsize=9.5)
    _styled_table(
        ax_tbl,
        ["Scenario", "Total Ret", "Ann Ret", "Ann Vol", "Max DD", "Sharpe", "95% VaR"],
        tbl_rows,
        [0.34, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
        fontsize=7.6,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _monte(
    plt: Any,
    pdf: Any,
    trace: Any,
    monte: dict[str, Any],
    stamp: str,
    stem: str,
) -> None:
    edges = [float(x) for x in monte.get("edges") or []]
    counts = [float(x) for x in monte.get("counts") or []]
    if len(edges) < 2 or not counts:
        return
    fig = plt.figure(figsize=(8.5, 11))
    mc_sub = (
        f"Monte Carlo Simulation: Wealth Fan & Sharpe Distribution "
        f"({monte.get('dgp')}, {monte.get('n_paths')} Paths)"
    )
    _chrome(
        fig,
        mc_sub,
        stamp,
        stem,
        _question(trace),
    )

    raw_fan = monte.get("fan")
    fan: dict[str, Any] = raw_fan if isinstance(raw_fan, dict) else {}
    q05 = fan.get("q05") or []
    q25 = fan.get("q25") or []
    q50 = fan.get("q50") or []
    q75 = fan.get("q75") or []
    q95 = fan.get("q95") or []
    if len(q50) >= 8:
        ax_fan = fig.add_axes((0.11, 0.55, 0.79, 0.33))
        xs = np.arange(len(q50))
        for sp in (fan.get("sample_paths") or [])[:12]:
            if len(sp) == len(q50):
                ax_fan.plot(xs, sp, color="#94A3B8", lw=0.6, alpha=0.35)
        ax_fan.fill_between(xs, q05, q95, color="#0072B2", alpha=0.16, label="5th–95th percentile cone")
        ax_fan.fill_between(xs, q25, q75, color="#0072B2", alpha=0.30, label="25th–75th percentile cone")
        ax_fan.plot(xs, q50, color="#0F172A", lw=2.0, label="Median simulated wealth (q50)")
        ax_fan.axhline(1.0, color="#64748B", lw=0.7, ls="--")
        ax_fan.set_xlim(0, len(q50) - 1)
        ax_fan.set_title(
            f"Simulated Cumulative Wealth Fan ({monte.get('horizon_days')} Trading Observations)",
            loc="left",
        )
        ax_fan.set_xlabel("simulated observation")
        ax_fan.set_ylabel("growth of 1")
        ax_fan.legend(loc="upper left", fontsize=7.6)

    ax_sh = fig.add_axes((0.11, 0.23, 0.36, 0.24))
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]
    width = edges[1] - edges[0]
    ax_sh.bar(centers[: len(counts)], counts, width=width * 0.88, color="#0072B2", edgecolor="white")
    for mark, label, col in (
        (monte.get("sharpe_q10"), "q10", "#D95F02"),
        (monte.get("sharpe_q50"), "q50", "#0F172A"),
        (monte.get("sharpe_q90"), "q90", "#009E73"),
    ):
        if isinstance(mark, (int, float)):
            ax_sh.axvline(float(mark), color=col, lw=1.4, ls="--", label=f"{label}: {mark}")
    ax_sh.set_title("Simulated Sharpe Distribution", loc="left", fontsize=9.5)
    ax_sh.set_xlabel("annualized Sharpe")
    ax_sh.set_ylabel("paths")
    ax_sh.legend(loc="upper left", fontsize=7.0)

    t_edges = [float(x) for x in monte.get("terminal_edges") or []]
    t_counts = [float(x) for x in monte.get("terminal_counts") or []]
    if len(t_edges) >= 2 and t_counts:
        ax_tr = fig.add_axes((0.54, 0.23, 0.36, 0.24))
        t_centers = [(t_edges[i] + t_edges[i + 1]) / 2.0 for i in range(len(t_edges) - 1)]
        t_width = t_edges[1] - t_edges[0]
        ax_tr.bar(
            t_centers[: len(t_counts)],
            t_counts,
            width=t_width * 0.88,
            color="#009E73",
            edgecolor="white",
        )
        for mark, label, col in (
            (monte.get("terminal_q05_pct"), "q05", "#D95F02"),
            (monte.get("terminal_q50_pct"), "q50", "#0F172A"),
            (monte.get("terminal_q95_pct"), "q95", "#0072B2"),
        ):
            if isinstance(mark, (int, float)):
                ax_tr.axvline(float(mark), color=col, lw=1.4, ls="--", label=f"{label}: {mark}%")
        ax_tr.set_title("Terminal Return Distribution (%)", loc="left", fontsize=9.5)
        ax_tr.set_xlabel("terminal cumulative return (%)")
        ax_tr.set_ylabel("paths")
        ax_tr.legend(loc="upper left", fontsize=7.0)

    ax_tbl = fig.add_axes((0.075, 0.07, 0.85, 0.10))
    ax_tbl.axis("off")
    sh_band = (
        f"{_shown(monte.get('sharpe_q10'))} / "
        f"{_shown(monte.get('sharpe_q50'))} / "
        f"{_shown(monte.get('sharpe_q90'))}"
    )
    summary_rows = [
        [
            str(monte.get("dgp") or "gbm"),
            _shown(monte.get("n_paths")),
            f"{_shown(monte.get('horizon_days'))}d",
            _fmt_pct(monte.get("terminal_q05_pct")),
            _fmt_pct(monte.get("terminal_q50_pct")),
            _fmt_pct(monte.get("terminal_q95_pct")),
            _fmt_pct(monte.get("drawdown_q50_pct")),
            _fmt_pct(monte.get("drawdown_q95_pct")),
            sh_band,
            _fmt_pct(monte.get("positive_pct")),
        ]
    ]
    _styled_table(
        ax_tbl,
        [
            "DGP",
            "Paths",
            "Horizon",
            "Ret q05",
            "Ret q50",
            "Ret q95",
            "DD q50",
            "DD q95",
            "Sharpe q10/q50/q90",
            "Sharpe > 0",
        ],
        summary_rows,
        [0.07, 0.07, 0.08, 0.09, 0.09, 0.09, 0.09, 0.09, 0.22, 0.11],
        fontsize=7.4,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _bakeoff(
    plt: Any,
    pdf: Any,
    trace: Any,
    bakeoff: dict[str, Any],
    stamp: str,
    stem: str,
) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(
        fig,
        f"Multi-Pass GAN Strategy Bakeoff & Referee Tournament ({bakeoff.get('n_passes') or 4} Passes)",
        stamp,
        stem,
        _question(trace),
    )
    banner = (
        f"Survivor: {bakeoff.get('survivor')}  |  "
        f"In-Sample Sharpe: {bakeoff.get('survivor_sharpe')}  |  "
        f"Post-Attack Standing: {bakeoff.get('attacked_sharpe')}  |  "
        f"Deflated Sharpe: {bakeoff.get('deflated_sharpe')} ({bakeoff.get('verdict')})\n"
        f"Passes: {bakeoff.get('n_passes') or 4}  |  "
        f"Candidates: {bakeoff.get('n_trials')}  |  "
        f"Adversarial Attacks: {', '.join(bakeoff.get('attacks') or []) or 'none'}"
    )
    fig.text(0.075, 0.895, banner, fontsize=8.0, fontweight="bold", color="#0F172A", va="top")

    curves = list(bakeoff.get("curves") or [])
    if curves:
        ax1 = fig.add_axes((0.09, 0.48, 0.44, 0.34))
        ax1.set_title("Cumulative Wealth Across Passes", loc="left", fontsize=9.5)
        survivor = str(bakeoff.get("survivor") or "")
        for index, curve in enumerate(curves):
            label = str(curve.get("label") or "")
            is_win = label == survivor
            ax1.plot(
                curve.get("growth") or [],
                color="#0F172A" if is_win else _COLORS[index % len(_COLORS)],
                lw=2.2 if is_win else 1.2,
                alpha=1.0 if is_win else 0.84,
                label=f"{label}*" if is_win else label,
            )
        ax1.axhline(1.0, color="#64748B", lw=0.7, ls="--")
        ax1.set_xlabel("trading observation")
        ax1.set_ylabel("growth of 1")
        ax1.legend(loc="upper left", ncol=2, fontsize=6.8)

        ax2 = fig.add_axes((0.62, 0.48, 0.30, 0.34))
        labels = [str(c.get("label") or "") for c in curves][::-1]
        in_sh = [float(c.get("sharpe") or 0.0) for c in curves][::-1]
        st_sh = [
            float(c.get("standing") if c.get("standing") is not None else (c.get("sharpe") or 0.0))
            for c in curves
        ][::-1]
        y_pos = np.arange(len(labels))
        ax2.barh(y_pos + 0.17, in_sh, height=0.32, color="#0072B2", label="In-Sample")
        ax2.barh(y_pos - 0.17, st_sh, height=0.32, color="#D95F02", label="Post-Attack")
        ax2.axvline(0.0, color="#475569", lw=0.8)
        ax2.set_yticks(y_pos, labels, fontsize=7.2)
        ax2.set_title("In-Sample vs. Attacked Sharpe", loc="left", fontsize=9.5)
        ax2.set_xlabel("period Sharpe")
        ax2.legend(loc="lower right", fontsize=7.0)

        tbl_rows: list[list[str]] = []
        survivor = str(bakeoff.get("survivor") or "")
        for idx, curve in enumerate(curves, start=1):
            label = str(curve.get("label") or "")
            status = "SURVIVOR" if label == survivor else "eliminated"
            p_num = curve.get("round") or ((idx - 1) // 2 + 1)
            tbl_rows.append(
                [
                    f"P{p_num}",
                    label,
                    _shown(curve.get("sharpe")),
                    _shown(curve.get("standing")),
                    str(curve.get("worst_attack") or "in_sample"),
                    _shown(curve.get("annualized_sharpe")),
                    _fmt_pct(curve.get("total_return_pct")),
                    _fmt_pct(curve.get("max_drawdown_pct")),
                    status,
                ]
            )
        ax_tbl = fig.add_axes((0.075, 0.07, 0.85, 0.34))
        ax_tbl.axis("off")
        ax_tbl.set_title("Multi-Pass Referee Candidate Scorecard", loc="left", fontsize=9.5)
        _styled_table(
            ax_tbl,
            [
                "Pass",
                "Candidate Rule",
                "Period SR",
                "Attacked SR",
                "Worst Attack",
                "Ann SR",
                "Total Ret",
                "Max DD",
                "Status",
            ],
            tbl_rows,
            [0.06, 0.17, 0.10, 0.11, 0.14, 0.10, 0.10, 0.10, 0.12],
            fontsize=7.3,
        )
    else:
        trials = list(bakeoff.get("trials") or [])
        if trials:
            fig.text(
                0.08,
                0.78,
                "\n".join(trials[:16]),
                fontsize=8.5,
                family="monospace",
                color="#1E293B",
                va="top",
            )
    pdf.savefig(fig)
    plt.close(fig)


def _pairs(
    plt: Any,
    pdf: Any,
    trace: Any,
    board: dict[str, Any],
    stamp: str,
    stem: str,
) -> None:
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
                "yes" if row.get("cointegrated") else "no",
            ]
        )
    if not rows:
        return
    fig = plt.figure(figsize=(8.5, 11))
    _chrome(
        fig,
        f"Cointegration Screen ({board.get('n_cointegrated')} of {board.get('n_pairs')} Pairs Survive FDR)",
        stamp,
        stem,
        _question(trace),
    )
    ax = fig.add_axes((0.075, 0.12, 0.85, 0.76))
    ax.axis("off")
    _styled_table(
        ax,
        ["Leg A", "Leg B", "ADF p-value", "Half-Life (obs)", "FDR Survivor"],
        rows,
        [0.20, 0.20, 0.20, 0.20, 0.20],
        fontsize=8.0,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _styled_table(
    ax: Any,
    col_labels: list[str],
    rows: list[list[str]],
    col_widths: list[float],
    *,
    fontsize: float = 7.8,
) -> Any:
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="upper left",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    base_h = 0.042
    for (row_i, _col_j), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.6)
        if row_i == 0:
            cell.set_facecolor("#1E293B")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(base_h * 1.15)
        else:
            line_count = 1
            if 1 <= row_i <= len(rows):
                line_count = max(str(val).count("\n") + 1 for val in rows[row_i - 1])
            cell.set_facecolor("#F8FAFC" if row_i % 2 == 1 else "#FFFFFF")
            cell.set_text_props(color="#0F172A")
            cell.set_height(base_h * (0.75 + 0.42 * line_count))
    return table


def _chrome(fig: Any, subtitle: str, stamp: str, stem: str, note: str) -> None:
    fig.text(0.075, 0.955, subtitle, fontsize=11, fontweight="bold", color="#0F172A")
    fig.text(
        0.925,
        0.955,
        f"Report {stem}   |   {stamp}",
        fontsize=8,
        color="#475569",
        ha="right",
    )
    line_top = fig.add_axes((0.075, 0.942, 0.85, 0.0015))
    line_top.axhspan(0, 1, color="#CBD5E1")
    line_top.axis("off")

    line_bot = fig.add_axes((0.075, 0.042, 0.85, 0.001))
    line_bot.axhspan(0, 1, color="#E2E8F0")
    line_bot.axis("off")

    fig.text(0.075, 0.022, f"alphaengine {__version__}", fontsize=7.8, color="#64748B")
    if note:
        fig.text(0.925, 0.022, _clip_words(note, 78), fontsize=7.8, color="#64748B", ha="right")


def _fmt_pct(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return "n/a"
    return f"{value}%"


def _shown(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def _question(trace: Any) -> str:
    return str(getattr(trace, "question", "") or "")


def _clip_words(text: str, width: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= width:
        return clean
    cut = clean[: max(1, width - 1)]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "…"


def _paragraphs(text: str, width: int) -> list[list[str]]:
    raw_paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if not raw_paras:
        return [[""]]
    out: list[list[str]] = []
    for para in raw_paras:
        clean = " ".join(para.split())
        wrapped = textwrap.wrap(
            clean,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        out.append(wrapped or [""])
    return out
