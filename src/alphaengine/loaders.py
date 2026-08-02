"""Get your data into a run without writing a Python module for it.

    alphaengine run screen_universe --data prices.csv
    alphaengine run size_position   --data returns.csv

WHY THIS EXISTS

    Until now data reached a workflow exactly one way: `--project <module>`,
    importing a module that exposes `data` and `backtest_fn`. That is the right
    door for a quant with a research package and the wrong one for everybody
    else, including the same quant on the day they want to check a CSV somebody
    emailed them. "Write a Python module first" is a real barrier in front of the
    thing this product is for.

    A CSV is not a lesser path. `validate_study` genuinely needs a simulator and
    a simulator is code; the other three workflows need numbers, and numbers live
    in files.

NOTHING HERE TOUCHES THE NETWORK, AND THAT IS THE POINT

    This reads a local file into memory and hands it to the executor, which runs
    in-process. The data does not move. There is deliberately no fetch, no cache
    and no upload: `--data` is the same custody story as `--project`, in a
    different format.

THE SHAPES, AND WHY DETECTION IS NARROW

    wide     date,AAPL,MSFT,NVDA        -> {symbol: [closes]}    a universe
    long     date,symbol,close          -> {symbol: [rows]}      a universe
    series   date,close  /  one column  -> [floats]              returns or prices

    Three shapes, recognised by their HEADER and nothing else. No sniffing of
    value ranges, no "looks like returns because they are small". A loader that
    guesses is a loader that eventually guesses wrong on somebody's book, and the
    failure — a screen ranking returns as prices — produces plausible numbers
    rather than an error. When the header does not decide it, this raises and
    names the three shapes.

NO NEW DEPENDENCY

    stdlib `csv` and numpy, which is already required. A loader that dragged in
    pandas would break the two-dependency promise on behalf of people who may
    never call it.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

__all__ = ["load_csv", "load_delimited", "DataShapeError", "coverage"]

#: Header names that mean "the date column", lowercased.
_DATE = {"date", "time", "timestamp", "dt", "day", "asof", "as_of"}
#: Header names that mean "the value column" in a long or series file.
_VALUE = {"close", "price", "value", "adj_close", "adjclose", "return", "returns", "ret", "pnl"}
#: Header names that mean "which instrument", in a long file.
_SYMBOL = {"symbol", "ticker", "name", "instrument", "asset", "id"}


class DataShapeError(ValueError):
    """The file could not be read as any of the three supported shapes.

    Raised rather than guessed at. The failure from a wrong guess is not an
    error, it is a plausible number — a screen that ranked returns as prices
    reports a shortlist nobody can tell is wrong.
    """


def _clean(name: str) -> str:
    return (name or "").strip().lstrip("﻿").lower()


def _number(raw: str) -> float | None:
    text = (raw or "").strip().replace(",", "")
    if not text or text.lower() in ("na", "n/a", "nan", "null", "none", "-"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_delimited(text: str, *, delimiter: str | None = None) -> Any:
    """Parse already-read text. `load_csv` is the file wrapper around this."""
    sample = text[:8192]
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","

    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        raise DataShapeError("the file is empty")

    header = [_clean(c) for c in rows[0]]
    body = rows[1:]
    if not body:
        raise DataShapeError("the file has a header and no rows")

    date_at = next((i for i, h in enumerate(header) if h in _DATE), None)
    symbol_at = next((i for i, h in enumerate(header) if h in _SYMBOL), None)
    value_at = next((i for i, h in enumerate(header) if h in _VALUE), None)

    # LONG first. A long file has all three columns, and checking it first means
    # a file with date+symbol+close is never mistaken for a two-name wide file.
    if date_at is not None and symbol_at is not None and value_at is not None:
        return _long(body, date_at, symbol_at, value_at)

    # SERIES: a date and one column whose header is a VALUE NAME, or a single
    # bare column of numbers.
    if date_at is not None and value_at is not None and len(header) == 2:
        return [v for v in (_number(r[value_at]) for r in body if len(r) > value_at) if v is not None]
    if len(header) == 1:
        first = _number(rows[0][0])
        start = body if first is None else rows  # an unnamed column is data, not a header
        return [v for v in (_number(r[0]) for r in start if r) if v is not None]

    # WIDE: a date column, and every other column is an instrument.
    #
    # `>= 2`, so `date,AAPL` is a ONE-NAME UNIVERSE rather than undecidable. The
    # second column's header is what separates the two cases and it separates
    # them cleanly: a series file names its column `close`/`price`/`returns`, and
    # that spelling was already claimed by the branch above. Anything else in
    # that position is a ticker.
    if date_at is not None and len(header) >= 2:
        return _wide(header, body, date_at)

    raise DataShapeError(
        "could not tell what shape this file is. Three are understood, decided by "
        "the HEADER:\n"
        "  wide    date,AAPL,MSFT,...   one column per name\n"
        "  long    date,symbol,close    one row per name per day\n"
        "  series  date,close           a single series, or one bare column of numbers\n"
        f"got columns: {header}"
    )


def _long(
    body: list[list[str]], date_at: int, symbol_at: int, value_at: int
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in body:
        if len(row) <= max(date_at, symbol_at, value_at):
            continue
        value = _number(row[value_at])
        if value is None:
            continue
        symbol = (row[symbol_at] or "").strip().upper()
        if not symbol:
            continue
        out.setdefault(symbol, []).append({"date": (row[date_at] or "").strip(), "close": value})
    # Sorted by date because the file's order is the file's business and every
    # metric downstream reads the LAST element as the most recent.
    for rows in out.values():
        rows.sort(key=lambda r: str(r["date"]))
    return out


def _wide(header: list[str], body: list[list[str]], date_at: int) -> dict[str, list[dict[str, Any]]]:
    columns = [(i, h) for i, h in enumerate(header) if i != date_at and h]
    out: dict[str, list[dict[str, Any]]] = {h.upper(): [] for _, h in columns}
    ordered = sorted(body, key=lambda r: str(r[date_at]).strip() if len(r) > date_at else "")
    for row in ordered:
        if len(row) <= date_at:
            continue
        date = (row[date_at] or "").strip()
        for i, name in columns:
            if i >= len(row):
                continue
            value = _number(row[i])
            if value is None:
                # A GAP IS A GAP, not a zero and not a carried-forward value.
                # Filling it here would invent observations the file does not
                # contain, and every coverage figure downstream would be a lie
                # about data that was never supplied.
                continue
            out[name.upper()].append({"date": date, "close": value})
    return {k: v for k, v in out.items() if v}


def load_csv(path: str | Path, *, delimiter: str | None = None) -> Any:
    """Read a local delimited file into the shape a workflow wants.

    Returns `{symbol: rows}` for a universe file and `list[float]` for a single
    series. Never fetches, never caches, never uploads.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise DataShapeError(f"no file at {p}")
    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DataShapeError(f"{p} is not text this can read: {exc}") from exc
    try:
        return load_delimited(text, delimiter=delimiter)
    except DataShapeError as exc:
        raise DataShapeError(f"{p}: {exc}") from exc


def coverage(data: Any, symbols: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Narrow a universe to the names a definition asks for, and say what is missing.

    Returns `(kept, missing)`. **Missing names are RETURNED, not logged and
    dropped.** A universe of 500 screened against a file holding 40 of them is a
    different result from a universe of 40, and the only place that difference
    can still be seen is here.
    """
    if not isinstance(data, dict):
        raise DataShapeError(
            "a universe definition needs a universe file (wide or long), and this data is a single series."
        )
    wanted = [str(s).strip().upper() for s in symbols if str(s).strip()]
    kept = {s: data[s] for s in wanted if s in data}
    missing = [s for s in wanted if s not in data]
    return kept, missing
