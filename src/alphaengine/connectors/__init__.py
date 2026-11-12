"""Optional data ingress. Lazy: importing alphaengine never pulls httpx or pyarrow.

Local parquet/CSV panels first. HTTP is opt-in and only to a URL the caller
passes — AlphaEngine still does not fetch a universe on their behalf by ticker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load_panel", "load_parquet", "fetch_url"]


def load_panel(path: str | Path) -> Any:
    """CSV or parquet, decided by suffix. Same three shapes as `loaders.load_csv`."""
    suffix = Path(path).suffix.lower()
    if suffix in {".parquet", ".pq", ".arrow", ".feather"}:
        return load_parquet(path)
    from ..loaders import load_csv

    return load_csv(path)


def load_parquet(path: str | Path) -> Any:
    """Read a parquet/arrow table into the CSV loader's three shapes.

    Requires `pyarrow` (`pip install 'alphaengine[connectors]'`). Columns are
    decoded to text and handed to `load_delimited`, so a parquet file with a
    `date,symbol,close` schema is a long panel, exactly as the CSV would be.
    """
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - extra missing
        raise ModuleNotFoundError(
            "parquet needs pyarrow. Install it with:  pip install 'alphaengine[connectors]'"
        ) from exc

    table = pq.read_table(str(path))
    names = list(table.column_names)
    rows: list[list[str]] = [names]
    for i in range(table.num_rows):
        row = []
        for name in names:
            value = table.column(name)[i].as_py()
            row.append("" if value is None else str(value))
        rows.append(row)

    # Rebuild as CSV text so `load_delimited` owns the shape decision — one
    # place, not two loaders that can disagree about "wide vs long".
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    from ..loaders import load_delimited

    return load_delimited(buf.getvalue())


#: Upper bound on one in-memory HTTP fetch. A larger file belongs on disk.
MAX_FETCH_BYTES = 256 * 1024 * 1024


def fetch_url(url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0) -> bytes:
    """GET a URL the caller named. Requires httpx. No ticker-list helper.

    The URL is the caller's already-paid vendor, not a universe we invent.
    """
    try:
        import httpx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "HTTP fetch needs httpx. Install it with:  pip install 'alphaengine[connectors]'"
        ) from exc
    from urllib.parse import urlsplit

    scheme = urlsplit(str(url)).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"fetch_url only fetches http(s) URLs, not {scheme or 'a bare path'!r}.")
    chunks: list[bytes] = []
    total = 0
    with httpx.stream("GET", url, headers=headers or {}, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > MAX_FETCH_BYTES:
                raise ValueError(
                    f"{url!r} returned more than {MAX_FETCH_BYTES // (1024 * 1024)} MiB; "
                    "download it to a file and load it with --data instead."
                )
            chunks.append(chunk)
    return b"".join(chunks)
