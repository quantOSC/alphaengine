"""Local dataset registry (`.alphaengine/data/datasets.jsonl`).

Registers CSV and Parquet files locally with full 64-hex SHA-256 hashes, row
counts, date ranges, column schemas, and data-hygiene checks:
- Duplicate timestamps (`duplicate_timestamps`)
- Missing values / calendar gaps (`gaps`)
- Look-ahead hints (`lookahead_hints`: future-dated rows relative to `--asof-col`)
- Survivorship hints (`survivorship_hints`: symbols whose history ends before the panel end date)

Data stays on the local machine; nothing is uploaded.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import io
from datetime import timezone
from pathlib import Path
from typing import Any

from .loaders import _DATE, _SYMBOL, _clean, _number, load_delimited
from .project import Project

__all__ = [
    "add_dataset",
    "list_datasets",
    "get_dataset",
    "dataset_lineage",
]


def _read_table_rows(path: Path) -> tuple[list[str], list[list[str]], str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq", ".arrow", ".feather"}:
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError as exc:
            raise ModuleNotFoundError(
                "parquet needs pyarrow. Install it with:  pip install 'alphaengine[connectors]'"
            ) from exc
        table = pq.read_table(str(path))
        header = [str(c) for c in table.column_names]
        body: list[list[str]] = []
        for i in range(table.num_rows):
            row = []
            for name in header:
                val = table.column(name)[i].as_py()
                row.append("" if val is None else str(val))
            body.append(row)
        return header, body, "parquet"

    text = path.read_text(encoding="utf-8-sig")
    sample = text[:8192]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        raise ValueError(f"{path} is empty")
    first_num = _number(rows[0][0]) if len(rows[0]) == 1 else None
    if len(rows[0]) == 1 and first_num is not None:
        return ["value"], rows, "csv"
    return [c.strip().lstrip("\ufeff") for c in rows[0]], rows[1:], "csv"


def _infer_col_schema(
    header: list[str],
    body: list[list[str]],
    date_idx: int | None,
    sym_idx: int | None,
    asof_idx: int | None,
) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for idx, name in enumerate(header):
        n_missing = 0
        n_numeric = 0
        n_present = 0
        for row in body:
            cell = (row[idx] if idx < len(row) else "").strip()
            if not cell or cell.lower() in ("na", "n/a", "nan", "null", "none", "-"):
                n_missing += 1
                continue
            n_present += 1
            if _number(cell) is not None:
                n_numeric += 1
        if idx in (date_idx, asof_idx):
            dtype = "date"
        elif idx == sym_idx:
            dtype = "symbol"
        elif n_present > 0 and n_numeric == n_present:
            dtype = "float"
        else:
            dtype = "string"
        schema.append({"name": name, "dtype": dtype, "n_missing": n_missing})
    return schema


def _parse_iso_date(val: str) -> datetime.date | None:
    clean = (val or "").strip()[:10]
    try:
        return datetime.date.fromisoformat(clean)
    except ValueError:
        return None


def _run_hygiene_checks(
    header: list[str],
    body: list[list[str]],
    *,
    date_idx: int | None,
    sym_idx: int | None,
    asof_col: str | None = None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    norm_header = [_clean(h) for h in header]
    asof_idx: int | None = None
    asof_cutoff: str | None = None
    if asof_col:
        wanted = _clean(asof_col)
        asof_idx = next((i for i, h in enumerate(norm_header) if h == wanted), None)
        if asof_idx is None:
            asof_cutoff = asof_col.strip()
    else:
        asof_idx = next(
            (
                i
                for i, h in enumerate(norm_header)
                if h in ("asof", "as_of", "as_of_date", "known_at") and i != date_idx
            ),
            None,
        )

    dates_all: list[str] = []
    seen_keys: set[tuple[str, str]] = set()
    dup_samples: list[str] = []
    dup_count = 0

    future_rows = 0
    future_samples: list[str] = []

    sym_dates: dict[str, list[str]] = {}
    missing_cells = 0

    for row in body:
        d_val = (row[date_idx] if (date_idx is not None and date_idx < len(row)) else "").strip()
        s_val = (row[sym_idx] if (sym_idx is not None and sym_idx < len(row)) else "").strip().upper()
        if d_val:
            dates_all.append(d_val)
            key = (s_val or "_ALL_", d_val)
            if key in seen_keys:
                dup_count += 1
                if len(dup_samples) < 5:
                    dup_samples.append(f"{s_val}:{d_val}" if s_val else d_val)
            else:
                seen_keys.add(key)

        if asof_idx is not None and date_idx is not None and asof_idx < len(row) and date_idx < len(row):
            a_val = (row[asof_idx] or "").strip()
            if d_val and a_val and d_val > a_val:
                future_rows += 1
                if len(future_samples) < 5:
                    future_samples.append(f"date={d_val} > asof={a_val}")
        elif asof_cutoff and d_val and d_val > asof_cutoff:
            future_rows += 1
            if len(future_samples) < 5:
                future_samples.append(f"date={d_val} > cutoff={asof_cutoff}")

        # Track per-symbol date coverage for survivorship and gap checks
        if sym_idx is not None and d_val and s_val:
            sym_dates.setdefault(s_val, []).append(d_val)
        elif sym_idx is None and date_idx is not None and d_val:
            for col_i, col_name in enumerate(header):
                if col_i in (date_idx, asof_idx):
                    continue
                cell = (row[col_i] if col_i < len(row) else "").strip()
                if _number(cell) is not None:
                    sym_dates.setdefault(col_name.upper(), []).append(d_val)
                else:
                    missing_cells += 1

    unique_dates = sorted(set(dates_all))
    start_date = unique_dates[0] if unique_dates else None
    end_date = unique_dates[-1] if unique_dates else None

    # Calendar gap check (> 5 calendar days between consecutive observations)
    calendar_gaps = 0
    parsed_dates = [d for d in (_parse_iso_date(x) for x in unique_dates) if d is not None]
    for i in range(1, len(parsed_dates)):
        if (parsed_dates[i] - parsed_dates[i - 1]).days > 5:
            calendar_gaps += 1

    # Survivorship hints: symbols whose last observation is earlier than `end_date`
    early_ending: list[str] = []
    if end_date and len(sym_dates) >= 2:
        for sym, s_dates in sorted(sym_dates.items()):
            if s_dates and max(s_dates) < end_date:
                early_ending.append(sym)

    checks = {
        "duplicate_timestamps": {
            "count": dup_count,
            "samples": dup_samples,
        },
        "gaps": {
            "missing_cells": missing_cells,
            "calendar_gaps_gt_5d": calendar_gaps,
        },
        "lookahead_hints": {
            "asof_col": header[asof_idx] if asof_idx is not None else asof_cutoff,
            "n_future_rows": future_rows,
            "samples": future_samples,
        },
        "survivorship_hints": {
            "n_symbols": len(sym_dates),
            "n_ends_early": len(early_ending),
            "early_ending_symbols": early_ending[:20],
        },
    }
    return start_date, end_date, checks


def add_dataset(
    proj: Project,
    file_path: str | Path,
    *,
    name: str | None = None,
    asof_col: str | None = None,
) -> dict[str, Any]:
    """Register a local CSV or Parquet dataset in `.alphaengine/data/datasets.jsonl`."""
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"No dataset file at {p}")

    raw_bytes = p.read_bytes()
    file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    header, body, fmt = _read_table_rows(p)
    norm_header = [_clean(h) for h in header]
    date_idx = next((i for i, h in enumerate(norm_header) if h in _DATE), None)
    sym_idx = next((i for i, h in enumerate(norm_header) if h in _SYMBOL), None)
    asof_idx = (
        next((i for i, h in enumerate(norm_header) if h == _clean(asof_col)), None)
        if asof_col
        else next(
            (
                i
                for i, h in enumerate(norm_header)
                if h in ("asof", "as_of", "as_of_date", "known_at") and i != date_idx
            ),
            None,
        )
    )

    columns = _infer_col_schema(header, body, date_idx, sym_idx, asof_idx)
    start_date, end_date, checks = _run_hygiene_checks(
        header,
        body,
        date_idx=date_idx,
        sym_idx=sym_idx,
        asof_col=asof_col,
    )

    # Also compute the parsed panel hash so runs using `load_panel(p)` match by hash
    panel_hash = file_sha256
    try:
        from .connectors import load_panel
        from .ledger import hash_dataset_any

        # If an extra asof column is present in a wide/long file, strip it if needed or hash the loaded panel
        try:
            loaded = load_panel(p)
        except Exception:  # noqa: BLE001
            if asof_idx is not None:
                kept_cols = [i for i in range(len(header)) if i != asof_idx]
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([header[i] for i in kept_cols])
                for r in body:
                    writer.writerow([r[i] if i < len(r) else "" for i in kept_cols])
                loaded = load_delimited(buf.getvalue())
            else:
                loaded = None
        if loaded is not None:
            hashes = hash_dataset_any(loaded)
            if hashes:
                panel_hash = hashes[0]
    except Exception:  # noqa: BLE001
        pass

    ds_name = (name or "").strip() or p.stem
    try:
        rel_path = str(p.relative_to(proj.root))
    except ValueError:
        rel_path = str(p)

    record: dict[str, Any] = {
        "name": ds_name,
        "path": rel_path,
        "abs_path": str(p),
        "format": fmt,
        "sha256": file_sha256,
        "panel_hash": panel_hash,
        "n_rows": len(body),
        "start_date": start_date,
        "end_date": end_date,
        "columns": columns,
        "checks": checks,
        "registered_at": datetime.datetime.now(timezone.utc).isoformat(),
    }
    obj_sha = proj.append_jsonl("data/datasets.jsonl", record)
    record["object_sha256"] = obj_sha
    return record


def list_datasets(proj: Project) -> list[dict[str, Any]]:
    """Return the latest registration record for each dataset name in `proj`."""
    rows = proj.read_jsonl("data/datasets.jsonl")
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        nm = str(row.get("name") or "")
        if nm:
            by_name[nm] = row
    return list(by_name.values())


def get_dataset(proj: Project, name_or_sha: str) -> dict[str, Any]:
    """Resolve a registered dataset by name or SHA-256 prefix."""
    q = (name_or_sha or "").strip()
    rows = list(reversed(proj.read_jsonl("data/datasets.jsonl")))
    for row in rows:
        if str(row.get("name") or "") == q:
            return row
    for row in rows:
        sha = str(row.get("sha256") or "")
        psha = str(row.get("panel_hash") or "")
        if (q and sha.startswith(q)) or (q and psha.startswith(q)):
            return row
    raise KeyError(f"No dataset named {name_or_sha!r} in {proj.dot_dir / 'data/datasets.jsonl'}")


def dataset_lineage(proj: Project, run_ref: str) -> dict[str, Any]:
    """Return the datasets consumed by `<run>` and all runs that shared those datasets."""
    from .ledger import resolve_run

    run = resolve_run(proj, run_ref)
    run_hashes = set(str(h) for h in (run.get("dataset_hashes") or []) if h)
    spec = run.get("repro_spec") or {}
    data_path = str(spec.get("data_path") or "")

    datasets = list_datasets(proj)
    matched_datasets: list[dict[str, Any]] = []
    matched_hashes = set(run_hashes)
    for ds in datasets:
        ds_hashes = {str(ds.get("sha256") or ""), str(ds.get("panel_hash") or "")} - {""}
        if (ds_hashes & run_hashes) or (
            data_path and data_path in {str(ds.get("name")), str(ds.get("path")), str(ds.get("abs_path"))}
        ):
            matched_datasets.append(ds)
            matched_hashes.update(ds_hashes)

    related_runs: list[dict[str, Any]] = []
    for r in proj.read_jsonl("ledger/runs.jsonl"):
        r_hashes = set(str(h) for h in (r.get("dataset_hashes") or []) if h)
        if r_hashes & matched_hashes:
            related_runs.append(r)

    return {
        "run_id": run.get("run_id"),
        "command": run.get("command"),
        "dataset_hashes": list(run.get("dataset_hashes") or []),
        "datasets": matched_datasets,
        "related_runs": [
            {
                "run_id": r.get("run_id"),
                "command": r.get("command"),
                "timestamp": r.get("timestamp"),
                "verdict": r.get("verdict"),
            }
            for r in related_runs
        ],
    }
