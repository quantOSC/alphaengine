"""A side channel for one inspectable line per statistic.

Core functions return the same numbers they always have. When a collector is
bound, they also record the formula, the scalar inputs, the series that line
was computed on, and the result. Nothing here is read by the functions
themselves, so an unbound collector is a no-op and cannot move a figure.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

_current: ContextVar[Collector | None] = ContextVar("alphaengine_math_trace", default=None)


@dataclass
class MathLine:
    """One statistic a person would open: formula, inputs, series, result."""

    id: str
    formula: str
    inputs: dict[str, Any]
    series: dict[str, list[float | None]]
    result: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "formula": self.formula,
            "inputs": self.inputs,
            "series": self.series,
            "result": self.result,
        }


@dataclass
class Collector:
    lines: list[MathLine] = field(default_factory=list)

    def add(self, line: MathLine) -> None:
        self.lines.append(line)


def bind(collector: Collector) -> Token[Collector | None]:
    return _current.set(collector)


def reset(token: Token[Collector | None]) -> None:
    _current.reset(token)


@contextmanager
def collecting() -> Iterator[Collector]:
    """Bind a fresh collector for the dynamic extent of the block."""
    collector = Collector()
    token = bind(collector)
    try:
        yield collector
    finally:
        reset(token)


def record(
    line_id: str,
    formula: str,
    *,
    inputs: dict[str, Any] | None = None,
    series: dict[str, Any] | None = None,
    result: Any = None,
) -> None:
    """Record one line if a collector is bound. No-op otherwise."""
    collector = _current.get()
    if collector is None:
        return
    collector.add(
        MathLine(
            id=str(line_id),
            formula=str(formula),
            inputs=_inputs(inputs or {}),
            series={str(k): _series(v) for k, v in (series or {}).items()},
            result=_value(result),
        )
    )


def _inputs(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in raw.items():
        cleaned = _value(value)
        if isinstance(cleaned, (dict, list)):
            continue
        out[str(key)] = cleaned
    return out


def _series(values: Any) -> list[float | None]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if not isinstance(values, list) or (values and isinstance(values[0], (list, tuple, dict))):
        return []
    return [_finite(item) for item in values]


def _finite(item: Any) -> float | None:
    if isinstance(item, bool) or item is None:
        return None
    try:
        number = float(item)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _value(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item) and not isinstance(value, (str, bytes, list, dict)):
        with contextlib.suppress(ValueError, TypeError):
            value = value.item()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items() if not isinstance(_value(v), (dict, list))}
    return value if isinstance(value, (int, float, str, bool)) or value is None else str(value)
