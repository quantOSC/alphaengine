"""Contract fixtures shared between AlphaEngine and quantOS."""

from __future__ import annotations

from pathlib import Path

__all__ = ["hash_golden_path"]


def hash_golden_path() -> Path:
    return Path(__file__).resolve().parent / "hash_golden.json"
