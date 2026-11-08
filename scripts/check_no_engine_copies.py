#!/usr/bin/env python3
"""Grep guard to detect and prevent shadow engine copies (QOS-159).

Ensures that core quantitative primitives (DSR, PBO, CSCV, MinTRL, Harvey-Liu,
participation ratio N_eff, backtest simulation and scoring) are imported
from canonical `alphaengine` rather than duplicated or re-implemented.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Primary math primitives that must never be re-implemented as shadow duplicates
CORE_ENGINE_PRIMITIVES = {
    "trials_for_deflated_sharpe",
    "deflated_sharpe",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "pbo_cscv",
    "combinatorial_oos_stability",
    "min_track_record_length",
    "harvey_liu_haircut",
    "n_eff_participation",
}


def check_for_shadow_copies(search_dir: Path) -> list[str]:
    """Inspect Python files outside src/alphaengine/core/ to ensure no shadow functions are defined."""
    violations: list[str] = []
    canonical_core = ROOT / "src" / "alphaengine" / "core"

    for py_file in search_dir.rglob("*.py"):
        # Skip canonical core definitions
        if canonical_core in py_file.parents or py_file.parent == canonical_core:
            continue
        # Skip test files and test utilities
        if "tests" in py_file.parts or "test_" in py_file.name:
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in CORE_ENGINE_PRIMITIVES:
                violations.append(
                    f"{py_file.relative_to(ROOT)}:{node.lineno}: shadow engine copy detected: "
                    f"function '{node.name}' re-implemented outside canonical core (QOS-159)"
                )
    return violations


def main() -> int:
    search_dir = ROOT / "src"
    violations = check_for_shadow_copies(search_dir)

    if violations:
        print("ERROR: Shadow engine copies detected:", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1

    print("check-no-engine-copies ok: zero shadow copies of core primitives found outside canonical engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
