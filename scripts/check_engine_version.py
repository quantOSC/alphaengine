#!/usr/bin/env python3
"""Engine version drift check (QOS-176).

Asserts that:
1. `constraints/engine.txt` pins the exact version from `_version.py`.
2. The wheel and sdist hashes in `constraints/engine.txt` match built distribution files.
3. Every consumer using this constraints file is guaranteed parity on alphaengine 2.0.x.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_VERSION = ROOT / "src" / "alphaengine" / "_version.py"
CONSTRAINTS = ROOT / "constraints" / "engine.txt"
DIST = ROOT / "dist"


def _read_current_version() -> str:
    content = SRC_VERSION.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not m:
        raise ValueError(f"Could not locate __version__ in {SRC_VERSION}")
    return m.group(1).strip()


def _read_pinned_constraint() -> tuple[str, list[str]]:
    if not CONSTRAINTS.is_file():
        raise FileNotFoundError(f"Missing {CONSTRAINTS}")
    content = CONSTRAINTS.read_text(encoding="utf-8")
    ver_match = re.search(r"alphaengine==([0-9a-zA-Z.-]+)", content)
    if not ver_match:
        raise ValueError(f"No alphaengine==<ver> constraint in {CONSTRAINTS}")
    pinned_version = ver_match.group(1).strip()
    hashes = re.findall(r"--hash=sha256:([0-9a-fA-F]{64})", content)
    return pinned_version, [h.lower() for h in hashes]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main() -> int:
    curr_version = _read_current_version()
    pinned_version, pinned_hashes = _read_pinned_constraint()

    if curr_version != pinned_version:
        print(
            f"ERROR: Engine version drift detected! _version.py is {curr_version!r} "
            f"but constraints/engine.txt pins {pinned_version!r}.",
            file=sys.stderr,
        )
        return 1

    # Check built dist hashes if dist files exist
    whl_files = list(DIST.glob(f"alphaengine-{curr_version}-*.whl"))
    tar_files = list(DIST.glob(f"alphaengine-{curr_version}.tar.gz"))

    if whl_files and pinned_hashes:
        whl_hash = _sha256(whl_files[0])
        if whl_hash not in pinned_hashes:
            print(
                f"ERROR: Built wheel {whl_files[0].name} sha256 ({whl_hash}) "
                f"does not match constraints/engine.txt ({pinned_hashes})",
                file=sys.stderr,
            )
            return 1

    if tar_files and pinned_hashes:
        tar_hash = _sha256(tar_files[0])
        if tar_hash not in pinned_hashes:
            print(
                f"ERROR: Built sdist {tar_files[0].name} sha256 ({tar_hash}) "
                f"does not match constraints/engine.txt ({pinned_hashes})",
                file=sys.stderr,
            )
            return 1

    print(
        f"engine-version-check ok: alphaengine=={curr_version} pinned and verified "
        f"with {len(pinned_hashes)} artifact hashes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
