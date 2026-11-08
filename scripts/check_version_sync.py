#!/usr/bin/env python3
"""Verify single-source version agreement across repo metadata (`QOS-176`).

Checks that:
1. `src/alphaengine/_version.py` declares `__version__`.
2. `pyproject.toml` uses Hatch dynamic versioning pointing to `src/alphaengine/_version.py`
   (never a static `version = ...` that can drift).
3. `CHANGELOG.md`'s first `## [x.y.z]` release header matches `__version__` (or base version
   for `rc` pre-releases such as `2.0.0rc1` -> `2.0.0`).
4. When `GITHUB_REF` is `refs/tags/v*` (or `--tag` is supplied), the tag matches `v{__version__}`.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "src" / "alphaengine" / "_version.py"
PYPROJECT_FILE = ROOT / "pyproject.toml"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def _read_declared_version() -> str:
    text = VERSION_FILE.read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, flags=re.MULTILINE)
    if not m:
        raise RuntimeError(f"No literal __version__ assignment found in {VERSION_FILE}")
    return m.group(1).strip()


def _check_pyproject() -> list[str]:
    errors: list[str] = []
    text = PYPROJECT_FILE.read_text(encoding="utf-8")
    if re.search(r'^version\s*=\s*["\']', text, flags=re.MULTILINE):
        errors.append("pyproject.toml defines a static `version = ...`; use `dynamic = ['version']`.")
    if 'path = "src/alphaengine/_version.py"' not in text:
        errors.append(
            'pyproject.toml is missing `[tool.hatch.version] path = "src/alphaengine/_version.py"`.'
        )
    return errors


def _base_version(ver: str) -> str:
    return re.split(r"(?:a|b|rc|\.post|\.dev)", ver, maxsplit=1)[0]


def _check_changelog(declared: str) -> list[str]:
    text = CHANGELOG_FILE.read_text(encoding="utf-8")
    m = re.search(r"^## \[([^\]]+)\]", text, flags=re.MULTILINE)
    if not m:
        return ["CHANGELOG.md has no `## [<version>]` section header."]
    head = m.group(1).strip()
    if head != declared and head != _base_version(declared):
        return [
            f"CHANGELOG.md top section is [{head}], which does not match "
            f"_version.py __version__ = {declared!r}."
        ]
    return []


def _check_tag(declared: str, explicit_tag: str | None) -> list[str]:
    tag = explicit_tag
    if not tag:
        ref = os.environ.get("GITHUB_REF", "")
        if ref.startswith("refs/tags/"):
            tag = ref.removeprefix("refs/tags/")
    if not tag:
        return []
    expected = f"v{declared}"
    if tag != expected:
        return [f"Git tag {tag!r} does not match expected {expected!r} from _version.py."]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check AlphaEngine version consistency.")
    parser.add_argument("--tag", help="Explicit git tag (e.g. v2.0.0) to validate against __version__.")
    args = parser.parse_args(argv)

    declared = _read_declared_version()
    errors: list[str] = []
    errors.extend(_check_pyproject())
    errors.extend(_check_changelog(declared))
    errors.extend(_check_tag(declared, args.tag))

    if errors:
        for err in errors:
            sys.stderr.write(f"version-sync error: {err}\n")
        return 1

    sys.stdout.write(f"version-sync ok: alphaengine {declared}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
