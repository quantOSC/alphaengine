"""Build the AlphaEngine wheel and run the golden suite against the installed wheel (`QOS-157`).

Why this script exists:
- Editable installs (`pip install -e .`) read directly from `src/` and can hide
  missing subpackages, missing `.json` contract files, or accidental relative-path
  reads from the repository working tree.
- This script builds a real PEP 517 `.whl` (`python -m build --wheel --no-isolation`),
  unpacks it into an isolated site directory outside the repository, strips the
  repository and any editable-install path hooks from `sys.path` / `sys.meta_path`,
  changes `cwd` to an empty temporary directory, verifies `alphaengine.__file__`
  resolves inside the unpacked wheel, and runs the golden test suite.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_ISOLATED_RUNNER = """
import os
import sys
from pathlib import Path

repo_root = Path(os.environ["AE_REPO_ROOT"]).resolve()
wheel_site = Path(os.environ["AE_WHEEL_SITE"]).resolve()
work_dir = Path(os.environ["AE_WORK_DIR"]).resolve()

os.chdir(work_dir)

# Remove editable hooks and any path pointing at the repository or src/
sys.meta_path = [
    finder for finder in sys.meta_path
    if "editable" not in type(finder).__name__.lower()
    and " alphaengine" not in repr(finder).lower()
]

clean_paths = [str(wheel_site)]
for entry in sys.path:
    if not entry:
        continue
    try:
        resolved = Path(entry).resolve()
    except OSError:
        continue
    if resolved == repo_root or resolved == (repo_root / "src") or repo_root in resolved.parents:
        # Keep .venv site-packages even when .venv lives inside repo_root
        parts_lower = [p.lower() for p in resolved.parts]
        if "site-packages" not in parts_lower:
            continue
    clean_paths.append(entry)
sys.path[:] = clean_paths

# Purge any preloaded alphaengine modules just in case
for mod_name in [k for k in list(sys.modules) if k == "alphaengine" or k.startswith("alphaengine.")]:
    sys.modules.pop(mod_name, None)

import alphaengine
from alphaengine._version import __version__

ae_file = Path(alphaengine.__file__).resolve()
if wheel_site not in ae_file.parents:
    raise SystemExit(
        f"FAIL: alphaengine imported from {ae_file}, expected inside unpacked wheel {wheel_site}"
    )

print(f"Verified wheel import: {ae_file} (version {__version__})")
print(f"Isolated cwd:          {Path.cwd()}")

import pytest

rc = pytest.main(
    [
        str(repo_root / "tests" / "golden"),
        str(repo_root / "tests" / "test_goldens.py"),
        "-m",
        "golden",
        "-q",
    ]
)
raise SystemExit(int(rc))
"""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ae_wheel_golden_") as tmp_str:
        tmp = Path(tmp_str).resolve()
        dist_dir = tmp / "dist"
        wheel_site = tmp / "wheel_site"
        work_dir = tmp / "work"
        cfg_dir = tmp / "cfg"
        data_dir = tmp / "data"
        for d in (dist_dir, wheel_site, work_dir, cfg_dir, data_dir):
            d.mkdir(parents=True, exist_ok=True)

        build_cmd = [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
            str(REPO_ROOT),
        ]
        print(f"Building wheel: {' '.join(build_cmd)}")
        build_proc = subprocess.run(build_cmd, cwd=str(REPO_ROOT), check=False)
        if build_proc.returncode != 0:
            return build_proc.returncode

        wheels = sorted(dist_dir.glob("alphaengine-*.whl"))
        if not wheels:
            print("FAIL: no alphaengine-*.whl artifact produced.", file=sys.stderr)
            return 1
        wheel_path = wheels[0]
        print(f"Unpacking wheel: {wheel_path.name}")
        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(wheel_site)

        env = dict(os.environ)
        env["AE_REPO_ROOT"] = str(REPO_ROOT)
        env["AE_WHEEL_SITE"] = str(wheel_site)
        env["AE_WORK_DIR"] = str(work_dir)
        env["XDG_CONFIG_HOME"] = str(cfg_dir)
        env["APPDATA"] = str(cfg_dir)
        env["XDG_DATA_HOME"] = str(data_dir)
        env["LOCALAPPDATA"] = str(data_dir)
        env.pop("PYTHONPATH", None)

        run_proc = subprocess.run(
            [sys.executable, "-c", _ISOLATED_RUNNER],
            cwd=str(work_dir),
            env=env,
            check=False,
        )
        return int(run_proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
