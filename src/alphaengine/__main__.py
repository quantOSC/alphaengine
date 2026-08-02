"""`python -m alphaengine` — the entry point that always exists.

THE CONSOLE SCRIPT IS NOT ALWAYS ON PATH and that is not a bug we can fix. The
compute steps run in-process against your DataFrames, so this has to live in
your project venv, and a venv's `Scripts/`/`bin/` is on PATH only while it is
activated. Somebody who has just run `pip install -e .` and typed `alphaengine`
gets "command not found", which reads as a broken install and is not one.

`python -m alphaengine` needs no PATH entry at all — the interpreter that can
import the package can always run it, which makes this the form that works in
every shell, in a Dockerfile, in a Makefile, and inside a venv nobody activated:

    .venv/Scripts/python.exe -m alphaengine        # Windows, unactivated
    .venv/bin/python -m alphaengine                # POSIX, unactivated
    python -m alphaengine                          # anywhere, activated

It is deliberately a one-line delegation. Two entry points that could disagree
about anything would be worse than the PATH problem they solve.
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
