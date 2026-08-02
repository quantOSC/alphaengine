"""The terminal entry point.

WHAT THESE GUARD, in order of how much it would cost to get wrong:

  1. `import alphaengine` still touches no networking code. The CLI imports the
     client, so the lazy-import discipline in `cli.py` is now load-bearing for a
     promise made on the front page of the README.
  2. A stop exits 0. "This did not clear the bar" is the system working, and a
     non-zero exit would make every CI pipeline treat an honest refusal as a
     broken build — which is the pressure that gets honesty controls switched
     off.
  3. Offline says so, plainly, once, and points at the half that still works.
  4. Colour never reaches a pipe.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from alphaengine import cli

# ── the offline guarantee, which the CLI must not break ────────────────────


def test_importing_the_package_still_pulls_in_no_client():
    """`cli.py` imports the client INSIDE functions for exactly this reason."""
    out = subprocess.run(
        [sys.executable, "-c", "import alphaengine, sys; print('alphaengine.client' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False", "importing the package dragged in the client"


def test_importing_the_cli_module_alone_makes_no_network_import():
    """Even importing `alphaengine.cli` must not reach the transport: the
    console-script shim imports this module before parsing a single argument."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import alphaengine.cli, sys; "
            "print(any(m in sys.modules for m in ('httpx','requests','aiohttp')))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


# ── the project module ─────────────────────────────────────────────────────


def test_a_missing_project_module_says_what_to_pass(tmp_path, monkeypatch):
    """The realistic mistake is passing a FILE PATH, so the error names the fix."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(cli.ProjectError) as e:
        cli.load_project("research/momentum.py")
    assert "MODULE PATH" in str(e.value)


def test_a_project_module_without_data_or_backtest_is_refused(tmp_path, monkeypatch):
    (tmp_path / "empty_proj.py").write_text("x = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(cli.ProjectError) as e:
        cli.load_project("empty_proj")
    assert "neither" in str(e.value)


def test_a_project_module_hands_over_data_and_backtest(tmp_path, monkeypatch):
    (tmp_path / "good_proj.py").write_text(
        "data = [1.0, 2.0, 3.0]\ndef backtest_fn(*, data, fast=1):\n    return data\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    data, fn = cli.load_project("good_proj")
    assert data == [1.0, 2.0, 3.0]
    assert callable(fn)


def test_no_project_is_a_valid_state():
    """Not every workflow sweeps. A run that needs no local data must not be
    forced to invent a module."""
    assert cli.load_project(None) == (None, None)


# ── exit codes carry meaning ───────────────────────────────────────────────


class _Run:
    def __init__(self, status, stopped=None, artifact=None):
        self.status = status
        self.stopped = stopped
        self.artifact = artifact
        self.run_id = "r_1"


def test_a_stop_exits_zero():
    """THE ONE THAT MATTERS. A refusal is a result, not a build failure."""
    assert cli._report(_Run("stopped", stopped={"reason": "the surface is a knife edge"})) == 0


def test_a_closed_run_exits_zero():
    assert cli._report(_Run("closed", artifact={"workflow": "validate_study@1.0.0"})) == 0


def test_an_abandoned_run_exits_nonzero():
    """This one IS a failure: the build could not execute a step it was given."""
    assert cli._report(_Run("abandoned", stopped={"op": "compute.nope"})) == 1


# ── output discipline ──────────────────────────────────────────────────────


def test_colour_never_reaches_a_pipe():
    """Captured output is not a TTY, so nothing here may contain an escape."""
    out = subprocess.run(
        [sys.executable, "-m", "alphaengine.cli", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "\033[" not in out.stdout
    assert out.stdout.startswith("alphaengine ")


# The line that actually broke: `run` prints `<workflow> · <url>` BEFORE it
# contacts anything, so an unreachable URL still exercises it.
_RUN_HEADER = ["-m", "alphaengine.cli", "run", "validate_study", "--url", "http://127.0.0.1:9"]


def test_separators_degrade_to_ascii_on_a_stream_that_cannot_encode_them():
    """OBSERVED ON WINDOWS, not theorised.

    `alphaengine run validate_study` printed `validate_study � https://...`
    because stdout defaulted to a legacy code page. The glyphs are decoration
    and the words either side carry the meaning, so a stream that cannot encode
    them gets ASCII rather than a replacement character — or, worse, raises
    UnicodeEncodeError and takes the run down over a separator.
    """
    out = subprocess.run(
        [sys.executable, *_RUN_HEADER],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "ascii:strict"},
    )
    # It must not have died encoding its own output.
    assert b"UnicodeEncodeError" not in out.stderr, out.stderr.decode("ascii", "replace")
    text = out.stdout.decode("ascii")  # raises if a non-ASCII byte got through
    assert "validate_study" in text
    assert "�" not in text


def test_the_glyphs_are_the_real_ones_when_the_stream_can_take_them():
    """The fallback must not become permanent — a UTF-8 terminal gets the mid-dot."""
    out = subprocess.run(
        [sys.executable, *_RUN_HEADER],
        capture_output=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert "·".encode() in out.stdout, out.stdout.decode("utf-8", "replace")


def test_offline_names_the_host_and_the_half_that_still_works():
    """A workflow server being absent is not an error state for this package."""
    out = subprocess.run(
        [sys.executable, "-m", "alphaengine.cli", "workflows", "--url", "http://127.0.0.1:9"],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 2
    assert "127.0.0.1:9" in out.stdout
    assert "sweep" in out.stdout, "offline must point at the half that needs no server"


def test_the_parser_offers_the_documented_commands():
    p = cli.build_parser()
    for argv in (["version"], ["workflows"], ["run", "validate_study"]):
        assert p.parse_args(argv).command == argv[0]
    # A bare invocation opens a session rather than printing usage and leaving.
    assert p.parse_args([]).command is None
