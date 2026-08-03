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


def test_a_project_s_declared_grid_is_recorded_for_the_run(tmp_path, monkeypatch):
    """The grid is the caller's parameter space and the trial count is derived
    from it. Until it travelled, `validate_study` received `grid={}` and the
    sweep failed on the built-in example."""
    (tmp_path / "grid_proj.py").write_text(
        "data = [1.0, 2.0]\nGRID = {'fast': [5, 10], 'slow': [50]}\n"
        "def backtest_fn(*, data, fast=1, slow=2):\n    return data\n"
    )
    (tmp_path / "gridless_proj.py").write_text("data = [1.0, 2.0]\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))

    cli.load_project("grid_proj")
    assert cli.project_grid() == {"fast": [5, 10], "slow": [50]}
    # A gridless module CLEARS the record — a stale grid from the previous
    # project would attach the wrong parameter space to the next run.
    cli.load_project("gridless_proj")
    assert cli.project_grid() is None


def test_the_shipped_demo_declares_the_grid_the_readme_documents():
    from alphaengine import demo

    cli.load_project("alphaengine.demo")
    assert cli.project_grid() == demo.GRID


# ── one name out of a loaded universe ──────────────────────────────────────


def test_a_symbol_turns_a_universe_into_its_return_series():
    """The loop with no exit: told `size_position needs returns`, the user
    loaded their universe, passed it again, and got the same refusal. A named
    symbol's closes become the one series the workflow measures."""
    prices = {"MU": [100.0, 110.0, 99.0], "AMD": [50.0, 55.0]}
    got = cli._returns_from_universe(prices, "mu")
    assert got == [0.1, -0.1]
    # {date, close} rows — the shape a wide CSV loads as — work identically.
    rows = {"MU": [{"date": "a", "close": 100.0}, {"date": "b", "close": 110.0}]}
    assert cli._returns_from_universe(rows, "MU") == [0.1]
    assert cli._returns_from_universe(prices, "TSLA") is None
    assert cli._returns_from_universe([1.0, 2.0], "MU") is None


def test_a_question_naming_one_symbol_picks_it_and_two_is_ambiguity():
    data = {"MU": [1.0], "AMD": [1.0]}
    assert cli._symbol_in("size a position in MU for me", data) == "MU"
    assert cli._symbol_in("compare MU and AMD", data) is None
    assert cli._symbol_in("size something", data) is None


def test_preflight_with_a_universe_loaded_names_the_symbol_move():
    """The refusal must not send somebody to load the thing they loaded."""
    catalogue = [{"name": "size_position", "requires": ["returns"]}]
    prices = {"MU": [100.0, 110.0], "AMD": [50.0, 55.0]}
    gap = cli.preflight(catalogue, "size_position", data=prices, backtest_fn=None)
    assert gap is not None
    assert "--symbol AMD" in gap
    # The generic "Load some" doors are the loop with no exit; they must not
    # render when a universe is already loaded.
    assert "Load some" not in gap


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


# ── the boot screen ────────────────────────────────────────────────────────


def test_python_dash_m_works_without_the_console_script_on_path():
    """THE ENTRY POINT THAT ALWAYS EXISTS.

    The console script lives in the venv's Scripts/bin directory and is on PATH
    only while that venv is activated, so a fresh `pip install -e .` followed by
    `alphaengine` gives "command not found" — which reads as a broken install.
    `python -m alphaengine` needs no PATH entry at all.
    """
    out = subprocess.run(
        [sys.executable, "-m", "alphaengine", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.startswith("alphaengine ")


def test_the_banner_is_suppressed_when_stdout_is_not_a_tty():
    """A CI log wants a transcript, not an ANSI painting."""
    out = subprocess.run(
        [sys.executable, "-m", "alphaengine", "version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "▁" not in out.stdout  # no block glyphs
    assert "╭" not in out.stdout  # no box corner
    assert len(out.stdout.strip().splitlines()) == 1


def test_boot_never_prints_the_caller_s_data(capsys, monkeypatch):
    """THE ONE THAT MATTERS ON THIS SCREEN.

    A boot banner that echoes the first rows of somebody's price series puts
    their data in the scrollback, in any recording of the session, and in any
    screenshot they paste into a ticket. Shape only — never values.
    """
    monkeypatch.setattr(cli, "_tty", lambda: True)
    secret = 1234.5678
    cli.boot(
        "https://example.invalid",
        project="research.momentum",
        data={"AAPL": [secret] * 100, "MSFT": [secret] * 100},
        keyed=True,
    )
    printed = capsys.readouterr().out
    assert "1234" not in printed, "a price reached the boot screen"
    assert "AAPL" not in printed, "a ticker reached the boot screen"
    assert "2 series" in printed, "the SHAPE should still be reported"


def test_boot_warns_when_there_is_no_key(capsys, monkeypatch):
    """Discovered at boot it costs nothing; discovered mid-run it costs the run."""
    monkeypatch.setattr(cli, "_tty", lambda: True)
    cli.boot("https://example.invalid", project=None, data=None, keyed=False)
    printed = capsys.readouterr().out
    assert "QUANTOS_API_KEY" in printed
    assert "key quantos" in printed


def test_a_long_server_url_does_not_overflow_the_box(monkeypatch):
    """The default URL is 51 characters and broke the box on a fresh install —
    the first thing an unconfigured user would ever see, misrendered."""
    monkeypatch.setattr(cli, "_tty", lambda: True)
    rows = [("server", "https://alpha-backend-production-51df.up.railway.app")]
    lines = cli._boxed(rows, width=62)
    widths = {cli._visible_len(ln) for ln in lines}
    assert widths == {62}, f"box lines are ragged: {sorted(widths)}"


def test_fit_keeps_both_ends_because_the_tail_identifies_the_host():
    got = cli._fit("https://alpha-backend-production-51df.up.railway.app", 30)
    assert cli._visible_len(got) == 30
    assert got.startswith("https://")
    assert got.endswith("railway.app")


def test_fit_preserves_colour_codes():
    """Truncation counts VISIBLE characters; an escape sequence costs no width
    and must survive intact or the rest of the line inherits the colour."""
    coloured = "\033[32m" + "x" * 80 + "\033[0m"
    got = cli._fit(coloured, 10)
    assert cli._visible_len(got) == 10
    assert got.startswith("\033[32m") and got.endswith("\033[0m")


def test_the_two_marks_are_the_same_width():
    """They sit on consecutive lines and the text after them must start at the
    same column, or the comparison reads as a misprint."""
    assert len(cli._PLATEAU_U) == len(cli._EDGE_U)
    assert len(cli._PLATEAU_A) == len(cli._EDGE_A)


def test_the_parser_offers_the_documented_commands():
    p = cli.build_parser()
    for argv in (["version"], ["workflows"], ["run", "validate_study"]):
        assert p.parse_args(argv).command == argv[0]
    # A bare invocation opens a session rather than printing usage and leaving.
    assert p.parse_args([]).command is None


# ── the capability ladder ──────────────────────────────────────────────────
#
# THREE RUNGS AND THE FIRST IS FREE FOREVER. The rung a user is on has to be
# visible at boot, because "No API key" with no further detail reads as "this
# tool does not work" when the truth is "two thirds of it works and always
# will".


def _ladder(monkeypatch, *, keyed: bool, model: str | None):
    monkeypatch.setattr(cli, "_tty", lambda: True)
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    if model:
        monkeypatch.setenv(model, "sk-test")
    return "\n".join(cli.ladder_lines(keyed=keyed))


def test_the_math_rung_is_unlocked_with_no_keys_at_all(monkeypatch):
    """The open core is not a trial and never expires."""
    out = _ladder(monkeypatch, keyed=False, model=None)
    assert "yours, offline, always" in out


def test_without_a_quantos_key_the_ladder_names_the_variable(monkeypatch):
    out = _ladder(monkeypatch, keyed=False, model=None)
    # The rungs stay in plain language; the ONE actionable line names the
    # command and the variable, so a locked reader is never left guessing.
    assert "key quantos" in out
    assert "QUANTOS_API_KEY" in out
    # And the agent rung says WHY it is out of reach, rather than naming a
    # second credential the user cannot use yet.
    assert "sign in first" in out


def test_with_a_quantos_key_the_agent_rung_names_the_model_variable(monkeypatch):
    out = _ladder(monkeypatch, keyed=True, model=None)
    assert "key anthropic" in out or "key openai" in out


def test_with_both_keys_every_rung_is_unlocked(monkeypatch):
    """The SDK is stubbed present because this is testing the LADDER, not this
    machine's install state. Before 2026-08-02 the rung lit on the env var
    alone, which meant it lit on a provider that would then raise -- a boot
    screen that lied, when telling a user what they cannot do is its whole job."""
    from alphaengine import model

    monkeypatch.setattr(model, "_sdk_present", lambda _label: True)
    out = _ladder(monkeypatch, keyed=True, model="ANTHROPIC_API_KEY")
    assert out.count("ready") >= 2
    assert "anthropic" in out


def test_a_model_key_alone_does_not_unlock_the_agent(monkeypatch):
    """The rungs nest: the agent drives the harness, so it cannot come first."""
    out = _ladder(monkeypatch, keyed=False, model="ANTHROPIC_API_KEY")
    assert "sign in first" in out


def test_the_model_adapter_reports_nothing_available_with_no_keys(monkeypatch):
    from alphaengine.model import NoModelConfigured, available_models, build_think

    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    assert available_models() == []
    with pytest.raises(NoModelConfigured) as e:
        build_think()
    # The message has to say the free path still works, or it reads as a wall.
    assert "scripted path" in str(e.value)


def test_importing_the_model_adapter_pulls_in_no_provider_sdk():
    """`pip install alphaengine` installs numpy and scipy. Nothing else."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import alphaengine.model, sys; "
            "print(any(m in sys.modules for m in ('anthropic','openai','httpx')))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


# ── the demo, which must reach a pip-only install ──────────────────────────


def test_the_demo_ships_inside_the_package():
    """It lived in `examples/`, which the wheel does not carry — so a
    `pip install` gave you a CLI with nothing to point `--project` at, and the
    README's first instruction was `git clone`."""
    out = subprocess.run(
        [sys.executable, "-m", "alphaengine", "demo"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "derived_from_grid" in out.stdout
    assert "9" in out.stdout  # the grid is 3x3 and the count is derived


def test_the_demo_is_also_a_valid_project_module():
    """So `--project alphaengine.demo` works without writing a module first."""
    from alphaengine import cli as c

    data, fn = c.load_project("alphaengine.demo")
    assert data is not None and callable(fn)


# ── authentication is a prompt, not a dead end ─────────────────────────────


class _Refusal(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def test_a_401_explains_that_a_browser_login_is_not_a_cli_credential(capsys, monkeypatch):
    """THE REPORTED BUG. A paying user, signed into the portal, read
    "Authentication required" as the product being broken. It is not: this
    process has no cookie jar and should never acquire one."""
    monkeypatch.setattr(cli, "_tty", lambda: True)
    cli.refused(_Refusal(401, "Authentication required"))
    out = capsys.readouterr().out
    assert "portal login does not reach this process" in out
    assert "Data" in out and "API keys" in out, "it must say where the key comes from"
    assert "key quantos" in out, "and how to enter one"


def test_a_403_is_treated_the_same_way(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_tty", lambda: True)
    cli.refused(_Refusal(403, "Forbidden"))
    assert "Not authenticated" in capsys.readouterr().out


def test_a_non_auth_refusal_keeps_the_plain_message(capsys, monkeypatch):
    """The fix for a 422 is not a credential, so it must not offer one."""
    monkeypatch.setattr(cli, "_tty", lambda: True)
    cli.refused(_Refusal(422, "workflow inputs invalid"))
    out = capsys.readouterr().out
    assert "workflow inputs invalid" in out
    assert "key quantos" not in out


def test_is_auth_error_distinguishes_the_two_cases():
    assert cli.is_auth_error(_Refusal(401, "x")) is True
    assert cli.is_auth_error(_Refusal(403, "x")) is True
    assert cli.is_auth_error(_Refusal(404, "x")) is False
    assert cli.is_auth_error(_Refusal(500, "x")) is False


def test_the_key_prompt_never_fires_on_a_pipe(monkeypatch):
    """A prompt in a CI log is a hang, not a question."""
    monkeypatch.setattr(cli, "_tty", lambda: False)

    def _boom(*a, **k):
        raise AssertionError("the CLI tried to prompt on a non-interactive stream")

    monkeypatch.setattr("builtins.input", _boom)
    assert cli.offer_key("quantos") is False


# ── staying signed in ──────────────────────────────────────────────────────


def test_credentials_never_land_in_the_project_directory(tmp_path, monkeypatch):
    """A key in a project folder is a key in somebody's git history eventually."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    from alphaengine import auth

    p = auth.save_key("QUANTOS_API_KEY", "ae_live_secret")
    assert (tmp_path / "cfg") in p.parents
    assert "credentials.json" not in [f.name for f in tmp_path.iterdir() if f.is_file()]


def test_a_stored_key_is_applied_but_never_overrides_the_shell(tmp_path, monkeypatch):
    """CI, a VPC deploy and a colleague's machine all set the variable. A stale
    file silently winning is a bug that takes a day to find."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from alphaengine import auth

    auth.save_key("QUANTOS_API_KEY", "from_file")

    # `apply_stored` writes os.environ directly, which monkeypatch does not
    # track — so this test cleans up after itself or it leaks a credential into
    # every test that runs later. (It did: test_smoke started failing.)
    monkeypatch.delenv("QUANTOS_API_KEY", raising=False)
    try:
        assert auth.apply_stored() == ["QUANTOS_API_KEY"]
        assert os.environ["QUANTOS_API_KEY"] == "from_file"

        os.environ["QUANTOS_API_KEY"] = "from_shell"
        assert auth.apply_stored() == []
        assert os.environ["QUANTOS_API_KEY"] == "from_shell"
    finally:
        os.environ.pop("QUANTOS_API_KEY", None)


def test_logout_deletes_the_file_rather_than_flagging_it(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from alphaengine import auth

    auth.save_key("QUANTOS_API_KEY", "x")
    assert auth.config_path().exists()
    assert auth.clear_stored() is True
    assert not auth.config_path().exists()
    assert auth.clear_stored() is False  # idempotent


def test_a_corrupt_credentials_file_signs_you_out_rather_than_crashing(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from alphaengine import auth

    auth.config_path().parent.mkdir(parents=True, exist_ok=True)
    auth.config_path().write_text("{not json", encoding="utf-8")
    assert auth.load_stored() == {}


def test_only_known_credentials_can_be_stored(tmp_path, monkeypatch):
    """The file is not a general-purpose config store."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from alphaengine import auth

    with pytest.raises(ValueError):
        auth.save_key("AWS_SECRET_ACCESS_KEY", "nope")


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits; Windows uses profile ACLs")
def test_the_file_is_mode_0600(tmp_path, monkeypatch):
    """Created with those bits, not chmod'ed after — a file that was briefly
    world-readable was briefly world-readable."""
    import stat as _stat

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from alphaengine import auth

    p = auth.save_key("QUANTOS_API_KEY", "x")
    assert _stat.S_IMODE(p.stat().st_mode) == 0o600


# ── preflight: refuse before the run, not after it churns ──────────────────

_CAT = [
    {"name": "validate_study", "version": "1.0.0", "requires": ["data", "backtest_fn"]},
    {"name": "no_inputs", "version": "1.0.0", "requires": []},
]


def test_a_sweep_workflow_without_a_backtest_is_refused_up_front():
    """THE FAILURE THIS REPLACES: the run started, `compute.sweep` raised, the
    server correctly re-offered the step, the second identical failure abandoned
    it — and the user watched a workflow grind and die naming an op they had
    never heard of. It was knowable before the first request."""
    gap = cli.preflight(_CAT, "validate_study", data=None, backtest_fn=None)
    assert gap and "backtest_fn" in gap
    # And it says what to actually type, including a path that needs nothing.
    assert "--project" in gap
    # The example is `alphaengine demo` now, not a --project pointed at the demo
    # module: one word, no arguments, and it is the fastest way to see the thing
    # work at all.
    assert "alphaengine demo" in gap


def test_data_alone_is_not_enough_for_a_sweep():
    gap = cli.preflight(_CAT, "validate_study", data={"close": [1.0]}, backtest_fn=None)
    assert gap and "backtest_fn" in gap and "`data`" not in gap


def test_nothing_missing_means_no_complaint():
    assert cli.preflight(_CAT, "validate_study", data={"c": [1]}, backtest_fn=lambda **k: []) is None


def test_a_workflow_needing_nothing_runs_with_nothing():
    assert cli.preflight(_CAT, "no_inputs", data=None, backtest_fn=None) is None


def test_an_unknown_workflow_lists_what_the_workspace_offers():
    gap = cli.preflight(_CAT, "nope", data=None, backtest_fn=None)
    assert gap and "validate_study" in gap and "no_inputs" in gap


def test_an_unpublished_requirement_is_not_invented():
    """A server that publishes no `requires` must not be second-guessed by the
    client — the contract is the server's to state."""
    assert cli.preflight([{"name": "x", "version": "1"}], "x", data=None, backtest_fn=None) is None


# ── preflight knows the SHAPE, not only the presence ───────────────────────
#
# `universe` and `returns` are both "data". A caller who brings the wrong one
# used to discover it from an UnsupportedOp raised inside a running workflow,
# which is the failure preflight exists to prevent, arriving one level finer.

_SHAPED = [
    {"name": "screen_universe", "version": "1.0.0", "requires": ["universe"]},
    {"name": "size_position", "version": "1.0.0", "requires": ["returns"]},
]


def test_a_return_series_pointed_at_a_screen_is_refused_up_front():
    gap = cli.preflight(_SHAPED, "screen_universe", data=[0.01, -0.02, 0.03], backtest_fn=None)
    assert gap and "universe" in gap
    # And it says what the shape should be, once, in the message that fires.
    assert "{symbol: prices}" in gap


def test_a_universe_pointed_at_a_sizing_workflow_is_refused_up_front():
    """And the refusal names the MOVE, not a re-load of the thing already
    loaded — "load a universe" to somebody with a universe loaded was a loop
    with no exit, which is how this message read in production."""
    universe = {"AAPL": [1.0, 2.0], "MSFT": [3.0, 4.0]}
    gap = cli.preflight(_SHAPED, "size_position", data=universe, backtest_fn=None)
    assert gap and "ONE return series" in gap
    assert "--symbol AAPL" in gap


def test_the_right_shape_passes_without_comment():
    assert cli.preflight(_SHAPED, "screen_universe", data={"AAPL": [1.0, 2.0]}, backtest_fn=None) is None
    assert cli.preflight(_SHAPED, "size_position", data=[0.01, -0.02], backtest_fn=None) is None


def test_returns_under_a_key_on_a_mapping_still_counts_as_returns():
    """The two spellings a research module actually uses. A mapping is not
    automatically a universe."""
    for key in ("returns", "pnl"):
        assert cli.preflight(_SHAPED, "size_position", data={key: [0.01, 0.02]}, backtest_fn=None) is None
        gap = cli.preflight(_SHAPED, "screen_universe", data={key: [0.01, 0.02]}, backtest_fn=None)
        assert gap, f"a {key} mapping is not a universe"


def test_nothing_loaded_still_names_every_gap():
    gap = cli.preflight(_SHAPED, "screen_universe", data=None, backtest_fn=None)
    assert gap and "universe" in gap and "--project" in gap


# ── the run answers the question it was asked ──────────────────────────────
#
# Before this the agentic path produced step narration and a verdict and NO
# ANSWER, because no call had saying anything as its job. `_answer` is that
# call, and these hold the two properties that make it safe to have: it is a
# separate pass from the one that drove the run, and it cannot quote a figure
# the run did not produce.


class _Answered:
    def __init__(self, figures, stopped=None):
        self.figures = figures
        self.stopped = stopped


_FIGS = {
    "sweep": {"n_trials": 240, "n_trials_source": "derived_from_grid"},
    "validate": {"deflated_sharpe": 0.61},
}


def test_the_answer_reaches_the_screen(capsys):
    cli._answer("did it hold up?", _Answered(_FIGS), lambda _p: "A deflated Sharpe of 0.61 over 240 trials.")
    out = capsys.readouterr().out
    assert "0.61" in out and "240" in out


def test_the_caveats_reach_the_screen_with_it(capsys):
    """Every honesty control in this product exists to survive exactly this last
    step. An answer rendered without them has reintroduced the failure."""
    run = _Answered({"v": {"n_trials": None, "n_trials_source": "not_recorded", "sharpe_annualized": 1.4}})
    cli._answer("good?", run, lambda _p: "The annualised Sharpe is 1.4.")
    out = capsys.readouterr().out
    assert "1.4" in out
    assert "trial count" in out.lower()
    assert "inconclusive" in out


def test_an_invented_figure_is_named_rather_than_swallowed(capsys):
    """A user who sees nothing cannot tell the guard firing from a model that
    had nothing to say."""
    cli._answer("how did it do?", _Answered(_FIGS), lambda _p: "A Sharpe of 3.99.")
    out = capsys.readouterr().out
    assert "refused" in out.lower()
    assert "3.99" in out


def test_a_stop_is_printed_as_written_and_the_model_is_not_asked(capsys):
    def never(_prompt):
        raise AssertionError("the model was asked to rewrite a refusal")

    cli._answer(
        "is it good?",
        _Answered({"resolve": {"n_obs": 30}}, stopped={"reason": "The series is too short."}),
        never,
    )
    assert "too short" in capsys.readouterr().out


def test_the_prompt_carries_the_figures_and_not_the_workflow():
    seen = {}

    def capture(prompt):
        seen["p"] = prompt
        return "240 trials ran."

    cli._answer("what happened?", _Answered(_FIGS), capture)
    assert "sweep.n_trials = 240" in seen["p"]
    assert "what happened?" in seen["p"]
    for leak in ("stage", "gate", "threshold"):
        assert leak not in seen["p"].lower()


def test_a_broken_model_loses_the_answer_and_not_the_run(capsys):
    """The figures are already printed by `_report`, so the worst case is the
    behaviour that shipped before this existed."""

    def boom(_prompt):
        raise RuntimeError("model is down")

    cli._answer("?", _Answered(_FIGS), boom)
    assert "No answer could be composed" in capsys.readouterr().out


# ── three doors for data, not one ──────────────────────────────────────────
#
# `--project` was the ONLY way in. It is the right door for a quant with a
# research package and the wrong one for the same quant on the day somebody
# emails them a CSV. Three of the four workflows want numbers, not code.


class _Universes:
    """A session whose universes are registered BROWSER-ONLY: symbols, no stored
    series. The series call 404s, which is what the portal does for one of
    these — stubbing it as missing entirely would pass the test with an
    AttributeError for the wrong reason."""

    def __init__(self, rows):
        self._rows = rows

    def universes(self):
        return self._rows

    def universe_series(self, universe_id, *, window=0):
        from alphaengine.client import ServerError

        raise ServerError(404, "Universe has no stored series (browser-only registration)")


def _args(**kw):
    import argparse

    fields = {"project": None, "data": None, "universe": None}
    fields.update(kw)
    return argparse.Namespace(**fields)


def test_a_csv_becomes_the_runs_data(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text("date,AAPL,MSFT\n2026-01-01,100,200\n2026-01-02,101,201\n", encoding="utf-8")
    data, backtest_fn = cli.resolve_data(_args(data=str(p)))
    assert set(data) == {"AAPL", "MSFT"}
    # No simulator, and that is correct: a file cannot carry one, which is why
    # `validate_study` still needs --project and the other three do not.
    assert backtest_fn is None


def test_a_universe_narrows_the_file_to_the_names_it_lists(tmp_path, capsys):
    p = tmp_path / "prices.csv"
    p.write_text("date,AAPL,MSFT,NVDA\n2026-01-01,100,200,300\n", encoding="utf-8")
    session = _Universes([{"id": "u1", "name": "core", "symbols": ["AAPL", "NVDA", "TSLA"]}])

    data, _ = cli.resolve_data(_args(data=str(p), universe="core"), session)
    assert set(data) == {"AAPL", "NVDA"}
    # SAID, NOT SWALLOWED. A universe of 500 screened against a file holding 40
    # is a different result from a universe of 40.
    assert "TSLA" in capsys.readouterr().out


def test_a_universe_may_be_named_by_id_too(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text("date,AAPL\n2026-01-01,100\n", encoding="utf-8")
    session = _Universes([{"id": "u_7", "name": "core", "symbols": ["AAPL"]}])
    data, _ = cli.resolve_data(_args(data=str(p), universe="u_7"), session)
    assert set(data) == {"AAPL"}


def test_a_universe_with_no_stored_series_and_no_file_says_what_to_do():
    """SUPERSEDED 2026-08-02, and the reason is worth keeping.

    This used to assert that `--universe` alone ALWAYS failed, on the reasoning
    that a universe is a definition and never a series. The first half is right
    and the conclusion was too broad: a universe registered WITH prices has them
    stored in the portal, encrypted, and the portal decrypts them back to the
    same account. Refusing to fetch the user's own upload was not a data
    boundary, it was a missing function wearing one.

    What survives is this case — a browser-only registration, symbols and no
    stored series — where there genuinely is nothing to fetch.
    """
    session = _Universes([{"id": "u1", "name": "core", "symbols": ["AAPL"]}])
    with pytest.raises(ValueError) as e:
        cli.resolve_data(_args(universe="core"), session)
    assert "--data" in str(e.value)


def test_an_unknown_universe_lists_the_ones_you_have(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text("date,AAPL\n2026-01-01,100\n", encoding="utf-8")
    session = _Universes([{"id": "u1", "name": "core", "symbols": ["AAPL"]}])
    with pytest.raises(ValueError) as e:
        cli.resolve_data(_args(data=str(p), universe="nope"), session)
    assert "core" in str(e.value)


def test_a_universe_sharing_no_names_with_the_file_is_refused(tmp_path):
    p = tmp_path / "prices.csv"
    p.write_text("date,AAPL\n2026-01-01,100\n", encoding="utf-8")
    session = _Universes([{"id": "u1", "name": "core", "symbols": ["TSLA", "AMZN"]}])
    with pytest.raises(ValueError) as e:
        cli.resolve_data(_args(data=str(p), universe="core"), session)
    assert "none of the 2 names" in str(e.value)


def test_a_project_module_still_wins_where_they_overlap(tmp_path, monkeypatch):
    """The module is the more specific statement. Silently replacing it would
    make two flags fight where the user can see neither winning."""
    import sys
    import types

    mod = types.ModuleType("fake_project")
    mod.data = {"FROM_MODULE": [1.0, 2.0]}
    mod.backtest_fn = lambda **k: [0.01]
    monkeypatch.setitem(sys.modules, "fake_project", mod)

    p = tmp_path / "prices.csv"
    p.write_text("date,AAPL\n2026-01-01,100\n", encoding="utf-8")

    data, backtest_fn = cli.resolve_data(_args(project="fake_project", data=str(p)))
    assert set(data) == {"FROM_MODULE"}
    assert backtest_fn is not None


def test_nothing_supplied_is_not_an_error_here():
    """Preflight is what refuses a run with nothing loaded, and it does so
    naming the workflow. Raising here would pre-empt a better message."""
    assert cli.resolve_data(_args()) == (None, None)


# ── a configuration problem is not a crash ─────────────────────────────────
#
# THE BUG: `build_think` returned a closure whose SDK import ran on first call,
# so a machine with ANTHROPIC_API_KEY set and the SDK missing got a valid-looking
# callable and failed inside `driver.pick()` -- three frames past the
# `except NoModelConfigured` in `_ask` that exists to handle exactly this. The
# user saw two full tracebacks with the one actionable line buried underneath.


def test_a_key_with_no_sdk_raises_where_the_handler_is(monkeypatch):
    """At BUILD time, not on first call. A callable that cannot work is not a
    callable."""
    from alphaengine import model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(model, "_sdk_present", lambda _label: False)

    with pytest.raises(model.NoModelConfigured) as e:
        model.build_think()
    assert "pip install anthropic" in str(e.value)
    # And it points at the extra that installs everything the agent path needs.
    assert "alphaengine[agents]" in str(e.value)


def test_a_missing_sdk_falls_through_to_a_provider_that_works(monkeypatch):
    """A machine with both keys and only openai installed should USE openai
    rather than refusing."""
    from alphaengine import model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setattr(model, "_sdk_present", lambda label: label == "openai")

    _think, label = model.build_think()
    assert label.startswith("openai/")


def test_the_ladder_does_not_light_a_rung_that_would_raise(monkeypatch):
    """A boot screen that lights on a provider which then raises is a boot
    screen that lied, and telling a user what they cannot do is its whole job."""
    from alphaengine import model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(model, "_sdk_present", lambda _label: False)

    assert model.available_models() == []
    assert model.keys_without_sdk() == ["anthropic"]


def test_the_ladder_says_which_step_is_missing(monkeypatch):
    """ "Add your own model key" told somebody who already had one to go and get
    one, which reads as the product being broken."""
    from alphaengine import model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(model, "_sdk_present", lambda _label: False)

    text = "\n".join(cli.ladder_lines(keyed=True))
    assert "pip install anthropic" in text
    assert "add your own model key" not in text


def test_main_never_shows_a_traceback_for_a_configuration_problem(monkeypatch, capsys):
    """The backstop. No future path gets to put a traceback in front of a user
    for a missing key or a missing install."""
    from alphaengine import cli as cli_mod
    from alphaengine.model import NoModelConfigured

    def boom(_args):
        raise NoModelConfigured("a key is set and the SDK is not installed")

    # `main` restores the signed-in credential before dispatching, which reads
    # the REAL file on this machine and would leak QUANTOS_API_KEY into
    # os.environ for every test that runs after this one. Stubbed rather than
    # tolerated: a test that touches the developer's own credentials behaves
    # differently on CI, and this one made an unrelated smoke test fail.
    monkeypatch.setattr("alphaengine.auth.apply_stored", lambda: None)
    monkeypatch.setattr(cli_mod, "cmd_version", boom)

    code = cli_mod.main(["version"])
    out = capsys.readouterr().out
    assert code == 2
    assert "Traceback" not in out
    assert "SDK is not installed" in out


# ── the built-in example covers every workflow, not just the sweep ─────────
#
# The preflight message ends "or see it work on the built-in example first", and
# that was only true of `validate_study`: `demo.data` is ONE instrument, so
# pointing a screen at it produced a universe of one called "close" -- which the
# screen then refuses as a handful of names. A worked example that works for one
# of four workflows tells three quarters of readers the product is broken.


def test_the_demo_universe_is_a_universe_a_screen_would_accept():
    from alphaengine.core.screen import screen_universe
    from alphaengine.demo_universe import data

    assert len(data) >= 20, "below the screen's own floor, so the example would be refused"
    out = screen_universe(data, rank_by="return_pct", top_n=5)
    assert out["n_evaluated"] == out["universe_size"], "the example cannot be fully measured"
    assert len(out["rows"]) == 5


def test_the_demo_returns_are_a_series_the_measuring_ops_accept():
    from alphaengine.client import StepExecutor
    from alphaengine.demo_returns import data

    ex = StepExecutor(data=data)
    assert ex.execute("compute.performance_report", {})["n_obs"] == len(data)
    assert ex.execute("compute.compute_var_cvar", {})["parametric"]["var_pct"] > 0


def test_every_built_in_example_reproduces():
    """Fixed seeds. The example must give the same figures on every machine, or
    two people reading the same docs see different numbers."""
    from alphaengine.demo import _returns, _universe

    assert _universe() == _universe()
    assert _returns() == _returns()


def test_the_returns_come_from_the_universe_so_the_example_is_consistent():
    """A reader can screen the universe and then size a name the screen
    returned. Two unrelated fixtures would make that story a lie."""
    from alphaengine.demo import universe
    from alphaengine.demo_returns import data as returns

    assert len(returns) == len(universe["SYM42"]) - 1


def test_the_demo_modules_expose_exactly_what_project_reads():
    """`--project` reads `data` off a module. A demo module that exposed
    something else would be a worked example nobody can run."""
    from alphaengine import demo_returns, demo_universe

    for mod in (demo_universe, demo_returns):
        assert hasattr(mod, "data"), f"{mod.__name__} has nothing --project can read"


# ── a universe registered WITH prices travels to the OS ────────────────────
#
# THE GAP: `--universe` fetched the SYMBOL LIST and still demanded `--data` for
# the closes. So somebody who uploaded an S&P universe to the portal was told to
# go and find the same CSV on disk before the OS would look at it -- which is not
# a data boundary, it is a missing function wearing one.
#
# §9 is that WE never fetch market data on the user's behalf and never hold a
# series as our own. This is neither: it is their upload, encrypted at rest,
# decrypted back to the same account. The portal endpoint's own docstring calls
# it "another device of the SAME account", and the CLI is that device.


class _Portal:
    """A session that answers both universe calls."""

    def __init__(self, rows, series=None, status=None):
        self._rows = rows
        self._series = series or {}
        self._status = status
        self.series_calls = 0

    def universes(self):
        return self._rows

    def universe_series(self, universe_id, *, window=0):
        self.series_calls += 1
        if self._status:
            from alphaengine.client import ServerError

            raise ServerError(self._status, "no stored series")
        return {"universe_id": universe_id, "prices": self._series}


_SP = [{"id": "u_sp", "name": "sp500", "symbols": ["AAPL", "MSFT", "NVDA"]}]
_PRICES = {"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0], "NVDA": [300.0, 301.0]}


def test_a_universe_with_stored_prices_needs_no_local_file(capsys):
    """THE ONE THAT WAS BROKEN."""
    session = _Portal(_SP, series=_PRICES)
    data, _ = cli.resolve_data(_args(universe="sp500"), session)

    assert set(data) == {"AAPL", "MSFT", "NVDA"}
    assert session.series_calls == 1
    out = capsys.readouterr().out
    # It says where the data came from. A run whose source is invisible is the
    # shape this codebase keeps finding bugs in.
    assert "from the portal" in out and "3 names" in out


def test_a_local_file_still_wins_over_the_stored_copy(tmp_path):
    """`--data` is the more explicit statement, and silently preferring a stored
    copy over the file somebody just pointed at is helpfulness nobody can
    debug."""
    p = tmp_path / "mine.csv"
    p.write_text("date,AAPL,MSFT\n2026-01-01,1,2\n", encoding="utf-8")

    session = _Portal(_SP, series=_PRICES)
    data, _ = cli.resolve_data(_args(data=str(p), universe="sp500"), session)

    assert session.series_calls == 0, "it fetched when a local file was supplied"
    assert data["AAPL"][0]["close"] == 1.0, "the stored copy overwrote the local file"


def test_a_browser_only_universe_says_what_is_missing():
    """Symbols with no stored series is a real and common registration, not a
    fault, and the message has to say which one this is."""
    session = _Portal(_SP, status=404)
    with pytest.raises(ValueError) as e:
        cli.resolve_data(_args(universe="sp500"), session)
    message = str(e.value)
    assert "names on your account and not its prices" in message
    # BOTH REPAIRS, because they are for two different people: the one who can
    # fix it on the portal in a click, and the one who has the CSV right here.
    assert "Store on my account" in message
    assert "--data" in message


def test_a_server_error_that_is_not_a_404_is_not_swallowed():
    """A 500 from the portal is an outage and must not be reported as "you
    registered this without prices"."""
    from alphaengine.client import ServerError

    session = _Portal(_SP, status=503)
    with pytest.raises(ServerError):
        cli.resolve_data(_args(universe="sp500"), session)


# ── the session accepts what its own messages tell you to type ─────────────
#
# THE LOOP, reported from a real terminal and the worst kind of bug there is.
# `run screen_universe --universe sp100` inside the session was not parsed as a
# run at all: the guard wanted a SINGLE TOKEN after `run`, so anything with a
# flag fell through to the agent as a question. The agent then correctly chose
# screen_universe, preflight correctly refused for want of data, and the refusal
# printed `alphaengine run screen_universe --universe <name>` as the remedy --
# which is what had just been typed.
#
# A tool that answers a refusal with instructions it cannot itself accept has
# stopped being a tool.


def test_run_with_flags_is_a_run_and_not_a_question():
    """The parse that was missing."""
    assert cli._split_run("screen_universe --universe sp100") == (
        "screen_universe",
        ["--universe", "sp100"],
    )
    assert cli._split_run("screen_universe") == ("screen_universe", [])
    assert cli._split_run("") == ("", [])


def test_the_session_loads_a_universe_from_a_run_flag():
    session = _Portal(_SP, series=_PRICES)
    data, _ = cli._apply_flags(["--universe", "sp500"], session, None, None)
    assert set(data) == {"AAPL", "MSFT", "NVDA"}


def test_the_session_loads_a_file_from_a_run_flag(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("date,AAPL,MSFT" + chr(10) + "2026-01-01,1,2" + chr(10), encoding="utf-8")
    data, _ = cli._apply_flags(["--data", str(p)], _Portal(_SP), None, None)
    assert set(data) == {"AAPL", "MSFT"}


def test_a_flag_the_session_does_not_know_says_which_it_takes():
    """And does not exit the process. A session that dies on a typo is worse
    than one that says so."""
    with pytest.raises(ValueError) as e:
        cli._apply_flags(["--nope", "x"], _Portal(_SP), None, None)
    assert "--universe" in str(e.value) and "--data" in str(e.value)


def test_the_refusal_speaks_the_session_s_language_not_the_shell_s():
    """THE LOOP-BREAKER. Inside the session `alphaengine` is not a word the
    prompt knows, so printing shell syntax sent the user round again."""
    gap = "screen_universe needs `universe`, and none was loaded." + chr(10) + "  more"
    out = cli._repl_gap(gap, "screen_universe")

    assert "universe <name>" in out
    assert "data <file.csv>" in out
    assert "run screen_universe --universe" in out
    # The one thing it must NOT say in here.
    assert "alphaengine run" not in out


def test_the_source_is_named_so_the_status_line_can_show_it():
    assert cli._describe_source(["--universe", "sp100"]) == "universe:sp100"
    assert cli._describe_source(["--data", "p.csv"]) == "data:p.csv"
    assert cli._describe_source([]) is None


# ── language and commands are one input, not two modes ────────────────────
#
# The prompt FORKED: a known verb ran a command, anything else went to the
# agent. So `screen my sp100 universe --data prices.csv` was a question with
# noise on the end, and there was no way to say a sentence AND point at data in
# the same breath. Every plain-English request therefore died at preflight, for
# want of data the sentence had already named.


def test_flags_come_out_of_any_line_and_the_prose_survives():
    prose, flags = cli._extract_flags("screen my book for high rsi --universe sp100")
    assert prose == "screen my book for high rsi"
    assert flags == ["--universe", "sp100"]


def test_a_line_with_no_flags_is_untouched():
    prose, flags = cli._extract_flags("what changed since yesterday")
    assert prose == "what changed since yesterday"
    assert flags == []


def test_flags_alone_are_a_complete_instruction():
    prose, flags = cli._extract_flags("--universe sp100")
    assert prose == ""
    assert flags == ["--universe", "sp100"]


def test_a_sentence_containing_dashes_is_not_mangled():
    """Only the three data flags are extracted. A sentence with `--` in it is
    far more likely to be a sentence than a mistyped flag, and stripping tokens
    somebody meant to say is worse than passing one through."""
    prose, flags = cli._extract_flags("compare pre-- and post-- earnings drift")
    assert flags == []
    assert "pre--" in prose


class _Named:
    def __init__(self, names):
        self._names = names

    def universes(self):
        return [{"id": f"u_{n}", "name": n, "symbols": ["AAPL"]} for n in self._names]


def test_a_question_that_names_your_own_universe_finds_it():
    """DETERMINISTIC, NO MODEL CALL. Spending an inference on a substring match
    would be slower, less reliable, and impossible to explain when it went
    wrong."""
    s = _Named(["sp100", "sp500"])
    assert cli._universe_named_in("screen my sp100 universe for high RSI", s) == "sp100"


def test_the_longest_match_wins():
    s = _Named(["sp500", "sp500-ex-financials"])
    assert cli._universe_named_in("deep dive sp500-ex-financials", s) == "sp500-ex-financials"


def test_it_can_only_recognise_a_universe_you_own():
    """Never invents one. It either matches something registered or returns
    nothing, so a question mentioning a name you do not have cannot conjure it."""
    s = _Named(["sp100"])
    assert cli._universe_named_in("screen the russell 2000", s) is None


def test_an_unreachable_portal_is_not_an_error_here():
    """Offline or unauthenticated is a normal state at this point in the line,
    and turning it into a raise would break asking a question without a key."""

    class _Down:
        def universes(self):
            raise RuntimeError("offline")

    assert cli._universe_named_in("screen my sp100 universe", _Down()) is None
