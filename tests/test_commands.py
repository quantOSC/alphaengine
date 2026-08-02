"""One declaration, four renderings, and a guard against the fifth copy.

── WHAT WENT WRONG ────────────────────────────────────────────────────────────

The command list lived in three hand-maintained places and none agreed. `--data`
and `--universe` shipped and appeared in one of the three. `demo` appeared in
two. The docs page on the site described the library in fourteen sections and
listed **no CLI commands at all** — so the answer to "how do I actually use
this" was not written down anywhere.

Every copy of a list looks authoritative on its own. A reader who finds the
stale one concludes the feature does not exist, which is worse than finding
nothing.

These tests hold the renderings to the declaration, and `--check` in CI holds
the README to it too.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from alphaengine import cli
from alphaengine.commands import COMMANDS, DATA_DOORS, FLAGS, GROUPS, by_verb, in_group

ROOT = Path(__file__).resolve().parents[1]


# ── the declaration is coherent ────────────────────────────────────────────


def test_every_command_lands_in_a_declared_group():
    """A group nobody declared renders nowhere, so the verb silently vanishes
    from the directory while still existing in the parser."""
    groups = {g for g, _ in GROUPS}
    for c in COMMANDS:
        assert c.group in groups, f"{c.verb} is in group {c.group!r}, which nothing renders"


def test_every_command_says_what_it_is_for_in_one_line():
    for c in COMMANDS:
        assert c.purpose, f"{c.verb} has no purpose line"
        assert "\n" not in c.purpose, f"{c.verb}'s purpose is a paragraph; the directory shows one line"
        assert len(c.purpose) <= 80, f"{c.verb}'s purpose is {len(c.purpose)} chars and will wrap"


def test_every_scope_is_one_the_renderers_understand():
    for c in COMMANDS:
        assert c.scope in ("cli", "repl", "both"), f"{c.verb} has scope {c.scope!r}"


def test_every_named_flag_exists():
    """A command referencing a flag nobody declared raises a KeyError inside the
    renderer — at the moment somebody asks for help, which is the worst time."""
    for c in COMMANDS:
        for f in c.flags:
            assert f in FLAGS, f"{c.verb} names flag {f!r}, which is not declared"


def test_examples_are_real_rather_than_placeholders():
    """An example with a placeholder nobody can resolve teaches nothing. The
    shell ones must actually start with the binary."""
    for c in COMMANDS:
        for ex in c.examples:
            assert "<your" not in ex, f"{c.verb}: {ex!r} is a placeholder"
            if c.scope == "cli":
                assert ex.startswith("alphaengine"), f"{c.verb}: {ex!r} is not runnable in a shell"


# ── the declaration matches the parser, in both directions ─────────────────


def test_every_subcommand_the_parser_accepts_is_documented():
    """THE DRIFT THAT ACTUALLY HAPPENED, in the direction that hurts: a command
    exists, works, and appears in no list, so nobody finds it."""
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None) and "run" in (a.choices or {}))
    documented = {c.verb for c in COMMANDS}
    for verb in sub.choices:
        assert verb in documented, f"`alphaengine {verb}` works and is in no command list"


def test_every_documented_shell_command_the_parser_accepts():
    """And the other direction: a directory entry for something that does not
    run is worse than a missing one, because somebody types it."""
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if getattr(a, "choices", None) and "run" in (a.choices or {}))
    for c in COMMANDS:
        if c.scope == "cli" or (c.scope == "both" and c.verb.isalpha()):
            assert c.verb in sub.choices, f"the directory lists `{c.verb}` and the parser refuses it"


def test_every_flag_the_parser_accepts_is_documented():
    parser = cli.build_parser()
    declared = {f.name for f in FLAGS.values()}
    seen = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt.startswith("--") and opt != "--help":
                seen.add(opt)
    for sub in (a for a in parser._actions if getattr(a, "choices", None)):
        for p in (sub.choices or {}).values():
            for action in p._actions:
                for opt in action.option_strings:
                    if opt.startswith("--") and opt != "--help":
                        seen.add(opt)
    assert seen <= declared, f"undocumented flags: {sorted(seen - declared)}"


# ── the renderings ─────────────────────────────────────────────────────────


def test_the_directory_prints_every_group_and_every_verb(capsys):
    import argparse

    cli.cmd_commands(argparse.Namespace(verb=None))
    out = capsys.readouterr().out
    for _, title in GROUPS:
        assert title in out
    for c in COMMANDS:
        assert c.verb in out, f"{c.verb} is missing from the directory"
    # The doors are flags rather than verbs, and they are what people are
    # actually hunting for when they open a directory at all.
    for door, _ in DATA_DOORS:
        assert door.split()[0] in out


def test_expanding_one_command_shows_its_flags_and_examples(capsys):
    import argparse

    cli.cmd_commands(argparse.Namespace(verb="run"))
    out = capsys.readouterr().out
    assert "--universe" in out and "--project" in out
    assert "alphaengine run screen_universe" in out
    # The distinction a reader cannot infer from a signature.
    assert "SCRIPTED" in out


def test_an_unknown_verb_suggests_rather_than_shrugs(capsys):
    import argparse

    code = cli.cmd_commands(argparse.Namespace(verb="wrokflows"))
    out = capsys.readouterr().out
    assert code == 2
    assert "commands" in out


def test_the_short_help_covers_every_session_verb():
    text = cli._help_text()
    for c in COMMANDS:
        if c.scope in ("both", "repl"):
            assert c.verb in text, f"{c.verb} works in a session and is not in `help`"


def test_nothing_the_cli_prints_carries_an_em_dash():
    """The console degrades `·` and `→` to ASCII because cp1252 mangles them.
    An em dash in a printed string has the same problem and no such handling —
    observed as `?` on a stock Windows terminal."""
    for c in COMMANDS:
        for text in (c.purpose, c.body, *c.examples):
            assert "—" not in text, f"{c.verb} prints an em dash"
    for f in FLAGS.values():
        assert "—" not in f.purpose


# ── the README cannot go stale ─────────────────────────────────────────────


def test_the_readme_command_block_is_current():
    """THE FIFTH COPY GUARD. Run in CI too. A verb added to `commands.py` and
    forgotten in the README fails a build rather than misleading somebody for a
    month."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_docs.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_readme_leads_with_the_os_and_not_the_library():
    """The ordering IS the confusion this change exists to fix: somebody
    arriving to use the OS met `sweep()` before they met a command."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.index("alphaengine demo") < text.index("from alphaengine import sweep")
    assert text.index("## What it answers") < text.index("## Writing your own")


@pytest.mark.parametrize("verb", ["run", "demo", "workflows", "commands"])
def test_the_readme_names_every_headline_command(verb):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"`{verb}" in text or f"alphaengine {verb}" in text


# ── helpers used by the directory ──────────────────────────────────────────


def test_by_verb_and_in_group_agree_with_the_declaration():
    assert by_verb("run") is not None
    assert by_verb("nope") is None
    assert {c.verb for c in in_group("data")} <= {c.verb for c in COMMANDS}
    assert all(c.scope in ("both", "repl") for c in in_group("data", scope="repl"))
