"""The study: what was tried, what came back, and on what.

The artifact that crosses the boundary between the person who ran the research
and the person who has to decide on it. It carries the claim WITH its
qualifiers, the trial count, the deflation, the data identity, because the
documented failure mode is a headline Sharpe outliving its caveats into a deck
where the footnotes did not follow.

Writes to disk by default. No account, no upload, no network. The hosted
platform makes a study shareable and durable across a firm; it is not what makes
one exist.

FORMAT
    JSON with an explicit `schema_version`, because a study written today has to
    parse in two years. JSON rather than a binary format so a PM can open it in
    a text editor and a reviewer can diff two of them, an artifact that needs
    our software to read is a worse artifact.

    Protobuf is the intended normative definition once the C++ path lands: the
    same schema then generates a writer for a C++ engine and a reader for our
    Python, instead of each side maintaining a bespoke parser. The JSON here is
    already field-stable, so that is an addition rather than a migration.

WHAT IS NOT IN IT
    No price series, no returns matrix, no parameter values unless the author
    opted in. A study is derived facts and references, the inputs stay with
    whoever owns them.
"""

from .schema import SCHEMA_VERSION, Study, load, save

__all__ = [
    "report",
    "ReportError","Study", "save", "load", "SCHEMA_VERSION"]


def __getattr__(name: str):  # PEP 562
    """`report` and `ReportError` load on first use.

    Eager import would pull urllib into `import alphaengine`, which the offline
    guarantee does not permit. Lazy keeps `from alphaengine.study import report`
    working for anyone who wants it, at no cost to everyone who does not.
    """
    if name in ("report", "ReportError"):
        # importlib, not `from . import report`: the submodule shares its name
        # with the function it exports, so the plain form re-enters this hook
        # and recurses.
        import importlib

        mod = importlib.import_module(".report", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
